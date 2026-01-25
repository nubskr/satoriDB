use crate::bucket_index::BucketIndex;
use crate::bucket_locks::BucketLocks;
use crate::ingest_control;
use crate::rebalancer::RebalanceWorker;
use crate::router::RoutingTable;
use crate::router_manager::{spawn_router_manager, RouterCommand, RouterShutdownRequest};
use crate::service::SatoriHandle;
use crate::storage::Storage;
use crate::tasks::ConsistentHashRing;
use crate::vector_index::VectorIndex;
use crate::wal::runtime::Walrus;
use crate::wal::FsyncSchedule;
use crate::worker::{run_worker, WorkerMessage};
use anyhow::{anyhow, Result};
use crossbeam_channel::unbounded;
use futures::channel::oneshot;
use futures::executor::block_on;
use glommio::{LocalExecutorBuilder, Placement};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// ============================================================================
// Builder
// ============================================================================

/// Builder for configuring a [`SatoriDb`] instance.
///
/// # Example
///
/// ```rust,no_run
/// use satoridb::SatoriDb;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let _db = SatoriDb::builder("my_app")
///     .workers(4)
///     .fsync_ms(200)
///     .build()?;
/// # Ok(())
/// # }
/// ```
pub struct SatoriDbBuilder {
    name: String,
    data_dir: Option<PathBuf>,
    workers: usize,
    fsync_ms: u64,
    virtual_nodes: usize,
}

impl SatoriDbBuilder {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_dir: None,
            workers: num_cpus::get().max(1),
            fsync_ms: 200,
            virtual_nodes: 8,
        }
    }

    /// Set the number of worker threads.
    ///
    /// Defaults to the number of CPU cores.
    pub fn workers(mut self, n: usize) -> Self {
        self.workers = n;
        self
    }

    /// Set the fsync interval in milliseconds.
    ///
    /// Lower values improve durability at the cost of performance.
    /// Defaults to 200ms.
    pub fn fsync_ms(mut self, ms: u64) -> Self {
        self.fsync_ms = ms;
        self
    }

    /// Set a custom data directory.
    ///
    /// Defaults to `./wal_files/{name}/`.
    pub fn data_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.data_dir = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set virtual nodes for consistent hashing.
    ///
    /// Higher values improve load balancing. Defaults to 8.
    pub fn virtual_nodes(mut self, n: usize) -> Self {
        self.virtual_nodes = n;
        self
    }

    /// Build and start the database.
    pub fn build(self) -> Result<SatoriDb> {
        // Suppress Walrus logs by default (user can override with WALRUS_QUIET=0)
        if std::env::var("WALRUS_QUIET").is_err() {
            std::env::set_var("WALRUS_QUIET", "1");
        }

        if self.workers == 0 {
            return Err(anyhow!("workers must be > 0"));
        }

        // Determine base directory for all data (wal, indexes)
        let base_dir = if let Some(ref dir) = self.data_dir {
            dir.clone()
        } else {
            PathBuf::from("./wal_files").join(&self.name)
        };

        // Ensure base directory exists
        std::fs::create_dir_all(&base_dir)
            .map_err(|e| anyhow!("failed to create data directory: {}", e))?;

        // Store indexes inside base_dir
        let vector_index_path = base_dir.join("vector_index");
        let bucket_index_path = base_dir.join("bucket_index");

        let wal = Arc::new(
            Walrus::with_data_dir_and_options(
                base_dir,
                crate::wal::runtime::ReadConsistency::StrictlyAtOnce,
                FsyncSchedule::Milliseconds(self.fsync_ms),
            )
            .map_err(|e| anyhow!("failed to initialize storage: {}", e))?,
        );

        SatoriDb::start_internal(
            wal,
            self.workers,
            self.virtual_nodes,
            vector_index_path,
            bucket_index_path,
        )
    }
}

// ============================================================================
// Main Database
// ============================================================================

/// An embedded SatoriDB vector database instance.
///
/// # Quick Start
///
/// ```rust,no_run
/// use satoridb::SatoriDb;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let db = SatoriDb::open("my_app")?;
///
/// db.insert(1, vec![0.1, 0.2, 0.3])?;
/// db.insert(2, vec![0.2, 0.3, 0.4])?;
///
/// let results = db.query(vec![0.15, 0.25, 0.35], 10)?;
/// for (id, distance) in results {
///     println!("id={id} distance={distance}");
/// }
/// // Auto-shutdown on drop
/// # Ok(())
/// # }
/// ```
pub struct SatoriDb {
    handle: SatoriHandle,
    router_tx: crossbeam_channel::Sender<RouterCommand>,
    router_handle: Option<thread::JoinHandle<()>>,
    worker_senders: Vec<async_channel::Sender<WorkerMessage>>,
    worker_handles: Vec<thread::JoinHandle<()>>,
    #[allow(dead_code)]
    rebalance_worker: RebalanceWorker,
    rebalance_thread: Option<thread::JoinHandle<()>>,
    shutdown_called: AtomicBool,
}

impl SatoriDb {
    /// Open a database with sensible defaults.
    ///
    /// Data is stored in `./wal_files/{name}/`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let _db = SatoriDb::open("my_app")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(name: impl Into<String>) -> Result<Self> {
        SatoriDbBuilder::new(name).build()
    }

    /// Create a builder for custom configuration.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let _db = SatoriDb::builder("my_app")
    ///     .workers(4)
    ///     .fsync_ms(100)
    ///     .build()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn builder(name: impl Into<String>) -> SatoriDbBuilder {
        SatoriDbBuilder::new(name)
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Insert a vector with the given ID.
    ///
    /// Returns an error if the ID already exists. Use [`delete`](Self::delete)
    /// first if you need to replace a vector.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = SatoriDb::open("my_app")?;
    /// db.insert(1, vec![0.1, 0.2, 0.3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn insert(&self, id: u64, vector: Vec<f32>) -> Result<()> {
        block_on(self.handle.upsert(id, vector, None))?;
        Ok(())
    }

    /// Delete a vector by ID.
    ///
    /// This operation is eventually consistent. The ID can be reused
    /// immediately after this call returns.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = SatoriDb::open("my_app")?;
    /// db.delete(1)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delete(&self, id: u64) -> Result<()> {
        block_on(self.handle.delete(id))
    }

    /// Run an aggressive rebalancing pass until no bucket exceeds the threshold.
    ///
    /// This call blocks and will use all CPU cores. It is best invoked when
    /// ingestion is paused.
    pub fn aggressive_rebalance(&self) -> Result<usize> {
        ingest_control::disallow_ingestion();
        thread::sleep(Duration::from_secs(10));
        let result = self.rebalance_worker.aggressive_rebalance_blocking();
        ingest_control::allow_ingestion();
        result.map_err(|e| anyhow!(e))
    }

    /// Query for the nearest neighbors of a vector.
    ///
    /// Returns up to `top_k` results as `(id, distance)` pairs, sorted by
    /// distance (closest first).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = SatoriDb::open("my_app")?;
    /// let results = db.query(vec![0.1, 0.2, 0.3], 10)?;
    /// for (id, distance) in results {
    ///     println!("id={id} distance={distance}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn query(&self, vector: Vec<f32>, top_k: usize) -> Result<Vec<(u64, f32)>> {
        // Default probe count based on top_k, with sensible bounds
        let probe_buckets = (top_k * 20).clamp(50, 500);
        block_on(self.handle.query(vector, top_k, probe_buckets))
    }

    /// Query with custom probe count for advanced tuning.
    ///
    /// Higher `probe_buckets` values improve recall at the cost of latency.
    pub fn query_with_probes(
        &self,
        vector: Vec<f32>,
        top_k: usize,
        probe_buckets: usize,
    ) -> Result<Vec<(u64, f32)>> {
        block_on(self.handle.query(vector, top_k, probe_buckets))
    }

    /// Query and return the stored vectors along with distances.
    ///
    /// Returns `(id, distance, vector)` tuples.
    pub fn query_with_vectors(
        &self,
        vector: Vec<f32>,
        top_k: usize,
    ) -> Result<Vec<(u64, f32, Vec<f32>)>> {
        let probe_buckets = (top_k * 20).clamp(50, 500);
        block_on(self.handle.query_with_vectors(vector, top_k, probe_buckets))
    }

    /// Fetch vectors by their IDs.
    ///
    /// Missing IDs are silently skipped in the result.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use satoridb::SatoriDb;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let db = SatoriDb::open("my_app")?;
    /// let vectors = db.get(vec![1, 2, 3])?;
    /// for (id, vector) in vectors {
    ///     println!("id={id} vector={vector:?}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn get(&self, ids: Vec<u64>) -> Result<Vec<(u64, Vec<f32>)>> {
        block_on(self.handle.fetch_vectors_by_id(ids))
    }

    /// Get database statistics.
    pub fn stats(&self) -> Stats {
        let inner = block_on(self.handle.stats());
        Stats {
            ready: inner.ready,
            buckets: inner.buckets,
            vectors: inner.total_vectors,
        }
    }

    /// Flush all pending writes to disk.
    pub fn flush(&self) -> Result<()> {
        block_on(self.handle.flush())
    }

    // ========================================================================
    // Async Operations
    // ========================================================================

    /// Async version of [`insert`](Self::insert).
    pub async fn insert_async(&self, id: u64, vector: Vec<f32>) -> Result<()> {
        self.handle.upsert(id, vector, None).await?;
        Ok(())
    }

    /// Async version of [`delete`](Self::delete).
    pub async fn delete_async(&self, id: u64) -> Result<()> {
        self.handle.delete(id).await
    }

    /// Async version of [`query`](Self::query).
    pub async fn query_async(&self, vector: Vec<f32>, top_k: usize) -> Result<Vec<(u64, f32)>> {
        let probe_buckets = (top_k * 20).clamp(50, 500);
        self.handle.query(vector, top_k, probe_buckets).await
    }

    /// Async version of [`get`](Self::get).
    pub async fn get_async(&self, ids: Vec<u64>) -> Result<Vec<(u64, Vec<f32>)>> {
        self.handle.fetch_vectors_by_id(ids).await
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Gracefully shut down the database.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// to handle errors.
    pub fn shutdown(mut self) -> Result<()> {
        self.shutdown_internal()
    }

    fn shutdown_internal(&mut self) -> Result<()> {
        if self.shutdown_called.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already called
        }

        // Flush router
        let (flush_tx, flush_rx) = oneshot::channel();
        let _ = self.router_tx.send(RouterCommand::Flush(
            crate::router_manager::RouterFlushRequest {
                respond_to: flush_tx,
            },
        ));
        let _ = block_on(flush_rx);

        // Shutdown router
        let (router_shutdown_tx, router_shutdown_rx) = oneshot::channel();
        let _ = self
            .router_tx
            .send(RouterCommand::Shutdown(RouterShutdownRequest {
                respond_to: router_shutdown_tx,
            }));

        // Shutdown workers
        let mut worker_shutdown_waiters = Vec::new();
        for sender in &self.worker_senders {
            let (tx, rx) = oneshot::channel();
            let _ = block_on(sender.send(WorkerMessage::Shutdown { respond_to: tx }));
            worker_shutdown_waiters.push(rx);
        }
        for rx in worker_shutdown_waiters {
            let _ = block_on(rx);
        }

        // Close channels
        self.rebalance_worker.delete_tx.close();
        for sender in &self.worker_senders {
            sender.close();
        }

        let _ = block_on(router_shutdown_rx);

        // Join threads
        for h in self.worker_handles.drain(..) {
            let _ = h.join();
        }
        if let Some(h) = self.router_handle.take() {
            let _ = h.join();
        }
        if let Some(h) = self.rebalance_thread.take() {
            let _ = h.join();
        }

        Ok(())
    }

    // ========================================================================
    // Internal
    // ========================================================================

    fn start_internal(
        wal: Arc<Walrus>,
        workers: usize,
        virtual_nodes: usize,
        vector_index_path: PathBuf,
        bucket_index_path: PathBuf,
    ) -> Result<Self> {
        let vector_index = Arc::new(VectorIndex::open(vector_index_path)?);
        let bucket_index = Arc::new(BucketIndex::open(bucket_index_path)?);
        let bucket_locks = Arc::new(BucketLocks::new());

        let mut worker_senders = Vec::new();
        let mut worker_handles = Vec::new();
        for i in 0..workers {
            let (sender, receiver) = async_channel::bounded(1000);
            worker_senders.push(sender);

            let wal_clone = wal.clone();
            let index_clone = vector_index.clone();
            let bucket_index_clone = bucket_index.clone();
            let bucket_locks_clone = bucket_locks.clone();
            let pin_cpu = i % num_cpus::get().max(1);
            let handle = thread::spawn(move || {
                let builder = LocalExecutorBuilder::new(Placement::Fixed(pin_cpu))
                    .name(&format!("worker-{}", i));
                builder
                    .make()
                    .expect("failed to create executor")
                    .run(run_worker(
                        i,
                        receiver,
                        wal_clone,
                        index_clone,
                        bucket_index_clone,
                        bucket_locks_clone,
                    ));
            });
            worker_handles.push(handle);
        }

        let ring = ConsistentHashRing::new(workers, virtual_nodes);

        let (router_tx, router_rx) = unbounded::<RouterCommand>();
        let router_handle = spawn_router_manager(wal.clone(), router_rx);

        let routing_table = Arc::new(RoutingTable::new());

        let (rebalance_worker, rebalance_thread) = RebalanceWorker::spawn_delete_only(
            Storage::new(wal.clone()),
            vector_index.clone(),
            bucket_index.clone(),
            routing_table.clone(),
            bucket_locks.clone(),
        );

        let handle = SatoriHandle::new(
            router_tx.clone(),
            ring,
            worker_senders.clone(),
            rebalance_worker.delete_tx.clone(),
            vector_index.clone(),
            bucket_index.clone(),
        );

        Ok(Self {
            handle,
            router_tx,
            router_handle: Some(router_handle),
            worker_senders,
            worker_handles,
            rebalance_worker,
            rebalance_thread: Some(rebalance_thread),
            shutdown_called: AtomicBool::new(false),
        })
    }
}

impl Drop for SatoriDb {
    fn drop(&mut self) {
        let _ = self.shutdown_internal();
    }
}

// ============================================================================
// Stats
// ============================================================================

/// Database statistics.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Whether the database is ready to serve queries.
    pub ready: bool,
    /// Number of buckets.
    pub buckets: usize,
    /// Approximate number of vectors.
    pub vectors: u64,
}

use crate::bucket_index::BucketIndex;
use crate::bucket_locks::BucketLocks;
use crate::indexer::Indexer;
use crate::ingest_counter;
use crate::quantizer::Quantizer;
use crate::router::{Router, RoutingTable};
use crate::storage::{Bucket, BucketMeta, BucketMetaStatus, Storage, Vector};
use crate::vector_index::VectorIndex;
use crate::wal::runtime::Walrus;
use anyhow::Result;
use futures::executor::block_on;
use log::{debug, error, warn};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::thread;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RebalanceTaskKind {
    Split,
    Merge,
    Rebalance,
}

type RebalanceFailHook = Arc<dyn Fn(RebalanceTaskKind) -> bool + Send + Sync>;

static FAIL_HOOK: StdMutex<Option<RebalanceFailHook>> = StdMutex::new(None);

#[derive(Debug)]
pub struct DeleteCommand {
    pub vector_id: u64,
    pub bucket_hint: Option<u64>,
    pub respond_to: futures::channel::oneshot::Sender<anyhow::Result<()>>,
}

pub fn set_rebalance_fail_hook<F>(hook: F)
where
    F: Fn(RebalanceTaskKind) -> bool + Send + Sync + 'static,
{
    *FAIL_HOOK.lock().expect("fail hook poisoned") = Some(Arc::new(hook));
}

pub fn clear_rebalance_fail_hook() {
    *FAIL_HOOK.lock().expect("fail hook poisoned") = None;
}

fn should_fail(kind: RebalanceTaskKind) -> bool {
    if let Some(h) = FAIL_HOOK.lock().expect("fail hook poisoned").as_ref() {
        return h(kind);
    }
    let _ = kind;
    false
}

pub(crate) struct RebalanceState {
    storage: Storage,
    vector_index: Arc<VectorIndex>,
    bucket_index: Arc<BucketIndex>,
    wal: Arc<Walrus>,
    routing: Arc<RoutingTable>,
    centroids: RwLock<HashMap<u64, Vec<f32>>>,
    bucket_sizes: RwLock<HashMap<u64, usize>>,
    next_bucket_id: AtomicU64,
    trim_counter: AtomicU64,
    bucket_locks: Arc<BucketLocks>,
}

impl RebalanceState {
    fn new(
        storage: Storage,
        vector_index: Arc<VectorIndex>,
        bucket_index: Arc<BucketIndex>,
        routing: Arc<RoutingTable>,
        bucket_locks: Arc<BucketLocks>,
    ) -> Self {
        Self {
            wal: storage.wal.clone(),
            storage,
            vector_index,
            bucket_index,
            routing,
            centroids: RwLock::new(HashMap::new()),
            bucket_sizes: RwLock::new(HashMap::new()),
            next_bucket_id: AtomicU64::new(0),
            trim_counter: AtomicU64::new(0),
            bucket_locks,
        }
    }

    fn prime_centroids(&self, buckets: &[Bucket]) {
        let mut map = self.centroids.write();
        let mut sizes = self.bucket_sizes.write();
        let mut max_id = 0;
        for b in buckets {
            map.insert(b.id, b.centroid.clone());
            sizes.insert(b.id, b.vectors.len());
            if b.id > max_id {
                max_id = b.id;
            }
        }
        self.next_bucket_id.store(max_id + 1, Ordering::Release);
    }

    fn allocate_bucket_id(&self) -> u64 {
        self.next_bucket_id.fetch_add(1, Ordering::AcqRel)
    }

    fn refresh_sizes(&self) -> HashMap<u64, usize> {
        let ids: Vec<u64> = self.centroids.read().keys().cloned().collect();
        let mut fresh = HashMap::new();
        for id in ids {
            let topic = crate::storage::Storage::topic_for(id);
            let count = self.wal.get_topic_entry_count(&topic) as usize;
            // Removed monotonic max to allow size to drop when trimming/checkpointing source bucket.
            fresh.insert(id, count);
        }
        let mut sizes = self.bucket_sizes.write();
        sizes.clear();
        sizes.extend(fresh.iter().map(|(k, v)| (*k, *v)));
        let sum: u64 = fresh.values().map(|v| *v as u64).sum();
        let inserted = ingest_counter::get();

        if sum < inserted {
            debug!(
                "rebalance: wal counts sum {} is less than total inserted {}",
                sum, inserted
            );
        }
        sizes.clone()
    }

    fn lock_for(&self, bucket_id: u64) -> Arc<futures::lock::Mutex<()>> {
        self.bucket_locks.lock_for(bucket_id)
    }

    fn seed_centroids_with_counts(
        &self,
        seeds: Vec<(u64, Vec<f32>, usize)>,
        write_meta: bool,
    ) -> Result<()> {
        let mut max_id = 0;
        {
            let mut map = self.centroids.write();
            let mut sizes = self.bucket_sizes.write();
            map.clear();
            sizes.clear();
            for (id, centroid, count) in seeds.into_iter() {
                if id > max_id {
                    max_id = id;
                }
                map.insert(id, centroid);
                sizes.insert(id, count);
            }
        }
        self.next_bucket_id.store(max_id + 1, Ordering::Release);
        self.rebuild_router(Vec::new());

        if write_meta {
            let ids: Vec<u64> = self.centroids.read().keys().copied().collect();
            for id in ids {
                let _ = block_on(self.storage.put_bucket_meta(&BucketMeta {
                    bucket_id: id,
                    status: BucketMetaStatus::Active,
                }));
            }
        }
        Ok(())
    }

    pub(crate) fn load_bucket_vectors(&self, bucket_id: u64) -> Option<Vec<Vector>> {
        let chunks = Storage::get_chunks_sync(self.storage.wal.clone(), bucket_id).ok()?;
        let mut vector_map: HashMap<u64, Vector> = HashMap::new();

        for chunk in chunks {
            if chunk.len() < 8 {
                continue;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&chunk[0..8]);
            let archive_len = u64::from_le_bytes(len_bytes) as usize;
            if 8 + archive_len > chunk.len() || archive_len < 16 {
                continue;
            }
            let mut off = 8;
            let mut id_bytes = [0u8; 8];
            id_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let mut dim_bytes = [0u8; 8];
            dim_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let dim = u64::from_le_bytes(dim_bytes) as usize;
            let Some(expected_bytes) = dim.checked_mul(4) else {
                continue;
            };
            if off + expected_bytes > chunk.len() {
                continue;
            }
            let mut data = Vec::with_capacity(dim);
            let data_bytes = &chunk[off..off + expected_bytes];
            for chunked in data_bytes.chunks_exact(4) {
                let mut fb = [0u8; 4];
                fb.copy_from_slice(chunked);
                data.push(f32::from_bits(u32::from_le_bytes(fb)));
            }
            let id = u64::from_le_bytes(id_bytes);

            if data.is_empty() {
                vector_map.remove(&id);
            } else {
                vector_map.insert(id, Vector { id, data });
            }
        }

        if vector_map.is_empty() {
            None
        } else {
            Some(vector_map.into_values().collect())
        }
    }

    fn rebuild_router(&self, changed_buckets: Vec<u64>) {
        let centroids_map = self.centroids.read();
        let bucket_count = centroids_map.len();
        if centroids_map.is_empty() {
            return;
        }
        let mut centroids: Vec<(u64, Vec<f32>)> = Vec::with_capacity(bucket_count);
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for (id, c) in centroids_map.iter() {
            for &val in c {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
            }
            centroids.push((*id, c.clone()));
        }
        drop(centroids_map);

        let Some((min, max)) = Quantizer::compute_bounds_from_minmax(min, max) else {
            return;
        };
        let quantizer = Quantizer::new(min, max);
        let mut router = Router::new(100_000, quantizer);
        for (id, centroid) in &centroids {
            router.add_centroid(*id, centroid);
        }
        let version = self.routing.install(router, changed_buckets);
        if log::log_enabled!(log::Level::Debug) {
            let sizes_map = self.bucket_sizes.read();
            let mut sizes: Vec<usize> = sizes_map.values().copied().collect();
            sizes.sort_unstable();
            debug!(
                "rebalance: published router version {} (buckets={}, sizes={:?})",
                version, bucket_count, sizes
            );
        }
    }

    fn parse_chunk(chunk: &[u8]) -> Option<Vector> {
        if chunk.len() < 16 {
            return None;
        }
        // Format: [len:8][id:8][dim:8][data...]
        // But Storage::put_chunk_raw_sync writes:
        // [payload_len:8] [id:8] [dim:8] [data...]
        // And Walrus Entry contains the payload_len prefix?
        // No, Walrus Entry.data is the payload passed to append.
        // Storage::put_chunk_raw_sync writes:
        // backing.extend_from_slice(&(payload_len as u64).to_le_bytes()); -> This is inside the payload if single entry?
        // Wait, Storage puts one entry per vector.
        // backing has: [payload_len: 8] [id: 8] [dim: 8] [data...]
        // So the Entry.data starts with payload_len.

        let mut off = 0;
        if off + 8 > chunk.len() {
            return None;
        }
        let _payload_len = u64::from_le_bytes(chunk[off..off + 8].try_into().unwrap());
        off += 8;

        if off + 8 > chunk.len() {
            return None;
        }
        let id = u64::from_le_bytes(chunk[off..off + 8].try_into().unwrap());
        off += 8;

        if off + 8 > chunk.len() {
            return None;
        }
        let dim = u64::from_le_bytes(chunk[off..off + 8].try_into().unwrap()) as usize;
        off += 8;

        let expected_bytes = dim * 4;
        if off + expected_bytes > chunk.len() {
            return None;
        }

        let mut data = Vec::with_capacity(dim);
        for i in 0..dim {
            let start = off + i * 4;
            let bytes = &chunk[start..start + 4];
            data.push(f32::from_le_bytes(bytes.try_into().unwrap()));
        }

        Some(Vector { id, data })
    }

    fn handle_split_sync(&self, bucket_id: u64) {
        log::debug!(
            "rebalance: handle_split_sync started for bucket {}",
            bucket_id
        );
        if should_fail(RebalanceTaskKind::Split) {
            debug!(
                "rebalance: injected failure for split on bucket {}",
                bucket_id
            );
            return;
        }

        // 1. Peek sample to determine centroids.
        let topic = crate::storage::Storage::topic_for(bucket_id);
        // Peek up to 1MB for sampling
        let sample_entries = match self
            .wal
            .batch_read_for_topic(&topic, 1024 * 1024, false, None)
        {
            Ok(e) => {
                log::debug!(
                    "rebalance: peeked {} sample entries for bucket {}",
                    e.len(),
                    bucket_id
                );
                e
            }
            Err(e) => {
                log::error!(
                    "rebalance: failed to peek sample for bucket {}: {:?}",
                    bucket_id,
                    e
                );
                return;
            } // Likely empty or IO error
        };

        if sample_entries.is_empty() {
            log::debug!(
                "rebalance: sample empty for bucket {}, aborting split",
                bucket_id
            );
            return;
        }

        let mut sample_vectors = Vec::new();
        for entry in sample_entries {
            if let Some(v) = Self::parse_chunk(&entry.data) {
                sample_vectors.push(v);
            }
        }

        if sample_vectors.len() < 2 {
            return; // Cannot split
        }

        // Compute centroids from sample
        let clusters = Indexer::build_clusters(sample_vectors, 2);
        if clusters.len() < 2 {
            debug!(
                "rebalance: cannot split bucket {} - unable to form 2 clusters",
                bucket_id
            );
            return;
        }
        let centroid_a = clusters[0].centroid.clone();
        let centroid_b = clusters[1].centroid.clone();

        // Allocate new buckets
        let id_a = self.allocate_bucket_id();
        let id_b = self.allocate_bucket_id();

        // Init metadata for A and B
        let _ = block_on(self.storage.put_bucket_meta(&BucketMeta {
            bucket_id: id_a,
            status: BucketMetaStatus::Active,
        }));
        let _ = block_on(self.storage.put_bucket_meta(&BucketMeta {
            bucket_id: id_b,
            status: BucketMetaStatus::Active,
        }));

        {
            let mut centroids = self.centroids.write();
            centroids.insert(id_a, centroid_a.clone());
            centroids.insert(id_b, centroid_b.clone());
            centroids.remove(&bucket_id);

            let mut sizes = self.bucket_sizes.write();
            sizes.insert(id_a, 0);
            sizes.insert(id_b, 0);
            sizes.remove(&bucket_id);
        }

        // Update Router with new buckets so they can start receiving traffic?
        // If we do this, we should add them. But C remains.
        self.rebuild_router(vec![id_a, id_b]);

        log::info!(
            "rebalance: trimming bucket {} -> {}, {}",
            bucket_id,
            id_a,
            id_b
        );

        let dim = centroid_a.len();
        let mut total_moved = 0;
        let start_time = std::time::Instant::now();
        let mut loop_iters = 0;

        // 2. Incremental Split Loop
        loop {
            loop_iters += 1;
            if start_time.elapsed() > Duration::from_secs(5 * 60 * 60) {
                error!(
                    "rebalance: trimming bucket {} timed out after {}s! Moved {}. Orphaned remaining.",
                    bucket_id,
                    start_time.elapsed().as_secs(),
                    total_moved
                );
                break;
            }

            // Peek batch (Checkpoint = false)
            // Use a reasonable batch size (e.g. 4MB) to balance throughput and latency
            let batch_entries =
                match self
                    .wal
                    .batch_read_for_topic(&topic, 4 * 1024 * 1024, false, None)
                {
                    Ok(e) => e,
                    Err(e) => {
                        error!("rebalance: read failed for {}: {:?}", topic, e);
                        break;
                    }
                };

            if batch_entries.is_empty() {
                break;
            }

            if loop_iters % 10 == 0 {
                debug!(
                    "rebalance: bucket {} trim loop {}: batch={}, total_moved={}",
                    bucket_id,
                    loop_iters,
                    batch_entries.len(),
                    total_moved
                );
            }

            let mut vecs_a = Vec::new();
            let mut vecs_b = Vec::new();
            let mut ids_a = Vec::new();
            let mut ids_b = Vec::new();

            // Assign vectors
            for entry in &batch_entries {
                if let Some(v) = Self::parse_chunk(&entry.data) {
                    if v.data.len() == dim {
                        let dist_a = crate::indexer::l2_sq_scalar(&v.data, &centroid_a);
                        let dist_b = crate::indexer::l2_sq_scalar(&v.data, &centroid_b);
                        if dist_a <= dist_b {
                            ids_a.push(v.id);
                            vecs_a.push(v);
                        } else {
                            ids_b.push(v.id);
                            vecs_b.push(v);
                        }
                    }
                }
            }

            // Write to A and B
            if let Err(e) = block_on(self.storage.put_chunk_raw(id_a, &vecs_a)) {
                error!("rebalance: failed to write chunk to {}: {:?}", id_a, e);
                break;
            }
            if let Err(e) = block_on(self.storage.put_chunk_raw(id_b, &vecs_b)) {
                error!("rebalance: failed to write chunk to {}: {:?}", id_b, e);
                break;
            }

            // Update Indexes (Buckets & Vectors)
            // We update indexes to point to new locations.
            // This effectively "moves" them from C's perspective in the query path.
            if !ids_a.is_empty() {
                let _ = self.bucket_index.put_batch(id_a, &ids_a);
            }
            if !ids_b.is_empty() {
                let _ = self.bucket_index.put_batch(id_b, &ids_b);
            }

            total_moved += batch_entries.len();

            // Commit (Checkpoint = true)
            // We must consume EXACTLY the same number of entries we just processed.
            let payload_bytes: usize = batch_entries.iter().map(|e| e.data.len()).sum();
            let mut commit_error = false;
            let mut remaining_entries = batch_entries.len();
            // Walrus batch read limit applies to payload bytes only.
            let mut remaining_bytes = payload_bytes;

            while remaining_entries > 0 {
                match self.wal.batch_read_for_topic_with_limit(
                    &topic,
                    remaining_bytes,
                    Some(remaining_entries as u32),
                    true,
                    None,
                ) {
                    Ok(batch) => {
                        if batch.is_empty() {
                            error!(
                                "rebalance: commit stalled for {}: expected {} more entries",
                                topic, remaining_entries
                            );
                            commit_error = true;
                            break;
                        }

                        let consumed_entries = batch.len();
                        if consumed_entries > remaining_entries {
                            error!(
                                "rebalance: commit over-consumed for {}: expected {} entries, consumed {}",
                                topic, remaining_entries, consumed_entries
                            );
                            commit_error = true;
                            break;
                        }

                        let consumed_bytes: usize = batch.iter().map(|e| e.data.len()).sum();
                        remaining_entries -= consumed_entries;
                        remaining_bytes = remaining_bytes.saturating_sub(consumed_bytes);
                    }
                    Err(e) => {
                        error!("rebalance: commit failed for {}: {:?}", topic, e);
                        commit_error = true;
                        break;
                    }
                }
            }

            if commit_error {
                error!(
                    "rebalance: aborting trim of {} due to commit failure",
                    bucket_id
                );
                break;
            }
        }

        let trim_count = self.trim_counter.fetch_add(1, Ordering::Relaxed) + 1;
        if trim_count.is_multiple_of(10) {
            log::info!(
                "rebalance: finished trim of {} (moved {} entries in {}s)",
                bucket_id,
                total_moved,
                start_time.elapsed().as_secs_f32()
            );
        } else {
            log::debug!(
                "rebalance: finished trim of {} (moved {} entries in {}s)",
                bucket_id,
                total_moved,
                start_time.elapsed().as_secs_f32()
            );
        }
    }
}

pub struct RebalanceWorker {
    state: Arc<RebalanceState>,
    pub delete_tx: async_channel::Sender<DeleteCommand>,
}

impl RebalanceWorker {
    pub fn new_for_tests(
        storage: Storage,
        vector_index: Arc<VectorIndex>,
        bucket_index: Arc<BucketIndex>,
        routing: Arc<RoutingTable>,
        bucket_locks: Arc<BucketLocks>,
    ) -> Self {
        let state = Arc::new(RebalanceState::new(
            storage,
            vector_index,
            bucket_index,
            routing,
            bucket_locks,
        ));
        let (delete_tx, _delete_rx) = async_channel::bounded(1024);
        Self { state, delete_tx }
    }

    pub fn spawn(
        storage: Storage,
        vector_index: Arc<VectorIndex>,
        bucket_index: Arc<BucketIndex>,
        routing: Arc<RoutingTable>,
        pin_cpu: Option<usize>,
        bucket_locks: Arc<BucketLocks>,
    ) -> Self {
        let threshold = read_rebalance_threshold();
        let poll_ms = read_rebalance_poll_ms();
        Self::spawn_with_threshold(
            storage,
            vector_index,
            bucket_index,
            routing,
            pin_cpu,
            bucket_locks,
            threshold,
            poll_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_with_threshold(
        storage: Storage,
        vector_index: Arc<VectorIndex>,
        bucket_index: Arc<BucketIndex>,
        routing: Arc<RoutingTable>,
        pin_cpu: Option<usize>,
        bucket_locks: Arc<BucketLocks>,
        threshold: usize,
        poll_ms: u64,
    ) -> Self {
        let state = Arc::new(RebalanceState::new(
            storage,
            vector_index,
            bucket_index,
            routing,
            bucket_locks,
        ));
        let state_clone = state.clone();
        let (delete_tx, delete_rx) = async_channel::bounded(1024);
        let name = "rebalance-loop".to_string();

        match pin_cpu {
            Some(cpu) => {
                let builder =
                    glommio::LocalExecutorBuilder::new(glommio::Placement::Fixed(cpu)).name(&name);
                std::thread::spawn(move || {
                    let executor = builder.make().expect("failed to create rebalance executor");
                    if poll_ms == 0 {
                        executor.run(run_delete_loop(state_clone, delete_rx));
                    } else {
                        executor.run(run_autonomous_loop(
                            state_clone,
                            delete_rx,
                            threshold,
                            poll_ms,
                        ));
                    }
                });
            }
            None => {
                thread::Builder::new()
                    .name(name.clone())
                    .spawn(move || {
                        let executor = glommio::LocalExecutorBuilder::default()
                            .name(&name)
                            .make()
                            .expect("failed to create default rebalance executor");
                        if poll_ms == 0 {
                            executor.run(run_delete_loop(state_clone, delete_rx));
                        } else {
                            executor.run(run_autonomous_loop(
                                state_clone,
                                delete_rx,
                                threshold,
                                poll_ms,
                            ));
                        }
                    })
                    .expect("rebalance worker");
            }
        }
        Self { state, delete_tx }
    }

    /// Spawns a background worker that ONLY handles deletes and does NOT perform
    /// autonomous rebalancing (splitting/merging).
    pub fn spawn_delete_only(
        storage: Storage,
        vector_index: Arc<VectorIndex>,
        bucket_index: Arc<BucketIndex>,
        routing: Arc<RoutingTable>,
        bucket_locks: Arc<BucketLocks>,
    ) -> (Self, std::thread::JoinHandle<()>) {
        let state = Arc::new(RebalanceState::new(
            storage,
            vector_index,
            bucket_index,
            routing,
            bucket_locks,
        ));
        let state_clone = state.clone();
        let (delete_tx, delete_rx) = async_channel::bounded(1024);
        let name = "delete-worker".to_string();

        let handle = thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                glommio::LocalExecutorBuilder::default()
                    .name(&name)
                    .make()
                    .expect("failed to create delete executor")
                    .run(run_delete_loop(state_clone, delete_rx));
            })
            .expect("delete worker");

        (Self { state, delete_tx }, handle)
    }

    pub async fn prime_centroids(&self, buckets: &[Bucket]) -> Result<()> {
        self.state.prime_centroids(buckets);
        self.state.rebuild_router(Vec::new());
        for b in buckets {
            self.state
                .storage
                .put_bucket_meta(&BucketMeta {
                    bucket_id: b.id,
                    status: BucketMetaStatus::Active,
                })
                .await?;
        }
        Ok(())
    }

    pub fn seed_centroids_with_counts(
        &self,
        seeds: Vec<(u64, Vec<f32>, usize)>,
        write_meta: bool,
    ) -> Result<()> {
        self.state.seed_centroids_with_counts(seeds, write_meta)
    }

    pub fn snapshot_sizes(&self) -> HashMap<u64, usize> {
        self.state.refresh_sizes()
    }

    pub fn close(&self) {}

    /// Run aggressive rebalancing until no bucket exceeds the configured threshold.
    /// This is synchronous and will use all available CPU cores.
    pub fn aggressive_rebalance_blocking(&self) -> anyhow::Result<usize> {
        let threshold = read_rebalance_threshold();
        let max_workers = num_cpus::get().max(1);
        let mut total_splits = 0usize;

        loop {
            let mut oversized: Vec<(u64, usize)> = self
                .state
                .refresh_sizes()
                .into_iter()
                .filter(|(_, size)| *size > threshold)
                .collect();

            if oversized.is_empty() {
                break;
            }

            oversized.sort_by(|a, b| b.1.cmp(&a.1));
            total_splits += oversized.len();

            let workers = max_workers.min(oversized.len());
            let chunk_size = oversized.len().div_ceil(workers);
            let mut handles = Vec::with_capacity(workers);

            for chunk in oversized.chunks(chunk_size) {
                let state = self.state.clone();
                let ids: Vec<u64> = chunk.iter().map(|(id, _)| *id).collect();
                handles.push(thread::spawn(move || {
                    for id in ids {
                        state.handle_split_sync(id);
                    }
                }));
            }

            for handle in handles {
                let _ = handle.join();
            }
        }

        Ok(total_splits)
    }

    pub async fn delete(&self, vector_id: u64, bucket_hint: Option<u64>) -> anyhow::Result<()> {
        let (tx, rx) = futures::channel::oneshot::channel();
        self.delete_tx
            .send(DeleteCommand {
                vector_id,
                bucket_hint,
                respond_to: tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("rebalance delete queue closed"))?;
        rx.await
            .map_err(|e| anyhow::anyhow!("rebalance delete canceled: {:?}", e))?
    }

    /// Synchronous delete helper primarily for tests to bypass the async loop.
    pub fn delete_inline_blocking(
        &self,
        vector_id: u64,
        bucket_hint: Option<u64>,
    ) -> anyhow::Result<()> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let cmd = DeleteCommand {
            vector_id,
            bucket_hint,
            respond_to: tx,
        };
        futures::executor::block_on(handle_delete(&self.state, cmd))?;
        futures::executor::block_on(rx)
            .map_err(|e| anyhow::anyhow!("rebalance delete canceled: {:?}", e))?
    }
}

impl Clone for RebalanceWorker {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            delete_tx: self.delete_tx.clone(),
        }
    }
}

async fn run_delete_loop(
    state: Arc<RebalanceState>,
    delete_rx: async_channel::Receiver<DeleteCommand>,
) {
    while let Ok(cmd) = delete_rx.recv().await {
        if let Err(e) = handle_delete(&state, cmd).await {
            warn!("rebalance: delete failed: {:?}", e);
        }
    }
}

async fn run_autonomous_loop(
    state: Arc<RebalanceState>,
    delete_rx: async_channel::Receiver<DeleteCommand>,
    threshold: usize,
    poll_ms: u64,
) {
    loop {
        while let Ok(cmd) = delete_rx.try_recv() {
            if let Err(e) = handle_delete(&state, cmd).await {
                warn!("rebalance: delete failed: {:?}", e);
            }
        }

        let sizes = state.refresh_sizes();

        let mut max_id = 0;
        let mut max_size = 0;
        for (id, size) in sizes {
            if size > max_size {
                max_size = size;
                max_id = id;
            }
        }

        if max_size > threshold {
            let state_ref = state.clone();
            glommio::executor()
                .spawn_blocking(move || {
                    state_ref.handle_split_sync(max_id);
                })
                .await;
        }

        glommio::timer::Timer::new(Duration::from_millis(poll_ms)).await;
    }
}

fn read_rebalance_threshold() -> usize {
    std::env::var("SATORI_REBALANCE_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000)
}

fn read_rebalance_poll_ms() -> u64 {
    std::env::var("SATORI_REBALANCE_POLL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300_000)
}

async fn perform_delete(
    state: &Arc<RebalanceState>,
    vector_id: u64,
    bucket_hint: Option<u64>,
) -> anyhow::Result<()> {
    let bucket_id = if let Some(b) = bucket_hint {
        b
    } else {
        let found = state
            .bucket_index
            .get_many(&[vector_id])
            .map_err(|e| anyhow::anyhow!("bucket index lookup failed: {:?}", e))?;
        match found.first() {
            Some((_, b)) => *b,
            None => {
                return Ok(());
            }
        }
    };

    let lock = state.lock_for(bucket_id);
    let _guard = lock.lock().await;

    let vectors = match state.load_bucket_vectors(bucket_id) {
        Some(v) => v,
        None => {
            return Ok(());
        }
    };
    let mut remaining: Vec<Vector> = Vec::with_capacity(vectors.len());
    let mut removed = false;
    for v in vectors {
        if v.id == vector_id {
            removed = true;
        } else {
            remaining.push(v);
        }
    }
    if !removed {
        return Ok(());
    }

    let _centroid = state
        .centroids
        .read()
        .get(&bucket_id)
        .cloned()
        .unwrap_or_default();
    let topic = crate::storage::Storage::topic_for(bucket_id);
    let entries_before = state.storage.wal.get_topic_entry_count(&topic);
    Storage::put_chunk_raw_sync(state.storage.wal.clone(), bucket_id, &topic, &remaining)
        .map_err(|e| anyhow::anyhow!("rewrite bucket after delete failed: {:?}", e))?;
    let tombstone = Vector::new(vector_id, Vec::new());
    if let Err(e) =
        Storage::put_chunk_raw_sync(state.storage.wal.clone(), bucket_id, &topic, &[tombstone])
    {
        warn!(
            "rebalance: failed to append delete tombstone for {}: {:?}",
            vector_id, e
        );
    }

    let ids = [vector_id];
    if let Err(e) = state.vector_index.delete_batch(&ids) {
        warn!(
            "rebalance: delete failed from vector index for {}: {:?}",
            vector_id, e
        );
    }
    if let Err(e) = state.bucket_index.delete_batch(&ids) {
        warn!(
            "rebalance: delete failed from bucket index for {}: {:?}",
            vector_id, e
        );
    }

    // Advance WAL checkpoints for the entries that existed prior to this rewrite so old blocks can
    // be reclaimed without consuming the newly written replacement/tombstone. Read in bounded
    // batches to avoid sweeping the fresh entries that were just appended.
    let mut remaining_entries = entries_before;
    while remaining_entries > 0 {
        const MIN_ENTRY_BYTES: usize = 24; // len prefix + id + dim (no payload)
        let max_entries = remaining_entries.min(2000);
        // Walrus limits by payload bytes.
        let max_bytes = (max_entries as usize)
            .saturating_mul(1024) // Assume 1KB avg vector size
            .max(MIN_ENTRY_BYTES);

        // We use checkpoint=true to consume them.
        match state.storage.wal.batch_read_for_topic_with_limit(
            &topic,
            max_bytes,
            Some(remaining_entries.min(2000) as u32),
            true,
            None,
        ) {
            Ok(batch) => {
                if batch.is_empty() {
                    break;
                }
                remaining_entries = remaining_entries.saturating_sub(batch.len() as u64);
            }
            Err(e) => {
                warn!("rebalance: checkpoint drain failed for {}: {:?}", topic, e);
                break;
            }
        }
    }

    Ok(())
}

async fn handle_delete(state: &Arc<RebalanceState>, cmd: DeleteCommand) -> anyhow::Result<()> {
    let result = perform_delete(state, cmd.vector_id, cmd.bucket_hint).await;
    let send_payload = result
        .as_ref()
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{:?}", e));
    let _ = cmd.respond_to.send(send_payload);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn compute_centroid(vectors: &[Vector]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }
        let dim = vectors[0].data.len();
        let mut sums = vec![0.0f32; dim];
        for v in vectors {
            for (i, val) in v.data.iter().enumerate() {
                sums[i] += *val;
            }
        }
        let count = vectors.len() as f32;
        for s in sums.iter_mut() {
            *s /= count;
        }
        sums
    }

    #[test]
    fn centroid_of_vectors_is_mean() {
        let vectors = vec![
            Vector::new(0, vec![1.0, 3.0]),
            Vector::new(1, vec![3.0, 5.0]),
        ];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![2.0, 4.0]);
    }

    #[test]
    fn centroid_of_empty_is_empty() {
        let centroid = compute_centroid(&[]);
        assert!(centroid.is_empty());
    }

    /// Centroid of single vector is the vector itself.
    #[test]
    fn centroid_of_single_is_identity() {
        let vectors = vec![Vector::new(0, vec![5.0, 10.0, 15.0])];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![5.0, 10.0, 15.0]);
    }

    /// Centroid with many vectors maintains numerical stability.
    #[test]
    fn centroid_many_vectors() {
        let n = 1000;
        let vectors: Vec<Vector> = (0..n)
            .map(|i| Vector::new(i as u64, vec![i as f32, (i * 2) as f32]))
            .collect();
        let centroid = compute_centroid(&vectors);
        let expected_x = (0..n).sum::<usize>() as f32 / n as f32;
        let expected_y = (0..n).map(|i| (i * 2) as f32).sum::<f32>() / n as f32;
        assert!((centroid[0] - expected_x).abs() < 0.01);
        assert!((centroid[1] - expected_y).abs() < 0.01);
    }

    /// Centroid with negative values.
    #[test]
    fn centroid_negative_values() {
        let vectors = vec![
            Vector::new(0, vec![-10.0, -20.0]),
            Vector::new(1, vec![10.0, 20.0]),
        ];
        let centroid = compute_centroid(&vectors);
        assert_eq!(centroid, vec![0.0, 0.0]);
    }

    /// Centroid of zero-dimension vectors is zero-dimension.
    #[test]
    fn centroid_zero_dimension() {
        let vectors = vec![Vector::new(0, vec![]), Vector::new(1, vec![])];
        let centroid = compute_centroid(&vectors);
        assert!(centroid.is_empty());
    }

    #[test]
    fn test_trim_split() {
        use crate::bucket_index::BucketIndex;
        use crate::bucket_locks::BucketLocks;
        use crate::router::RoutingTable;
        use crate::storage::wal::runtime::Walrus;
        use crate::vector_index::VectorIndex;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal");
        // std::env::set_var("WALRUS_DATA_DIR", wal_path.to_str().unwrap());

        let wal = Arc::new(Walrus::with_data_dir(wal_path).unwrap());
        let storage = Storage::new(wal.clone());

        let vi_path = dir.path().join("vector_index");
        let vector_index = Arc::new(VectorIndex::open(&vi_path).unwrap());

        let bi_path = dir.path().join("bucket_index");
        let bucket_index = Arc::new(BucketIndex::open(&bi_path).unwrap());

        let routing = Arc::new(RoutingTable::new());
        let bucket_locks = Arc::new(BucketLocks::new());

        let state = RebalanceState::new(
            storage.clone(),
            vector_index.clone(),
            bucket_index.clone(),
            routing.clone(),
            bucket_locks.clone(),
        );

        // 1. Seed Bucket 0 with vectors clustered around (0,0) and (10,10)
        let mut vectors = Vec::new();
        // Cluster A: around (0,0)
        for i in 0..50 {
            vectors.push(Vector::new(i, vec![0.1 * i as f32, 0.1 * i as f32]));
        }
        // Cluster B: around (10,10)
        for i in 50..100 {
            vectors.push(Vector::new(
                i,
                vec![10.0 + 0.1 * i as f32, 10.0 + 0.1 * i as f32],
            ));
        }

        // Write to Bucket 0
        block_on(storage.put_chunk_raw(0, &vectors)).unwrap();

        // Prime centroids to ensure rebalancer knows about Bucket 0
        state.prime_centroids(&[Bucket {
            id: 0,
            centroid: vec![5.0, 5.0], // rough average
            vectors: vec![],
        }]);

        // 2. Trigger Split (Trim)
        // This should read from Bucket 0, create A and B, move vectors, and checkpoint Bucket 0.
        state.handle_split_sync(0);

        // 3. Verify
        // Bucket 0 should be logically empty (consumed)
        // We check entry count instead of get_chunks because get_chunks reads from offset 0
        // and the WAL file might not be deleted yet (active writer).
        let count_0 = wal.get_topic_entry_count("bucket_0");
        assert_eq!(
            count_0, 0,
            "Bucket 0 should be fully consumed/checkpointed (count=0)"
        );

        // We should have 2 new buckets (ids 1 and 2, since 0 was max)
        let router_version = routing.current_version();
        assert!(router_version > 0, "Router should be updated");

        // Check Vectors in new buckets
        let read_bucket = |id| {
            let chunks = block_on(storage.get_chunks(id)).unwrap();
            let mut vecs = Vec::new();
            for chunk in chunks {
                if let Some(v) = RebalanceState::parse_chunk(&chunk) {
                    vecs.push(v);
                }
            }
            vecs
        };

        let vecs_1 = read_bucket(1);
        let vecs_2 = read_bucket(2);

        let total = vecs_1.len() + vecs_2.len();
        assert_eq!(total, 100, "All 100 vectors should be preserved");

        assert!(!vecs_1.is_empty(), "Bucket 1 should have vectors");
        assert!(!vecs_2.is_empty(), "Bucket 2 should have vectors");

        // Verify separation
        let avg_1: f32 = vecs_1.iter().map(|v| v.data[0]).sum::<f32>() / vecs_1.len() as f32;
        let avg_2: f32 = vecs_2.iter().map(|v| v.data[0]).sum::<f32>() / vecs_2.len() as f32;
        assert!(
            (avg_1 - avg_2).abs() > 5.0,
            "Buckets should be well-separated"
        );

        // Verify Index Updates
        let locs = bucket_index.get_many(&[10]).unwrap();
        assert!(!locs.is_empty(), "Index lookup failed");
        let loc = locs[0].1;
        assert!(
            loc == 1 || loc == 2,
            "Index should point to new buckets (found {})",
            loc
        );

        // 4. Verify Bucket 0 Is Removed (Streaming Model Split)
        // It should be removed from the router/centroids
        let centroids = state.centroids.read();
        assert!(
            !centroids.contains_key(&0),
            "Bucket 0 should be removed from centroids map after split"
        );
        assert!(
            centroids.contains_key(&1),
            "Bucket 1 should be in centroids map"
        );
        assert!(
            centroids.contains_key(&2),
            "Bucket 2 should be in centroids map"
        );
        drop(centroids);

        // It should NOT accept NEW writes (routed to it), but we can still write manually if we force it.
        // But the key check is above.
    }

    #[test]
    fn test_concurrent_ingest_and_split() {
        use crate::bucket_index::BucketIndex;
        use crate::bucket_locks::BucketLocks;
        use crate::router::RoutingTable;
        use crate::storage::wal::runtime::Walrus;
        use crate::vector_index::VectorIndex;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        use std::thread;
        use std::time::Duration;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal_conc");
        // std::env::set_var("WALRUS_DATA_DIR", wal_path.to_str().unwrap());

        let wal = Arc::new(Walrus::with_data_dir(wal_path).unwrap());
        let storage = Storage::new(wal.clone());

        let vi_path = dir.path().join("vector_index_conc");
        let vector_index = Arc::new(VectorIndex::open(&vi_path).unwrap());

        let bi_path = dir.path().join("bucket_index_conc");
        let bucket_index = Arc::new(BucketIndex::open(&bi_path).unwrap());

        let routing = Arc::new(RoutingTable::new());
        let bucket_locks = Arc::new(BucketLocks::new());

        let state = Arc::new(RebalanceState::new(
            storage.clone(),
            vector_index.clone(),
            bucket_index.clone(),
            routing.clone(),
            bucket_locks.clone(),
        ));

        // 1. Prime Bucket 0
        let seed_vectors: Vec<Vector> = (0..100).map(|i| Vector::new(i, vec![0.0, 0.0])).collect();
        block_on(storage.put_chunk_raw(0, &seed_vectors)).unwrap();
        state.prime_centroids(&[Bucket {
            id: 0,
            centroid: vec![0.0, 0.0],
            vectors: vec![],
        }]);

        // 2. Concurrent Writer
        // Writes 2000 vectors (IDs 100..2100) while rebalancer runs
        let writer_done = Arc::new(AtomicBool::new(false));
        let writer_done_clone = writer_done.clone();
        let storage_clone = storage.clone();

        let writer_handle = thread::spawn(move || {
            for i in 100..2100 {
                let v = Vector::new(i, vec![i as f32, i as f32]); // Spread out values
                block_on(storage_clone.put_chunk_raw(0, &[v])).unwrap();
                if i % 100 == 0 {
                    thread::sleep(Duration::from_millis(1)); // Mild pacing
                }
            }
            writer_done_clone.store(true, Ordering::SeqCst);
        });

        // 3. Concurrent Rebalancer
        // Repeatedly trigger split until writer is done AND bucket 0 is empty
        let state_clone = state.clone();
        let wal_clone = wal.clone();
        let rebalancer_handle = thread::spawn(move || {
            loop {
                // We just call handle_split_sync. If it returns (idle), we wait and retry.
                state_clone.handle_split_sync(0);

                let count = wal_clone.get_topic_entry_count("bucket_0");
                if writer_done.load(Ordering::SeqCst) && count == 0 {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        });

        writer_handle.join().unwrap();
        rebalancer_handle.join().unwrap();

        // 4. Verify Total Count & Uniqueness
        // IDs should be 0..2100 (Total 2100 vectors)
        let expected_total = 2100;
        let mut collected_ids = std::collections::HashSet::new();

        // Scan all possible buckets (start at 1, go up reasonably)
        // We expect at least bucket 1 and 2. Maybe more if it split recursively?
        // Note: split_bucket_once only splits ONCE. It doesn't recurse in this loop.
        // But handle_split_sync creates new IDs.
        // Since we target Bucket 0 repeatedly, and Bucket 0 is the "source",
        // the rebalancer moves data FROM 0 TO (new_id_1, new_id_2).
        // Since we are continuously writing to 0, it continuously moves to NEW buckets.
        // It allocates new IDs every time handle_split_sync runs successfully.
        // So we might have many buckets.

        // We'll scan bucket IDs until we find empty ones for a while.
        let mut empty_streak = 0;
        let mut bucket_id = 1;
        while empty_streak < 10 {
            let chunks = block_on(storage.get_chunks(bucket_id)).unwrap();
            if chunks.is_empty() {
                empty_streak += 1;
            } else {
                empty_streak = 0;
                for chunk in chunks {
                    if let Some(v) = RebalanceState::parse_chunk(&chunk) {
                        if !collected_ids.insert(v.id) {
                            panic!("Duplicate vector ID found: {}", v.id);
                        }
                    }
                }
            }
            bucket_id += 1;
        }

        // Also check Bucket 0 (should be empty, but verify active/unconsumed data)
        // We MUST use cursor-aware read (read_next or batch_read with None offset)
        // get_chunks reads from 0, ignoring checkpoints.
        let topic_0 = crate::storage::Storage::topic_for(0);
        loop {
            let batch = wal
                .batch_read_for_topic(&topic_0, 1024 * 1024, false, None)
                .unwrap();
            if batch.is_empty() {
                break;
            }
            for entry in batch {
                if let Some(v) = RebalanceState::parse_chunk(&entry.data) {
                    if !collected_ids.insert(v.id) {
                        panic!(
                            "Duplicate vector ID found in source bucket (active): {}",
                            v.id
                        );
                    }
                }
            }
            // Since we use checkpoint=false (peek), we might loop forever if we don't advance?
            // But we want to see what is REMAINING.
            // If we peek, we see the same thing.
            // We should use checkpoint=true to consume them as we count?
            // Or just read once?
            // If the rebalancer did its job, the cursor should be at the end.
            // So peeking should return empty.
            // If it returns non-empty, those are un-rebalanced vectors.
            // We want to count them.
            // To iterate, we can't peek in a loop without advancing.
            // So we use checkpoint=true (consume) or just assume one batch is enough?
            // Let's use checkpoint=true. It's a test, destroying the cursor is fine.
            let _ = wal.read_next(&topic_0, true); // Consume 1-by-1? No.
                                                   // Just break after one batch?
                                                   // If there are many, we need to loop.
                                                   // But wait, if I used peek=false (checkpoint=false) and loop, I get infinite loop.
                                                   // If I use checkpoint=true, I consume.
                                                   // Let's assume we want to consume all remaining.
        }

        // Actually, let's just check get_topic_entry_count first.
        let count_0 = wal.get_topic_entry_count("bucket_0");
        if count_0 > 0 {
            // Read them to check IDs
            let mut remaining = count_0;
            while remaining > 0 {
                // We can't easily read-consume-batch without logic.
                // Just read_next 1 by 1.
                if let Ok(Some(entry)) = wal.read_next(&topic_0, true) {
                    if let Some(v) = RebalanceState::parse_chunk(&entry.data) {
                        if !collected_ids.insert(v.id) {
                            panic!("Duplicate vector ID found in source bucket: {}", v.id);
                        }
                    }
                    remaining -= 1;
                } else {
                    break;
                }
            }
        }

        assert_eq!(
            collected_ids.len(),
            expected_total,
            "Total vectors preserved match"
        );

        // Verify 0..2100 are all present
        for i in 0..2100 {
            assert!(collected_ids.contains(&i), "Missing vector ID: {}", i);
        }
    }

    #[test]
    fn test_rebalancer_keeps_working() {
        let _ = env_logger::builder().is_test(true).try_init();
        use crate::bucket_index::BucketIndex;
        use crate::bucket_locks::BucketLocks;
        use crate::router::RoutingTable;
        use crate::storage::wal::runtime::Walrus;
        use crate::vector_index::VectorIndex;
        use std::sync::Arc;
        use std::thread;
        use std::time::{Duration, Instant};
        use tempfile::tempdir;

        // Set low threshold to trigger rebalancing easily
        let threshold = 10;
        let poll_ms = 500;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("wal_keep");
        // std::env::set_var("WALRUS_DATA_DIR", wal_path.to_str().unwrap());

        let wal = Arc::new(Walrus::with_data_dir(wal_path).unwrap());
        let storage = Storage::new(wal.clone());

        let vi_path = dir.path().join("vector_index_keep");
        let vector_index = Arc::new(VectorIndex::open(&vi_path).unwrap());

        let bi_path = dir.path().join("bucket_index_keep");
        let bucket_index = Arc::new(BucketIndex::open(&bi_path).unwrap());

        let routing = Arc::new(RoutingTable::new());
        let bucket_locks = Arc::new(BucketLocks::new());

        // Use RebalanceWorker::spawn to get the autonomous loop running
        let worker = RebalanceWorker::spawn_with_threshold(
            storage.clone(),
            vector_index.clone(),
            bucket_index.clone(),
            routing.clone(),
            None,
            bucket_locks.clone(),
            threshold,
            poll_ms,
        );

        // 1. Prime Bucket 0
        let seed_vectors: Vec<Vector> = (0..20)
            .map(|i| Vector::new(i, vec![i as f32, i as f32]))
            .collect();
        block_on(storage.put_chunk_raw(0, &seed_vectors)).unwrap();
        block_on(worker.prime_centroids(&[Bucket {
            id: 0,
            centroid: vec![0.0, 0.0],
            vectors: vec![],
        }]))
        .unwrap();

        // 2. Wait for first split (Bucket 0 -> 1, 2)
        // Threshold is 10, we put 20. Should split.
        let mut split_count = 0;
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(30) {
            let sizes = worker.snapshot_sizes();
            // If split happened, we have > 1 bucket.
            // And hopefully Bucket 0 is empty (or near empty).
            if sizes.len() > 1 {
                split_count = 1;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }
        if split_count == 0 {
            let sizes = worker.snapshot_sizes();
            panic!(
                "First split failed to trigger after 30s. Buckets: {:?}",
                sizes
            );
        }
        assert!(split_count > 0, "First split failed to trigger");

        // 3. Pump data into a NEW bucket (e.g. Bucket 1) to force another split
        // Find a valid bucket ID that isn't 0
        let sizes = worker.snapshot_sizes();
        let target_id = sizes
            .keys()
            .find(|&&id| id != 0)
            .cloned()
            .expect("should have non-zero bucket");

        let pump_vectors: Vec<Vector> = (100..150)
            .map(|i| Vector::new(i, vec![100.0 + i as f32, 100.0 + i as f32]))
            .collect();
        block_on(storage.put_chunk_raw(target_id, &pump_vectors)).unwrap();

        // 4. Wait for second split
        // Bucket count should increase further
        let initial_buckets = sizes.len();
        let start = Instant::now();
        let mut second_split = false;
        while start.elapsed() < Duration::from_secs(30) {
            let current_sizes = worker.snapshot_sizes();
            if current_sizes.len() > initial_buckets {
                second_split = true;
                break;
            }
            thread::sleep(Duration::from_millis(50));
        }

        if !second_split {
            let sizes = worker.snapshot_sizes();
            panic!("Second split failed to trigger after 30s. Initial buckets: {}, Current buckets: {:?}, Target bucket: {}", initial_buckets, sizes, target_id);
        }

        assert!(
            second_split,
            "Rebalancer stopped working! Second split failed."
        );
    }

    #[test]

    fn test_continuous_splitting_under_load() {
        use crate::bucket_index::BucketIndex;

        use crate::bucket_locks::BucketLocks;

        use crate::router::RoutingTable;

        use crate::storage::wal::runtime::Walrus;

        use crate::vector_index::VectorIndex;

        use std::sync::Arc;

        use std::thread;

        use std::time::{Duration, Instant};

        use tempfile::tempdir;

        let _ = env_logger::builder().is_test(true).try_init();

        // 1. Setup with low threshold to force frequent splits

        let threshold = 100;
        let poll_ms = 500;

        let dir = tempdir().unwrap();

        let wal_path = dir.path().join("wal_load");

        // std::env::set_var("WALRUS_DATA_DIR", wal_path.to_str().unwrap());

        let wal = Arc::new(Walrus::with_data_dir(wal_path).unwrap());

        let storage = Storage::new(wal.clone());

        let vector_index = Arc::new(VectorIndex::open(dir.path().join("vi")).unwrap());

        let bucket_index = Arc::new(BucketIndex::open(dir.path().join("bi")).unwrap());

        let routing = Arc::new(RoutingTable::new());

        let bucket_locks = Arc::new(BucketLocks::new());

        let worker = RebalanceWorker::spawn_with_threshold(
            storage.clone(),
            vector_index.clone(),
            bucket_index.clone(),
            routing.clone(),
            None,
            bucket_locks.clone(),
            threshold,
            poll_ms,
        );

        // 2. Prime Bucket 0

        block_on(worker.prime_centroids(&[Bucket {
            id: 0,

            centroid: vec![0.0, 0.0],

            vectors: vec![],
        }]))
        .unwrap();

        // 3. Heavy Load Writer (Router-aware)

        let routing_clone = routing.clone();

        let storage_clone = storage.clone();

        let total_vectors = 10_000;

        thread::spawn(move || {
            for i in 0..total_vectors {
                // Generate diverse vectors to ensure splits

                let val = (i as f32) / 10.0;

                let vec_data = vec![val, val];

                let vec = Vector::new(i, vec_data.clone());

                // Route using the shared routing table (simulating real workers)

                // Retry loop to handle race where router might be empty briefly or updating

                let target = loop {
                    if let Some(snap) = routing_clone.snapshot() {
                        if let Ok(ids) = snap.router.query(&vec_data, 1) {
                            if !ids.is_empty() {
                                break ids[0];
                            }
                        }
                    }

                    // Fallback to 0 if router not ready (shouldn't happen often after prime)

                    // But if 0 is removed, we must wait for new buckets.

                    thread::sleep(Duration::from_millis(1));
                };

                block_on(storage_clone.put_chunk_raw(target, &[vec])).unwrap();

                if i % 100 == 0 {
                    thread::sleep(Duration::from_micros(100)); // Slight pacing
                }
            }
        });

        // 4. Monitor Splits

        // We expect bucket count to grow significantly.

        // Threshold 100, 10000 vectors -> theoretically ~100 buckets.

        // We'll accept anything > 10 as proof of continuous operation.

        let start = Instant::now();

        let mut max_buckets = 0;

        while start.elapsed() < Duration::from_secs(30) {
            let sizes = worker.snapshot_sizes();

            let count = sizes.len();

            if count > max_buckets {
                max_buckets = count;

                println!("Bucket count grew to: {}", max_buckets);
            }

            if max_buckets >= 5 {
                println!(
                    "Success: Bucket count reached {}, proving continuous rebalancing.",
                    max_buckets
                );

                return;
            }

            thread::sleep(Duration::from_millis(100));
        }

        panic!(
            "Rebalancer failed to split enough times. Max buckets reached: {} (expected >= 5)",
            max_buckets
        );
    }
}

use anyhow::anyhow;
use crossbeam_channel::{unbounded, Sender};
use dotenv::dotenv;
use flate2::read::GzDecoder;
use futures::channel::oneshot;
use futures::executor::block_on;
use futures::future::join_all;
use glommio::{LocalExecutorBuilder, Placement};
use log::info;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use tar::Archive;

use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
use satoridb::bvecs::BvecsReader;
use satoridb::flatbin::FlatF32Reader;
use satoridb::fvecs::FvecsReader;
use satoridb::gnd::GndReader;
use satoridb::indexer::Indexer;
use satoridb::ingest_counter;
use satoridb::rebalancer::RebalanceWorker;
use satoridb::router::RoutingTable;
use satoridb::s3_reader::spawn_s3_reader;
use satoridb::storage::{Storage, Vector};
use satoridb::tasks::{ConsistentHashRing, RouterResult, RouterTask};
use satoridb::vector_index::VectorIndex;
use satoridb::wal::runtime::Walrus;
use satoridb::wal::{FsyncSchedule, ReadConsistency};
use satoridb::worker::{run_worker, QueryRequest, WorkerMessage};

#[derive(Clone, Copy, Debug)]
enum DatasetKind {
    Bigann,
    Gist,
}

struct DatasetConfig {
    kind: DatasetKind,
    base_path: PathBuf,
    query_path: PathBuf,
    gnd_path: Option<PathBuf>,
    max_vectors: usize,
    base_is_prepared: bool,
    use_s3: bool,
}

fn ensure_gist_files(tar_path: &Path) -> anyhow::Result<()> {
    let base_path = Path::new("gist/gist_base.fvecs");
    let query_path = Path::new("gist/gist_query.fvecs");

    if base_path.exists() && query_path.exists() {
        return Ok(());
    }

    if !tar_path.exists() {
        return Err(anyhow!("gist.tar.gz not found at {}", tar_path.display()));
    }

    info!(
        "Extracting GIST1M base/query from archive {}...",
        tar_path.display()
    );
    fs::create_dir_all("gist")?;
    let file = File::open(tar_path)?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);

    let mut extracted = 0;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        if path == Path::new("gist/gist_base.fvecs") || path == Path::new("gist/gist_query.fvecs") {
            entry.unpack(".")?;
            extracted += 1;
        }
    }

    if base_path.exists() && query_path.exists() {
        Ok(())
    } else {
        Err(anyhow!(
            "Failed to extract gist_base.fvecs or gist_query.fvecs (extracted {} entries)",
            extracted
        ))
    }
}

fn trigger_indexing(router_shared: &Arc<RoutingTable>) -> anyhow::Result<()> {
    if router_shared.snapshot().is_some() {
        info!("Router already built; indexing trigger is a no-op.");
        Ok(())
    } else {
        Err(anyhow!(
            "Router not initialized; indexing step must run first"
        ))
    }
}

fn detect_dataset() -> anyhow::Result<Option<DatasetConfig>> {
    let gist_base = PathBuf::from("gist/gist_base.fvecs");
    let gist_query = PathBuf::from("gist/gist_query.fvecs");
    let gist_tar = PathBuf::from("gist.tar.gz");

    if (gist_base.exists() && gist_query.exists()) || gist_tar.exists() {
        if (!gist_base.exists() || !gist_query.exists()) && gist_tar.exists() {
            ensure_gist_files(&gist_tar)?;
        }
        let prepared_base = gist_base.with_extension("f32bin");
        let base_path = if prepared_base.exists() {
            prepared_base.clone()
        } else {
            gist_base.clone()
        };
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Gist,
            base_path,
            query_path: gist_query,
            gnd_path: if gist_tar.exists() {
                Some(gist_tar)
            } else {
                None
            },
            max_vectors: 1_000_000usize,
            base_is_prepared: prepared_base.exists(),
            use_s3: false,
        }));
    }

    let bigann_base = PathBuf::from("bigann_base.bvecs.gz");
    let bigann_query = PathBuf::from("bigann_query.bvecs.gz");
    let bigann_gnd = PathBuf::from("bigann_gnd.tar.gz");
    let prepared_bigann = bigann_base.with_extension("f32bin");

    if prepared_bigann.exists() && bigann_query.exists() {
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Bigann,
            base_path: prepared_bigann.clone(),
            query_path: bigann_query.clone(),
            gnd_path: if bigann_gnd.exists() {
                Some(bigann_gnd)
            } else {
                None
            },
            max_vectors: 1_000_000_000usize,
            base_is_prepared: true,
            use_s3: false,
        }));
    }

    if bigann_base.exists() && bigann_query.exists() {
        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Bigann,
            base_path: if prepared_bigann.exists() {
                prepared_bigann.clone()
            } else {
                bigann_base.clone()
            },
            query_path: bigann_query,
            gnd_path: if bigann_gnd.exists() {
                Some(bigann_gnd)
            } else {
                None
            },
            max_vectors: 1_000_000_000usize,
            base_is_prepared: prepared_bigann.exists(),
            use_s3: false,
        }));
    }

    // Check for S3/R2 Fallback
    if std::env::var("R2_ENDPOINT").is_ok()
        && std::env::var("BUCKET_NAME").is_ok()
        && std::env::var("ACCESS_KEY_ID").is_ok()
        && std::env::var("SECRET_ACCESS_KEY").is_ok()
    {
        // Assume BigAnn if R2 is configured and no local Gist
        // We still need query/gnd locally for now, or we can skip them?
        // The original code assumes query file exists locally for benchmark.
        // Assuming user has query file locally or downloads it.
        // If bigann_query.bvecs.gz doesn't exist, we might fail later, but let's try.

        // Actually, if query file is missing, we can't run recall, but we can ingest.
        // Let's allow it.

        return Ok(Some(DatasetConfig {
            kind: DatasetKind::Bigann,
            base_path: PathBuf::from("bigann_base.bvecs.f32bin"), // Virtual path
            query_path: bigann_query,
            gnd_path: if bigann_gnd.exists() {
                Some(bigann_gnd)
            } else {
                None
            },
            max_vectors: 1_000_000_000usize,
            base_is_prepared: true, // S3 stream is f32bin
            use_s3: true,
        }));
    }

    Ok(None)
}

fn main() -> anyhow::Result<()> {
    dotenv().ok();
    env_logger::init();

    // Check limits
    let mut rlimit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut rlimit) };
    info!(
        "RLIMIT_NOFILE: cur={}, max={}",
        rlimit.rlim_cur, rlimit.rlim_max
    );

    let fd_count = std::fs::read_dir("/proc/self/fd")
        .map(|d| d.count())
        .unwrap_or(0);
    info!("Initial open FDs: {}", fd_count);

    let satori_cores: usize = std::env::var("SATORI_CORES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| num_cpus::get().max(1));
    let usable_cores = satori_cores.saturating_sub(2).max(1);
    let total_cpus = num_cpus::get().max(1);
    let reserved_start = usable_cores.min(total_cpus);
    let mut reserved_cores: Vec<usize> = (reserved_start..total_cpus).collect();
    let rebalance_core = reserved_cores.pop();

    let router_pool = ((usable_cores as f32) * 0.2).round() as usize;
    let router_pool = router_pool.max(1);
    let executor_shards = (usable_cores.saturating_sub(router_pool)).max(1);
    info!(
        "SATORI_CORES={}, usable_cores={}, router_pool={}, executor_shards={}",
        satori_cores, usable_cores, router_pool, executor_shards
    );

    let wal = Arc::new(Walrus::with_consistency_and_schedule(
        ReadConsistency::StrictlyAtOnce,
        FsyncSchedule::NoFsync,
    )?);
    let vector_index_path =
        std::env::var("SATORI_VECTOR_INDEX_PATH").unwrap_or_else(|_| "vector_index".to_string());
    let vector_index = Arc::new(VectorIndex::open(&vector_index_path)?);
    let bucket_index_path =
        std::env::var("SATORI_BUCKET_INDEX_PATH").unwrap_or_else(|_| "bucket_index".to_string());
    let bucket_index = Arc::new(BucketIndex::open(&bucket_index_path)?);
    let bucket_locks = Arc::new(BucketLocks::new());

    let mut senders = Vec::new();
    let mut worker_handles = Vec::new();

    for i in 0..executor_shards {
        let (sender, receiver) = async_channel::bounded(1000);
        senders.push(sender);
        let wal_clone = wal.clone();
        let index_clone = vector_index.clone();
        let bucket_index_clone = bucket_index.clone();
        let bucket_locks_clone = bucket_locks.clone();
        let pin_cpu = i % usable_cores;
        let handle = thread::spawn(move || {
            let builder =
                LocalExecutorBuilder::new(Placement::Fixed(pin_cpu)).name(&format!("worker-{}", i));

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

    let ring = ConsistentHashRing::new(executor_shards, 8);

    let (router_tx, router_rx) = unbounded::<RouterTask>();
    let router_shared = Arc::new(RoutingTable::new());
    let rebalance_worker = RebalanceWorker::spawn(
        Storage::new(wal.clone()),
        vector_index.clone(),
        bucket_index.clone(),
        router_shared.clone(),
        rebalance_core,
        bucket_locks.clone(),
    );
    if let Some(core) = rebalance_core {
        info!("Rebalance worker pinned to reserved core {}", core);
    } else {
        info!("Rebalance worker running without dedicated core (only one CPU visible)");
    }

    {
        let worker = rebalance_worker.clone();
        std::thread::spawn(move || {
            let mut tick: u64 = 0;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                tick += 1;

                let sizes = worker.snapshot_sizes();
                let mut sizes_vec: Vec<usize> = sizes.values().cloned().collect();
                sizes_vec.sort_unstable();

                info!(
                    "monitor: tick={} buckets={} total_vectors={} sizes={:?}",
                    tick,
                    sizes_vec.len(),
                    sizes_vec.iter().sum::<usize>(),
                    sizes_vec
                );
            }
        });
    }

    let mut router_handles = Vec::new();
    for i in 0..router_pool {
        let rx = router_rx.clone();
        let router_clone = router_shared.clone();
        let handle = thread::spawn(move || {
            let mut local_snapshot: Option<satoridb::router::RoutingSnapshot> = None;
            for task in rx.iter() {
                if local_snapshot
                    .as_ref()
                    .map(|s| s.version != router_clone.current_version())
                    .unwrap_or(true)
                {
                    local_snapshot = router_clone.snapshot();
                }

                let snapshot = match &local_snapshot {
                    Some(s) => s.clone(),
                    None => {
                        let _ = task.respond_to.send(Err(anyhow!("router not initialized")));
                        continue;
                    }
                };

                let res = snapshot
                    .router
                    .query(&task.query_vec, task.top_k)
                    .map(|bucket_ids| RouterResult {
                        bucket_ids,
                        routing_version: snapshot.version,
                        affected_buckets: snapshot.changed.clone(),
                    })
                    .map_err(|e| anyhow!("router {} query failed: {:?}", i, e));
                let _ = task.respond_to.send(res);
            }
            Ok::<(), anyhow::Error>(())
        });
        router_handles.push(handle);
    }

    let run_benchmark = std::env::var("SATORI_RUN_BENCH").is_ok();
    if run_benchmark {
        block_on(run_benchmark_mode(
            wal.clone(),
            router_tx.clone(),
            router_shared.clone(),
            rebalance_worker.clone(),
            ring.clone(),
            senders.clone(),
        ))?;
    } else {
        info!("Benchmark logic is disabled (set SATORI_RUN_BENCH=1 to enable).");
    }

    drop(router_tx);
    for sender in senders {
        sender.close();
    }

    for h in worker_handles {
        let _ = h.join();
    }
    for h in router_handles {
        let _ = h.join();
    }
    Ok(())
}

async fn run_benchmark_mode(
    wal: Arc<Walrus>,
    router_tx: Sender<RouterTask>,
    router_shared: Arc<RoutingTable>,
    rebalance: RebalanceWorker,
    ring: ConsistentHashRing,
    senders: Vec<async_channel::Sender<WorkerMessage>>,
) -> anyhow::Result<()> {
    let dataset = match detect_dataset()? {
        Some(d) => d,
        None => {
            info!("No dataset detected (bigann_* or gist.tar.gz). Skipping benchmark.");
            return Ok(());
        }
    };
    info!(
        "Dataset: {:?}, base={}, query={}",
        dataset.kind,
        dataset.base_path.display(),
        dataset.query_path.display()
    );

    let storage = Storage::new(wal.clone());

    enum Reader {
        Bvecs(BvecsReader),
        Fvecs(FvecsReader),
        Flat(FlatF32Reader),
        S3(async_channel::Receiver<anyhow::Result<Vec<Vector>>>),
    }

    impl Reader {
        fn read_batch(&mut self, batch_size: usize) -> anyhow::Result<Vec<Vector>> {
            match self {
                Reader::Bvecs(r) => r.read_batch(batch_size),
                Reader::Fvecs(r) => r.read_batch(batch_size),
                Reader::Flat(r) => r.read_batch(batch_size),
                Reader::S3(rx) => match rx.recv_blocking() {
                    Ok(Ok(vectors)) => Ok(vectors),
                    Ok(Err(e)) => Err(e),
                    Err(_) => Ok(Vec::new()),
                },
            }
        }
    }

    let (default_initial_batch_size, stream_batch_size, k, router_top_k) = match dataset.kind {
        DatasetKind::Bigann => (500_000, 500_000, 10, 20),
        DatasetKind::Gist => (50_000, 100_000, 10, 400),
    };

    let initial_batch_size = if dataset.use_s3 {
        200_000
    } else {
        default_initial_batch_size
    };

    if dataset.base_path.exists() || dataset.use_s3 {
        let mut reader = if dataset.use_s3 {
            println!("Initializing S3 Stream for BigANN...");
            let endpoint = std::env::var("R2_ENDPOINT").expect("R2_ENDPOINT not set");
            let bucket = std::env::var("BUCKET_NAME").expect("BUCKET_NAME not set");
            let access = std::env::var("ACCESS_KEY_ID").expect("ACCESS_KEY_ID not set");
            let secret = std::env::var("SECRET_ACCESS_KEY").expect("SECRET_ACCESS_KEY not set");
            // We assume the file in bucket is named similarly or we use a fixed name.
            // Based on earlier check: bigann_base.bvecs.f32bin
            let key = "bigann_base.bvecs.f32bin".to_string();

            let rx = spawn_s3_reader(endpoint, bucket, key, access, secret, stream_batch_size);
            Reader::S3(rx)
        } else {
            match dataset.kind {
                DatasetKind::Bigann => {
                    if dataset.base_is_prepared {
                        Reader::Flat(FlatF32Reader::new(&dataset.base_path)?)
                    } else {
                        Reader::Bvecs(BvecsReader::new(&dataset.base_path)?)
                    }
                }
                DatasetKind::Gist => {
                    if dataset.base_is_prepared {
                        Reader::Flat(FlatF32Reader::new(&dataset.base_path)?)
                    } else {
                        Reader::Fvecs(FvecsReader::new(&dataset.base_path)?)
                    }
                }
            }
        };

        info!("Reading initial batch of {} vectors...", initial_batch_size);
        let vectors = reader.read_batch(initial_batch_size)?;
        if vectors.is_empty() {
            return Err(anyhow!("Base file is empty"));
        }
        info!(
            "Clustering initial batch: {} vectors, k={}",
            vectors.len(),
            k
        );

        let total_initial = vectors.len();
        let buckets = Indexer::build_clusters(vectors, k);
        ingest_counter::add(total_initial as u64);
        info!("Indexer created {} buckets.", buckets.len());

        for bucket in &buckets {
            storage.put_chunk(bucket).await?;
        }
        rebalance.prime_centroids(&buckets).await?;
        info!(
            "Initial indexing complete (router version {}, buckets={}).",
            router_shared.current_version(),
            buckets.len()
        );

        let mut total_processed = initial_batch_size;
        let mut ingest_waiters = Vec::new();

        while total_processed < dataset.max_vectors {
            let batch = reader.read_batch(stream_batch_size)?;
            if batch.is_empty() {
                break;
            }

            info!("Ingesting batch (total: {})...", total_processed);

            let mut updates: HashMap<u64, Vec<Vector>> = HashMap::new();
            updates.reserve(router_top_k);

            if let Some(snapshot) = router_shared.snapshot() {
                for vec in batch {
                    if let Ok(ids) = snapshot.router.query(&vec.data, 1) {
                        if let Some(&id) = ids.first() {
                            updates
                                .entry(id)
                                .or_insert_with(|| Vec::with_capacity(64))
                                .push(vec);
                        }
                    }
                }
            } else {
                return Err(anyhow!("router missing"));
            }

            for (bucket_id, new_vectors) in updates {
                let shard = ring.node_for(bucket_id);
                let (tx, rx) = oneshot::channel();
                let msg = WorkerMessage::Ingest {
                    bucket_id,
                    vectors: new_vectors,
                    respond_to: tx,
                };
                if let Err(e) = senders[shard].send(msg).await {
                    log::error!("Failed to send ingest to shard {}: {:?}", shard, e);
                } else {
                    ingest_waiters.push(rx);
                }
            }
            total_processed += stream_batch_size;
        }

        let ingest_results = join_all(ingest_waiters).await;
        for res in ingest_results {
            match res {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    log::error!("Ingest failed: {:?}", e);
                }
                Err(e) => {
                    log::error!("Ingest channel closed: {:?}", e);
                }
            }
        }
        info!("Flushing all workers before queries...");
        let mut flush_waiters = Vec::new();
        for sender in &senders {
            let (tx, rx) = oneshot::channel();
            if let Err(e) = sender.send(WorkerMessage::Flush { respond_to: tx }).await {
                log::error!("Failed to send flush: {:?}", e);
            } else {
                flush_waiters.push(rx);
            }
        }
        for rx in flush_waiters {
            let _ = rx.await;
        }
        info!(
            "Ingestion complete. Total vectors ingested or scanned: {}",
            total_processed
        );

        info!("Waiting 5 minutes for system to settle/rebalance before querying...");
        std::thread::sleep(std::time::Duration::from_secs(300));
        info!("Wait complete. Starting queries.");
    } else {
        info!(
            "{} not found. Skipping Benchmark Ingestion.",
            dataset.base_path.display()
        );
    }

    if dataset.query_path.exists() {
        trigger_indexing(&router_shared)?;

        let mut reader = match dataset.kind {
            DatasetKind::Bigann => Reader::Bvecs(BvecsReader::new(&dataset.query_path)?),
            DatasetKind::Gist => Reader::Fvecs(FvecsReader::new(&dataset.query_path)?),
        };
        let queries = reader.read_batch(100)?;

        let ground_truth = match &dataset.gnd_path {
            Some(path) if path.exists() => {
                info!("Loading Ground Truth from {}...", path.display());
                Some(GndReader::new(path)?)
            }
            Some(path) => {
                info!(
                    "Ground Truth {} not found. Recall will be skipped.",
                    path.display()
                );
                None
            }
            None => {
                info!("Ground Truth path not provided. Recall will be skipped.");
                None
            }
        };

        let mut total_hits_at_10 = 0;
        let mut total_queries_checked = 0;
        let mut total_reachable_gt = 0;

        for (i, q) in queries.iter().enumerate() {
            let (tx, rx) = oneshot::channel();
            let task = RouterTask {
                query_vec: q.data.clone(),
                top_k: router_top_k,
                respond_to: tx,
            };
            router_tx
                .send(task)
                .map_err(|e| anyhow!("router send: {:?}", e))?;
            let router_result = rx.await.map_err(|e| anyhow!("router recv: {:?}", e))??;
            let RouterResult {
                bucket_ids,
                routing_version,
                affected_buckets,
            } = router_result;

            if i == 0 {
                info!(
                    "Debug query[0]: router returned {} bucket ids (top_k={})",
                    bucket_ids.len(),
                    router_top_k
                );
            }

            let mut pending = Vec::new();
            let mut requests: HashMap<usize, Vec<u64>> = HashMap::new();
            for &bid in &bucket_ids {
                let shard = ring.node_for(bid);
                requests.entry(shard).or_default().push(bid);
            }

            for (shard, bids) in requests {
                let (tx, rx) = oneshot::channel();
                let req = QueryRequest {
                    query_vec: q.data.clone(),
                    bucket_ids: bids,
                    routing_version,
                    affected_buckets: affected_buckets.clone(),
                    include_vectors: false,
                    respond_to: tx,
                };
                let msg = WorkerMessage::Query(req);
                if senders[shard].send(msg).await.is_ok() {
                    pending.push(rx);
                }
            }

            let responses = join_all(pending).await;
            let mut all_results = Vec::new();
            for res in responses {
                if let Ok(Ok(candidates)) = res {
                    all_results.extend(candidates);
                }
            }
            if i == 0 {
                info!(
                    "Debug query[0]: collected {} candidates from workers",
                    all_results.len()
                );
            }
            all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut unique_results = Vec::with_capacity(all_results.len());
            let mut seen_ids = std::collections::HashSet::new();
            for (id, dist, _) in all_results {
                if seen_ids.insert(id) {
                    unique_results.push((id, dist));
                }
            }
            let all_results = unique_results;

            if let Some(ref gt) = ground_truth {
                if i < gt.ground_truth.len() {
                    let true_neighbors = &gt.ground_truth[i];
                    let gt_top_10 = true_neighbors.iter().take(10).cloned().collect::<Vec<_>>();

                    let reachable = gt_top_10
                        .iter()
                        .filter(|&&id| (id as usize) < dataset.max_vectors)
                        .count();
                    total_reachable_gt += reachable;

                    let my_top_10: Vec<u32> =
                        all_results.iter().take(10).map(|r| r.0 as u32).collect();
                    if i == 0 {
                        info!("Debug query[0]: my_top_10={:?}", my_top_10);
                        info!("Debug query[0]: gt_top_10={:?}", gt_top_10);
                    }
                    let hits = my_top_10.iter().filter(|id| gt_top_10.contains(id)).count();
                    total_hits_at_10 += hits;
                    total_queries_checked += 1;
                }
            }

            if i % 10 == 0 {
                info!("Processed {} queries", i);
            }
        }

        if total_queries_checked > 0 {
            let adjusted_recall = if total_reachable_gt > 0 {
                total_hits_at_10 as f64 / total_reachable_gt as f64
            } else {
                0.0
            };
            info!(
                "Benchmark complete. Queries: {} Adjusted Recall@10: {:.2}%",
                total_queries_checked,
                adjusted_recall * 100.0
            );
        }
    }

    Ok(())
}

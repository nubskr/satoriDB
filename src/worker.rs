use crate::bucket_index::BucketIndex;
use crate::bucket_locks::BucketLocks;
use crate::executor::{Executor, WorkerCache};
use crate::ingest_control;
use crate::ingest_counter;
use crate::storage::wal::runtime::Walrus;
use crate::storage::{Bucket, Storage, StorageExecMode, Vector};
use crate::vector_index::VectorIndex;
use async_channel::Receiver;
use futures::channel::oneshot;
use log::{error, info};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

type QueryResult = anyhow::Result<Vec<(u64, f32, Option<Vec<f32>>)>>;

pub struct QueryRequest {
    pub query_vec: Vec<f32>,
    pub bucket_ids: Vec<u64>,
    pub routing_version: u64,
    pub affected_buckets: std::sync::Arc<Vec<u64>>,
    pub include_vectors: bool,
    pub respond_to: oneshot::Sender<QueryResult>,
}

type FetchVectorsResult = anyhow::Result<Vec<(u64, Vec<f32>)>>;

pub enum WorkerMessage {
    Query(QueryRequest),
    Upsert {
        bucket_id: u64,
        vector: Vector,
        respond_to: oneshot::Sender<anyhow::Result<()>>,
    },
    Ingest {
        bucket_id: u64,
        vectors: Vec<Vector>,
        respond_to: oneshot::Sender<anyhow::Result<()>>,
    },
    Flush {
        respond_to: oneshot::Sender<()>,
    },
    FetchVectors {
        bucket_id: u64,
        ids: Vec<u64>,
        respond_to: oneshot::Sender<FetchVectorsResult>,
    },
    Shutdown {
        respond_to: oneshot::Sender<()>,
    },
}

pub async fn run_worker(
    id: usize,
    receiver: Receiver<WorkerMessage>,
    wal: Arc<Walrus>,
    vector_index: Arc<VectorIndex>,
    bucket_index: Arc<BucketIndex>,
    bucket_locks: Arc<BucketLocks>,
) {
    info!("Worker {} started.", id);

    let storage = Storage::new(wal.clone()).with_mode(StorageExecMode::Offload);
    let cache_max_buckets: usize = std::env::var("SATORI_WORKER_CACHE_BUCKETS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(128);
    let cache_bucket_mb: usize = std::env::var("SATORI_WORKER_CACHE_BUCKET_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(128);
    let cache_bucket_bytes = cache_bucket_mb * 1024 * 1024;
    let cache_total_bytes = cache_max_buckets
        .saturating_mul(cache_bucket_bytes)
        .max(cache_bucket_bytes);
    let skip_dup_checks = std::env::var("SATORI_SKIP_DUP_CHECKS").is_ok()
        || std::env::var("SATORI_RUN_BENCH").is_ok();
    let cache = WorkerCache::new(cache_max_buckets, cache_bucket_bytes, cache_total_bytes);
    let executor = Rc::new(Executor::new(storage.clone(), cache));
    Storage::prewarm_thread_locals(2048, 1024);

    let max_concurrency: usize = std::env::var("SATORI_WORKER_MAX_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(512);
    let (limit_tx, limit_rx) = async_channel::bounded(max_concurrency);

    while let Ok(msg) = receiver.recv().await {
        match msg {
            WorkerMessage::Query(req) => {
                limit_tx.send(()).await.unwrap();
                let limit_rx = limit_rx.clone();
                let executor = executor.clone();
                glommio::spawn_local(async move {
                    let result = executor
                        .query(
                            &req.query_vec,
                            &req.bucket_ids,
                            100,
                            req.routing_version,
                            req.affected_buckets,
                            req.include_vectors,
                        )
                        .await;
                    if req.respond_to.send(result).is_err() {
                        error!("Worker {} failed to send response back.", id);
                    }
                    let _ = limit_rx.recv().await;
                })
                .detach();
            }
            WorkerMessage::Upsert {
                bucket_id,
                vector,
                respond_to,
            } => {
                if !ingest_control::ingestion_allowed() {
                    let _ = respond_to.send(Err(anyhow::anyhow!("ingestion disabled")));
                    continue;
                }
                if !skip_dup_checks {
                    // Check for duplicate id before acquiring lock
                    match vector_index.exists(vector.id) {
                        Ok(true) => {
                            let _ = respond_to
                                .send(Err(anyhow::anyhow!("id {} already exists", vector.id)));
                            continue;
                        }
                        Err(e) => {
                            let _ = respond_to.send(Err(e));
                            continue;
                        }
                        Ok(false) => {}
                    }
                }

                let lock = bucket_locks.lock_for(bucket_id);
                let _guard = lock.lock().await;
                let mut bucket = Bucket::new(bucket_id, Vec::new());
                bucket.vectors = vec![vector];
                let result = storage.put_chunk(&bucket).await;
                if result.is_ok() {
                    if let Err(e) = vector_index.put_batch(&bucket.vectors) {
                        error!(
                            "Worker {} failed to update vector index for bucket {}: {:?}",
                            id, bucket_id, e
                        );
                    }
                    if let Err(e) = bucket_index.put_batch(bucket_id, &[bucket.vectors[0].id]) {
                        error!(
                            "Worker {} failed to update bucket index for bucket {}: {:?}",
                            id, bucket_id, e
                        );
                    }
                }
                let _ = respond_to.send(result);
            }
            WorkerMessage::FetchVectors {
                bucket_id,
                ids,
                respond_to,
            } => {
                let res = executor
                    .fetch_vectors(bucket_id, &ids)
                    .await
                    .map(|vectors| vectors.into_iter().map(|v| (v.id, v.data)).collect());
                let _ = respond_to.send(res);
            }
            WorkerMessage::Ingest {
                bucket_id,
                vectors,
                respond_to,
            } => {
                if !ingest_control::ingestion_allowed() {
                    let _ = respond_to.send(Err(anyhow::anyhow!("ingestion disabled")));
                    continue;
                }
                let mut ids = Vec::with_capacity(vectors.len());
                let mut seen = std::collections::HashSet::with_capacity(vectors.len());
                let mut duplicate_in_batch = None;
                for v in &vectors {
                    if !seen.insert(v.id) {
                        duplicate_in_batch = Some(v.id);
                        break;
                    }
                    ids.push(v.id);
                }

                limit_tx.send(()).await.unwrap();
                let limit_rx = limit_rx.clone();
                let storage = storage.clone();
                let vector_index = vector_index.clone();
                let bucket_index = bucket_index.clone();
                let bucket_locks = bucket_locks.clone();
                glommio::spawn_local(async move {
                    let lock = bucket_locks.lock_for(bucket_id);
                    let _guard = lock.lock().await;
                    if let Some(dup) = duplicate_in_batch {
                        let _ = respond_to
                            .send(Err(anyhow::anyhow!("duplicate id {} in batch", dup)));
                        let _ = limit_rx.recv().await;
                        return;
                    }
                    if !skip_dup_checks {
                        match vector_index.first_existing(&ids) {
                            Ok(Some(existing)) => {
                                let _ = respond_to
                                    .send(Err(anyhow::anyhow!("id {} already exists", existing)));
                                let _ = limit_rx.recv().await;
                                return;
                            }
                            Err(e) => {
                                let _ = respond_to.send(Err(e));
                                let _ = limit_rx.recv().await;
                                return;
                            }
                            Ok(None) => {}
                        }
                    }

                    let topic = Storage::topic_for(bucket_id);
                    loop {
                        match storage
                            .put_chunk_raw_with_topic(bucket_id, &topic, &vectors)
                            .await
                        {
                            Ok(_) => {
                                ingest_counter::add(vectors.len() as u64);
                                match vector_index.put_batch(&vectors) {
                                    Ok(_) => match bucket_index.put_batch(bucket_id, &ids) {
                                        Ok(_) => {
                                            let _ = respond_to.send(Ok(()));
                                        }
                                        Err(e) => {
                                            error!(
                                                "Worker {} failed to update bucket index for bucket {}: {:?}",
                                                id, bucket_id, e
                                            );
                                            let _ = respond_to.send(Err(e));
                                        }
                                    },
                                    Err(e) => {
                                        error!(
                                            "Worker {} failed to update vector index for bucket {}: {:?}",
                                            id, bucket_id, e
                                        );
                                        let _ = respond_to.send(Err(e));
                                    }
                                }
                                break;
                            }
                            Err(e) => {
                                let is_would_block = e
                                    .downcast_ref::<std::io::Error>()
                                    .map(|io_err| io_err.kind() == std::io::ErrorKind::WouldBlock)
                                    .unwrap_or(false);

                                if is_would_block {
                                    glommio::timer::Timer::new(Duration::from_millis(5)).await;
                                    continue;
                                }

                                error!(
                                    "Worker {} failed to persist chunk for bucket {}: {:?}",
                                    id, bucket_id, e
                                );
                                let _ = respond_to.send(Err(e));
                                break;
                            }
                        }
                    }
                    let _ = limit_rx.recv().await;
                })
                .detach();
            }
            WorkerMessage::Flush { respond_to } => {
                while !limit_rx.is_empty() {
                    glommio::timer::Timer::new(Duration::from_millis(10)).await;
                }
                let _ = respond_to.send(());
            }
            WorkerMessage::Shutdown { respond_to } => {
                let _ = respond_to.send(());
                break;
            }
        }
    }

    info!("Worker {} shutting down.", id);
}

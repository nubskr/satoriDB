use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::executor::block_on;
use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
use satoridb::executor::{Executor, WorkerCache};
use satoridb::rebalancer::RebalanceWorker;
use satoridb::router::RoutingTable;
use satoridb::storage::{Bucket, Storage, Vector};
use satoridb::vector_index::VectorIndex;
use satoridb::wal::runtime::Walrus;
use tempfile::TempDir;

fn init_wal(tempdir: &TempDir) -> Arc<Walrus> {
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

fn wait_until(timeout: Duration, mut f: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if f() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn split_updates_routing_and_executor_results() -> Result<()> {
    // Force split on small buckets (8 vectors > 4)
    let threshold = 4;

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let bucket_locks = Arc::new(BucketLocks::new());
    let worker = RebalanceWorker::spawn_with_threshold(
        storage.clone(),
        vector_index.clone(),
        bucket_index.clone(),
        routing.clone(),
        None,
        bucket_locks,
        threshold,
        100,
    );

    // One bucket containing two clear clusters: near (0,0) and near (100,100).
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..4u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32 * 0.01, 0.0])); // cluster A
    }
    for i in 0..4u64 {
        bucket.add_vector(Vector::new(100 + i, vec![100.0 + i as f32, 100.0])); // cluster B
    }
    block_on(storage.put_chunk(&bucket))?;
    block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Autonomous loop should split bucket 0

    // Wait for routing to advance and for sizes to show multiple buckets.
    // Also wait for rebalance to finish (total count settles back to 8).
    // During streaming rebalance, we might see duplicates (transiently 16, then 12, then 8).
    let progressed = wait_until(Duration::from_secs(3), || {
        let sizes = worker.snapshot_sizes();
        let total: usize = sizes.values().sum();
        sizes.len() > 1 && routing.current_version() > 0 && total == 8
    });
    assert!(progressed, "split did not finish in time");

    // Snapshot routing after split.
    let snap = routing.snapshot().expect("routing installed");
    assert!(
        !snap.changed.is_empty(),
        "routing snapshot should list changed buckets"
    );

    // Use executor to query near cluster A and ensure we get the small ids.
    let executor = Executor::new(
        storage.clone(),
        WorkerCache::new(8, 1024 * 1024, 1024 * 1024),
    );
    let bucket_ids = snap.router.query(&[0.0, 0.0], 2)?;
    let sizes = worker.snapshot_sizes();
    let total_after: usize = sizes.values().sum();
    assert_eq!(total_after, 8, "split should preserve total vector count");

    let results = block_on(executor.query(
        &[0.0, 0.0],
        &bucket_ids,
        2,
        snap.version,
        snap.changed.clone(),
        false,
    ))?;
    assert!(
        results.iter().any(|(id, _, _)| *id < 50),
        "expected a vector from the low-id cluster, got {:?}",
        results
    );

    Ok(())
}

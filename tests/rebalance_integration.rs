use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use futures::executor::block_on;
use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
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
        thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn rebalance_split_increases_buckets_and_router_version() -> Result<()> {
    // Force aggressive splitting for test
    let threshold = 10;

    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let storage = Storage::new(wal);
    let routing = Arc::new(RoutingTable::new());
    let bucket_locks = Arc::new(BucketLocks::new());
    let worker = RebalanceWorker::spawn_with_threshold(
        storage.clone(),
        vector_index,
        bucket_index,
        routing.clone(),
        None,
        bucket_locks,
        threshold,
        100,
    );

    // Persist a bucket with enough vectors to split.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..16u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, 0.0]));
    }
    block_on(storage.put_chunk(&bucket))?;
    block_on(worker.prime_centroids(&[bucket]))?;

    let initial_version = routing.current_version();
    // No manual enqueue; autonomous loop should pick it up since 16 > 10.

    let progressed = wait_until(Duration::from_secs(2), || {
        let sizes = worker.snapshot_sizes();
        sizes.len() > 1
    });
    assert!(progressed, "split did not produce new buckets");
    assert!(
        routing.current_version() > initial_version,
        "router version did not advance after split"
    );

    drop(worker);
    Ok(())
}

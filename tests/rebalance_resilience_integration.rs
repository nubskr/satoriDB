use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
use satoridb::rebalancer::RebalanceWorker;
use satoridb::router::RoutingTable;
use satoridb::storage::{Storage, Vector};
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
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Fast-path split condition: when a bucket grows beyond target_size*2, ensure a split is enqueued and routing advances.
#[test]
fn rebalance_fast_path_split_triggers_on_oversized_bucket() -> Result<()> {
    // Force split at 25
    let threshold = 25;

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

    // Create one large bucket and prime centroids.
    let mut big = satoridb::storage::Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..50u64 {
        big.add_vector(Vector::new(i, vec![i as f32, 0.0]));
    }
    futures::executor::block_on(storage.put_chunk(&big))?;
    futures::executor::block_on(worker.prime_centroids(&[big.clone()]))?;

    // Autonomous loop will detect 50 > 25 and split.

    let progressed = wait_until(Duration::from_secs(3), || {
        worker.snapshot_sizes().len() > 1 && routing.current_version() > 0
    });
    assert!(
        progressed,
        "fast-path split should complete and advance routing"
    );
    Ok(())
}

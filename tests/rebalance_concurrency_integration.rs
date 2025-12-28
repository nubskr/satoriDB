use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
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
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Concurrent routing snapshots during a split should not panic or deadlock.
#[test]
fn routing_snapshots_survive_concurrent_split() -> Result<()> {
    // Force split with 10 vectors
    let threshold = 5;

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

    // Prime with one bucket.
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..10u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, 0.0]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Spawn a thread that repeatedly takes routing snapshots while split is happening.
    let routing_clone = routing.clone();
    let snapshot_thread = std::thread::spawn(move || {
        for _ in 0..200 {
            let _ = routing_clone.snapshot();
            std::thread::sleep(Duration::from_millis(5));
        }
    });

    // Autonomous loop will trigger split on bucket 0.

    let progressed = wait_until(Duration::from_secs(3), || {
        worker.snapshot_sizes().len() > 1 && routing.current_version() > 0
    });
    snapshot_thread
        .join()
        .expect("snapshot thread should not panic");
    assert!(
        progressed,
        "split did not finish in time with concurrent snapshots"
    );

    Ok(())
}

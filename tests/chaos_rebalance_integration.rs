use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use satoridb::bucket_index::BucketIndex;
use satoridb::bucket_locks::BucketLocks;
use satoridb::executor::{
    clear_executor_fail_load_hook, set_executor_fail_load_hook, Executor, WorkerCache,
};
use satoridb::rebalancer::{
    clear_rebalance_fail_hook, set_rebalance_fail_hook, RebalanceTaskKind, RebalanceWorker,
};
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

#[test]
fn rebalance_survives_split_failures() -> Result<()> {
    // Force split at 10
    let threshold = 10;

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
        bucket_index,
        routing.clone(),
        None,
        bucket_locks,
        threshold,
        100,
    );

    // Seed one bucket with enough vectors to split (32 > 10).
    let mut bucket = Bucket::new(0, vec![0.0, 0.0]);
    for i in 0..32u64 {
        bucket.add_vector(Vector::new(i, vec![i as f32, i as f32 + 0.5]));
    }
    futures::executor::block_on(storage.put_chunk(&bucket))?;
    futures::executor::block_on(worker.prime_centroids(&[bucket.clone()]))?;

    // Fail every third split attempt to simulate flaky work.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_rebalance_fail_hook(move |kind| {
        if kind != RebalanceTaskKind::Split {
            return false;
        }
        let n = c.fetch_add(1, Ordering::SeqCst);
        n.is_multiple_of(3)
    });

    // Autonomous loop will try to split. It might fail, but it should retry.

    let ok = wait_until(Duration::from_secs(15), || {
        worker.snapshot_sizes().len() >= 2 && routing.current_version() > 0
    });
    clear_rebalance_fail_hook();
    assert!(ok, "rebalance should progress despite injected failures");
    Ok(())
}

#[test]
fn executor_cache_survives_version_churn() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let cache = WorkerCache::new(32, 4 * 1024 * 1024, usize::MAX);
    let executor = Arc::new(Executor::new(storage.clone(), cache));

    // Seed a few buckets.
    let mut buckets = Vec::new();
    for id in 0..4u64 {
        let mut b = Bucket::new(id, vec![id as f32, id as f32]);
        for j in 0..8u64 {
            b.add_vector(Vector::new(id * 100 + j, vec![j as f32, j as f32 + 1.0]));
        }
        futures::executor::block_on(storage.put_chunk(&b))?;
        buckets.push(b);
    }

    let version = Arc::new(AtomicUsize::new(1));
    let bucket_ids: Vec<u64> = buckets.iter().map(|b| b.id).collect();

    std::thread::scope(|s| {
        for t in 0..4 {
            let exec = executor.clone();
            let version = version.clone();
            let ids = bucket_ids.clone();
            s.spawn(move || {
                for i in 0..40 {
                    // Occasionally bump the router version and mark one bucket as changed.
                    let v = if i % 7 == 0 {
                        version.fetch_add(1, Ordering::SeqCst) + 1
                    } else {
                        version.load(Ordering::SeqCst)
                    };
                    let changed = if i % 5 == 0 {
                        Arc::new(vec![ids[(i + t) % ids.len()]])
                    } else {
                        Arc::new(Vec::new())
                    };
                    let q = vec![1.0, 2.0];
                    let res = futures::executor::block_on(
                        exec.query(&q, &ids, 4, v as u64, changed, false),
                    );
                    let res = res.expect("query succeeds");
                    assert!(
                        !res.is_empty(),
                        "query should return candidates even during churn"
                    );
                    for (_id, dist, _) in res {
                        assert!(dist.is_finite());
                    }
                }
            });
        }
    });

    // Final cache version should match the latest router version we tracked.
    let final_version = version.load(Ordering::SeqCst) as u64;
    assert_eq!(executor.cache_version(), final_version);
    Ok(())
}

#[test]
fn executor_load_failures_do_not_panic() -> Result<()> {
    let tmp = tempfile::tempdir()?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let cache = WorkerCache::new(8, 2 * 1024 * 1024, usize::MAX);
    let executor = Executor::new(storage.clone(), cache);

    // Seed one bucket.
    let mut b = Bucket::new(0, vec![0.0, 0.0]);
    b.add_vector(Vector::new(1, vec![1.0, 2.0]));
    futures::executor::block_on(storage.put_chunk(&b))?;

    // Fail the first two loads.
    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();
    set_executor_fail_load_hook(move |_id| {
        let n = c.fetch_add(1, Ordering::SeqCst);
        n < 2
    });

    // Try a few times; first two loads are injected failures, the third should succeed.
    let mut last = Vec::new();
    for _ in 0..3 {
        let res = futures::executor::block_on(executor.query(
            &[1.0, 2.0],
            &[0],
            1,
            1,
            Arc::new(Vec::new()),
            false,
        ))?;
        if !res.is_empty() {
            last = res;
            break;
        }
    }
    clear_executor_fail_load_hook();
    // After failures, a subsequent load should succeed and return at least one candidate.
    assert!(!last.is_empty(), "eventual load should succeed");
    Ok(())
}

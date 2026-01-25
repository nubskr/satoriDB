use std::collections::HashSet;
use std::sync::Arc;

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
    std::env::set_var("WALRUS_QUIET", "1");
    Arc::new(Walrus::with_data_dir(tempdir.path().to_path_buf()).expect("walrus init"))
}

fn decode_bucket_vectors(storage: &Storage, bucket_id: u64) -> Vec<Vector> {
    let chunks = block_on(storage.get_chunks(bucket_id)).expect("get chunks");
    // Walk newest-to-oldest and keep the freshest copy per id to mirror executor semantics.
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for chunk in chunks.iter().rev() {
        if chunk.len() < 16 {
            continue;
        }
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&chunk[0..8]);
        let payload_len = u64::from_le_bytes(len_bytes) as usize;
        if payload_len < 16 || 8 + payload_len > chunk.len() {
            continue;
        }
        let mut off = 8;
        while off + 16 <= chunk.len() {
            let mut id_bytes = [0u8; 8];
            id_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let mut dim_bytes = [0u8; 8];
            dim_bytes.copy_from_slice(&chunk[off..off + 8]);
            off += 8;
            let dim = u64::from_le_bytes(dim_bytes) as usize;
            let bytes_needed = dim.saturating_mul(4);
            if off + bytes_needed > chunk.len() {
                break;
            }
            let mut data = Vec::with_capacity(dim);
            for fb in chunk[off..off + bytes_needed].chunks_exact(4) {
                let mut buf = [0u8; 4];
                buf.copy_from_slice(fb);
                data.push(f32::from_bits(u32::from_le_bytes(buf)));
            }
            let id = u64::from_le_bytes(id_bytes);
            if seen.insert(id) && !data.is_empty() {
                out.push(Vector { id, data });
            }
            off += bytes_needed;
        }
    }
    out
}

#[test]
fn delete_removes_vector_and_indexes() -> Result<()> {
    std::fs::create_dir_all(".tmp")?;
    let tmp = tempfile::tempdir_in(".tmp")?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let routing = Arc::new(RoutingTable::new());
    let bucket_locks = Arc::new(BucketLocks::new());
    let worker = RebalanceWorker::new_for_tests(
        storage.clone(),
        vector_index.clone(),
        bucket_index.clone(),
        routing,
        bucket_locks,
    );

    let mut bucket = Bucket::new(42, vec![0.0, 0.0]);
    for id in [10u64, 11, 12] {
        bucket.add_vector(Vector::new(id, vec![id as f32, 1.0]));
    }
    block_on(storage.put_chunk(&bucket))?;
    let ids: Vec<u64> = bucket.vectors.iter().map(|v| v.id).collect();
    vector_index.put_batch(&bucket.vectors)?;
    bucket_index.put_batch(bucket.id, &ids)?;
    block_on(worker.prime_centroids(&[bucket.clone()]))?;

    worker.delete_inline_blocking(11, None)?;

    let stored = decode_bucket_vectors(&storage, bucket.id);
    assert_eq!(stored.len(), 2);
    assert!(stored.iter().all(|v| v.id != 11));
    assert!(vector_index.get_many(&[11])?.is_empty());
    assert!(bucket_index.get_many(&[11])?.is_empty());
    Ok(())
}

#[test]
fn delete_queue_drains_under_burst_load() -> Result<()> {
    std::fs::create_dir_all(".tmp")?;
    let tmp = tempfile::tempdir_in(".tmp")?;
    let wal = init_wal(&tmp);
    let storage = Storage::new(wal);
    let vector_index = Arc::new(VectorIndex::open(tmp.path().join("vectors"))?);
    let bucket_index = Arc::new(BucketIndex::open(tmp.path().join("buckets"))?);
    let routing = Arc::new(RoutingTable::new());
    let bucket_locks = Arc::new(BucketLocks::new());
    let worker = RebalanceWorker::new_for_tests(
        storage.clone(),
        vector_index.clone(),
        bucket_index.clone(),
        routing,
        bucket_locks,
    );

    let mut bucket = Bucket::new(7, vec![0.5, 0.5]);
    for id in 100u64..180 {
        bucket.add_vector(Vector::new(id, vec![id as f32, id as f32 + 1.0]));
    }
    let all_ids: Vec<u64> = bucket.vectors.iter().map(|v| v.id).collect();
    block_on(storage.put_chunk(&bucket))?;
    vector_index.put_batch(&bucket.vectors)?;
    bucket_index.put_batch(bucket.id, &all_ids)?;
    block_on(worker.prime_centroids(&[bucket.clone()]))?;

    let to_delete: Vec<u64> = (120u64..170).collect();
    for id in &to_delete {
        let hint = if id % 2 == 0 { Some(bucket.id) } else { None };
        // Use inline delete to avoid relying on background scheduling in this regression test.
        worker.delete_inline_blocking(*id, hint)?;
    }

    let remaining = decode_bucket_vectors(&storage, bucket.id);
    let remaining_ids: HashSet<u64> = remaining.iter().map(|v| v.id).collect();
    for id in &to_delete {
        assert!(
            !remaining_ids.contains(id),
            "deleted id {} should not remain in bucket",
            id
        );
        assert!(vector_index.get_many(&[*id])?.is_empty());
        assert!(bucket_index.get_many(&[*id])?.is_empty());
    }
    let expected = all_ids.len() - to_delete.len();
    assert_eq!(
        remaining_ids.len(),
        expected,
        "expected {} survivors after deletions",
        expected
    );
    Ok(())
}

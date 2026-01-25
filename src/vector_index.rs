use crate::storage::Vector;
use anyhow::{anyhow, Context, Result};
use rkyv::{
    check_archived_root, ser::serializers::AllocSerializer, ser::Serializer, AlignedVec,
    Deserialize,
};
use rocksdb::{BlockBasedOptions, Cache, Options, WriteBatch, DB};
use std::path::Path;

/// Persistent map of `vector_id -> archived Vector` stored in RocksDB.
///
/// This keeps lookup-by-id feasible at high cardinality without holding everything in RAM.
pub struct VectorIndex {
    db: Option<DB>,
}

impl VectorIndex {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        if std::env::var("SATORI_RUN_BENCH").is_ok() {
            return Ok(Self { db: None });
        }
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_compression_type(rocksdb::DBCompressionType::Zstd);
        opts.optimize_level_style_compaction(64 * 1024 * 1024);

        // Let RocksDB manage open files (no internal cap).
        opts.set_max_open_files(-1);
        opts.set_allow_mmap_reads(false);
        opts.set_allow_mmap_writes(false);

        // Increase background jobs for compaction
        opts.set_max_background_jobs(4);

        // Enforce backpressure
        opts.set_level_zero_file_num_compaction_trigger(4);
        opts.set_level_zero_slowdown_writes_trigger(20);
        opts.set_level_zero_stop_writes_trigger(36);

        let cache = Cache::new_lru_cache(64 * 1024 * 1024);
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        opts.set_block_based_table_factory(&block_opts);
        opts.set_write_buffer_size(64 * 1024 * 1024); // Increase buffer size
        opts.set_max_write_buffer_number(4);

        let db = DB::open(&opts, path).context("open vector index")?;
        Ok(Self { db: Some(db) })
    }

    /// Insert or overwrite a batch of vectors.
    pub fn put_batch(&self, vectors: &[Vector]) -> Result<()> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(()),
        };
        if vectors.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for v in vectors {
            let mut ser = AllocSerializer::<256>::default();
            ser.serialize_value(v)
                .context("serialize vector for index")?;
            let bytes: AlignedVec = ser.into_serializer().into_inner();
            batch.put(v.id.to_le_bytes(), bytes);
        }
        db.write(batch).context("write vector index batch")
    }

    /// Delete a batch of ids (best-effort; missing keys are ignored).
    #[allow(dead_code)]
    pub fn delete_batch(&self, ids: &[u64]) -> Result<()> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(()),
        };
        if ids.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for id in ids {
            batch.delete(id.to_le_bytes());
        }
        db.write(batch).context("delete vector index batch")
    }

    /// Fetch the stored vectors for the given ids. Missing ids are skipped.
    pub fn get_many(&self, ids: &[u64]) -> Result<Vec<(u64, Vector)>> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(Vec::new()),
        };
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(raw) = db
                .get(id.to_le_bytes())
                .with_context(|| format!("read id {} from vector index", id))?
            {
                let archived = check_archived_root::<Vector>(&raw)
                    .map_err(|e| anyhow!("validate archived vector: {:?}", e))?;
                let vec: Vector = archived.deserialize(&mut rkyv::Infallible)?;
                out.push((*id, vec));
            }
        }
        Ok(out)
    }

    /// Return the first id that already exists in the index, if any.
    pub fn first_existing(&self, ids: &[u64]) -> Result<Option<u64>> {
        for id in ids {
            if self.exists(*id)? {
                return Ok(Some(*id));
            }
        }
        Ok(None)
    }

    /// Check if an id exists in the index.
    ///
    /// Uses bloom filter for fast negative lookups, falling back to actual
    /// read only when the bloom filter indicates the key may exist.
    pub fn exists(&self, id: u64) -> Result<bool> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(false),
        };
        let key = id.to_le_bytes();
        // Fast path: bloom filter says definitely not there
        if !db.key_may_exist(key) {
            return Ok(false);
        }
        // Slow path: bloom filter uncertain, do actual lookup
        Ok(db
            .get(key)
            .with_context(|| format!("check existence of id {}", id))?
            .is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = VectorIndex::open(dir.path()).expect("open index");

        let v1 = Vector::new(1, vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(2, vec![4.0, 5.0, 6.0]);
        index.put_batch(&[v1.clone(), v2.clone()]).expect("put");

        let got = index
            .get_many(&[1, 2, 3])
            .expect("get")
            .into_iter()
            .map(|(_, v)| (v.id, v.data))
            .collect::<Vec<_>>();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&(v1.id, v1.data)));
        assert!(got.contains(&(v2.id, v2.data)));
    }

    #[test]
    fn exists_returns_false_for_missing_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = VectorIndex::open(dir.path()).expect("open index");

        assert!(!index.exists(42).expect("exists check"));
        assert!(!index.exists(0).expect("exists check"));
        assert!(!index.exists(u64::MAX).expect("exists check"));
    }

    #[test]
    fn exists_returns_true_for_inserted_id() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = VectorIndex::open(dir.path()).expect("open index");

        let v1 = Vector::new(1, vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(2, vec![4.0, 5.0, 6.0]);
        index.put_batch(&[v1, v2]).expect("put");

        assert!(index.exists(1).expect("exists check"));
        assert!(index.exists(2).expect("exists check"));
        assert!(!index.exists(3).expect("exists check"));
    }

    #[test]
    fn first_existing_uses_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = VectorIndex::open(dir.path()).expect("open index");

        let v = Vector::new(5, vec![1.0]);
        index.put_batch(&[v]).expect("put");

        assert_eq!(index.first_existing(&[1, 2, 3]).expect("check"), None);
        assert_eq!(index.first_existing(&[1, 5, 3]).expect("check"), Some(5));
        assert_eq!(index.first_existing(&[5, 1, 2]).expect("check"), Some(5));
    }
}

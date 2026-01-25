use anyhow::{Context, Result};
use rocksdb::{BlockBasedOptions, Cache, Options, WriteBatch, DB};
use std::path::Path;

/// Persistent map of `vector_id -> bucket_id` stored in RocksDB.
///
/// This keeps the reverse lookup lightweight without holding everything in RAM.
pub struct BucketIndex {
    db: Option<DB>,
}

impl BucketIndex {
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

        // Increase background jobs
        opts.set_max_background_jobs(4);

        let cache = Cache::new_lru_cache(64 * 1024 * 1024);
        let mut block_opts = BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        opts.set_block_based_table_factory(&block_opts);
        opts.set_write_buffer_size(16 * 1024 * 1024);
        opts.set_max_write_buffer_number(2);
        let db = DB::open(&opts, path).context("open bucket index")?;
        Ok(Self { db: Some(db) })
    }

    /// Insert or overwrite a batch of `id -> bucket` mappings.
    pub fn put_batch(&self, bucket_id: u64, ids: &[u64]) -> Result<()> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(()),
        };
        if ids.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::default();
        for id in ids {
            batch.put(id.to_le_bytes(), bucket_id.to_le_bytes());
        }
        db.write(batch).context("write bucket index batch")
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
        db.write(batch).context("delete bucket index batch")
    }

    /// Fetch the stored bucket ids for the given vector ids. Missing ids are skipped.
    pub fn get_many(&self, ids: &[u64]) -> Result<Vec<(u64, u64)>> {
        let db = match &self.db {
            Some(db) => db,
            None => return Ok(Vec::new()),
        };
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(raw) = db
                .get(id.to_le_bytes())
                .with_context(|| format!("read id {} from bucket index", id))?
            {
                if raw.len() == 8 {
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&raw);
                    let bucket = u64::from_le_bytes(buf);
                    out.push((*id, bucket));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_index_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let index = BucketIndex::open(dir.path()).expect("open index");
        let ids = vec![1u64, 2u64, 3u64];
        index.put_batch(7, &ids).expect("put");

        let got = index
            .get_many(&[1, 3, 4])
            .expect("get")
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(got.len(), 2);
        assert!(got.contains(&(1, 7)));
        assert!(got.contains(&(3, 7)));
    }
}

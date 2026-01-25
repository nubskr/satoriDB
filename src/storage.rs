use crate::storage::wal::block::Entry;
use crate::storage::wal::runtime::Walrus;
use anyhow::Result;
use rkyv::{Archive, Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::Arc;

pub mod wal;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Bucket {
    pub id: u64,
    pub centroid: Vec<f32>,
    pub vectors: Vec<Vector>,
}

impl Vector {
    pub fn new(id: u64, data: Vec<f32>) -> Self {
        Self { id, data }
    }
}

impl Bucket {
    pub fn new(id: u64, centroid: Vec<f32>) -> Self {
        Self {
            id,
            centroid,
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum StorageExecMode {
    Direct,
    Offload,
}

#[derive(Clone)]
pub struct Storage {
    pub(crate) wal: Arc<Walrus>,
    pub(crate) mode: StorageExecMode,
}

thread_local! {
    static TL_BACKING: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static TL_RANGES: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum BucketMetaStatus {
    Active,
    Retired,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct BucketMeta {
    pub bucket_id: u64,
    pub status: BucketMetaStatus,
}

impl Storage {
    pub fn new(wal: Arc<Walrus>) -> Self {
        Self {
            wal,
            mode: StorageExecMode::Direct,
        }
    }

    pub fn with_mode(mut self, mode: StorageExecMode) -> Self {
        self.mode = mode;
        self
    }

    /// Pre-reserve thread-local buffers used during ingestion to avoid growth reallocations.
    /// `max_entries` is the expected max batch size; `approx_f32_per_vector` is an estimate
    /// of vector dimensionality to size the backing buffer conservatively.
    pub fn prewarm_thread_locals(max_entries: usize, approx_f32_per_vector: usize) {
        let backing_bytes =
            max_entries.saturating_mul(16 + approx_f32_per_vector.saturating_mul(4));
        TL_BACKING.with(|b| {
            let mut buf = b.borrow_mut();
            let cap = buf.capacity();
            if cap < backing_bytes {
                buf.reserve(backing_bytes - cap);
            }
        });
        TL_RANGES.with(|r| {
            let mut ranges = r.borrow_mut();
            let cap = ranges.capacity();
            if cap < max_entries {
                ranges.reserve(max_entries - cap);
            }
        });
    }

    pub(crate) fn topic_for(bucket_id: u64) -> String {
        format!("bucket_{}", bucket_id)
    }

    /// Persists a bucket as an append-only wal entry.
    pub async fn put_chunk(&self, bucket: &Bucket) -> Result<()> {
        // One entry per vector to keep WAL entries granular and allow per-vector recovery.
        // Serialize into a single backing buffer to reduce per-entry allocations.
        if bucket.vectors.is_empty() {
            return Ok(());
        }

        let topic = Self::topic_for(bucket.id);
        self.put_chunk_raw_with_topic(bucket.id, &topic, &bucket.vectors)
            .await
    }

    /// Fast path when only vectors are provided; avoids centroid clone.
    pub async fn put_chunk_raw(&self, bucket_id: u64, vectors: &[Vector]) -> Result<()> {
        let topic = Self::topic_for(bucket_id);
        self.put_chunk_raw_with_topic(bucket_id, &topic, vectors)
            .await
    }

    /// Same as `put_chunk_raw` but uses a precomputed topic string to avoid per-call formatting.
    pub async fn put_chunk_raw_with_topic(
        &self,
        bucket_id: u64,
        topic: &str,
        vectors: &[Vector],
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        match self.mode {
            StorageExecMode::Direct => {
                Self::put_chunk_raw_sync(self.wal.clone(), bucket_id, topic, vectors)
            }
            StorageExecMode::Offload => {
                let wal = self.wal.clone();
                let vectors = vectors.to_vec();
                let topic = topic.to_string();
                glommio::executor()
                    .spawn_blocking(move || {
                        Self::put_chunk_raw_sync(wal, bucket_id, &topic, &vectors)
                    })
                    .await
            }
        }
    }

    pub(crate) fn put_chunk_raw_sync(
        wal: Arc<Walrus>,
        bucket_id: u64,
        topic: &str,
        vectors: &[Vector],
    ) -> Result<()> {
        TL_BACKING.with(|backing_cell| {
            TL_RANGES.with(|ranges_cell| -> Result<()> {
                let mut backing = backing_cell.borrow_mut();
                backing.clear();
                // Each entry: len prefix + id + count + data bytes.
                let est: usize = vectors.iter().map(|v| v.data.len() * 4 + 24).sum();
                backing.reserve(est);

                let mut ranges = ranges_cell.borrow_mut();
                ranges.clear();
                ranges.reserve(vectors.len());

                for v in vectors {
                    let start = backing.len();
                    // payload_len excludes the outer length prefix itself.
                    let payload_len = 16 + v.data.len() * 4;
                    backing.extend_from_slice(&(payload_len as u64).to_le_bytes());
                    backing.extend_from_slice(&v.id.to_le_bytes());
                    backing.extend_from_slice(&(v.data.len() as u64).to_le_bytes());
                    let data_bytes = unsafe {
                        std::slice::from_raw_parts(v.data.as_ptr() as *const u8, v.data.len() * 4)
                    };
                    backing.extend_from_slice(data_bytes);
                    let end = backing.len();
                    ranges.push((start, end));
                }

                for (s, e) in ranges.iter() {
                    let slice = &backing[*s..*e];
                    wal.append_for_topic(topic, slice).map_err(|e| {
                        anyhow::anyhow!("Walrus append failed for bucket {}: {:?}", bucket_id, e)
                    })?;
                }
                Ok(())
            })
        })
    }

    /// Emits a bucket metadata record (active/retired) for housekeeping.
    pub async fn put_bucket_meta(&self, meta: &BucketMeta) -> Result<()> {
        match self.mode {
            StorageExecMode::Direct => Self::put_bucket_meta_sync(self.wal.clone(), meta),
            StorageExecMode::Offload => {
                let wal = self.wal.clone();
                let meta = meta.clone();
                glommio::executor()
                    .spawn_blocking(move || Self::put_bucket_meta_sync(wal, &meta))
                    .await
            }
        }
    }

    fn put_bucket_meta_sync(wal: Arc<Walrus>, meta: &BucketMeta) -> Result<()> {
        let bytes = rkyv::to_bytes::<_, 256>(meta)
            .map_err(|e| anyhow::anyhow!("Failed to serialize bucket meta: {}", e))?;
        let mut payload = Vec::with_capacity(8 + bytes.len());
        payload.extend_from_slice(&bytes.len().to_le_bytes());
        payload.extend_from_slice(bytes.as_slice());
        wal.append_for_topic("bucket_meta", &payload)
            .map_err(|e| anyhow::anyhow!("Walrus append failed (bucket_meta): {:?}", e))
    }

    /// Retrieves all chunks for a bucket.
    pub async fn get_chunks(&self, bucket_id: u64) -> Result<Vec<Vec<u8>>> {
        match self.mode {
            StorageExecMode::Direct => Self::get_chunks_sync(self.wal.clone(), bucket_id),
            StorageExecMode::Offload => {
                let wal = self.wal.clone();
                glommio::executor()
                    .spawn_blocking(move || Self::get_chunks_sync(wal, bucket_id))
                    .await
            }
        }
    }

    pub(crate) fn get_chunks_sync(wal: Arc<Walrus>, bucket_id: u64) -> Result<Vec<Vec<u8>>> {
        let topic = Self::topic_for(bucket_id);
        let topic_size = wal.get_topic_size(&topic) as usize;
        if topic_size == 0 {
            return Ok(Vec::new());
        }

        let entries: Vec<Entry> = wal
            .batch_read_for_topic(&topic, topic_size + 1024, false, Some(0))
            .map_err(|e| anyhow::anyhow!("Walrus read failed for {}: {:?}", topic, e))?;

        Ok(entries.into_iter().map(|e| e.data).collect())
    }
}

use futures::lock::Mutex;
use parking_lot::Mutex as MapMutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Shared per-bucket locks used to serialize worker writes and deletes.
pub struct BucketLocks {
    map: MapMutex<HashMap<u64, Arc<Mutex<()>>>>,
}

impl BucketLocks {
    pub fn new() -> Self {
        Self {
            map: MapMutex::new(HashMap::new()),
        }
    }

    /// Get (or create) the lock for a bucket id.
    pub fn lock_for(&self, bucket_id: u64) -> Arc<Mutex<()>> {
        let mut guard = self.map.lock();
        guard
            .entry(bucket_id)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }
}

impl Default for BucketLocks {
    fn default() -> Self {
        Self::new()
    }
}

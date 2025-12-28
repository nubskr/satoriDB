use futures::channel::oneshot;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct RouterResult {
    pub bucket_ids: Vec<u64>,
    pub routing_version: u64,
    pub affected_buckets: std::sync::Arc<Vec<u64>>,
}

pub struct RouterTask {
    pub query_vec: Vec<f32>,
    pub top_k: usize,
    pub respond_to: oneshot::Sender<anyhow::Result<RouterResult>>,
}

pub enum RouterMessage {
    Query(RouterTask),
    Refresh(oneshot::Sender<()>),
}

#[derive(Clone)]
pub struct ConsistentHashRing {
    ring: Vec<(u64, usize)>,
}

impl ConsistentHashRing {
    pub fn new(nodes: usize, virtual_nodes: usize) -> Self {
        let mut ring = Vec::new();
        for node in 0..nodes {
            for v in 0..virtual_nodes.max(1) {
                let mut hasher = DefaultHasher::new();
                (node as u64, v as u64).hash(&mut hasher);
                ring.push((hasher.finish(), node));
            }
        }
        ring.sort_by_key(|(h, _)| *h);
        Self { ring }
    }

    pub fn node_for(&self, key: u64) -> usize {
        if self.ring.is_empty() {
            return 0;
        }
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let h = hasher.finish();
        for (hash, node) in &self.ring {
            if *hash >= h {
                return *node;
            }
        }
        self.ring[0].1
    }
}

#[cfg(test)]
mod tests {
    use super::ConsistentHashRing;

    #[test]
    fn hash_ring_is_deterministic() {
        let ring = ConsistentHashRing::new(4, 8);
        assert_eq!(ring.node_for(42), ring.node_for(42));
    }

    #[test]
    fn hash_ring_handles_single_node() {
        let ring = ConsistentHashRing::new(1, 1);
        assert_eq!(ring.node_for(99), 0);
    }

    /// Empty ring (0 nodes) should return node 0 without panicking.
    #[test]
    fn hash_ring_empty_returns_zero() {
        let ring = ConsistentHashRing::new(0, 8);
        assert_eq!(ring.node_for(42), 0);
        assert_eq!(ring.node_for(0), 0);
        assert_eq!(ring.node_for(u64::MAX), 0);
    }

    /// Zero virtual nodes should still work (max(1) ensures at least 1).
    #[test]
    fn hash_ring_zero_virtual_nodes() {
        let ring = ConsistentHashRing::new(4, 0);
        let node = ring.node_for(42);
        assert!(node < 4, "node should be in valid range");
    }

    /// Multiple keys should distribute across nodes (not all to one).
    #[test]
    fn hash_ring_distributes_keys() {
        let ring = ConsistentHashRing::new(4, 8);
        let mut counts = [0usize; 4];
        for key in 0..1000u64 {
            let node = ring.node_for(key);
            counts[node] += 1;
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "all nodes should receive some keys"
        );
    }

    /// Ring should handle wraparound: hash >= all ring hashes → first entry.
    #[test]
    fn hash_ring_wraparound() {
        let ring = ConsistentHashRing::new(2, 1);
        for key in [0, u64::MAX / 2, u64::MAX] {
            let node = ring.node_for(key);
            assert!(node < 2, "node should be valid for key {}", key);
        }
    }

    /// Same key always maps to same node across ring instances with same params.
    #[test]
    fn hash_ring_consistent_across_instances() {
        let ring1 = ConsistentHashRing::new(8, 16);
        let ring2 = ConsistentHashRing::new(8, 16);
        for key in 0..100u64 {
            assert_eq!(
                ring1.node_for(key),
                ring2.node_for(key),
                "same params should yield same mapping"
            );
        }
    }

    /// Large number of nodes should not panic or misbehave.
    #[test]
    fn hash_ring_many_nodes() {
        let ring = ConsistentHashRing::new(100, 10);
        for key in 0..1000u64 {
            let node = ring.node_for(key);
            assert!(node < 100, "node {} out of range for key {}", node, key);
        }
    }
}

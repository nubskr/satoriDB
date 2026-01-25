use crate::rng::{sample_k_indices, shuffle, SplitMix64};
use crate::storage::{Bucket, Vector};
use log::debug;

pub struct Indexer;

#[derive(Clone, Copy)]
struct BuildOpts {
    rebalance: bool,
    reindex: bool,
    max_iters: usize,
    log_iters: bool,
    init: InitMode,
    force_two_buckets_k2: bool,
}

#[derive(Clone, Copy)]
enum InitMode {
    Random,
    FarthestPairK2,
}

impl Indexer {
    /// basic k-means implementation
    /// consumes vectors and returns buckets
    pub fn build_clusters(vectors: Vec<Vector>, k: usize) -> Vec<Bucket> {
        Self::build_clusters_with_opts(
            vectors,
            k,
            BuildOpts {
                rebalance: true,
                reindex: true,
                max_iters: 20,
                log_iters: true,
                init: InitMode::Random,
                force_two_buckets_k2: false,
            },
        )
    }

    /// Split a bucket into exactly two buckets (no recursive rebalancing / reindexing).
    /// Optimized for the rebalance path.
    pub fn split_bucket_once(bucket: Bucket) -> Vec<Bucket> {
        Self::build_clusters_with_opts(
            bucket.vectors,
            2,
            BuildOpts {
                rebalance: false,
                reindex: false,
                max_iters: 8,
                log_iters: false,
                init: InitMode::FarthestPairK2,
                force_two_buckets_k2: true,
            },
        )
    }

    fn build_clusters_with_opts(vectors: Vec<Vector>, k: usize, opts: BuildOpts) -> Vec<Bucket> {
        if vectors.is_empty() || k == 0 {
            return vec![];
        }

        let n = vectors.len();
        let dim = vectors[0].data.len();
        if dim == 0 {
            return vec![];
        }

        if vectors.iter().any(|v| v.data.len() != dim) {
            return vec![];
        }

        let rng = SplitMix64::seeded_from_time();

        let simd = SimdCaps::detect();

        let mut centroids = vec![0.0f32; k * dim];
        match opts.init {
            InitMode::FarthestPairK2 if k == 2 && n >= 2 => {
                let a0 = &vectors[0].data;
                let mut b = 1usize;
                let mut best = -1.0f32;
                for (i, v) in vectors.iter().enumerate().skip(1) {
                    let d = l2_sq_scalar(a0, &v.data);
                    if d > best {
                        best = d;
                        b = i;
                    }
                }

                let ab = &vectors[b].data;
                let mut a = 0usize;
                best = -1.0f32;
                for (i, v) in vectors.iter().enumerate() {
                    let d = l2_sq_scalar(ab, &v.data);
                    if d > best {
                        best = d;
                        a = i;
                    }
                }
                if a == b {
                    a = 0;
                    if a == b {
                        a = 1;
                    }
                }

                centroids[0..dim].copy_from_slice(&vectors[a].data);
                centroids[dim..2 * dim].copy_from_slice(&vectors[b].data);
            }
            _ => {
                let mut chosen_idx = sample_k_indices(&rng, vectors.len(), k);
                shuffle(&rng, &mut chosen_idx);
                for (c, idx) in chosen_idx.into_iter().take(k).enumerate() {
                    centroids[c * dim..(c + 1) * dim].copy_from_slice(&vectors[idx].data);
                }
                for c in vectors.len()..k {
                    let base = c * dim;
                    for d in 0..dim {
                        centroids[base + d] = rng.next_f32();
                    }
                }
            }
        }

        let mut assignments = vec![usize::MAX; n];

        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];

        let mut centroids_t = Vec::<f32>::new();

        let max_iters = opts.max_iters.max(1);

        for iter in 0..max_iters {
            if opts.log_iters && log::log_enabled!(log::Level::Debug) {
                debug!("kmeans iter {}/{}", iter + 1, max_iters);
            }

            sums.fill(0.0);
            counts.fill(0);

            let use_block8 = simd.avx2_fma && k >= 8;
            let use_k2 = simd.avx2_fma && k == 2;

            if use_block8 {
                transpose_centroids_aos_to_soa(&centroids, k, dim, &mut centroids_t);
            }

            let mut changed = false;

            for (i, v) in vectors.iter().enumerate() {
                let x = &v.data;

                let best = if use_k2 {
                    unsafe { nearest_centroid_k2_avx2_fma(x, &centroids, dim) }
                } else if use_block8 {
                    unsafe { nearest_centroid_block8_avx2_fma(x, &centroids_t, k, dim) }
                } else if simd.avx2 {
                    unsafe { nearest_centroid_pairwise_avx2(x, &centroids, k, dim, simd.avx2_fma) }
                } else {
                    nearest_centroid_scalar(x, &centroids, k, dim)
                };

                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }

                let base = best * dim;
                for (d, sum) in sums[base..base + dim].iter_mut().enumerate() {
                    *sum += x[d];
                }
                counts[best] += 1;
            }

            let mut reseeded = false;
            for j in 0..k {
                if counts[j] > 0 {
                    let inv = 1.0 / counts[j] as f32;
                    let base = j * dim;
                    for d in 0..dim {
                        centroids[base + d] = sums[base + d] * inv;
                    }
                } else if !vectors.is_empty() {
                    let idx = rng.gen_range(vectors.len());
                    centroids[j * dim..(j + 1) * dim].copy_from_slice(&vectors[idx].data);
                    reseeded = true;
                }
            }

            if !changed && !reseeded {
                break;
            }
        }

        let mut final_counts = vec![0usize; k];
        for &a in &assignments {
            final_counts[a] += 1;
        }

        if opts.force_two_buckets_k2
            && k == 2
            && n >= 2
            && (final_counts[0] == 0 || final_counts[1] == 0)
        {
            let mid = n / 2;
            let mut b0 = Vec::with_capacity(mid);
            let mut b1 = Vec::with_capacity(n - mid);
            for (i, v) in vectors.into_iter().enumerate() {
                if i < mid {
                    b0.push(v);
                } else {
                    b1.push(v);
                }
            }
            let c0 = centroid_of(&b0, dim);
            let c1 = centroid_of(&b1, dim);
            let mut out = Vec::with_capacity(2);
            let mut bb0 = Bucket::new(0, c0);
            bb0.vectors = b0;
            out.push(bb0);
            let mut bb1 = Bucket::new(1, c1);
            bb1.vectors = b1;
            out.push(bb1);
            return out;
        }

        let mut buckets_data: Vec<Vec<Vector>> = (0..k)
            .map(|j| Vec::with_capacity(final_counts[j]))
            .collect();

        for (i, vec) in vectors.into_iter().enumerate() {
            let cluster_idx = assignments[i];
            buckets_data[cluster_idx].push(vec);
        }

        let mut buckets = Vec::new();
        for (i, b_vectors) in buckets_data.into_iter().enumerate() {
            if !b_vectors.is_empty() {
                let centroid_vec = centroids[i * dim..(i + 1) * dim].to_vec();
                let mut b = Bucket::new(i as u64, centroid_vec);
                b.vectors = b_vectors;
                buckets.push(b);
            }
        }

        let mut out = buckets;
        if opts.rebalance {
            out = Self::rebalance_buckets(out);
        }

        if opts.reindex {
            let mut reindexed = Vec::with_capacity(out.len());
            for (i, b) in out.into_iter().enumerate() {
                let mut nb = Bucket::new(i as u64, b.centroid);
                nb.vectors = b.vectors;
                reindexed.push(nb);
            }
            return reindexed;
        }

        out
    }

    pub fn split_bucket(bucket: Bucket) -> Vec<Bucket> {
        Self::build_clusters(bucket.vectors, 2)
    }

    fn rebalance_buckets(buckets: Vec<Bucket>) -> Vec<Bucket> {
        if buckets.is_empty() {
            return buckets;
        }
        let total: usize = buckets.iter().map(|b| b.vectors.len()).sum();
        let avg = (total as f32 / buckets.len() as f32).max(1.0);
        let mut balanced = Vec::new();

        for bucket in buckets {
            if bucket.vectors.len() as f32 > avg * 1.2 && bucket.vectors.len() > 2 {
                let mut splits = Self::split_bucket(bucket);
                balanced.append(&mut splits);
            } else {
                balanced.push(bucket);
            }
        }

        balanced
    }
}

fn centroid_of(vectors: &[Vector], dim: usize) -> Vec<f32> {
    if vectors.is_empty() || dim == 0 {
        return Vec::new();
    }
    let mut sums = vec![0.0f32; dim];
    for v in vectors {
        for (sum, val) in sums.iter_mut().zip(&v.data) {
            *sum += val;
        }
    }
    let inv = 1.0 / vectors.len() as f32;
    for s in &mut sums {
        *s *= inv;
    }
    sums
}

#[inline(always)]
pub(crate) fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

#[derive(Copy, Clone)]
struct SimdCaps {
    avx2: bool,
    avx2_fma: bool,
}

impl SimdCaps {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = std::arch::is_x86_feature_detected!("avx2");
            let fma = std::arch::is_x86_feature_detected!("fma");
            Self {
                avx2,
                avx2_fma: avx2 && fma,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                avx2_fma: false,
            }
        }
    }
}

fn transpose_centroids_aos_to_soa(centroids_aos: &[f32], k: usize, dim: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(dim * k, 0.0);
    for d in 0..dim {
        let row = d * k;
        for c in 0..k {
            out[row + c] = centroids_aos[c * dim + d];
        }
    }
}

fn nearest_centroid_scalar(x: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;

    for c in 0..k {
        let base = c * dim;
        let mut acc = 0f32;
        for d in 0..dim {
            let diff = x[d] - centroids[base + d];
            acc += diff * diff;
        }
        if acc < best_dist {
            best_dist = acc;
            best = c;
        }
    }
    best
}

#[cfg(target_arch = "x86_64")]
unsafe fn nearest_centroid_pairwise_avx2(
    x: &[f32],
    centroids: &[f32],
    k: usize,
    dim: usize,
    use_fma: bool,
) -> usize {
    let mut best = 0usize;
    let mut best_dist = f32::INFINITY;

    let xp = x.as_ptr();

    for c in 0..k {
        let cp = centroids.as_ptr().add(c * dim);
        let dist = if use_fma {
            l2_dist_sq_avx2_fma_dim(xp, cp, dim)
        } else {
            l2_dist_sq_avx2_dim(xp, cp, dim)
        };
        if dist < best_dist {
            best_dist = dist;
            best = c;
        }
    }

    best
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_dist_sq_avx2_dim(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= dim {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        let sq = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, sq);
        i += 8;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();

    while i < dim {
        let d = *a.add(i) - *b.add(i);
        total += d * d;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_dist_sq_avx2_fma_dim(a: *const f32, b: *const f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0usize;

    while i + 8 <= dim {
        let va = _mm256_loadu_ps(a.add(i));
        let vb = _mm256_loadu_ps(b.add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum);
    let mut total = tmp.iter().sum::<f32>();

    while i < dim {
        let d = *a.add(i) - *b.add(i);
        total += d * d;
        i += 1;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nearest_centroid_k2_avx2_fma(x: &[f32], centroids: &[f32], dim: usize) -> usize {
    use std::arch::x86_64::*;

    let c0 = centroids.as_ptr();
    let c1 = centroids.as_ptr().add(dim);
    let xp = x.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let mut i = 0usize;
    while i + 8 <= dim {
        let vx = _mm256_loadu_ps(xp.add(i));
        let v0 = _mm256_loadu_ps(c0.add(i));
        let v1 = _mm256_loadu_ps(c1.add(i));

        let d0 = _mm256_sub_ps(vx, v0);
        let d1 = _mm256_sub_ps(vx, v1);

        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        i += 8;
    }

    let mut t0 = [0f32; 8];
    let mut t1 = [0f32; 8];
    _mm256_storeu_ps(t0.as_mut_ptr(), acc0);
    _mm256_storeu_ps(t1.as_mut_ptr(), acc1);

    let mut dist0 = t0.iter().sum::<f32>();
    let mut dist1 = t1.iter().sum::<f32>();

    while i < dim {
        let xv = *xp.add(i);
        let d0 = xv - *c0.add(i);
        let d1 = xv - *c1.add(i);
        dist0 += d0 * d0;
        dist1 += d1 * d1;
        i += 1;
    }

    if dist0 <= dist1 {
        0
    } else {
        1
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn nearest_centroid_block8_avx2_fma(
    x: &[f32],
    centroids_t: &[f32],
    k: usize,
    dim: usize,
) -> usize {
    use std::arch::x86_64::*;

    let mut best_c = 0usize;
    let mut best_dist = f32::INFINITY;

    let mut c0 = 0usize;
    let mut tmp = [0f32; 8];

    while c0 + 8 <= k {
        let base = c0;
        let mut acc = _mm256_setzero_ps();

        for d in 0..dim {
            let p = _mm256_set1_ps(*x.get_unchecked(d));
            let cvals = _mm256_loadu_ps(centroids_t.as_ptr().add(d * k + base));
            let diff = _mm256_sub_ps(p, cvals);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);

        for (lane, val) in tmp.iter().enumerate() {
            let dist = *val;
            if dist < best_dist {
                best_dist = dist;
                best_c = base + lane;
            }
        }

        c0 += 8;
    }

    while c0 < k {
        let mut accs = 0f32;
        for d in 0..dim {
            let cd = *centroids_t.get_unchecked(d * k + c0);
            let diff = *x.get_unchecked(d) - cd;
            accs += diff * diff;
        }
        if accs < best_dist {
            best_dist = accs;
            best_c = c0;
        }
        c0 += 1;
    }

    best_c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_reasonable_clusters() {
        let vectors = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![0.0, 1.0]),
            Vector::new(2, vec![10.0, 10.0]),
            Vector::new(3, vec![10.0, 11.0]),
        ];
        let buckets = Indexer::build_clusters(vectors, 2);
        assert_eq!(buckets.len(), 2);
        let total_vectors: usize = buckets.iter().map(|b| b.vectors.len()).sum();
        assert_eq!(total_vectors, 4);
    }

    #[test]
    fn handles_empty() {
        let buckets = Indexer::build_clusters(Vec::new(), 2);
        assert!(buckets.is_empty());
    }

    #[test]
    fn split_bucket_once_returns_two_buckets() {
        let mut b = Bucket::new(0, vec![]);
        b.vectors = vec![
            Vector::new(0, vec![1.0, 1.0]),
            Vector::new(1, vec![1.0, 1.0]),
            Vector::new(2, vec![1.0, 1.0]),
            Vector::new(3, vec![1.0, 1.0]),
        ];
        let splits = Indexer::split_bucket_once(b);
        assert_eq!(splits.len(), 2);
        let total: usize = splits.iter().map(|s| s.vectors.len()).sum();
        assert_eq!(total, 4);
        assert!(splits.iter().all(|s| !s.vectors.is_empty()));
    }

    /// Splitting two vectors should yield two buckets with one vector each.
    #[test]
    fn split_two_vectors_yields_two_buckets() {
        let mut b = Bucket::new(0, vec![0.0, 0.0]);
        b.vectors = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![10.0, 10.0]),
        ];
        let splits = Indexer::split_bucket_once(b);
        assert_eq!(splits.len(), 2, "two vectors should split into two buckets");
        let total: usize = splits.iter().map(|s| s.vectors.len()).sum();
        assert_eq!(total, 2);
        assert!(
            splits.iter().all(|s| s.vectors.len() == 1),
            "each bucket should have one vector"
        );
    }

    /// Splitting a single-vector bucket should still return 2 buckets (via fallback).
    /// This ensures the split never returns a single bucket.
    #[test]
    fn split_single_vector_uses_fallback() {
        let mut b = Bucket::new(0, vec![5.0, 5.0]);
        b.vectors = vec![Vector::new(0, vec![5.0, 5.0])];
        let splits = Indexer::split_bucket_once(b);
        assert!(
            splits.is_empty() || splits.len() == 1,
            "single vector cannot be split into two"
        );
    }

    /// k=0 should return empty (no clusters requested).
    #[test]
    fn k_zero_returns_empty() {
        let vectors = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![1.0, 1.0]),
        ];
        let buckets = Indexer::build_clusters(vectors, 0);
        assert!(buckets.is_empty(), "k=0 should return empty");
    }

    /// Ragged dimensions (vectors with different lengths) should return empty.
    #[test]
    fn ragged_dimensions_returns_empty() {
        let vectors = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![1.0, 1.0, 1.0]),
        ];
        let buckets = Indexer::build_clusters(vectors, 2);
        assert!(buckets.is_empty(), "ragged dimensions should return empty");
    }

    /// Zero-dimension vectors should return empty.
    #[test]
    fn zero_dimension_returns_empty() {
        let vectors = vec![Vector::new(0, vec![]), Vector::new(1, vec![])];
        let buckets = Indexer::build_clusters(vectors, 2);
        assert!(
            buckets.is_empty(),
            "zero-dimension vectors should return empty"
        );
    }

    /// Total vector count is preserved after clustering.
    #[test]
    fn total_vectors_preserved() {
        let n = 100;
        let vectors: Vec<Vector> = (0..n)
            .map(|i| Vector::new(i as u64, vec![i as f32, (i * 2) as f32]))
            .collect();
        let buckets = Indexer::build_clusters(vectors, 5);
        let total: usize = buckets.iter().map(|b| b.vectors.len()).sum();
        assert_eq!(total, n, "all vectors should be preserved after clustering");
    }
}

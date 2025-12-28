# SatoriDB Architecture

## Overview

SatoriDB is a billion-scale embedded vector database with a two-tier architecture:

1. **Tier 1 - Routing (in-memory):** HNSW index over quantized bucket centroids
2. **Tier 2 - Scanning (on-disk):** Parallel bucket scanning with exact L2 distance

The design achieves 95%+ recall at billion-vector scale on a single machine by trading routing precision for throughput—the router finds *candidate buckets* approximately, then exact scanning finds the actual nearest neighbors.

```
Query flow:

  Query vector
       │
       ▼
  ┌─────────────────────────┐
  │  Router (HNSW, 8-bit)   │  ← Tier 1: "which ~500 buckets to probe?"
  └───────────┬─────────────┘
              │
              ▼
  ┌─────────────────────────┐
  │  Parallel Bucket Scan   │  ← Tier 2: exact L2 on ~1M vectors
  │  (N workers, cached)    │
  └───────────┬─────────────┘
              │
              ▼
       Top-K results
```

---

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              SatoriDb                                   │
│                  (lifecycle, public API, thread ownership)              │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            SatoriHandle                                 │
│                    (cloneable, stateless coordinator)                   │
└────┬──────────────────┬──────────────────┬──────────────────┬───────────┘
     │                  │                  │                  │
     ▼                  ▼                  ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐
│RouterManager │  │ HashRing      │  │ Workers (N)   │  │ RebalanceWorker │
│(1 thread)    │  │ (stateless)   │  │ (N threads)   │  │ (1 thread)      │
├──────────────┤  ├───────────────┤  ├───────────────┤  ├─────────────────┤
│ HNSW index   │  │ bucket →      │  │ query exec    │  │ split/merge     │
│ centroids    │  │ worker shard  │  │ upsert        │  │ delete          │
│ quantizer    │  │               │  │ LRU cache     │  │ centroid track  │
└──────┬───────┘  └───────────────┘  └───────┬───────┘  └────────┬────────┘
       │                                     │                   │
       │                                     ▼                   │
       │         ┌───────────────────────────────────────────────┼────────┐
       │         │                   Storage                     │        │
       │         ├───────────────────────────────────────────────┴────────┤
       │         │                    Walrus (WAL)                        │
       │         │               (topic per bucket, io_uring)             │
       │         └───────────────────────────┬────────────────────────────┘
       │                                     │
       │         ┌───────────────────────────┴────────────────────────────┐
       │         │                                                        │
       │         ▼                                                        ▼
       │  ┌─────────────────────┐                            ┌─────────────────────┐
       │  │    VectorIndex      │                            │    BucketIndex      │
       │  │     (RocksDB)       │                            │     (RocksDB)       │
       │  ├─────────────────────┤                            ├─────────────────────┤
       │  │ vector_id → vector  │                            │ vector_id → bucket  │
       │  └─────────────────────┘                            └─────────────────────┘
       │
       ▼
┌──────────────────┐
│  RoutingTable    │
│  (lock-free Arc) │
├──────────────────┤
│ atomic version   │
│ RwLock<Router>   │
│ changed buckets  │
└──────────────────┘
       ▲
       │
  (workers snapshot
   for cache invalidation)
```

---

## Communication Patterns

### Channels Everywhere

Components communicate via message passing, not shared mutable state:

```
SatoriHandle ──crossbeam channel──► RouterManager
             ──async_channel──────► Workers[0..N]
             ──async_channel──────► RebalanceWorker
```

Each component follows the same pattern:

```rust
loop {
    match receiver.recv().await {
        Message::DoThing { respond_to } => {
            let result = do_thing();
            respond_to.send(result);
        }
        Message::Shutdown { respond_to } => {
            respond_to.send(());
            break;
        }
    }
}
```

### Single Writer Principle

Every piece of mutable state has exactly one writer:

| State | Owner | Others |
|-------|-------|--------|
| HNSW index | RouterManager | read via RoutingTable snapshot |
| Worker cache | Worker (per-thread) | nobody else |
| Centroids map | RebalanceWorker | RouterManager reads via channel |
| WAL | Walrus (append-only) | readers don't conflict |

### Version-Based Invalidation

No explicit cache invalidation messages. Workers detect staleness via version numbers:

```rust
// RoutingTable
pub fn install(&self, router: Router, changed_buckets: Vec<u64>) -> u64 {
    let next = self.version.fetch_add(1, Ordering::AcqRel) + 1;
    *self.router.write() = Some(RoutingData { router, changed });
    next
}

// Executor (in worker)
if self.cache_version.load() != routing_version {
    cache.invalidate_many(&changed_buckets);
    self.cache_version.store(routing_version);
}
```

```
  Time ─────────────────────────────────────────────────────────────────────►

  RoutingTable        │ version=1 │     │ version=2 │         │ version=3 │
  version             └───────────┘     └───────────┘         └───────────┘
                                              │
  RebalanceWorker                             │
  splits bucket 42  ──────────────────────────┴────► install(router, [42, A, B])
                                                            │
                                                     changed_buckets = [42]
                                                            │
                                                            ▼
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                                                                           │
  │   Worker 0 (cache_version=1)          Worker 1 (cache_version=1)          │
  │                                                                           │
  │   on next query:                      on next query:                      │
  │   ┌──────────────────────────┐        ┌──────────────────────────┐        │
  │   │ routing_version = 2      │        │ routing_version = 2      │        │
  │   │ cache_version = 1        │        │ cache_version = 1        │        │
  │   │                          │        │                          │        │
  │   │ 2 != 1 → stale!          │        │ 2 != 1 → stale!          │        │
  │   │                          │        │                          │        │
  │   │ invalidate bucket 42     │        │ invalidate bucket 42     │        │
  │   │ cache_version = 2        │        │ cache_version = 2        │        │
  │   └──────────────────────────┘        └──────────────────────────┘        │
  │                                                                           │
  └───────────────────────────────────────────────────────────────────────────┘

  No messages sent. Workers lazily discover changes on next operation.
```

---

## Data Flow

### Query

```
1. SatoriHandle.query(vector, top_k)
2. → RouterManager.Query → HNSW search → bucket_ids
3. → HashRing.node_for(bucket_id) → worker shard assignment
4. → Workers receive QueryRequest with their bucket subset
5. → Each worker:
      a. Check LRU cache for bucket
      b. Cache miss → Storage.get_chunks() from WAL
      c. L2 distance scan (SIMD accelerated)
      d. Return local top-k
6. → Coordinator merges results, sorts, returns global top-k
```

```
                              Query Vector
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │    RouterManager    │
                        │  ┌───────────────┐  │
                        │  │  HNSW Index   │  │
                        │  │  (quantized)  │  │
                        │  └───────┬───────┘  │
                        └──────────┼──────────┘
                                   │
                          bucket_ids: [42, 17, 99, 203, ...]
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
            ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
            │  Worker 0   │ │  Worker 1   │ │  Worker 2   │
            │ buckets:    │ │ buckets:    │ │ buckets:    │
            │ [42, 203]   │ │ [17]        │ │ [99]        │
            ├─────────────┤ ├─────────────┤ ├─────────────┤
            │ cache hit?  │ │ cache hit?  │ │ cache hit?  │
            │     │       │ │     │       │ │     │       │
            │     ▼       │ │     ▼       │ │     ▼       │
            │  L2 scan    │ │  L2 scan    │ │  L2 scan    │
            │  (SIMD)     │ │  (SIMD)     │ │  (SIMD)     │
            └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                   │               │               │
                   └───────────────┼───────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Merge & Sort      │
                        │   Return top-k      │
                        └─────────────────────┘
```

### Insert

```
1. SatoriHandle.upsert(id, vector)
2. → RouterManager.RouteOrInit → get target bucket_id
3. → HashRing.node_for(bucket_id) → worker shard
4. → Worker:
      a. BucketLocks.lock_for(bucket_id).await
      b. VectorIndex.exists(id)? → reject duplicate
      c. Storage.put_chunk() → append to WAL
      d. VectorIndex.put_batch() → RocksDB
      e. BucketIndex.put_batch() → RocksDB
5. → RouterManager.ApplyUpsert → update running centroid
```

```
                         Insert(id=7, vector=[...])
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   RouterManager     │
                        │   RouteOrInit       │──────┐
                        └──────────┬──────────┘      │
                                   │                 │ (if no buckets exist,
                          bucket_id = 42             │  create bucket 0)
                                   │                 │
                                   ▼                 │
                        ┌─────────────────────┐      │
                        │   ConsistentHash    │      │
                        │   node_for(42) = 1  │      │
                        └──────────┬──────────┘      │
                                   │                 │
                                   ▼                 │
                        ┌─────────────────────┐      │
                        │     Worker 1        │      │
                        ├─────────────────────┤      │
                        │ lock bucket 42      │      │
                        │         │           │      │
                        │         ▼           │      │
                        │ ┌─────────────────┐ │      │
                        │ │ VectorIndex     │ │      │
                        │ │ exists(7)?  NO  │ │      │
                        │ └────────┬────────┘ │      │
                        │          │          │      │
                        │          ▼          │      │
                        │ ┌─────────────────┐ │      │
                        │ │ WAL.append()    │ │      │
                        │ └────────┬────────┘ │      │
                        │          │          │      │
                        │          ▼          │      │
                        │ ┌─────────────────┐ │      │
                        │ │ VectorIndex.put │ │      │
                        │ │ BucketIndex.put │ │      │
                        │ └─────────────────┘ │      │
                        └──────────┬──────────┘      │
                                   │                 │
                                   ▼                 │
                        ┌─────────────────────┐      │
                        │   RouterManager     │◄─────┘
                        │   ApplyUpsert       │
                        │   (update centroid) │
                        └─────────────────────┘
```

### Split (Rebalancing)

The split operation uses a **cut-over-then-drain** pattern:

```
1. RebalanceWorker detects oversized bucket (> threshold)
2. Sample vectors, k-means cluster → 2 centroids
3. Allocate new bucket IDs (A, B)
4. Update centroids map:
      - INSERT A, B
      - REMOVE old bucket
5. Rebuild router → new traffic goes to A, B immediately
6. Drain loop:
      while old_bucket not empty:
          a. Peek batch from old bucket's WAL
          b. Assign each vector to A or B (by distance)
          c. Write to A and B
          d. Update BucketIndex
          e. Checkpoint (consume) the batch from old bucket
7. Old bucket is now empty, effectively deleted
```

```
 BEFORE SPLIT                          AFTER ROUTER UPDATE
 ────────────                          ───────────────────

    Router                                  Router
  ┌────────┐                              ┌────────┐
  │ HNSW:  │                              │ HNSW:  │
  │ [0,1,2]│                              │ [0,1,A,B]  ◄── bucket 2 replaced
  └────┬───┘                              └────┬───┘
       │                                       │
       ▼                                       ▼
  ┌─────────┐                             ┌─────────┐
  │Bucket 0 │                             │Bucket 0 │
  │ (1000)  │                             │ (1000)  │
  ├─────────┤                             ├─────────┤
  │Bucket 1 │                             │Bucket 1 │
  │ (1500)  │                             │ (1500)  │
  ├─────────┤                             ├─────────┤
  │Bucket 2 │ ◄── oversized!              │Bucket A │ ◄── new, empty
  │ (5000)  │                             │ (0)     │
  └─────────┘                             ├─────────┤
                                          │Bucket B │ ◄── new, empty
                                          │ (0)     │
                                          └─────────┘

                                          │Bucket 2 │ ◄── zombie, draining
                                          │ (5000)  │     (invisible to router)
                                          └─────────┘


 DRAIN LOOP (runs until bucket 2 is empty)
 ──────────────────────────────────────────

  Iteration 1:
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Bucket 2 (WAL)          Bucket A           Bucket B           │
  │  ┌─────────────┐         ┌────────┐         ┌────────┐          │
  │  │ v1 v2 v3... │─peek───►│        │         │        │          │
  │  │             │         │        │         │        │          │
  │  │             │  assign │ v1, v3 │         │ v2     │          │
  │  │             │  by L2  │        │         │        │          │
  │  └─────────────┘         └────────┘         └────────┘          │
  │        │                                                        │
  │        ▼                                                        │
  │   checkpoint                                                    │
  │   (consume batch)                                               │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  Iteration 2:
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Bucket 2 (WAL)          Bucket A           Bucket B           │
  │  ┌─────────────┐         ┌────────┐         ┌────────┐          │
  │  │ v4 v5 v6... │─peek───►│ v1, v3 │         │ v2     │          │
  │  │             │         │ v4, v6 │         │ v5     │          │
  │  │             │         │        │         │        │          │
  │  └─────────────┘         └────────┘         └────────┘          │
  │        │                                                        │
  │        ▼                                                        │
  │   checkpoint                                                    │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

  ...repeat until bucket 2 is empty...

  Final state:
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Bucket 2 (WAL)          Bucket A           Bucket B           │
  │  ┌─────────────┐         ┌────────┐         ┌────────┐          │
  │  │   (empty)   │         │ ~2500  │         │ ~2500  │          │
  │  │             │         │ vectors│         │ vectors│          │
  │  └─────────────┘         └────────┘         └────────┘          │
  │        │                                                        │
  │        ▼                                                        │
  │   bucket 2 gone                                                 │
  │   (no tombstone needed)                                         │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

This is "graceful shutdown" for a data structure:
- Remove from load balancer (router)
- Drain in-flight work (remaining vectors)
- Terminate when idle (empty bucket)

---

## CPU Pinning & Thread Model

Workers are pinned to specific CPU cores using glommio's `LocalExecutor`:

```rust
// embedded.rs
let pin_cpu = i % num_cpus::get().max(1);
let builder = LocalExecutorBuilder::new(Placement::Fixed(pin_cpu))
    .name(&format!("worker-{}", i));
```

```
 CPU 0          CPU 1          CPU 2          CPU 3
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Worker 0 │   │ Worker 1 │   │ Worker 2 │   │ Worker 3 │
│          │   │          │   │          │   │          │
│ glommio  │   │ glommio  │   │ glommio  │   │ glommio  │
│ executor │   │ executor │   │ executor │   │ executor │
│          │   │          │   │          │   │          │
│ L1/L2    │   │ L1/L2    │   │ L1/L2    │   │ L1/L2    │
│ cache    │   │ cache    │   │ cache    │   │ cache    │
│ affinity │   │ affinity │   │ affinity │   │ affinity │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                           │
                    io_uring (shared)
```

### Why Pin?

1. **Cache locality**: Worker's LRU cache stays hot in L1/L2
2. **No migration overhead**: OS won't move thread between cores
3. **Predictable latency**: No cache invalidation from core switches
4. **NUMA awareness**: On multi-socket systems, memory stays local

### Thread Inventory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Thread Layout                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Main Thread                                                           │
│   └── SatoriDb (lifecycle, blocking API wrappers)                       │
│                                                                         │
│   Worker Threads (N, pinned to CPU 0..N-1)                              │
│   ├── Worker 0  ─── glommio LocalExecutor ─── channel receiver          │
│   ├── Worker 1  ─── glommio LocalExecutor ─── channel receiver          │
│   ├── Worker 2  ─── glommio LocalExecutor ─── channel receiver          │
│   └── ...                                                               │
│                                                                         │
│   Router Thread (unpinned)                                              │
│   └── RouterManager ─── crossbeam receiver ─── HNSW owner               │
│                                                                         │
│   Rebalancer Thread (unpinned, or pinned if configured)                 │
│   └── RebalanceWorker ─── glommio LocalExecutor ─── split/delete        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### glommio + io_uring

Workers use glommio, which is built on io_uring:

- **Async I/O without thread pool**: io_uring does kernel-side async
- **Single-threaded executors**: No work-stealing, no cross-core synchronization
- **Cooperative scheduling**: Tasks yield explicitly, no preemption
- **Linux only**: io_uring requires kernel 5.8+

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        glommio LocalExecutor                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Task Queue                    io_uring                                │
│  ┌──────────────┐            ┌──────────────┐                           │
│  │ task1        │            │ SQ (submit)  │──────► kernel             │
│  │ task2        │            ├──────────────┤                           │
│  │ task3        │            │ CQ (complete)│◄────── kernel             │
│  │ ...         │            └──────────────┘                           │
│  └──────────────┘                   │                                   │
│        │                            │                                   │
│        ▼                            ▼                                   │
│   ┌─────────────────────────────────────────┐                           │
│   │           Event Loop                    │                           │
│   │   1. Poll CQ for completions            │                           │
│   │   2. Wake tasks waiting on I/O          │                           │
│   │   3. Run ready tasks                    │                           │
│   │   4. Submit new I/O to SQ               │                           │
│   └─────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Shared-Nothing Design

Each worker is isolated:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Worker N                                │
├─────────────────────────────────────────────────────────────────┤
│  LocalExecutor (glommio, pinned to CPU N)                       │
│  ├── channel receiver (owned)                                   │
│  ├── Executor (owned)                                           │
│  │   └── WorkerCache (owned, LRU, no cross-worker sharing)      │
│  └── Storage handle (Arc to WAL, append-only so no conflicts)   │
└─────────────────────────────────────────────────────────────────┘
```

Workers share *references* to:
- `Arc<Walrus>` — append-only, no coordination needed
- `Arc<VectorIndex>` — RocksDB handles concurrency
- `Arc<BucketIndex>` — same
- `Arc<BucketLocks>` — per-bucket granularity, rarely collide

They share *no mutable state*. This is the actor model without the framework.

### Worker Cache (LRU)

Each worker has a fixed-size arena-allocated LRU cache:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WorkerCache                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HashMap<bucket_id, slot_index>     Doubly-linked list (LRU order)     │
│  ┌─────────────────────────┐        ┌─────────────────────────────┐     │
│  │ 42 → slot 0             │        │ head                        │     │
│  │ 17 → slot 1             │        │   ↓                         │     │
│  │ 99 → slot 2             │        │ [slot 2] ←→ [slot 0] ←→ [slot 1]  │
│  └─────────────────────────┘        │                         ↑   │     │
│                                     │                       tail  │     │
│                                     └─────────────────────────────┘     │
│                                                                         │
│   Arena (pre-allocated, fixed size)                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ slot 0          │ slot 1          │ slot 2          │ ...       │    │
│  │ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────┐ │           │    │
│  │ │ bucket 42   │ │ │ bucket 17   │ │ │ bucket 99   │ │           │    │
│  │ │ data...     │ │ │ data...     │ │ │ data...     │ │           │    │
│  │ │ (128MB max) │ │ │             │ │ │             │ │           │    │
│  │ └─────────────┘ │ └─────────────┘ │ └─────────────┘ │           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   On access: move to head                                               │
│   On eviction: remove tail, reuse slot                                  │
│   On version change: invalidate changed buckets                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Consistent Hash Ring

Buckets are assigned to workers via consistent hashing:

```
                        Hash Ring (virtual nodes)

                               0°
                               │
                       ┌───────┴───────┐
                      ╱                 ╲
                    W0                   W1
                   ╱                       ╲
                 90°                        270°
                  │                          │
                  W2                        W3
                   ╲                       ╱
                    W0                   W1
                      ╲                 ╱
                       └───────┬───────┘
                               │
                              180°


   bucket_id = 42
       │
       ▼
   hash(42) = 0x7A3F...  ──────►  lands between W1 and W2
       │                                    │
       ▼                                    ▼
   node_for(42) = Worker 2         (clockwise to next node)


   Distribution with 4 workers, 8 virtual nodes each:

   Worker 0: handles buckets hashing to ~25% of ring
   Worker 1: handles buckets hashing to ~25% of ring
   Worker 2: handles buckets hashing to ~25% of ring
   Worker 3: handles buckets hashing to ~25% of ring
```

---

## Lock Inventory

The system has very few locks:

| Lock | Location | Contention |
|------|----------|------------|
| `RwLock<Router>` | RoutingTable | Workers snapshot() and leave immediately |
| `Mutex<WorkerCache>` | Executor | Per-worker, zero cross-thread contention |
| `RwLock<HashMap>` | RebalanceState | Only touched by rebalancer thread |
| `DashMap<Mutex>` | BucketLocks | Per-bucket, different buckets = different locks |

The hot path has zero lock contention:

```
Query:
  RoutingTable.snapshot()     → clone Arc, release immediately
  HashRing.node_for()         → pure function
  Worker.cache.get()          → local mutex, no contention
  Storage.get_chunks()        → WAL read
  L2 scan                     → pure compute
```

---

## Durability Model

- All writes go through Walrus (WAL) before indexes
- fsync is scheduled on a timer (default 200ms)
- Insert returns after WAL append, before fsync
- Crash between append and fsync = data loss window
- Each bucket is a separate WAL topic (no global log contention)

```
                              WAL Structure (Walrus)

  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │   Topic: "bucket_0"              Topic: "bucket_1"                      │
  │  ┌─────────────────────┐        ┌─────────────────────┐                 │
  │  │ entry: [len][id][dim][data]  │ entry: [len][id][dim][data]           │
  │  │ entry: [len][id][dim][data]  │ entry: [len][id][dim][data]           │
  │  │ entry: [len][id][dim][data]  │ entry: ...                            │
  │  │ ...                 │        │                     │                 │
  │  │       ▲             │        │                     │                 │
  │  │       │ checkpoint  │        │                     │                 │
  │  │       │ cursor      │        │                     │                 │
  │  └───────┴─────────────┘        └─────────────────────┘                 │
  │                                                                         │
  │   Topic: "bucket_42"             Topic: "router_snapshot"               │
  │  ┌─────────────────────┐        ┌─────────────────────┐                 │
  │  │ entry: ...          │        │ [min][max][buckets...]                │
  │  │ entry: ...          │        │ (serialized via rkyv)                 │
  │  │                     │        │                     │                 │
  │  └─────────────────────┘        └─────────────────────┘                 │
  │                                                                         │
  └─────────────────────────────────────────────────────────────────────────┘

  Write path:
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   Worker                                                               │
  │      │                                                                 │
  │      ▼                                                                 │
  │   Storage.put_chunk(bucket_id, vectors)                                │
  │      │                                                                 │
  │      ▼                                                                 │
  │   Walrus.append_for_topic("bucket_{id}", serialized_entry)             │
  │      │                                                                 │
  │      ├──► write to memory buffer ──► return immediately (fast path)    │
  │      │                                                                 │
  │      └──► background: fsync every 200ms (configurable)                 │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘

  Crash recovery:
  ┌────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │   On startup:                                                          │
  │   1. Walrus scans all topic files                                      │
  │   2. RouterManager loads "router_snapshot" → rebuild HNSW              │
  │   3. RouterManager applies "router_updates" since snapshot             │
  │   4. Workers ready (buckets loaded on-demand)                          │
  │                                                                        │
  │   Data not fsync'd before crash is lost.                               │
  │   VectorIndex/BucketIndex (RocksDB) may be ahead of WAL.               │
  │   WAL is source of truth for bucket contents.                          │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## Scaling Characteristics

### What scales well:

| Aspect | Why |
|--------|-----|
| Vector count | Buckets are independent, parallel scanning |
| Query throughput | N workers, each with own cache |
| Write throughput | Per-bucket locking, no global contention |
| Memory | Quantized router (~1 byte/dim), bounded caches |

### Known ceilings:

| Aspect | Bottleneck |
|--------|------------|
| Query routing | RouterManager is single-threaded |
| Router rebuild | O(buckets), causes latency spikes every ~1000 inserts |
| Cold start | HNSW rebuilt from centroids on startup |
| Deletes | O(bucket_size) per delete (rewrite entire bucket) |

### Billion-scale math:

```
1B vectors ÷ 2000 per bucket = 500k buckets

Router memory:
  500k × 768 dims × 1 byte (quantized) ≈ 384MB
  + HNSW graph overhead ≈ 1-2GB total

Query (probing 500 buckets):
  500 buckets × 2000 vectors = 1M L2 distance calculations
  Distributed across N workers
  SIMD accelerated
```

---

## Design Decisions

The architecture is opinionated:

| Decision | Choice | Rejected alternative |
|----------|--------|----------------------|
| Threading | 1 thread per role, channels | Thread pool + shared state |
| WAL | Walrus, io_uring, Linux only | Cross-platform, pluggable |
| Indexes | RocksDB, hardcoded | Pluggable backend |
| Quantization | 8-bit scalar, fixed at init | Adaptive precision |
| Routing | HNSW always | Flat/IVF options |
| Splits | 2-way only | k-way, configurable |
| API | Embedded library | Server mode |

Configuration surface is minimal:

```rust
SatoriDb::builder("name")
    .workers(N)        // thread count
    .fsync_ms(N)       // durability interval
    .data_dir(path)    // storage location
    .build()
```

Power-user tuning via environment variables:

```
SATORI_REBALANCE_THRESHOLD=2000    # vectors before split
SATORI_WORKER_CACHE_BUCKETS=128    # LRU cache size
SATORI_ROUTER_REBUILD_EVERY=1000   # inserts between rebuilds
SATORI_REBALANCE_POLL_MS=300000    # rebalancer wake interval
```

---

## Key Insights

1. **Distributed patterns at small scale**: The architecture uses patterns from distributed systems (cut-over-then-drain, version-based invalidation, message passing) but applies them to threads in a single process. The patterns work because the *problem structure* is the same—concurrent actors with partial visibility.

2. **Boring components, clever composition**: Each component is simple (loop over channel, match on message). The sophistication is in how they're wired together.

3. **Optimistic concurrency everywhere**: Version numbers instead of locks. Stale reads are acceptable. Conflicts are rare.

4. **WAL as source of truth**: A bucket *is* its WAL topic. Splitting is log compaction. Deletion is rewriting. No separate "bucket store" to coordinate with.

5. **Escape complexity via ownership**: Instead of synchronizing access to shared state, each component owns its state exclusively. "Who writes X? One guy. Always."

//! SatoriDB: Billion-scale embedded vector database.
//!
//! Two-tier architecture: HNSW routing in RAM + parallel bucket scanning on disk.
//! Handles 1B+ vectors with 95%+ recall.
//!
//! # Quick Start
//!
//! ```no_run
//! use satoridb::SatoriDb;
//!
//! fn main() -> anyhow::Result<()> {
//!     let db = SatoriDb::open("my_app")?;
//!
//!     db.insert(1, vec![0.1, 0.2, 0.3])?;
//!     let results = db.query(vec![0.15, 0.25, 0.35], 10)?;
//!
//!     Ok(()) // auto-shutdown on drop
//! }
//! ```
//!
//! # Configuration
//!
//! ```no_run
//! use satoridb::SatoriDb;
//!
//! fn main() -> anyhow::Result<()> {
//!     let db = SatoriDb::builder("my_app")
//!         .workers(4)
//!         .fsync_ms(100)
//!         .data_dir("/custom/path")
//!         .build()?;
//!     # Ok(())
//! }
//! ```
//!
//! # Core Operations
//!
//! ```no_run
//! # use satoridb::SatoriDb;
//! # fn main() -> anyhow::Result<()> {
//! # let db = SatoriDb::open("my_app")?;
//! // Insert (rejects duplicates)
//! db.insert(1, vec![0.1, 0.2, 0.3])?;
//!
//! // Query nearest neighbors
//! let results = db.query(vec![0.1, 0.2, 0.3], 10)?;
//!
//! // Query with vectors returned
//! let results = db.query_with_vectors(vec![0.1, 0.2, 0.3], 10)?;
//!
//! // Fetch by ID
//! let vectors = db.get(vec![1, 2, 3])?;
//!
//! // Delete
//! db.delete(1)?;
//!
//! // Flush to disk
//! db.flush()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Async API
//!
//! ```no_run
//! # use satoridb::SatoriDb;
//! async fn example() -> anyhow::Result<()> {
//!     let db = SatoriDb::open("my_app")?;
//!
//!     db.insert_async(1, vec![0.1, 0.2, 0.3]).await?;
//!     let results = db.query_async(vec![0.1, 0.2, 0.3], 10).await?;
//!     db.delete_async(1).await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! # Platform & Stability
//!
//! - Linux-only (requires io_uring, kernel 5.8+)
//! - Pre-1.0: expect breaking API changes
//! - Single-process embedded database

// Public library exports so integration tests and external tooling can use the same modules.
#[doc(hidden)]
pub mod bucket_index;
#[doc(hidden)]
pub mod bucket_locks;
#[doc(hidden)]
pub mod bvecs;
#[doc(hidden)]
pub mod executor;
#[doc(hidden)]
pub mod flatbin;
#[doc(hidden)]
pub mod fvecs;
#[doc(hidden)]
pub mod gnd;
#[doc(hidden)]
pub mod indexer;
#[doc(hidden)]
pub mod ingest_control;
#[doc(hidden)]
pub mod ingest_counter;
#[doc(hidden)]
pub mod quantizer;
#[doc(hidden)]
pub mod rebalancer;
#[doc(hidden)]
pub mod rng;
#[doc(hidden)]
pub mod router;
#[doc(hidden)]
pub mod router_hnsw;
#[doc(hidden)]
pub mod router_manager;
#[doc(hidden)]
pub mod s3_reader;
#[doc(hidden)]
pub mod service;
#[doc(hidden)]
pub mod storage;
#[doc(hidden)]
pub mod tasks;
#[doc(hidden)]
pub mod vector_index;
pub mod wal;
#[doc(hidden)]
pub mod worker;

pub mod embedded;

pub use embedded::{SatoriDb, SatoriDbBuilder, Stats};

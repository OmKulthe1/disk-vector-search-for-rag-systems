//! # LightVector
//!
//! A production-grade, asynchronous, page-aligned disk-native vector search engine.

pub mod distance;
pub mod primitives;
pub mod quantization;
pub mod storage;

pub use distance::{cosine_distance, dot_product, l2_distance, l2_squared, DistanceMetric};
pub use primitives::{NodeId, Vector, VectorError};
pub use quantization::{AdcTable, ProductQuantizer, QuantizationError};
pub use storage::{AlignedSectorBuffer, DiskNode, DiskNode1536, StorageError, PAGE_SIZE};



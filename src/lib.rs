//! # LightVector
//!
//! A production-grade, asynchronous, page-aligned disk-native vector search engine.

pub mod distance;
pub mod primitives;

pub use distance::{cosine_distance, dot_product, l2_distance, l2_squared, DistanceMetric};
pub use primitives::{NodeId, Vector, VectorError};

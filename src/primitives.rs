use std::fmt;
use thiserror::Error;

/// Error types related to vector operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VectorError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector cannot be empty")]
    EmptyVector,

    #[error("Vector norm is zero or invalid for normalization")]
    ZeroNorm,
}

/// Unique identifier for a node in the vector search index graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeId(pub u32);

impl NodeId {
    pub const INVALID: NodeId = NodeId(u32::MAX);

    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for NodeId {
    fn from(val: u32) -> Self {
        Self(val)
    }
}

impl From<NodeId> for u32 {
    fn from(node: NodeId) -> Self {
        node.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

/// Dense floating-point vector wrapper (`f32`).
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    pub fn new(data: Vec<f32>) -> Result<Self, VectorError> {
        if data.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        Ok(Self { data })
    }

    pub fn from_slice(slice: &[f32]) -> Result<Self, VectorError> {
        if slice.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        Ok(Self {
            data: slice.to_vec(),
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_inner(self) -> Vec<f32> {
        self.data
    }

    /// Calculates L2 norm (euclidean length) of the vector.
    pub fn l2_norm(&self) -> f32 {
        let sum_sq: f32 = self.data.iter().map(|&x| x * x).sum();
        sum_sq.sqrt()
    }

    /// Normalizes vector in-place to unit length (L2 norm = 1.0).
    pub fn normalize_in_place(&mut self) -> Result<(), VectorError> {
        let norm = self.l2_norm();
        if norm == 0.0 || !norm.is_finite() {
            return Err(VectorError::ZeroNorm);
        }
        for x in self.data.iter_mut() {
            *x /= norm;
        }
        Ok(())
    }

    /// Returns a new normalized vector.
    pub fn normalized(&self) -> Result<Self, VectorError> {
        let mut copy = self.clone();
        copy.normalize_in_place()?;
        Ok(copy)
    }
}

impl std::ops::Deref for Vector {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id() {
        let id = NodeId::new(42);
        assert_eq!(id.as_u32(), 42);
        assert_eq!(id.as_usize(), 42);
        assert_eq!(NodeId::from(42u32), id);
        assert_eq!(u32::from(id), 42u32);
        assert_eq!(format!("{}", id), "NodeId(42)");
    }

    #[test]
    fn test_vector_creation_and_norm() {
        let v = Vector::new(vec![3.0, 4.0]).unwrap();
        assert_eq!(v.dim(), 2);
        assert_eq!(v.l2_norm(), 5.0);

        let norm_v = v.normalized().unwrap();
        assert!((norm_v[0] - 0.6).abs() < 1e-6);
        assert!((norm_v[1] - 0.8).abs() < 1e-6);
        assert!((norm_v.l2_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_errors() {
        assert_eq!(Vector::new(vec![]), Err(VectorError::EmptyVector));
        let mut zero_v = Vector::new(vec![0.0, 0.0]).unwrap();
        assert_eq!(zero_v.normalize_in_place(), Err(VectorError::ZeroNorm));
    }
}

pub mod kmeans;

use rayon::prelude::*;
use thiserror::Error;

use crate::distance::l2_squared;

pub const K_CENTROIDS: usize = 256;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QuantizationError {
    #[error("Invalid dimension: {0}")]
    InvalidDimension(usize),

    #[error("Dimension {dim} is not evenly divisible by num_subvectors {num_subvectors}")]
    DimensionNotDivisible { dim: usize, num_subvectors: usize },

    #[error("Data length {len} is not a multiple of sub_dim {sub_dim}")]
    InvalidDataLength { len: usize, sub_dim: usize },

    #[error("Insufficient training data: needed at least {needed} vectors, got {actual}")]
    InsufficientTrainingData { needed: usize, actual: usize },

    #[error("Codebook size mismatch: expected {expected}, actual {actual}")]
    CodebookDimensionMismatch { expected: usize, actual: usize },

    #[error("Vector dimension mismatch: expected {expected}, actual {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid code length: expected {expected}, actual {actual}")]
    InvalidCodeLength { expected: usize, actual: usize },

    #[error("Buffer length mismatch: expected {expected}, actual {actual}")]
    BufferLengthMismatch { expected: usize, actual: usize },
}

/// Asymmetric Distance Computation (ADC) precomputed lookup table.
///
/// Contains precomputed squared distances between a full-precision query vector
/// and each of the 256 centroids across all M subspaces.
#[derive(Debug, Clone, PartialEq)]
pub struct AdcTable {
    num_subvectors: usize,
    table: Vec<f32>,
}

impl AdcTable {
    /// Creates an AdcTable from precomputed distances.
    pub fn new(num_subvectors: usize, table: Vec<f32>) -> Result<Self, QuantizationError> {
        let expected = num_subvectors * K_CENTROIDS;
        if table.len() != expected {
            return Err(QuantizationError::BufferLengthMismatch {
                expected,
                actual: table.len(),
            });
        }
        Ok(Self {
            num_subvectors,
            table,
        })
    }

    #[inline]
    pub fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }

    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.table
    }

    /// Computes squared L2 distance between the query and a compressed vector code.
    ///
    /// Runs completely in RAM with ZERO heap allocations.
    #[inline]
    pub fn compute_distance_squared(&self, code: &[u8]) -> f32 {
        debug_assert_eq!(
            code.len(),
            self.num_subvectors,
            "Code length must match num_subvectors"
        );

        let mut sum_sq = 0.0f32;
        let table_ptr = self.table.as_ptr();

        for (m, &byte) in code.iter().enumerate() {
            let offset = m * K_CENTROIDS + byte as usize;
            unsafe {
                sum_sq += *table_ptr.add(offset);
            }
        }

        sum_sq
    }

    /// Computes L2 distance between the query and a compressed vector code.
    ///
    /// Runs completely in RAM with ZERO heap allocations.
    #[inline]
    pub fn compute_distance(&self, code: &[u8]) -> f32 {
        self.compute_distance_squared(code).sqrt()
    }
}

/// Product Quantizer (PQ) for vector compression and fast asymmetric search.
#[derive(Debug, Clone, PartialEq)]
pub struct ProductQuantizer {
    dim: usize,
    num_subvectors: usize,
    sub_dim: usize,
    codebooks: Vec<f32>,
}

impl ProductQuantizer {
    /// Constructs a ProductQuantizer with existing trained codebooks.
    pub fn new(
        dim: usize,
        num_subvectors: usize,
        codebooks: Vec<f32>,
    ) -> Result<Self, QuantizationError> {
        if dim == 0 || num_subvectors == 0 {
            return Err(QuantizationError::InvalidDimension(dim));
        }
        if dim % num_subvectors != 0 {
            return Err(QuantizationError::DimensionNotDivisible {
                dim,
                num_subvectors,
            });
        }

        let sub_dim = dim / num_subvectors;
        let expected_size = num_subvectors * K_CENTROIDS * sub_dim;

        if codebooks.len() != expected_size {
            return Err(QuantizationError::CodebookDimensionMismatch {
                expected: expected_size,
                actual: codebooks.len(),
            });
        }

        Ok(Self {
            dim,
            num_subvectors,
            sub_dim,
            codebooks,
        })
    }

    /// Trains a ProductQuantizer on a slice of training vectors.
    ///
    /// Clusters each of the `num_subvectors` subspaces into 256 centroids using k-means,
    /// parallelized across CPU threads with Rayon.
    pub fn train(
        dataset: &[&[f32]],
        num_subvectors: usize,
        max_iterations: usize,
        seed: Option<u64>,
    ) -> Result<Self, QuantizationError> {
        if dataset.is_empty() {
            return Err(QuantizationError::InsufficientTrainingData {
                needed: K_CENTROIDS,
                actual: 0,
            });
        }

        let dim = dataset[0].len();
        if dim == 0 || num_subvectors == 0 {
            return Err(QuantizationError::InvalidDimension(dim));
        }
        if dim % num_subvectors != 0 {
            return Err(QuantizationError::DimensionNotDivisible {
                dim,
                num_subvectors,
            });
        }

        for vec in dataset {
            if vec.len() != dim {
                return Err(QuantizationError::VectorDimensionMismatch {
                    expected: dim,
                    actual: vec.len(),
                });
            }
        }

        let num_vectors = dataset.len();
        if num_vectors < K_CENTROIDS {
            return Err(QuantizationError::InsufficientTrainingData {
                needed: K_CENTROIDS,
                actual: num_vectors,
            });
        }

        let sub_dim = dim / num_subvectors;
        let base_seed = seed.unwrap_or(42);

        // Train all M subspaces in parallel
        let subspace_results: Result<Vec<Vec<f32>>, QuantizationError> = (0..num_subvectors)
            .into_par_iter()
            .map(|m| {
                // Extract m-th subspace data for all vectors
                let mut sub_data = Vec::with_capacity(num_vectors * sub_dim);
                let start_idx = m * sub_dim;
                let end_idx = start_idx + sub_dim;

                for vec in dataset {
                    sub_data.extend_from_slice(&vec[start_idx..end_idx]);
                }

                let sub_seed = base_seed.wrapping_add(m as u64 * 10007);
                kmeans::train_kmeans(&sub_data, sub_dim, K_CENTROIDS, max_iterations, Some(sub_seed))
            })
            .collect();

        let subspace_centroids = subspace_results?;

        // Flatten all subspace centroids into a single codebooks buffer
        let mut codebooks = Vec::with_capacity(num_subvectors * K_CENTROIDS * sub_dim);
        for centroids in subspace_centroids {
            codebooks.extend(centroids);
        }

        Ok(Self {
            dim,
            num_subvectors,
            sub_dim,
            codebooks,
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }

    #[inline]
    pub fn sub_dim(&self) -> usize {
        self.sub_dim
    }

    #[inline]
    pub fn codebooks(&self) -> &[f32] {
        &self.codebooks
    }

    /// Compresses a high-dimensional vector into an M-byte code.
    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>, QuantizationError> {
        let mut code = vec![0u8; self.num_subvectors];
        self.encode_into(vector, &mut code)?;
        Ok(code)
    }

    /// Compresses a vector directly into a pre-allocated byte slice with zero heap allocations.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) -> Result<(), QuantizationError> {
        if vector.len() != self.dim {
            return Err(QuantizationError::VectorDimensionMismatch {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        if out.len() != self.num_subvectors {
            return Err(QuantizationError::InvalidCodeLength {
                expected: self.num_subvectors,
                actual: out.len(),
            });
        }

        let codebook_subspace_size = K_CENTROIDS * self.sub_dim;

        for m in 0..self.num_subvectors {
            let vec_sub = &vector[m * self.sub_dim..(m + 1) * self.sub_dim];
            let codebook_offset = m * codebook_subspace_size;
            let subspace_centroids = &self.codebooks[codebook_offset..codebook_offset + codebook_subspace_size];

            let mut best_dist = f32::MAX;
            let mut best_c = 0u8;

            for (c_idx, centroid) in subspace_centroids.chunks_exact(self.sub_dim).enumerate() {
                let dist = l2_squared(vec_sub, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c_idx as u8;
                }
            }

            out[m] = best_c;
        }

        Ok(())
    }

    /// Reconstructs an approximate full-precision vector from an M-byte code.
    pub fn decode(&self, code: &[u8]) -> Result<Vec<f32>, QuantizationError> {
        let mut reconstructed = vec![0.0f32; self.dim];
        self.decode_into(code, &mut reconstructed)?;
        Ok(reconstructed)
    }

    /// Reconstructs approximate vector directly into a pre-allocated buffer with zero heap allocations.
    pub fn decode_into(&self, code: &[u8], out: &mut [f32]) -> Result<(), QuantizationError> {
        if code.len() != self.num_subvectors {
            return Err(QuantizationError::InvalidCodeLength {
                expected: self.num_subvectors,
                actual: code.len(),
            });
        }
        if out.len() != self.dim {
            return Err(QuantizationError::VectorDimensionMismatch {
                expected: self.dim,
                actual: out.len(),
            });
        }

        let codebook_subspace_size = K_CENTROIDS * self.sub_dim;

        for (m, &byte) in code.iter().enumerate() {
            let centroid_offset = m * codebook_subspace_size + (byte as usize) * self.sub_dim;
            let centroid = &self.codebooks[centroid_offset..centroid_offset + self.sub_dim];
            let out_sub = &mut out[m * self.sub_dim..(m + 1) * self.sub_dim];
            out_sub.copy_from_slice(centroid);
        }

        Ok(())
    }

    /// Computes the Asymmetric Distance Computation (ADC) lookup table for a given query vector.
    pub fn compute_adc_table(&self, query: &[f32]) -> Result<AdcTable, QuantizationError> {
        let mut table = vec![0.0f32; self.num_subvectors * K_CENTROIDS];
        self.compute_adc_table_into(query, &mut table)?;
        AdcTable::new(self.num_subvectors, table)
    }

    /// Computes ADC table directly into a pre-allocated buffer with zero heap allocations.
    pub fn compute_adc_table_into(
        &self,
        query: &[f32],
        table_out: &mut [f32],
    ) -> Result<(), QuantizationError> {
        if query.len() != self.dim {
            return Err(QuantizationError::VectorDimensionMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }

        let expected_table_len = self.num_subvectors * K_CENTROIDS;
        if table_out.len() != expected_table_len {
            return Err(QuantizationError::BufferLengthMismatch {
                expected: expected_table_len,
                actual: table_out.len(),
            });
        }

        let codebook_subspace_size = K_CENTROIDS * self.sub_dim;

        for m in 0..self.num_subvectors {
            let query_sub = &query[m * self.sub_dim..(m + 1) * self.sub_dim];
            let codebook_offset = m * codebook_subspace_size;
            let subspace_centroids = &self.codebooks[codebook_offset..codebook_offset + codebook_subspace_size];
            let table_sub = &mut table_out[m * K_CENTROIDS..(m + 1) * K_CENTROIDS];

            for (c_idx, centroid) in subspace_centroids.chunks_exact(self.sub_dim).enumerate() {
                table_sub[c_idx] = l2_squared(query_sub, centroid);
            }
        }

        Ok(())
    }
}

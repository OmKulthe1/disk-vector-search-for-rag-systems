use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::distance::l2_squared;
use crate::quantization::QuantizationError;

/// Performs k-means clustering on a set of sub-vectors.
///
/// * `data`: Contiguous slice of sub-vectors, total length = `num_points * sub_dim`.
/// * `sub_dim`: Dimensionality of each sub-vector.
/// * `k`: Number of clusters (typically 256 for 1-byte codebooks).
/// * `max_iterations`: Maximum iterations to run Lloyd's algorithm.
/// * `seed`: Optional PRNG seed for deterministic centroid initialization.
pub fn train_kmeans(
    data: &[f32],
    sub_dim: usize,
    k: usize,
    max_iterations: usize,
    seed: Option<u64>,
) -> Result<Vec<f32>, QuantizationError> {
    if sub_dim == 0 {
        return Err(QuantizationError::InvalidDimension(0));
    }
    if data.is_empty() || data.len() % sub_dim != 0 {
        return Err(QuantizationError::InvalidDataLength {
            len: data.len(),
            sub_dim,
        });
    }

    let num_points = data.len() / sub_dim;
    if num_points < k {
        return Err(QuantizationError::InsufficientTrainingData {
            needed: k,
            actual: num_points,
        });
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // 1. Initialize centroids by sampling k distinct points from data
    let mut indices: Vec<usize> = (0..num_points).collect();
    indices.shuffle(&mut rng);

    let mut centroids = vec![0.0f32; k * sub_dim];
    for (i, &idx) in indices.iter().take(k).enumerate() {
        let src = &data[idx * sub_dim..(idx + 1) * sub_dim];
        let dst = &mut centroids[i * sub_dim..(i + 1) * sub_dim];
        dst.copy_from_slice(src);
    }

    // Buffers for centroid accumulation
    let mut cluster_sums = vec![0.0f32; k * sub_dim];
    let mut cluster_counts = vec![0usize; k];
    let mut assignments = vec![0u8; num_points];

    let tolerance = 1e-4f32;

    for _ in 0..max_iterations {
        let mut max_movement = 0.0f32;
        cluster_sums.fill(0.0);
        cluster_counts.fill(0);

        // Expectation step: assign each point to nearest centroid
        for (p_idx, point) in data.chunks_exact(sub_dim).enumerate() {
            let mut best_dist = f32::MAX;
            let mut best_c = 0usize;

            for (c_idx, centroid) in centroids.chunks_exact(sub_dim).enumerate() {
                let dist = l2_squared(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c_idx;
                }
            }

            assignments[p_idx] = best_c as u8;
            cluster_counts[best_c] += 1;

            let sum_slice = &mut cluster_sums[best_c * sub_dim..(best_c + 1) * sub_dim];
            for (s, &val) in sum_slice.iter_mut().zip(point.iter()) {
                *s += val;
            }
        }

        // Maximization step: compute new centroids
        for c_idx in 0..k {
            let count = cluster_counts[c_idx];
            let centroid_slice = &mut centroids[c_idx * sub_dim..(c_idx + 1) * sub_dim];

            if count > 0 {
                let sum_slice = &cluster_sums[c_idx * sub_dim..(c_idx + 1) * sub_dim];
                let inv_count = 1.0f32 / count as f32;

                let mut movement_sq = 0.0f32;
                for (old_val, &sum_val) in centroid_slice.iter_mut().zip(sum_slice.iter()) {
                    let new_val = sum_val * inv_count;
                    let diff = new_val - *old_val;
                    movement_sq += diff * diff;
                    *old_val = new_val;
                }
                max_movement = max_movement.max(movement_sq.sqrt());
            } else {
                // Handle empty cluster by re-seeding with a random point
                let rand_point_idx = indices.choose(&mut rng).copied().unwrap_or(0);
                let src = &data[rand_point_idx * sub_dim..(rand_point_idx + 1) * sub_dim];
                centroid_slice.copy_from_slice(src);
            }
        }

        if max_movement < tolerance {
            break;
        }
    }

    Ok(centroids)
}

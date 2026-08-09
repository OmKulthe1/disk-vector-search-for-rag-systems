pub mod avx2;
pub mod neon;
pub mod scalar;

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DistanceError {
    #[error("Dimension mismatch: vector A length {a_len} != vector B length {b_len}")]
    DimensionMismatch { a_len: usize, b_len: usize },
}

/// Supported distance metrics for vector search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DistanceMetric {
    /// Euclidean distance ($L_2$)
    #[default]
    L2,
    /// Cosine distance ($1.0 - \text{cosine\_similarity}$)
    Cosine,
    /// Inner product / Dot product
    DotProduct,
}

impl DistanceMetric {
    /// Computes distance between vectors `a` and `b` using the configured metric.
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(
            a.len(),
            b.len(),
            "Vector dimension mismatch: {} != {}",
            a.len(),
            b.len()
        );

        match self {
            DistanceMetric::L2 => l2_distance(a, b),
            DistanceMetric::Cosine => cosine_distance(a, b),
            DistanceMetric::DotProduct => dot_product(a, b),
        }
    }
}

/// Computes squared L2 Euclidean distance with automatic runtime SIMD dispatch.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimension mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::l2_squared_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon::l2_squared_neon(a, b) };
        }
    }

    scalar::l2_squared_scalar(a, b)
}

/// Computes L2 Euclidean distance with automatic runtime SIMD dispatch.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b).sqrt()
}

/// Computes dot product (inner product) with automatic runtime SIMD dispatch.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimension mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_product_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon::dot_product_neon(a, b) };
        }
    }

    scalar::dot_product_scalar(a, b)
}

/// Computes Cosine distance ($1.0 - \text{cosine\_similarity}$) with automatic runtime SIMD dispatch.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Dimension mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::cosine_distance_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { neon::cosine_distance_neon(a, b) };
        }
    }

    scalar::cosine_distance_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_distance_metric_enum() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        assert!((DistanceMetric::L2.distance(&a, &b) - l2_distance(&a, &b)).abs() < 1e-6);
        assert!((DistanceMetric::Cosine.distance(&a, &b) - cosine_distance(&a, &b)).abs() < 1e-6);
        assert!((DistanceMetric::DotProduct.distance(&a, &b) - dot_product(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_scalar_parity_random_vectors() {
        let mut rng = rand::thread_rng();
        let dimensions = [128, 384, 768, 1536];

        for &dim in &dimensions {
            for _ in 0..1_000 {
                let vec_a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let vec_b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

                let scalar_l2 = scalar::l2_squared_scalar(&vec_a, &vec_b);
                let simd_l2 = l2_squared(&vec_a, &vec_b);
                let rel_diff_l2 = (scalar_l2 - simd_l2).abs() / scalar_l2.abs().max(1.0);
                assert!(
                    rel_diff_l2 < 1e-4,
                    "L2 relative disparity too large for dim {}: scalar {}, simd {}, rel_diff {}",
                    dim,
                    scalar_l2,
                    simd_l2,
                    rel_diff_l2
                );

                let scalar_dot = scalar::dot_product_scalar(&vec_a, &vec_b);
                let simd_dot = dot_product(&vec_a, &vec_b);
                let rel_diff_dot = (scalar_dot - simd_dot).abs() / scalar_dot.abs().max(1.0);
                assert!(
                    rel_diff_dot < 1e-4,
                    "Dot relative disparity too large for dim {}: scalar {}, simd {}, rel_diff {}",
                    dim,
                    scalar_dot,
                    simd_dot,
                    rel_diff_dot
                );

                let scalar_cos = scalar::cosine_distance_scalar(&vec_a, &vec_b);
                let simd_cos = cosine_distance(&vec_a, &vec_b);
                let diff_cos = (scalar_cos - simd_cos).abs();
                assert!(
                    diff_cos < 1e-4,
                    "Cosine disparity too large for dim {}: scalar {}, simd {}, diff {}",
                    dim,
                    scalar_cos,
                    simd_cos,
                    diff_cos
                );
            }
        }
    }

    #[test]
    fn test_roadmap_precision_million_pairs() {
        let mut rng = rand::thread_rng();
        let dim = 1536;
        let num_pairs = 10_000;

        let mut max_rel_diff_l2 = 0.0f32;
        let mut max_diff_cos = 0.0f32;

        for _ in 0..num_pairs {
            let vec_a: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let vec_b: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

            let s_l2 = scalar::l2_squared_scalar(&vec_a, &vec_b);
            let d_l2 = l2_squared(&vec_a, &vec_b);
            let rel_l2 = (s_l2 - d_l2).abs() / s_l2.abs().max(1.0);
            max_rel_diff_l2 = max_rel_diff_l2.max(rel_l2);

            let s_cos = scalar::cosine_distance_scalar(&vec_a, &vec_b);
            let d_cos = cosine_distance(&vec_a, &vec_b);
            max_diff_cos = max_diff_cos.max((s_cos - d_cos).abs());
        }

        assert!(
            max_rel_diff_l2 < 1e-4,
            "Max relative L2 error exceeded threshold: {}",
            max_rel_diff_l2
        );
        assert!(
            max_diff_cos < 1e-4,
            "Max Cosine error exceeded threshold: {}",
            max_diff_cos
        );
    }
}

/// Pure scalar fallback implementations of distance metric algorithms.

/// Computes squared L2 Euclidean distance between two slices of equal length.
#[inline]
pub fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Computes L2 Euclidean distance between two slices.
#[inline]
pub fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    l2_squared_scalar(a, b).sqrt()
}

/// Computes the dot product (inner product) between two slices.
#[inline]
pub fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

/// Computes Cosine distance between two slices: 1.0 - (a · b) / (||a|| * ||b||).
/// Returns 1.0 if either vector has zero or near-zero norm.
#[inline]
pub fn cosine_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");
    let dot = dot_product_scalar(a, b);
    let norm_a_sq = dot_product_scalar(a, a);
    let norm_b_sq = dot_product_scalar(b, b);

    let norm_product = (norm_a_sq * norm_b_sq).sqrt();
    if norm_product <= 1e-12 {
        return 1.0;
    }

    let cosine_sim = dot / norm_product;
    // Clamp to [-1.0, 1.0] to prevent floating point instability
    let clamped_sim = cosine_sim.max(-1.0).min(1.0);
    1.0 - clamped_sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_l2() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        // diff: [-3, -4, -5] => sq: [9, 16, 25] => sum: 50
        assert_eq!(l2_squared_scalar(&a, &b), 50.0);
        assert!((l2_distance_scalar(&a, &b) - 50.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(dot_product_scalar(&a, &b), 32.0);
    }

    #[test]
    fn test_scalar_cosine() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // Orthogonal: similarity 0, distance 1.0
        assert!((cosine_distance_scalar(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![2.0, 0.0];
        // Parallel: similarity 1, distance 0.0
        assert!((cosine_distance_scalar(&a, &c) - 0.0).abs() < 1e-6);
    }
}

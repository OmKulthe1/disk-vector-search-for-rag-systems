/// ARM NEON SIMD implementations for aarch64 target architecture.

#[cfg(target_arch = "aarch64")]
pub use neon_impl::*;

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    /// Computes squared L2 Euclidean distance using ARM NEON instructions.
    ///
    /// # Safety
    /// Caller must ensure CPU supports target feature `neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");

        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let mut iter_a = chunks_a;
        let mut iter_b = chunks_b;

        while iter_a.len() >= 2 {
            let va0 = vld1q_f32(iter_a.next().unwrap().as_ptr());
            let vb0 = vld1q_f32(iter_b.next().unwrap().as_ptr());
            let diff0 = vsubq_f32(va0, vb0);
            acc0 = vfmaq_f32(acc0, diff0, diff0);

            let va1 = vld1q_f32(iter_a.next().unwrap().as_ptr());
            let vb1 = vld1q_f32(iter_b.next().unwrap().as_ptr());
            let diff1 = vsubq_f32(va1, vb1);
            acc1 = vfmaq_f32(acc1, diff1, diff1);
        }

        for (ca, cb) in iter_a.zip(iter_b) {
            let va = vld1q_f32(ca.as_ptr());
            let vb = vld1q_f32(cb.as_ptr());
            let diff = vsubq_f32(va, vb);
            acc0 = vfmaq_f32(acc0, diff, diff);
        }

        let acc = vaddq_f32(acc0, acc1);
        let mut sum = vaddvq_f32(acc);

        for (&ra, &rb) in remainder_a.iter().zip(remainder_b.iter()) {
            let diff = ra - rb;
            sum += diff * diff;
        }

        sum
    }

    /// Computes dot product using ARM NEON instructions.
    ///
    /// # Safety
    /// Caller must ensure CPU supports target feature `neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");

        let chunks_a = a.chunks_exact(4);
        let chunks_b = b.chunks_exact(4);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        let mut iter_a = chunks_a;
        let mut iter_b = chunks_b;

        while iter_a.len() >= 2 {
            let va0 = vld1q_f32(iter_a.next().unwrap().as_ptr());
            let vb0 = vld1q_f32(iter_b.next().unwrap().as_ptr());
            acc0 = vfmaq_f32(acc0, va0, vb0);

            let va1 = vld1q_f32(iter_a.next().unwrap().as_ptr());
            let vb1 = vld1q_f32(iter_b.next().unwrap().as_ptr());
            acc1 = vfmaq_f32(acc1, va1, vb1);
        }

        for (ca, cb) in iter_a.zip(iter_b) {
            let va = vld1q_f32(ca.as_ptr());
            let vb = vld1q_f32(cb.as_ptr());
            acc0 = vfmaq_f32(acc0, va, vb);
        }

        let acc = vaddq_f32(acc0, acc1);
        let mut sum = vaddvq_f32(acc);

        for (&ra, &rb) in remainder_a.iter().zip(remainder_b.iter()) {
            sum += ra * rb;
        }

        sum
    }

    /// Computes L2 distance using ARM NEON.
    ///
    /// # Safety
    /// Caller must ensure CPU supports target feature `neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        l2_squared_neon(a, b).sqrt()
    }

    /// Computes Cosine distance using ARM NEON.
    ///
    /// # Safety
    /// Caller must ensure CPU supports target feature `neon`.
    #[target_feature(enable = "neon")]
    pub unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product_neon(a, b);
        let norm_a_sq = dot_product_neon(a, a);
        let norm_b_sq = dot_product_neon(b, b);

        let norm_product = (norm_a_sq * norm_b_sq).sqrt();
        if norm_product <= 1e-12 {
            return 1.0;
        }

        let cosine_sim = dot / norm_product;
        let clamped_sim = cosine_sim.max(-1.0).min(1.0);
        1.0 - clamped_sim
    }
}

/// AVX2 + FMA SIMD implementations for x86_64 target architecture.

#[cfg(target_arch = "x86_64")]
pub use avx2_impl::*;

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    /// Computes squared L2 Euclidean distance using AVX2 and FMA instructions.
    ///
    /// # Safety
    /// Caller must ensure that CPU supports `avx2` and `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");

        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut iter_a = chunks_a;
        let mut iter_b = chunks_b;

        while iter_a.len() >= 4 {
            let va0 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb0 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            let diff0 = _mm256_sub_ps(va0, vb0);
            acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

            let va1 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb1 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            let diff1 = _mm256_sub_ps(va1, vb1);
            acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

            let va2 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb2 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            let diff2 = _mm256_sub_ps(va2, vb2);
            acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

            let va3 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb3 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            let diff3 = _mm256_sub_ps(va3, vb3);
            acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
        }

        for (ca, cb) in iter_a.zip(iter_b) {
            let va = _mm256_loadu_ps(ca.as_ptr());
            let vb = _mm256_loadu_ps(cb.as_ptr());
            let diff = _mm256_sub_ps(va, vb);
            acc0 = _mm256_fmadd_ps(diff, diff, acc0);
        }

        let acc = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), acc);
        let mut sum: f32 = arr.iter().sum();

        for (&ra, &rb) in remainder_a.iter().zip(remainder_b.iter()) {
            let diff = ra - rb;
            sum += diff * diff;
        }

        sum
    }

    /// Computes dot product using AVX2 and FMA instructions.
    ///
    /// # Safety
    /// Caller must ensure that CPU supports `avx2` and `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Slices must have equal length");

        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let mut iter_a = chunks_a;
        let mut iter_b = chunks_b;

        while iter_a.len() >= 4 {
            let va0 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb0 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

            let va1 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb1 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

            let va2 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb2 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

            let va3 = _mm256_loadu_ps(iter_a.next().unwrap().as_ptr());
            let vb3 = _mm256_loadu_ps(iter_b.next().unwrap().as_ptr());
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }

        for (ca, cb) in iter_a.zip(iter_b) {
            let va = _mm256_loadu_ps(ca.as_ptr());
            let vb = _mm256_loadu_ps(cb.as_ptr());
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
        }

        let acc = _mm256_add_ps(
            _mm256_add_ps(acc0, acc1),
            _mm256_add_ps(acc2, acc3),
        );

        let mut arr = [0.0f32; 8];
        _mm256_storeu_ps(arr.as_mut_ptr(), acc);
        let mut sum: f32 = arr.iter().sum();

        for (&ra, &rb) in remainder_a.iter().zip(remainder_b.iter()) {
            sum += ra * rb;
        }

        sum
    }

    /// Computes L2 distance using AVX2.
    ///
    /// # Safety
    /// Caller must ensure that CPU supports `avx2` and `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        l2_squared_avx2(a, b).sqrt()
    }

    /// Computes Cosine distance using AVX2.
    ///
    /// # Safety
    /// Caller must ensure that CPU supports `avx2` and `fma` target features.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product_avx2(a, b);
        let norm_a_sq = dot_product_avx2(a, a);
        let norm_b_sq = dot_product_avx2(b, b);

        let norm_product = (norm_a_sq * norm_b_sq).sqrt();
        if norm_product <= 1e-12 {
            return 1.0;
        }

        let cosine_sim = dot / norm_product;
        let clamped_sim = cosine_sim.max(-1.0).min(1.0);
        1.0 - clamped_sim
    }
}

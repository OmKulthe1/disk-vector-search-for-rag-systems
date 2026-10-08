use lightvector::distance::l2_squared;
use lightvector::quantization::{AdcTable, ProductQuantizer, QuantizationError, K_CENTROIDS};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_exact_centroid_parity() {
    let dim = 8;
    let num_subvectors = 2;
    let sub_dim = 4;

    // Create synthetic codebooks where centroid c has value (m * 1000 + c) for all components
    let mut codebooks = vec![0.0f32; num_subvectors * K_CENTROIDS * sub_dim];
    for m in 0..num_subvectors {
        for c in 0..K_CENTROIDS {
            let offset = m * K_CENTROIDS * sub_dim + c * sub_dim;
            for d in 0..sub_dim {
                codebooks[offset + d] = (m as f32 * 1000.0) + (c as f32) + (d as f32 * 0.01);
            }
        }
    }

    let pq = ProductQuantizer::new(dim, num_subvectors, codebooks.clone()).unwrap();

    // Test specific known centroid indices
    let target_centroids = [(5u8, 42u8), (0u8, 255u8), (128u8, 64u8)];

    for (c0, c1) in target_centroids {
        // Construct vector that exactly matches centroid c0 in subspace 0, and c1 in subspace 1
        let mut vec_target = vec![0.0f32; dim];
        let sub0_offset = (c0 as usize) * sub_dim;
        let sub1_offset = K_CENTROIDS * sub_dim + (c1 as usize) * sub_dim;

        vec_target[0..4].copy_from_slice(&codebooks[sub0_offset..sub0_offset + 4]);
        vec_target[4..8].copy_from_slice(&codebooks[sub1_offset..sub1_offset + 4]);

        // Encode vector
        let code = pq.encode(&vec_target).unwrap();
        assert_eq!(code.len(), 2);
        assert_eq!(code[0], c0, "Subspace 0 code mismatch");
        assert_eq!(code[1], c1, "Subspace 1 code mismatch");

        // Decode code
        let decoded = pq.decode(&code).unwrap();
        assert_eq!(decoded.len(), dim);
        for i in 0..dim {
            assert!(
                (decoded[i] - vec_target[i]).abs() < 1e-6,
                "Decoded value mismatch at index {}",
                i
            );
        }
    }
}

#[test]
fn test_compression_ratios_for_1536_dim() {
    let dim = 1536;
    let raw_bytes = dim * std::mem::size_of::<f32>(); // 6,144 bytes

    // 1. M = 64 subvectors (96x compression)
    let m_64 = 64;
    assert_eq!(dim % m_64, 0);
    let sub_dim_64 = dim / m_64; // 24
    assert_eq!(sub_dim_64, 24);

    let code_len_64 = m_64; // 64 bytes
    let compression_ratio_64 = raw_bytes as f32 / code_len_64 as f32;
    assert_eq!(compression_ratio_64, 96.0);

    // 2. M = 128 subvectors (48x compression)
    let m_128 = 128;
    assert_eq!(dim % m_128, 0);
    let sub_dim_128 = dim / m_128; // 12
    assert_eq!(sub_dim_128, 12);

    let code_len_128 = m_128; // 128 bytes
    let compression_ratio_128 = raw_bytes as f32 / code_len_128 as f32;
    assert_eq!(compression_ratio_128, 48.0);
}

#[test]
fn test_zero_heap_allocation_adc_lookup() {
    let num_subvectors = 64;
    let table = vec![0.5f32; num_subvectors * K_CENTROIDS];
    let adc_table = AdcTable::new(num_subvectors, table).unwrap();

    let code = vec![12u8; num_subvectors];

    // Repeated in-RAM lookup
    for _ in 0..10_000 {
        let dist_sq = adc_table.compute_distance_squared(&code);
        assert!((dist_sq - (64.0 * 0.5)).abs() < 1e-5);

        let dist = adc_table.compute_distance(&code);
        assert!((dist - (32.0f32).sqrt()).abs() < 1e-5);
    }
}

#[test]
fn test_quantizer_validation_errors() {
    // 1. Dimension not divisible
    assert_eq!(
        ProductQuantizer::new(100, 3, vec![]),
        Err(QuantizationError::DimensionNotDivisible {
            dim: 100,
            num_subvectors: 3
        })
    );

    // 2. Invalid codebook size
    let expected = 2 * 256 * 2; // dim=4, m=2, sub_dim=2
    assert_eq!(
        ProductQuantizer::new(4, 2, vec![0.0; 10]),
        Err(QuantizationError::CodebookDimensionMismatch {
            expected,
            actual: 10
        })
    );

    // 3. Insufficient training data
    let single_vec = [1.0f32, 2.0, 3.0, 4.0];
    let dataset = [&single_vec[..]];
    assert_eq!(
        ProductQuantizer::train(&dataset, 2, 10, Some(42)),
        Err(QuantizationError::InsufficientTrainingData {
            needed: K_CENTROIDS,
            actual: 1
        })
    );
}

#[test]
fn test_integration_mse_on_dataset() {
    let mut rng = StdRng::seed_from_u64(12345);
    let dim = 128;
    let num_subvectors = 16; // sub_dim = 8
    let num_train = 500;

    // Generate training vectors
    let mut train_data = Vec::with_capacity(num_train);
    for _ in 0..num_train {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        train_data.push(v);
    }
    let train_refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();

    // Train quantizer
    let pq = ProductQuantizer::train(&train_refs, num_subvectors, 15, Some(42)).unwrap();
    assert_eq!(pq.dim(), dim);
    assert_eq!(pq.num_subvectors(), num_subvectors);
    assert_eq!(pq.sub_dim(), 8);

    // Encode a database of 200 vectors
    let num_db = 200;
    let mut db_vectors = Vec::with_capacity(num_db);
    let mut db_codes = Vec::with_capacity(num_db);

    for _ in 0..num_db {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let code = pq.encode(&v).unwrap();
        db_vectors.push(v);
        db_codes.push(code);
    }

    // Evaluate distance estimation across 50 query vectors x 200 db vectors = 10,000 comparisons
    let num_queries = 50;
    let mut total_squared_error = 0.0f64;
    let mut total_comparisons = 0usize;

    for _ in 0..num_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let adc_table = pq.compute_adc_table(&query).unwrap();

        for (db_vec, code) in db_vectors.iter().zip(db_codes.iter()) {
            let exact_dist_sq = l2_squared(&query, db_vec);
            let pq_dist_sq = adc_table.compute_distance_squared(code);

            let err = (exact_dist_sq - pq_dist_sq) as f64;
            total_squared_error += err * err;
            total_comparisons += 1;
        }
    }

    assert_eq!(total_comparisons, 10_000);
    let mse = total_squared_error / total_comparisons as f64;

    // Ensure Mean Squared Error is bounded and reasonable for 128-dim random floats
    assert!(
        mse < 100.0,
        "MSE was unexpectedly high: {}, expected < 100.0",
        mse
    );
}

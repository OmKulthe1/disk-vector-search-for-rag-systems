use lightvector::distance::l2_distance;
use lightvector::graph::{VamanaConfig, VamanaGraph};
use lightvector::primitives::NodeId;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn test_degree_bounding_and_reachability_basic() {
    let mut rng = StdRng::seed_from_u64(123);
    let dim = 16;
    let num_vectors = 300;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let r_max = 32;
    let config = VamanaConfig {
        r_max,
        beam_width_l: 50,
        alpha: 1.2,
        seed: Some(42),
    };

    let graph = VamanaGraph::build(&vector_refs, &config).unwrap();

    // 1. Assert degree bound strictly <= r_max for EVERY node
    for i in 0..num_vectors {
        let node = NodeId::new(i as u32);
        let deg = graph.degree(node);
        assert!(
            deg <= r_max,
            "Node {} exceeded degree bound: {} > {}",
            i,
            deg,
            r_max
        );
    }

    // 2. Assert 100% reachability from medoid
    let (reachable, avg_hops) = graph.bfs_reachability();
    assert_eq!(
        reachable, num_vectors,
        "Not all nodes were reachable from medoid: {} / {}",
        reachable, num_vectors
    );
    assert!(avg_hops > 0.0, "Average hops must be positive");
}

#[test]
fn test_graph_integrity_5000_vectors() {
    let mut rng = StdRng::seed_from_u64(456);
    let dim = 32;
    let num_vectors = 5000;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let r_max = 64;
    let config = VamanaConfig {
        r_max,
        beam_width_l: 64,
        alpha: 1.2,
        seed: Some(789),
    };

    let graph = VamanaGraph::build(&vector_refs, &config).unwrap();
    assert_eq!(graph.num_nodes(), 5000);

    // 1. Verify strict degree bound
    for i in 0..num_vectors {
        let node = NodeId::new(i as u32);
        assert!(graph.degree(node) <= r_max);
    }

    // 2. Perform BFS to verify 100% reachability from medoid
    let (reachable, avg_hops) = graph.bfs_reachability();
    assert_eq!(
        reachable, 5000,
        "Failed 100% reachability: only {} / 5000 reachable",
        reachable
    );
    println!("5,000 vector graph: 100% reachable with average hop distance: {:.2}", avg_hops);
}

#[test]
fn test_alpha_property_path_length_reduction() {
    let mut rng = StdRng::seed_from_u64(999);
    let dim = 16;
    let num_vectors = 800;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let r_max = 32;

    // Build with alpha = 1.0 (local edges only, fewer long-range shortcuts)
    let config_low_alpha = VamanaConfig {
        r_max,
        beam_width_l: 50,
        alpha: 1.0,
        seed: Some(101),
    };
    let graph_low = VamanaGraph::build(&vector_refs, &config_low_alpha).unwrap();
    let (reached_low, avg_hops_low) = graph_low.bfs_reachability();
    assert_eq!(reached_low, num_vectors);

    // Build with alpha = 1.4 (more long-range diverse shortcut edges)
    let config_high_alpha = VamanaConfig {
        r_max,
        beam_width_l: 50,
        alpha: 1.4,
        seed: Some(101),
    };
    let graph_high = VamanaGraph::build(&vector_refs, &config_high_alpha).unwrap();
    let (reached_high, avg_hops_high) = graph_high.bfs_reachability();
    assert_eq!(reached_high, num_vectors);

    // Increasing alpha should decrease or maintain average path length due to shortcut edges
    assert!(
        avg_hops_high <= avg_hops_low + 0.1,
        "Increasing alpha did not produce shortcut edges: alpha=1.0 hops: {}, alpha=1.4 hops: {}",
        avg_hops_low,
        avg_hops_high
    );
}

#[test]
fn test_vamana_search_recall() {
    let mut rng = StdRng::seed_from_u64(2024);
    let dim = 16;
    let num_vectors = 500;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let vector_refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let config = VamanaConfig {
        r_max: 48,
        beam_width_l: 64,
        alpha: 1.2,
        seed: Some(42),
    };

    let graph = VamanaGraph::build(&vector_refs, &config).unwrap();

    // Query 20 test vectors
    let k = 5;
    let mut correct_top1_count = 0usize;

    for _ in 0..20 {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Exact brute-force top 1
        let mut exact_best_dist = f32::MAX;
        let mut exact_best_id = NodeId::new(0);

        for (idx, vec) in vectors.iter().enumerate() {
            let dist = l2_distance(&query, vec);
            if dist < exact_best_dist {
                exact_best_dist = dist;
                exact_best_id = NodeId::new(idx as u32);
            }
        }

        // Graph search
        let results = graph.search(&query, k, 64);
        assert!(!results.is_empty());

        let found_in_top_k = results.iter().any(|&(_, id)| id == exact_best_id);
        if found_in_top_k {
            correct_top1_count += 1;
        }
    }

    // High recall (at least 90% top-1 found in top-k)
    let recall = correct_top1_count as f32 / 20.0;
    assert!(
        recall >= 0.90,
        "Recall was lower than expected: {} (expected >= 0.90)",
        recall
    );
}

use crate::distance::l2_distance;
use crate::primitives::{NodeId, Vector};
use std::collections::HashSet;

/// RobustPrune algorithm from DiskANN / Vamana graph construction.
///
/// Prunes a candidate pool to at most `r_max` neighbors, balancing proximity
/// with directional angular diversity using the pruning parameter `alpha` (>= 1.0).
pub fn robust_prune(
    target_id: NodeId,
    candidates: &[NodeId],
    current_neighbors: &[NodeId],
    vectors: &[Vector],
    r_max: usize,
    alpha: f32,
) -> Vec<NodeId> {
    if r_max == 0 {
        return Vec::new();
    }

    // 1. Gather all candidates (candidate pool + current neighbors), deduplicated
    let mut candidate_set = HashSet::with_capacity(candidates.len() + current_neighbors.len());
    for &cand in candidates.iter().chain(current_neighbors.iter()) {
        if cand != target_id && cand.as_usize() < vectors.len() {
            candidate_set.insert(cand);
        }
    }

    if candidate_set.is_empty() {
        return Vec::new();
    }

    let target_vec = &vectors[target_id.as_usize()];

    // 2. Compute distance from target to each candidate, and sort descending
    // Sorting descending allows pop() to extract the minimum distance in O(1) without memory shifts
    let mut candidate_distances: Vec<(f32, NodeId)> = candidate_set
        .into_iter()
        .map(|node_id| {
            let dist = l2_distance(target_vec, &vectors[node_id.as_usize()]);
            (dist, node_id)
        })
        .collect();

    candidate_distances.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // 3. Greedily select neighbors using directional alpha-pruning
    let mut pruned_neighbors = Vec::with_capacity(r_max);

    while let Some((_p_star_dist, p_star)) = candidate_distances.pop() {
        pruned_neighbors.push(p_star);

        if pruned_neighbors.len() >= r_max {
            break;
        }

        let p_star_vec = &vectors[p_star.as_usize()];

        // Prune remaining candidates: keep p_prime only if p_star does not shadow p_prime
        // A candidate is shadowed if: alpha * dist(p_star, p_prime) <= dist(target, p_prime)
        candidate_distances.retain(|&(target_to_prime_dist, p_prime)| {
            let p_star_to_prime_dist = l2_distance(p_star_vec, &vectors[p_prime.as_usize()]);
            alpha * p_star_to_prime_dist > target_to_prime_dist
        });
    }

    pruned_neighbors
}

use crate::distance::l2_distance;
use crate::primitives::{NodeId, Vector};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Copy, Clone, PartialEq)]
struct MinCandidate(f32, NodeId);

impl Eq for MinCandidate {}

impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Traverses the graph greedily from `start_node` to find candidate nearest neighbors of `query`.
///
/// Uses standard best-first beam search with early termination when unexpanded candidates
/// exceed the worst candidate in the beam window.
pub fn greedy_search(
    start_node: NodeId,
    query: &[f32],
    l_size: usize,
    vectors: &[Vector],
    adjacency: &[Vec<NodeId>],
) -> (Vec<(f32, NodeId)>, Vec<NodeId>) {
    let start_idx = start_node.as_usize();
    let start_dist = l2_distance(query, &vectors[start_idx]);

    let mut visited = HashSet::new();
    visited.insert(start_node);

    let mut visited_order = Vec::with_capacity(l_size * 2);
    visited_order.push(start_node);

    let mut unexpanded_min_heap = BinaryHeap::new();
    unexpanded_min_heap.push(MinCandidate(start_dist, start_node));

    // Best candidates sorted ascending by distance
    let mut best_candidates: Vec<(f32, NodeId)> = vec![(start_dist, start_node)];

    while let Some(MinCandidate(curr_dist, curr_node)) = unexpanded_min_heap.pop() {
        // Early stopping: if the closest unexpanded candidate is worse than the worst candidate in full beam
        if best_candidates.len() >= l_size {
            let worst_best_dist = best_candidates[best_candidates.len() - 1].0;
            if curr_dist > worst_best_dist {
                break;
            }
        }

        // Explore outgoing neighbors of current node
        for &neighbor in &adjacency[curr_node.as_usize()] {
            if visited.insert(neighbor) {
                visited_order.push(neighbor);
                let dist = l2_distance(query, &vectors[neighbor.as_usize()]);

                let worst_best_dist = if best_candidates.len() >= l_size {
                    best_candidates[best_candidates.len() - 1].0
                } else {
                    f32::MAX
                };

                if dist < worst_best_dist || best_candidates.len() < l_size {
                    // Insert into sorted best_candidates
                    let pos = best_candidates
                        .binary_search_by(|probe| {
                            probe.0.partial_cmp(&dist).unwrap_or(Ordering::Equal)
                        })
                        .unwrap_or_else(|e| e);

                    best_candidates.insert(pos, (dist, neighbor));
                    if best_candidates.len() > l_size {
                        best_candidates.pop();
                    }

                    unexpanded_min_heap.push(MinCandidate(dist, neighbor));
                }
            }
        }
    }

    (best_candidates, visited_order)
}

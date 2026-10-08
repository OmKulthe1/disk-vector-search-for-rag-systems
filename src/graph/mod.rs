pub mod prune;
pub mod search;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::{HashSet, VecDeque};
use thiserror::Error;

use crate::distance::l2_distance;
use crate::primitives::{NodeId, Vector, VectorError};

#[derive(Debug, Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("Dataset cannot be empty")]
    EmptyDataset,

    #[error("Vector dimension mismatch: expected {expected}, actual {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },

    #[error("Vector conversion error: {0}")]
    VectorError(#[from] VectorError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Node not found: {0}")]
    NodeNotFound(u32),
}

/// Configuration parameters for Vamana graph indexing.
#[derive(Debug, Clone)]
pub struct VamanaConfig {
    /// Maximum out-degree bound per node (default: 64).
    pub r_max: usize,
    /// Search candidate pool size during graph construction (default: 100).
    pub beam_width_l: usize,
    /// Directional pruning threshold factor alpha >= 1.0 (default: 1.2).
    pub alpha: f32,
    /// Optional seed for deterministic index construction.
    pub seed: Option<u64>,
}

impl Default for VamanaConfig {
    fn default() -> Self {
        Self {
            r_max: 64,
            beam_width_l: 100,
            alpha: 1.2,
            seed: Some(42),
        }
    }
}

/// In-memory Vamana graph index.
#[derive(Debug, Clone)]
pub struct VamanaGraph {
    dim: usize,
    r_max: usize,
    medoid: NodeId,
    vectors: Vec<Vector>,
    adjacency: Vec<Vec<NodeId>>,
}

impl VamanaGraph {
    /// Constructs a single-layer Vamana graph index from a slice of vectors.
    pub fn build(vectors_raw: &[&[f32]], config: &VamanaConfig) -> Result<Self, GraphError> {
        if vectors_raw.is_empty() {
            return Err(GraphError::EmptyDataset);
        }
        if config.r_max == 0 {
            return Err(GraphError::InvalidConfig("r_max must be > 0".to_string()));
        }
        if config.alpha < 1.0 {
            return Err(GraphError::InvalidConfig("alpha must be >= 1.0".to_string()));
        }

        let dim = vectors_raw[0].len();
        if dim == 0 {
            return Err(GraphError::InvalidConfig("Dimension cannot be 0".to_string()));
        }

        let num_nodes = vectors_raw.len();
        let mut vectors = Vec::with_capacity(num_nodes);

        for &raw in vectors_raw {
            if raw.len() != dim {
                return Err(GraphError::VectorDimensionMismatch {
                    expected: dim,
                    actual: raw.len(),
                });
            }
            vectors.push(Vector::from_slice(raw)?);
        }

        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // 1. Find the medoid (vector closest to the dataset centroid)
        let medoid = Self::find_medoid(&vectors);

        // 2. Initialize random regular directed graph (bounded by r_max)
        let initial_degree = config.r_max.min(num_nodes.saturating_sub(1)).min(16);
        let mut adjacency: Vec<Vec<NodeId>> = Vec::with_capacity(num_nodes);

        for i in 0..num_nodes {
            let mut neighbors = Vec::with_capacity(config.r_max);
            let mut pool: Vec<usize> = (0..num_nodes).filter(|&idx| idx != i).collect();
            pool.shuffle(&mut rng);

            for &n_idx in pool.iter().take(initial_degree) {
                neighbors.push(NodeId::new(n_idx as u32));
            }
            adjacency.push(neighbors);
        }

        // 3. Vamana indexing loop: greedy traversal + directional alpha-pruning
        let mut permutation: Vec<usize> = (0..num_nodes).collect();
        permutation.shuffle(&mut rng);

        for &node_idx in &permutation {
            let node_id = NodeId::new(node_idx as u32);
            let query = &vectors[node_idx];

            // Greedy search from medoid
            let (_candidates, visited) = search::greedy_search(
                medoid,
                query,
                config.beam_width_l,
                &vectors,
                &adjacency,
            );

            // Prune candidate connections
            let new_neighbors = prune::robust_prune(
                node_id,
                &visited,
                &adjacency[node_idx],
                &vectors,
                config.r_max,
                config.alpha,
            );

            adjacency[node_idx] = new_neighbors.clone();

            // Add reverse edges to preserve graph connectivity
            for &neighbor in &new_neighbors {
                let n_idx = neighbor.as_usize();
                if !adjacency[n_idx].contains(&node_id) {
                    adjacency[n_idx].push(node_id);
                }
            }
        }

        // 4. Bound all node degrees strictly to r_max
        for idx in 0..num_nodes {
            if adjacency[idx].len() > config.r_max {
                let nid = NodeId::new(idx as u32);
                adjacency[idx] = prune::robust_prune(
                    nid,
                    &[],
                    &adjacency[idx],
                    &vectors,
                    config.r_max,
                    config.alpha,
                );
            }
        }

        // 5. Ensure 100% reachability from medoid entry point
        Self::ensure_reachability(medoid, &vectors, &mut adjacency, config.r_max, config.alpha);

        Ok(Self {
            dim,
            r_max: config.r_max,
            medoid,
            vectors,
            adjacency,
        })
    }

    /// Finds the dataset medoid (node closest to the centroid).
    fn find_medoid(vectors: &[Vector]) -> NodeId {
        let num_nodes = vectors.len();
        if num_nodes == 1 {
            return NodeId::new(0);
        }

        let dim = vectors[0].dim();
        let mut centroid = vec![0.0f32; dim];

        for vec in vectors {
            for (c, &v) in centroid.iter_mut().zip(vec.iter()) {
                *c += v;
            }
        }

        let inv_n = 1.0f32 / num_nodes as f32;
        for c in centroid.iter_mut() {
            *c *= inv_n;
        }

        let mut best_dist = f32::MAX;
        let mut best_idx = 0usize;

        for (idx, vec) in vectors.iter().enumerate() {
            let dist = l2_distance(&centroid, vec);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        NodeId::new(best_idx as u32)
    }

    /// Enforces 100% node reachability from the medoid entry point via BFS repair.
    fn ensure_reachability(
        medoid: NodeId,
        vectors: &[Vector],
        adjacency: &mut [Vec<NodeId>],
        r_max: usize,
        alpha: f32,
    ) {
        let num_nodes = adjacency.len();
        let mut visited = HashSet::with_capacity(num_nodes);
        let mut queue = VecDeque::new();

        visited.insert(medoid);
        queue.push_back(medoid);

        while let Some(curr) = queue.pop_front() {
            for &neighbor in &adjacency[curr.as_usize()] {
                if visited.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }

        if visited.len() < num_nodes {
            let reached_nodes: Vec<NodeId> = visited.iter().copied().collect();

            // Link every unreached node to the closest reached node
            for unreached_idx in 0..num_nodes {
                let unreached_node = NodeId::new(unreached_idx as u32);
                if !visited.contains(&unreached_node) {
                    let unreached_vec = &vectors[unreached_idx];

                    // Find closest reached node
                    let mut best_dist = f32::MAX;
                    let mut best_reached = medoid;

                    for &reached in &reached_nodes {
                        let dist = l2_distance(unreached_vec, &vectors[reached.as_usize()]);
                        if dist < best_dist {
                            best_dist = dist;
                            best_reached = reached;
                        }
                    }

                    // Add directed edge from best_reached to unreached_node
                    let r_idx = best_reached.as_usize();
                    if !adjacency[r_idx].contains(&unreached_node) {
                        adjacency[r_idx].push(unreached_node);
                        if adjacency[r_idx].len() > r_max {
                            adjacency[r_idx] = prune::robust_prune(
                                best_reached,
                                &[unreached_node],
                                &adjacency[r_idx],
                                vectors,
                                r_max,
                                alpha,
                            );
                            // If pruned away, forcefully retain it to guarantee reachability
                            if !adjacency[r_idx].contains(&unreached_node) {
                                adjacency[r_idx].pop();
                                adjacency[r_idx].push(unreached_node);
                            }
                        }
                    }
                    visited.insert(unreached_node);
                }
            }
        }
    }

    /// Performs nearest neighbor search on the Vamana graph index.
    pub fn search(&self, query: &[f32], k: usize, l_search: usize) -> Vec<(f32, NodeId)> {
        let l_size = l_search.max(k);
        let (candidates, _) = search::greedy_search(
            self.medoid,
            query,
            l_size,
            &self.vectors,
            &self.adjacency,
        );

        candidates.into_iter().take(k).collect()
    }

    /// Evaluates graph reachability from the medoid using breadth-first search (BFS).
    ///
    /// Returns `(reachable_node_count, average_hop_distance_from_medoid)`.
    pub fn bfs_reachability(&self) -> (usize, f64) {
        let num_nodes = self.vectors.len();
        let mut visited = HashSet::with_capacity(num_nodes);
        let mut queue = VecDeque::new();
        let mut total_hops = 0usize;

        visited.insert(self.medoid);
        queue.push_back((self.medoid, 0usize));

        while let Some((curr, depth)) = queue.pop_front() {
            total_hops += depth;

            for &neighbor in &self.adjacency[curr.as_usize()] {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }

        let reachable_count = visited.len();
        let avg_hops = if reachable_count > 0 {
            total_hops as f64 / reachable_count as f64
        } else {
            0.0
        };

        (reachable_count, avg_hops)
    }

    #[inline]
    pub fn medoid(&self) -> NodeId {
        self.medoid
    }

    #[inline]
    pub fn r_max(&self) -> usize {
        self.r_max
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.vectors.len()
    }

    #[inline]
    pub fn vector(&self, node: NodeId) -> Option<&Vector> {
        self.vectors.get(node.as_usize())
    }

    #[inline]
    pub fn neighbors(&self, node: NodeId) -> &[NodeId] {
        &self.adjacency[node.as_usize()]
    }

    #[inline]
    pub fn degree(&self, node: NodeId) -> usize {
        self.adjacency[node.as_usize()].len()
    }
}

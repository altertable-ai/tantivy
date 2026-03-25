//! HNSW search directly on the compact graph + distance closure.
//!
//! Replaces the previous approach of dumping to temp files and reloading through
//! `hnsw_rs::hnswio::HnswIo`.  The search algorithm is the standard HNSW greedy
//! descent + beam search from the original paper (Malkov & Yashunin, 2016).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::vector::io::CompactHnswGraph;

/// Wrapper returned by [`search`] — same fields as the old `hnsw_rs::prelude::Neighbour`
/// so the reader can convert to `(DocId, Score)` unchanged.
pub(crate) struct SearchResult {
    pub doc_id: u32,
    pub distance: f32,
}

/// Run an approximate k-NN search on the compact HNSW graph.
///
/// `distance_to(id)` returns the distance from the query to the vector at `id`.
/// The caller supplies storage (e.g. mmap'd quantized bytes + dequantize buffer).
pub(crate) fn search(
    graph: &CompactHnswGraph,
    distance_to: &mut impl FnMut(u32) -> f32,
    k: usize,
    ef: usize,
) -> Vec<SearchResult> {
    if graph.num_points == 0 || k == 0 {
        return Vec::new();
    }

    let mut current = graph.entry_point;
    let mut current_dist = distance_to(current);

    // Greedy descent: layers entry_layer → 1  (skip layer 0 — that gets the beam search).
    for layer in (1..=graph.entry_layer as usize).rev() {
        loop {
            let mut improved = false;
            for &neighbor in graph.neighbors(current, layer) {
                let d = distance_to(neighbor);
                if d < current_dist {
                    current = neighbor;
                    current_dist = d;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }

    // ef-bounded beam search at layer 0.
    let ef_actual = ef.max(k);
    let mut candidates: BinaryHeap<Reverse<DistId>> = BinaryHeap::new();
    let mut results: BinaryHeap<DistId> = BinaryHeap::new();
    let mut visited = VisitedSet::new(graph.num_points);

    visited.mark(current);
    candidates.push(Reverse(DistId(current_dist, current)));
    results.push(DistId(current_dist, current));

    while let Some(Reverse(DistId(c_dist, c_id))) = candidates.pop() {
        let worst = results.peek().map_or(f32::INFINITY, |d| d.0);
        if c_dist > worst && results.len() >= ef_actual {
            break;
        }

        for &neighbor in graph.neighbors(c_id, 0) {
            if !visited.mark(neighbor) {
                continue; // already visited
            }
            let d = distance_to(neighbor);
            let worst = results.peek().map_or(f32::INFINITY, |r| r.0);
            if d < worst || results.len() < ef_actual {
                candidates.push(Reverse(DistId(d, neighbor)));
                results.push(DistId(d, neighbor));
                if results.len() > ef_actual {
                    results.pop(); // remove farthest
                }
            }
        }
    }

    let mut out: Vec<SearchResult> = results
        .into_iter()
        .map(|DistId(d, id)| SearchResult {
            doc_id: id,
            distance: d,
        })
        .collect();
    out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    out.truncate(k);
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Ordered (distance, id) pair for BinaryHeap. Uses `total_cmp` so NaN is handled.
#[derive(Clone, Copy, PartialEq)]
struct DistId(f32, u32);

impl Eq for DistId {}

impl PartialOrd for DistId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0).then(self.1.cmp(&other.1))
    }
}

/// Bit-vector for O(1) visited checks instead of a `HashSet`.
struct VisitedSet {
    bits: Vec<u64>,
}

impl VisitedSet {
    fn new(n: u32) -> Self {
        let words = (n as usize).div_ceil(64);
        Self {
            bits: vec![0u64; words],
        }
    }

    /// Mark `id` as visited; returns `true` if it was **not** previously visited.
    #[inline]
    fn mark(&mut self, id: u32) -> bool {
        let word = id as usize / 64;
        let bit = 1u64 << (id % 64);
        if self.bits[word] & bit != 0 {
            return false;
        }
        self.bits[word] |= bit;
        true
    }
}

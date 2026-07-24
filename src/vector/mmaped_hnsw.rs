//! HNSW search directly on the compact graph + scalar-quantized flat vectors.
//!
//! Candidate vectors are dequantized row-by-row into a scratch buffer for distance
//! evaluation (no full flat decompression / allocation on the search path beyond
//! one `dim`-sized scratch buffer).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::schema::VectorDistance;
#[cfg(feature = "vector-simd")]
use crate::vector::io::dequant_distance_fn_for;
use crate::vector::io::CompactHnswGraph;
#[cfg(not(feature = "vector-simd"))]
use crate::vector::io::{dequantize_row_into, distance_fn_for};

/// Wrapper returned by [`search`] — same fields as the old `hnsw_rs::prelude::Neighbour`
/// so the reader can convert to `(DocId, Score)` unchanged.
pub(crate) struct SearchResult {
    pub doc_id: u32,
    pub distance: f32,
}

/// Run an approximate k-NN search on the compact HNSW graph over SQ8-stored vectors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    graph: &CompactHnswGraph,
    flat_u8: &[u8],
    mins: &[f32],
    scales: &[f32],
    dim: usize,
    dist: VectorDistance,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Vec<SearchResult> {
    if graph.num_points == 0 || k == 0 {
        return Vec::new();
    }

    // With vector-simd: fused dequantize+distance keeps values in SIMD registers,
    // no scratch buffer needed. Without: two-step dequantize → distance via scratch.
    #[cfg(feature = "vector-simd")]
    let dequant_distance = dequant_distance_fn_for(dist);
    #[cfg(not(feature = "vector-simd"))]
    let distance = distance_fn_for(dist);
    #[cfg(not(feature = "vector-simd"))]
    let mut scratch = vec![0f32; dim];

    macro_rules! row_distance {
        ($id:expr) => {{
            let start = $id as usize * dim;
            let row = &flat_u8[start..start + dim];
            #[cfg(feature = "vector-simd")]
            {
                dequant_distance(query, row, mins, scales)
            }
            #[cfg(not(feature = "vector-simd"))]
            {
                dequantize_row_into(row, mins, scales, &mut scratch);
                distance(query, &scratch)
            }
        }};
    }

    let mut current = graph.entry_point;
    let mut current_dist = row_distance!(current);

    // Greedy descent: layers entry_layer → 1  (skip layer 0 — that gets the beam search).
    for layer in (1..=graph.entry_layer as usize).rev() {
        loop {
            let mut improved = false;
            for &neighbor in graph.neighbors(current, layer) {
                let d = row_distance!(neighbor);
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
                continue;
            }
            let d = row_distance!(neighbor);
            let worst = results.peek().map_or(f32::INFINITY, |r| r.0);
            if d < worst || results.len() < ef_actual {
                candidates.push(Reverse(DistId(d, neighbor)));
                results.push(DistId(d, neighbor));
                if results.len() > ef_actual {
                    results.pop();
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

//! HNSW search on the compact mmap'd graph + 4-bit TurboQuant codes.
//!
//! The query is rotated once; each candidate is scored from packed nibbles
//! (cosine/dot) or reconstructed (euclidean). Neighbor lists are iterated
//! straight out of the mmap blob.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::schema::VectorDistance;
use crate::vector::io::CompactHnswGraph;
use crate::vector::turboquant::{
    dist_dot_packed, dist_dot_packed_batch, dist_l2_packed, packed_bytes, prepare_query, Codebook,
    PreparedQuery, TqPlus, FASTSCAN_N,
};

/// Wrapper returned by [`search`] — same fields as the old `hnsw_rs::prelude::Neighbour`
/// so the reader can convert to `(DocId, Score)` unchanged.
pub(crate) struct SearchResult {
    pub doc_id: u32,
    pub distance: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    graph: &CompactHnswGraph,
    packed: &[u8],
    renorm: &[f32],
    tqplus: &TqPlus,
    codebook: &Codebook,
    dim: usize,
    dist: VectorDistance,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Vec<SearchResult> {
    if graph.num_points == 0 || k == 0 {
        return Vec::new();
    }

    let padded = tqplus.shift.len();
    let stride = packed_bytes(padded);
    let prepared = prepare_query(query, dim, tqplus, codebook);

    if graph.num_points as usize <= k {
        return match dist {
            VectorDistance::Euclidean => exhaustive_search(graph, k, |id| {
                let start = id as usize * stride;
                let row = &packed[start..start + stride];
                let r = renorm.get(id as usize).copied().unwrap_or(1.0);
                dist_l2_packed(&prepared, row, dim, padded, codebook, tqplus, r)
            }),
            VectorDistance::Cosine | VectorDistance::DotProduct => {
                exhaustive_search(graph, k, |id| {
                    let start = id as usize * stride;
                    let row = &packed[start..start + stride];
                    let r = renorm.get(id as usize).copied().unwrap_or(1.0);
                    dist_dot_packed(&prepared, row, r)
                })
            }
        };
    }

    match dist {
        VectorDistance::Euclidean => search_graph(graph, k, ef, |id| {
            let start = id as usize * stride;
            let row = &packed[start..start + stride];
            let r = renorm.get(id as usize).copied().unwrap_or(1.0);
            dist_l2_packed(&prepared, row, dim, padded, codebook, tqplus, r)
        }),
        VectorDistance::Cosine | VectorDistance::DotProduct => {
            search_graph_dot(graph, k, ef, packed, stride, renorm, &prepared)
        }
    }
}

fn exhaustive_search(
    graph: &CompactHnswGraph,
    k: usize,
    row_distance: impl Fn(u32) -> f32,
) -> Vec<SearchResult> {
    let mut out: Vec<SearchResult> = (0..graph.num_points)
        .map(|id| SearchResult {
            doc_id: id,
            distance: row_distance(id),
        })
        .collect();
    out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    out.truncate(k);
    out
}

fn search_graph_dot(
    graph: &CompactHnswGraph,
    k: usize,
    ef: usize,
    packed: &[u8],
    stride: usize,
    renorm: &[f32],
    prepared: &PreparedQuery,
) -> Vec<SearchResult> {
    let score_one = |id: u32| {
        let start = id as usize * stride;
        debug_assert!(start + stride <= packed.len());
        debug_assert!((id as usize) < renorm.len());
        let row = unsafe { packed.get_unchecked(start..start + stride) };
        let r = unsafe { *renorm.get_unchecked(id as usize) };
        dist_dot_packed(prepared, row, r)
    };

    let mut current = graph.entry_point;
    let mut current_dist = score_one(current);

    for layer in (1..=graph.entry_layer as usize).rev() {
        loop {
            let mut improved = false;
            for neighbor in graph.neighbors(current, layer) {
                let d = score_one(neighbor);
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

    let ef_actual = ef.max(k);
    let mut candidates: BinaryHeap<Reverse<DistId>> = BinaryHeap::new();
    let mut results: BinaryHeap<DistId> = BinaryHeap::new();
    let mut visited = VisitedSet::new(graph.num_points);

    visited.mark(current);
    candidates.push(Reverse(DistId(current_dist, current)));
    results.push(DistId(current_dist, current));

    let mut buf = [0u32; FASTSCAN_N];
    let mut dists = [0.0f32; FASTSCAN_N];
    let mut buf_n = 0usize;

    let mut flush = |ids: &[u32],
                     results: &mut BinaryHeap<DistId>,
                     candidates: &mut BinaryHeap<Reverse<DistId>>| {
        if ids.is_empty() {
            return;
        }
        dist_dot_packed_batch(
            prepared,
            packed,
            stride,
            ids,
            renorm,
            &mut dists[..ids.len()],
        );
        for (i, &id) in ids.iter().enumerate() {
            let d = dists[i];
            let worst = results.peek().map_or(f32::INFINITY, |r| r.0);
            if d < worst || results.len() < ef_actual {
                candidates.push(Reverse(DistId(d, id)));
                results.push(DistId(d, id));
                if results.len() > ef_actual {
                    results.pop();
                }
            }
        }
    };

    while let Some(Reverse(DistId(c_dist, c_id))) = candidates.pop() {
        let worst = results.peek().map_or(f32::INFINITY, |d| d.0);
        if c_dist > worst && results.len() >= ef_actual {
            break;
        }

        for neighbor in graph.neighbors(c_id, 0) {
            if !visited.mark(neighbor) {
                continue;
            }
            buf[buf_n] = neighbor;
            buf_n += 1;
            if buf_n == FASTSCAN_N {
                flush(&buf, &mut results, &mut candidates);
                buf_n = 0;
            }
        }
    }
    if buf_n > 0 {
        flush(&buf[..buf_n], &mut results, &mut candidates);
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

fn search_graph(
    graph: &CompactHnswGraph,
    k: usize,
    ef: usize,
    row_distance: impl Fn(u32) -> f32,
) -> Vec<SearchResult> {
    let mut current = graph.entry_point;
    let mut current_dist = row_distance(current);

    for layer in (1..=graph.entry_layer as usize).rev() {
        loop {
            let mut improved = false;
            for neighbor in graph.neighbors(current, layer) {
                let d = row_distance(neighbor);
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

        for neighbor in graph.neighbors(c_id, 0) {
            if !visited.mark(neighbor) {
                continue;
            }
            let d = row_distance(neighbor);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::turboquant::{codebook_for_dim, encode_flat};

    /// Three isolated points: layer-0 walk from the entry point can only score id 0.
    fn isolated_three() -> CompactHnswGraph {
        CompactHnswGraph::new(0, 0, 3, vec![vec![Vec::new()]; 3])
    }

    fn search_isolated(dist: VectorDistance, k: usize) -> Vec<u32> {
        let dim = 4usize;
        let flat = [
            1.0, 0.0, 0.0, 0.0, // 0
            0.0, 1.0, 0.0, 0.0, // 1 — query
            0.0, 0.0, 1.0, 0.0, // 2
        ];
        let enc = encode_flat(3, dim, &flat);
        let codebook = codebook_for_dim(enc.padded_dim);
        search(
            &isolated_three(),
            &enc.packed,
            &enc.renorm,
            &enc.tqplus,
            &codebook,
            dim,
            dist,
            &[0.0, 1.0, 0.0, 0.0],
            k,
            1,
        )
        .into_iter()
        .map(|r| r.doc_id)
        .collect()
    }

    #[test]
    fn k_at_least_n_returns_isolated_nodes() {
        for dist in [
            VectorDistance::Cosine,
            VectorDistance::DotProduct,
            VectorDistance::Euclidean,
        ] {
            for k in [3, 10] {
                let ids = search_isolated(dist, k);
                assert_eq!(ids.len(), 3, "{dist:?} k={k}");
                assert_eq!(ids[0], 1, "{dist:?} k={k}: nearest must be the query row");
                let mut all = ids;
                all.sort_unstable();
                assert_eq!(all, [0, 1, 2], "{dist:?} k={k}");
            }
        }
    }
}

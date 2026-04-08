//! HNSW search on the compact graph + BBQ-stored vectors.
//!
//! The graph is built on BBQ-reconstructed `f32` rows. Search uses **asymmetric** distance
//! (closed form in [`crate::vector::bbq`]) that matches exact distance on those reconstructions,
//! without materializing a full `dim`-vector per candidate on the hot path.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::schema::VectorDistance;
use crate::vector::bbq::{
    bbq_binary_dots_block, bbq_bytes_per_row, bbq_dequantize_row, bbq_distance, bbq_distance_dot,
    bbq_distance_l2, bbq_dot_distances_block16, bbq_l2_distances_from_binary_dots, gather_bbq_rows,
    BbqDotCtx, BbqL2Ctx,
};
use crate::vector::io::{distance_fn_for, CompactHnswGraph};

/// Wrapper returned by [`search`] — same fields as the old `hnsw_rs::prelude::Neighbour`
/// so the reader can convert to `(DocId, Score)` unchanged.
pub(crate) struct SearchResult {
    pub doc_id: u32,
    pub distance: f32,
}

/// Gather + fused block distance pays off when enough neighbors amortize the bit loop.
const BBQ_BLOCK_MIN_NEIGHBORS: usize = 6;

/// Run approximate k-NN on the compact HNSW graph over BBQ-stored vectors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    graph: &CompactHnswGraph,
    bbq_bits: &[u8],
    bbq_lower: &[f32],
    bbq_upper: &[f32],
    centroid: &[f32],
    dim: usize,
    dist: VectorDistance,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Vec<SearchResult> {
    if graph.num_points == 0 || k == 0 {
        return Vec::new();
    }

    let bpr = bbq_bytes_per_row(dim);
    let dot_ctx = BbqDotCtx::new(query, centroid);
    let l2_ctx = BbqL2Ctx::new(query, centroid);

    // Euclidean upper layers still use f32 recon + metric fn (rare visits).
    let distance_f32 = distance_fn_for(dist);
    let mut scratch = vec![0f32; dim];

    let mut row_distance = |id: u32| -> f32 {
        let i = id as usize;
        let start = i * bpr;
        match dist {
            VectorDistance::Euclidean => {
                bbq_dequantize_row(
                    centroid,
                    &bbq_bits[start..start + bpr],
                    bbq_lower[i],
                    bbq_upper[i],
                    &mut scratch,
                );
                distance_f32(query, &scratch)
            }
            VectorDistance::Cosine | VectorDistance::DotProduct => bbq_distance(
                dist,
                query,
                &dot_ctx,
                &l2_ctx,
                &bbq_bits[start..start + bpr],
                bbq_lower[i],
                bbq_upper[i],
                dim,
            ),
        }
    };

    let mut current = graph.entry_point;
    let mut current_dist = row_distance(current);

    for layer in (1..=graph.entry_layer as usize).rev() {
        loop {
            let mut improved = false;
            for &neighbor in graph.neighbors(current, layer) {
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

    let mut gathered = vec![0u8; 16 * bpr];
    let mut binary_dots = vec![0f32; 16];
    let mut dist_block = vec![0f32; 16];
    let mut lowers16 = [0f32; 16];
    let mut uppers16 = [0f32; 16];

    while let Some(Reverse(DistId(c_dist, c_id))) = candidates.pop() {
        let worst = results.peek().map_or(f32::INFINITY, |d| d.0);
        if c_dist > worst && results.len() >= ef_actual {
            break;
        }

        let neighbors = graph.neighbors(c_id, 0);
        let mut nbr_batch: Vec<u32> = Vec::with_capacity(neighbors.len());
        for &nbr in neighbors {
            if visited.mark(nbr) {
                nbr_batch.push(nbr);
            }
        }

        for chunk in nbr_batch.chunks(16) {
            let n = chunk.len();

            if n < BBQ_BLOCK_MIN_NEIGHBORS {
                for i in 0..n {
                    let idx = chunk[i] as usize;
                    let start = idx * bpr;
                    let row_bits = &bbq_bits[start..start + bpr];
                    dist_block[i] = match dist {
                        VectorDistance::Euclidean => {
                            bbq_distance_l2(&l2_ctx, row_bits, bbq_lower[idx], bbq_upper[idx], dim)
                        }
                        VectorDistance::Cosine | VectorDistance::DotProduct => bbq_distance_dot(
                            &dot_ctx,
                            query,
                            row_bits,
                            bbq_lower[idx],
                            bbq_upper[idx],
                            dim,
                        ),
                    };
                }
            } else {
                gather_bbq_rows(bbq_bits, bpr, chunk, &mut gathered[..n * bpr]);
                for i in 0..n {
                    let idx = chunk[i] as usize;
                    lowers16[i] = bbq_lower[idx];
                    uppers16[i] = bbq_upper[idx];
                }
                binary_dots[..n].fill(0.0f32);

                match dist {
                    VectorDistance::Euclidean => {
                        bbq_binary_dots_block(
                            &l2_ctx.qc,
                            dim,
                            bpr,
                            &gathered[..n * bpr],
                            n,
                            &mut binary_dots,
                        );
                        bbq_l2_distances_from_binary_dots(
                            &l2_ctx,
                            &lowers16[..n],
                            &uppers16[..n],
                            &gathered[..n * bpr],
                            dim,
                            bpr,
                            n,
                            &binary_dots,
                            &mut dist_block,
                        );
                    }
                    VectorDistance::Cosine | VectorDistance::DotProduct => {
                        bbq_binary_dots_block(
                            query,
                            dim,
                            bpr,
                            &gathered[..n * bpr],
                            n,
                            &mut binary_dots,
                        );
                        bbq_dot_distances_block16(
                            &dot_ctx,
                            &lowers16[..n],
                            &uppers16[..n],
                            &binary_dots,
                            n,
                            &mut dist_block,
                        );
                    }
                }
            }

            for i in 0..n {
                let neighbor = chunk[i];
                let d = dist_block[i];
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

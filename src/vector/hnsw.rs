//! Build `hnsw_rs` graphs in memory for serialization. After building, the graph
//! topology is extracted into [`CompactHnswGraph`] via public iterators — no
//! `file_dump` needed.
//!
//! Distance during graph construction uses `anndists::DistL2` / `DistDot` with the crates'
//! `stdsimd` feature (portable SIMD `f32`), not the scalar `f64` path used by `DistCosine`.
//! Cosine-indexed fields pass L2-normalized vectors and [`DistDot`] so indexing matches the
//! `f32` search path in [`crate::vector::io`].

use hnsw_rs::prelude::*;
use rayon::current_num_threads;

use crate::schema::{VectorDistance, VectorOptions};
use crate::vector::io::CompactHnswGraph;
use crate::TantivyError;

pub(crate) const HNSW_DUMP_MAX_LAYER: usize = 16;

fn flat_rows_as_insert_slices(flat: &[f32], n: usize, dim: usize) -> Vec<(&[f32], usize)> {
    (0..n)
        .map(|doc| {
            let start = doc * dim;
            (&flat[start..start + dim], doc)
        })
        .collect()
}

fn parallel_insert_worthwhile(n: usize) -> bool {
    const MIN_DOCS_PER_RAYON_THREAD: usize = 1000;
    let threads = current_num_threads().max(1);
    n >= threads.saturating_mul(MIN_DOCS_PER_RAYON_THREAD)
}

/// Build HNSW with `hnsw_rs`, then extract the compact graph topology.
pub(crate) fn extract_compact_graph(
    options: &VectorOptions,
    max_doc: crate::DocId,
    flat: &[f32],
) -> crate::Result<CompactHnswGraph> {
    let dim = options.dimension;
    let n = max_doc as usize;
    if n == 0 {
        return Ok(CompactHnswGraph::new(0, 0, 0, Vec::new()));
    }
    if flat.len() != n * dim {
        return Err(TantivyError::InternalError(format!(
            "flat buffer len {} expected {}",
            flat.len(),
            n * dim
        )));
    }

    let max_layer = HNSW_DUMP_MAX_LAYER;
    let max_elements = n.max(1);
    let insert_slices_opt = if parallel_insert_worthwhile(n) {
        Some(flat_rows_as_insert_slices(flat, n, dim))
    } else {
        None
    };

    match options.distance {
        VectorDistance::Euclidean => {
            let h = build_hnsw::<DistL2>(
                options,
                max_elements,
                max_layer,
                n,
                dim,
                flat,
                &insert_slices_opt,
                DistL2 {},
            );
            graph_from_hnsw(&h, n)
        }
        VectorDistance::Cosine | VectorDistance::DotProduct => {
            let h = build_hnsw::<DistDot>(
                options,
                max_elements,
                max_layer,
                n,
                dim,
                flat,
                &insert_slices_opt,
                DistDot {},
            );
            graph_from_hnsw(&h, n)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_hnsw<'a, D: Distance<f32> + Send + Sync>(
    options: &VectorOptions,
    max_elements: usize,
    max_layer: usize,
    n: usize,
    dim: usize,
    flat: &'a [f32],
    insert_slices_opt: &Option<Vec<(&'a [f32], usize)>>,
    dist: D,
) -> Hnsw<'a, f32, D> {
    let mut h = Hnsw::<f32, D>::new(
        options.max_nb_connection,
        max_elements,
        max_layer,
        options.ef_construction,
        dist,
    );
    if n > 0 {
        if let Some(ref slices) = insert_slices_opt {
            h.parallel_insert_slice(slices);
        } else {
            for doc in 0..n {
                let start = doc * dim;
                h.insert((&flat[start..start + dim], doc));
            }
        }
    }
    h.set_searching_mode(true);
    h
}

/// Walk the built `Hnsw` via its public iterators and extract a [`CompactHnswGraph`].
fn graph_from_hnsw<D: Distance<f32> + Send + Sync>(
    hnsw: &Hnsw<'_, f32, D>,
    n: usize,
) -> crate::Result<CompactHnswGraph> {
    let pi = hnsw.get_point_indexation();
    // Pre-allocate adjacency: one entry per doc id.
    let mut adjacency: Vec<Vec<Vec<u32>>> = vec![Vec::new(); n];

    // Find entry point (a point at the highest layer).
    let mut entry_point: u32 = 0;
    let mut entry_layer: u8 = 0;

    // Layer 0 contains all points. Iterate it to get every point's full neighborhood.
    for point in pi.get_layer_iterator(0) {
        let origin_id = point.get_origin_id();
        if origin_id >= n {
            continue;
        }
        let pid = point.get_point_id();
        let point_layer = pid.0;

        if point_layer >= entry_layer {
            entry_layer = point_layer;
            entry_point = origin_id as u32;
        }

        let neighborhoods = point.get_neighborhood_id();
        let num_layers = (point_layer as usize + 1).min(neighborhoods.len());
        let mut layers = Vec::with_capacity(num_layers);
        for l in 0..num_layers {
            let neighbor_ids: Vec<u32> = neighborhoods
                .get(l)
                .map(|nbrs| nbrs.iter().map(|nb| nb.d_id as u32).collect())
                .unwrap_or_default();
            layers.push(neighbor_ids);
        }
        adjacency[origin_id] = layers;
    }

    // Points that were never visited (shouldn't happen, but be safe) get an empty
    // single-layer entry so graph.neighbors(id, 0) returns &[].
    for adj in &mut adjacency {
        if adj.is_empty() {
            adj.push(Vec::new());
        }
    }

    Ok(CompactHnswGraph::new(
        entry_point,
        entry_layer,
        n as u32,
        adjacency,
    ))
}

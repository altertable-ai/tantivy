//! Build `hnsw_rs` graphs in memory for serialization (indexing / merge). Search uses
//! [`crate::vector::mmaped_hnsw`] — always `HnswIo` reload with mmap on the data file, like other
//! Tantivy segment components backed by mmap’d bytes.

use hnsw_rs::prelude::*;
use rayon::current_num_threads;

use crate::schema::{VectorDistance, VectorOptions};
use crate::TantivyError;

/// Must match `hnsw_rs::hnsw::NB_LAYER_MAX`: `Description::dump` rejects other values, and
/// `file_dump` / reload expect this layer count.
pub(crate) const HNSW_DUMP_MAX_LAYER: usize = 16;

/// In-memory graph used only to call [`AnnT::file_dump`](hnsw_rs::api::AnnT::file_dump) while
/// building a segment. Search never holds this type.
pub(crate) enum BuiltHnsw {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
    Dot(Hnsw<'static, f32, DistDot>),
}

/// Row slices and doc ids for [`Hnsw::parallel_insert_slice`].
fn flat_rows_as_insert_slices(flat: &[f32], n: usize, dim: usize) -> Vec<(&[f32], usize)> {
    (0..n)
        .map(|doc| {
            let start = doc * dim;
            (&flat[start..start + dim], doc)
        })
        .collect()
}

/// Whether to use [`Hnsw::parallel_insert_slice`]: `hnsw_rs` uses Rayon and recommends batches
/// large enough to amortize threading — typically `1000 *` the number of Rayon threads.
fn parallel_insert_worthwhile(n: usize) -> bool {
    const MIN_DOCS_PER_RAYON_THREAD: usize = 1000;
    let threads = current_num_threads().max(1);
    n >= threads.saturating_mul(MIN_DOCS_PER_RAYON_THREAD)
}

/// Builds HNSW in memory from row-major vectors (for `file_dump` during indexing).
pub(crate) fn build_hnsw_for_flat(
    options: &VectorOptions,
    max_doc: crate::DocId,
    flat: &[f32],
) -> crate::Result<BuiltHnsw> {
    let dim = options.dimension;
    let n = max_doc as usize;
    if flat.len() != n * dim {
        return Err(TantivyError::InternalError(format!(
            "flat buffer len {} expected {}",
            flat.len(),
            n * dim
        )));
    }
    let max_layer = HNSW_DUMP_MAX_LAYER;
    let max_elements = n.max(1);
    let insert_slices_opt = if n > 0 && parallel_insert_worthwhile(n) {
        Some(flat_rows_as_insert_slices(flat, n, dim))
    } else {
        None
    };
    let inner = match options.distance {
        VectorDistance::Euclidean => {
            let mut h = Hnsw::<'_, f32, DistL2>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistL2 {},
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
            BuiltHnsw::L2(h)
        }
        VectorDistance::Cosine => {
            let mut h = Hnsw::<'_, f32, DistCosine>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistCosine {},
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
            BuiltHnsw::Cosine(h)
        }
        VectorDistance::DotProduct => {
            let mut h = Hnsw::<'_, f32, DistDot>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistDot {},
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
            BuiltHnsw::Dot(h)
        }
    };
    Ok(inner)
}

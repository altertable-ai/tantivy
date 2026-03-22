//! In-memory HNSW construction (shared by writer and reader).

use hnsw_rs::prelude::*;

use crate::schema::{VectorDistance, VectorOptions};
use crate::TantivyError;

/// Must match `hnsw_rs::hnsw::NB_LAYER_MAX`: `Description::dump` rejects other values, and
/// `file_dump` / reload expect this layer count.
pub(crate) const HNSW_DUMP_MAX_LAYER: usize = 16;

/// Loaded HNSW index for one distance type.
pub(crate) enum VectorIndexInner {
    L2(Hnsw<'static, f32, DistL2>),
    Cosine(Hnsw<'static, f32, DistCosine>),
    Dot(Hnsw<'static, f32, DistDot>),
}

/// Builds HNSW in memory from row-major vectors (no disk I/O).
pub(crate) fn build_hnsw_in_memory(
    options: &VectorOptions,
    max_doc: crate::DocId,
    flat: &[f32],
) -> crate::Result<VectorIndexInner> {
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
    let inner = match options.distance {
        VectorDistance::Euclidean => {
            let mut h = Hnsw::<'_, f32, DistL2>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistL2 {},
            );
            for doc in 0..n {
                let start = doc * dim;
                let slice = &flat[start..start + dim];
                h.insert((slice, doc));
            }
            h.set_searching_mode(true);
            VectorIndexInner::L2(h)
        }
        VectorDistance::Cosine => {
            let mut h = Hnsw::<'_, f32, DistCosine>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistCosine {},
            );
            for doc in 0..n {
                let start = doc * dim;
                let slice = &flat[start..start + dim];
                h.insert((slice, doc));
            }
            h.set_searching_mode(true);
            VectorIndexInner::Cosine(h)
        }
        VectorDistance::DotProduct => {
            let mut h = Hnsw::<'_, f32, DistDot>::new(
                options.max_nb_connection,
                max_elements,
                max_layer,
                options.ef_construction,
                DistDot {},
            );
            for doc in 0..n {
                let start = doc * dim;
                let slice = &flat[start..start + dim];
                h.insert((slice, doc));
            }
            h.set_searching_mode(true);
            VectorIndexInner::Dot(h)
        }
    };
    Ok(inner)
}

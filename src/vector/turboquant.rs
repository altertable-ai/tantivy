//! Thin adapter over turbovec's public quantization and SIMD search modules.
//!
//! Tantivy owns persistence and segment integration; this module only bridges
//! flat row-major vectors to turbovec's bit-plane codes and blocked search
//! layout.

use crate::schema::{VectorDistance, VectorOptions};
use crate::{DocId, TantivyError};

const MAX_INPUT_MAGNITUDE: f32 = 1e16;

pub(crate) struct EncodedTurboQuant {
    pub bit_width: usize,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    pub norms: Vec<f32>,
}

pub(crate) struct TurboQuantCache {
    pub rotation: Vec<f32>,
    pub centroids: Vec<f32>,
    pub blocked_codes: Vec<u8>,
    pub n_blocks: usize,
}

pub(crate) fn validate_options(options: &VectorOptions) -> crate::Result<()> {
    let dim = options.dimension;
    if dim == 0 || dim % 8 != 0 {
        return Err(TantivyError::InvalidArgument(format!(
            "TurboQuant vector dimension must be a positive multiple of 8, got {dim}"
        )));
    }
    if !matches!(options.bit_width, 2 | 4) {
        return Err(TantivyError::InvalidArgument(format!(
            "TurboQuant bit_width must be 2 or 4, got {}",
            options.bit_width
        )));
    }
    Ok(())
}

pub(crate) fn encode(
    options: &VectorOptions,
    flat: &[f32],
    num_docs: usize,
) -> crate::Result<EncodedTurboQuant> {
    validate_options(options)?;
    let dim = options.dimension;
    if flat.len() != num_docs * dim {
        return Err(TantivyError::InternalError(format!(
            "TurboQuant flat buffer len {} expected {}",
            flat.len(),
            num_docs * dim
        )));
    }
    validate_values(flat, dim)?;

    let norms = if options.distance == VectorDistance::DotProduct {
        flat.chunks_exact(dim).map(l2_norm).collect()
    } else {
        Vec::new()
    };
    let rotation = turbovec::rotation::make_rotation_matrix(dim);
    let (boundaries, centroids) = turbovec::codebook::codebook(options.bit_width, dim);
    let (packed_codes, scales, tqplus_shift, tqplus_scale) = turbovec::encode::encode(
        flat,
        num_docs,
        dim,
        &rotation,
        &boundaries,
        &centroids,
        options.bit_width,
        None,
    );

    Ok(EncodedTurboQuant {
        bit_width: options.bit_width,
        packed_codes,
        scales,
        tqplus_shift,
        tqplus_scale,
        norms,
    })
}

pub(crate) fn prepare_cache(
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    packed_codes: &[u8],
) -> crate::Result<TurboQuantCache> {
    if dim == 0 || dim % 8 != 0 {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(format!(
                "TurboQuant dimension must be a positive multiple of 8, got {dim}"
            )),
        ));
    }
    if !matches!(bit_width, 2 | 4) {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(format!(
                "TurboQuant bit_width must be 2 or 4, got {bit_width}"
            )),
        ));
    }
    let expected = packed_len(n_vectors, dim, bit_width);
    if packed_codes.len() != expected {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(format!(
                "TurboQuant packed codes len {} != expected {}",
                packed_codes.len(),
                expected
            )),
        ));
    }
    let rotation = turbovec::rotation::make_rotation_matrix(dim);
    let (_boundaries, centroids) = turbovec::codebook::codebook(bit_width, dim);
    let (blocked_codes, n_blocks) = turbovec::pack::repack(packed_codes, n_vectors, bit_width, dim);
    Ok(TurboQuantCache {
        rotation,
        centroids,
        blocked_codes,
        n_blocks,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn search(
    query: &[f32],
    cache: &TurboQuantCache,
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    bit_width: usize,
    dim: usize,
    n_vectors: usize,
    k: usize,
    mask: Option<&[u64]>,
) -> crate::Result<Vec<(DocId, f32)>> {
    validate_values(query, dim)?;
    let (scores, indices) = turbovec::search::search(
        query,
        1,
        &cache.rotation,
        &cache.blocked_codes,
        &cache.centroids,
        scales,
        tqplus_shift,
        tqplus_scale,
        bit_width,
        dim,
        n_vectors,
        cache.n_blocks,
        k,
        mask,
    );
    Ok(indices
        .into_iter()
        .zip(scores)
        .filter_map(|(idx, score)| {
            if idx < 0 {
                None
            } else {
                Some((idx as DocId, score))
            }
        })
        .collect())
}

pub(crate) fn build_alive_mask<I>(num_docs: usize, alive_docs: I) -> Vec<u64>
where
    I: IntoIterator<Item = DocId>,
{
    let mut mask = vec![0u64; num_docs.div_ceil(64)];
    for doc in alive_docs {
        let doc = doc as usize;
        if doc < num_docs {
            mask[doc / 64] |= 1u64 << (doc % 64);
        }
    }
    mask
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dequantize_flat(
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    norms: &[f32],
    bit_width: usize,
    dim: usize,
    num_docs: usize,
    distance: VectorDistance,
) -> crate::Result<Vec<f32>> {
    let expected = packed_len(num_docs, dim, bit_width);
    if packed_codes.len() != expected || scales.len() != num_docs {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only("TurboQuant payload length mismatch"),
        ));
    }
    let shift_identity;
    let scale_identity;
    let shift = if tqplus_shift.is_empty() {
        shift_identity = vec![0.0f32; dim];
        shift_identity.as_slice()
    } else {
        tqplus_shift
    };
    let scale_tq = if tqplus_scale.is_empty() {
        scale_identity = vec![1.0f32; dim];
        scale_identity.as_slice()
    } else {
        tqplus_scale
    };
    if shift.len() != dim || scale_tq.len() != dim {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only("TurboQuant TQ+ length mismatch"),
        ));
    }
    if distance == VectorDistance::DotProduct && norms.len() != num_docs {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only("TurboQuant norms length mismatch"),
        ));
    }

    let rotation = turbovec::rotation::make_rotation_matrix(dim);
    let (_boundaries, centroids) = turbovec::codebook::codebook(bit_width, dim);
    let mut out = vec![0f32; num_docs * dim];
    let mut rotated = vec![0f32; dim];
    let mut unit = vec![0f32; dim];
    for row in 0..num_docs {
        decode_rotated_row(
            &packed_codes[row * row_len(dim, bit_width)..(row + 1) * row_len(dim, bit_width)],
            &centroids,
            shift,
            scale_tq,
            bit_width,
            dim,
            &mut rotated,
        );
        inverse_rotate(&rotated, &rotation, &mut unit);
        let norm = if distance == VectorDistance::DotProduct {
            norms[row]
        } else {
            1.0
        };
        let base = row * dim;
        for d in 0..dim {
            out[base + d] = unit[d] * norm;
        }
    }
    Ok(out)
}

pub(crate) fn packed_len(num_docs: usize, dim: usize, bit_width: usize) -> usize {
    num_docs * row_len(dim, bit_width)
}

fn row_len(dim: usize, bit_width: usize) -> usize {
    bit_width * (dim / 8)
}

fn validate_values(values: &[f32], dim: usize) -> crate::Result<()> {
    if let Some((i, &value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || value.abs() >= MAX_INPUT_MAGNITUDE)
    {
        return Err(TantivyError::InvalidArgument(format!(
            "invalid TurboQuant value at vector {}, coord {}: {}",
            i / dim,
            i % dim,
            value
        )));
    }
    Ok(())
}

fn l2_norm(row: &[f32]) -> f32 {
    row.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn decode_rotated_row(
    packed_row: &[u8],
    centroids: &[f32],
    shift: &[f32],
    scale_tq: &[f32],
    bit_width: usize,
    dim: usize,
    out: &mut [f32],
) {
    let bytes_per_plane = dim / 8;
    for d in 0..dim {
        let byte_in_plane = d / 8;
        let bit_in_byte = 7 - (d % 8);
        let mut code = 0u8;
        for plane in 0..bit_width {
            let byte = packed_row[plane * bytes_per_plane + byte_in_plane];
            if byte & (1u8 << bit_in_byte) != 0 {
                code |= 1u8 << plane;
            }
        }
        out[d] = centroids[code as usize] / scale_tq[d] - shift[d];
    }
}

fn inverse_rotate(rotated: &[f32], rotation: &[f32], out: &mut [f32]) {
    let dim = rotated.len();
    debug_assert_eq!(rotation.len(), dim * dim);
    out.fill(0.0);
    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotated[j] * rotation[j * dim + i];
        }
        out[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_repack_search_smoke() -> crate::Result<()> {
        let dim = 64usize;
        let num_docs = 16usize;
        let mut flat = vec![0f32; num_docs * dim];
        for row in 0..num_docs {
            flat[row * dim + row] = 1.0;
        }
        let options = VectorOptions::new(dim).set_bit_width(4);
        let encoded = encode(&options, &flat, num_docs)?;
        let cache = prepare_cache(
            encoded.bit_width,
            dim,
            num_docs,
            encoded.packed_codes.as_slice(),
        )?;
        let hits = search(
            &flat[3 * dim..4 * dim],
            &cache,
            &encoded.scales,
            &encoded.tqplus_shift,
            &encoded.tqplus_scale,
            encoded.bit_width,
            dim,
            num_docs,
            3,
            None,
        )?;
        assert_eq!(hits.first().map(|(doc, _)| *doc), Some(3));
        Ok(())
    }
}

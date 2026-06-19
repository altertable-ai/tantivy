//! On-disk layout for the `.vec` segment file.
//!
//! **V5 format** (magic `TNVYVEC5`): per-field TurboQuant bit-plane codes
//! searched by turbovec's SIMD kernel.

use std::io::Write;
use std::ops::Range;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use crate::directory::{FileSlice, OwnedBytes};
use crate::schema::{VectorDistance, VectorOptions};
use crate::vector::turboquant;
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC5";
const VERSION: u32 = 5;
const CODEC_TURBOQUANT: u8 = 1;

// ---------------------------------------------------------------------------
// L2 normalization (cosine -> dot-product conversion)
// ---------------------------------------------------------------------------

/// In-place L2-normalize a single vector. Zero-norm vectors are left untouched.
///
/// With `vector-simd`: 8-wide portable SIMD.
/// Without: plain `f32` accumulation so the path auto-vectorizes at full NEON width.
#[inline]
pub(crate) fn l2_normalize(v: &mut [f32]) {
    #[cfg(feature = "vector-simd")]
    return simd_impl::l2_normalize(v);

    #[cfg(not(feature = "vector-simd"))]
    {
        let mut norm_sq = 0.0f32;
        for i in 0..v.len() {
            let x = v[i];
            norm_sq += x * x;
        }
        if norm_sq > 0.0f32 {
            let inv = 1.0f32 / norm_sq.sqrt();
            for x in v.iter_mut() {
                *x *= inv;
            }
        }
    }
}

/// Normalize every row of a row-major flat buffer in-place for Cosine fields.
pub(crate) fn normalize_flat_for_cosine(flat: &mut [f32], dim: usize, dist: VectorDistance) {
    if dist != VectorDistance::Cosine {
        return;
    }
    for row in flat.chunks_exact_mut(dim) {
        l2_normalize(row);
    }
}

// ---------------------------------------------------------------------------
// Bundle written during indexing / merge
// ---------------------------------------------------------------------------

pub(crate) struct VectorFieldBundle {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub bit_width: usize,
    pub packed_codes: Vec<u8>,
    pub scales: Vec<f32>,
    pub tqplus_shift: Vec<f32>,
    pub tqplus_scale: Vec<f32>,
    pub norms: Vec<f32>,
}

pub(crate) fn write_vec_file(
    writer: &mut dyn Write,
    fields: &[VectorFieldBundle],
) -> crate::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_u32::<LittleEndian>(VERSION)?;
    writer.write_u32::<LittleEndian>(fields.len() as u32)?;

    for bundle in fields {
        writer.write_u32::<LittleEndian>(bundle.field_id)?;

        let opts_json = serde_json::to_vec(&bundle.options)
            .map_err(|e| TantivyError::InternalError(format!("vector options json: {e}")))?;
        writer.write_u32::<LittleEndian>(opts_json.len() as u32)?;
        writer.write_all(&opts_json)?;

        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(bundle.options.dimension as u32)?;
        writer.write_all(&[CODEC_TURBOQUANT])?;
        write_turboquant_payload(
            writer,
            bundle.num_docs as usize,
            bundle.options.dimension,
            bundle.bit_width,
            &bundle.packed_codes,
            &bundle.scales,
            &bundle.tqplus_shift,
            &bundle.tqplus_scale,
            &bundle.norms,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_turboquant_payload(
    writer: &mut dyn Write,
    num_docs: usize,
    dim: usize,
    bit_width: usize,
    packed_codes: &[u8],
    scales: &[f32],
    tqplus_shift: &[f32],
    tqplus_scale: &[f32],
    norms: &[f32],
) -> crate::Result<()> {
    let expected_packed = turboquant::packed_len(num_docs, dim, bit_width);
    if packed_codes.len() != expected_packed || scales.len() != num_docs {
        return Err(TantivyError::InternalError(format!(
            "TurboQuant payload mismatch: packed {} expected {}, scales {} expected {}",
            packed_codes.len(),
            expected_packed,
            scales.len(),
            num_docs
        )));
    }
    if !(tqplus_shift.is_empty() && tqplus_scale.is_empty())
        && (tqplus_shift.len() != dim || tqplus_scale.len() != dim)
    {
        return Err(TantivyError::InternalError(
            "TurboQuant TQ+ payload length mismatch".to_string(),
        ));
    }
    if !norms.is_empty() && norms.len() != num_docs {
        return Err(TantivyError::InternalError(
            "TurboQuant norms length mismatch".to_string(),
        ));
    }
    writer.write_u32::<LittleEndian>(bit_width as u32)?;
    writer.write_all(packed_codes)?;
    for &scale in scales {
        writer.write_f32::<LittleEndian>(scale)?;
    }
    writer.write_u32::<LittleEndian>(tqplus_shift.len() as u32)?;
    for &shift in tqplus_shift {
        writer.write_f32::<LittleEndian>(shift)?;
    }
    writer.write_u32::<LittleEndian>(tqplus_scale.len() as u32)?;
    for &scale in tqplus_scale {
        writer.write_f32::<LittleEndian>(scale)?;
    }
    writer.write_u32::<LittleEndian>(norms.len() as u32)?;
    for &norm in norms {
        writer.write_f32::<LittleEndian>(norm)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lazy read: mmap-backed persisted bit-plane codes; blocked cache is built by reader.
// ---------------------------------------------------------------------------

pub(crate) struct LazyVectorField {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub dimension: usize,
    raw: OwnedBytes,
    bit_width: usize,
    packed_codes_range: Range<usize>,
    scales_range: Range<usize>,
    tqplus_shift_range: Range<usize>,
    tqplus_scale_range: Range<usize>,
    norms_range: Range<usize>,
}

impl LazyVectorField {
    pub(crate) fn turbo_bit_width(&self) -> usize {
        self.bit_width
    }

    pub(crate) fn turbo_packed_codes(&self) -> &[u8] {
        &self.raw[self.packed_codes_range.clone()]
    }

    pub(crate) fn parse_turbo_scales(&self) -> crate::Result<Vec<f32>> {
        f32_slice_from_bytes(&self.raw[self.scales_range.clone()], self.num_docs as usize)
    }

    pub(crate) fn parse_tqplus_shift(&self) -> crate::Result<Vec<f32>> {
        f32_slice_from_bytes(
            &self.raw[self.tqplus_shift_range.clone()],
            self.tqplus_shift_range.len() / 4,
        )
    }

    pub(crate) fn parse_tqplus_scale(&self) -> crate::Result<Vec<f32>> {
        f32_slice_from_bytes(
            &self.raw[self.tqplus_scale_range.clone()],
            self.tqplus_scale_range.len() / 4,
        )
    }

    pub(crate) fn parse_turbo_norms(&self) -> crate::Result<Vec<f32>> {
        f32_slice_from_bytes(
            &self.raw[self.norms_range.clone()],
            self.norms_range.len() / 4,
        )
    }

    /// Full dequantized flat vectors for merge / retrieval.
    pub(crate) fn dequantize_flat(&self) -> crate::Result<Vec<f32>> {
        turboquant::dequantize_flat(
            self.turbo_packed_codes(),
            &self.parse_turbo_scales()?,
            &self.parse_tqplus_shift()?,
            &self.parse_tqplus_scale()?,
            &self.parse_turbo_norms()?,
            self.turbo_bit_width(),
            self.dimension,
            self.num_docs as usize,
            self.options.distance,
        )
    }
}

pub(crate) fn read_vec_file_lazy(data: FileSlice) -> crate::Result<Vec<LazyVectorField>> {
    let bytes = data.read_bytes()?;
    let buf = bytes.as_slice();
    let mut pos = 0usize;

    check_len(buf, 0, 8, "magic")?;
    if &buf[..8] != MAGIC {
        return Err(corruption(format!(
            "unsupported .vec format (expected TNVYVEC5, got {:?})",
            std::str::from_utf8(&buf[..8]).unwrap_or("???")
        )));
    }
    pos += 8;

    let _version = read_u32_le(buf, &mut pos)?;
    let n_fields = read_u32_le(buf, &mut pos)? as usize;

    let mut out = Vec::with_capacity(n_fields);
    for _ in 0..n_fields {
        let field_id = read_u32_le(buf, &mut pos)?;

        let opt_len = read_u32_le(buf, &mut pos)? as usize;
        check_len(buf, pos, opt_len, "options")?;
        let options: VectorOptions = serde_json::from_slice(&buf[pos..pos + opt_len])
            .map_err(|e| corruption(format!("vector options: {e}")))?;
        pos += opt_len;

        let num_docs = read_u32_le(buf, &mut pos)? as usize;
        let dimension = read_u32_le(buf, &mut pos)? as usize;

        check_len(buf, pos, 1, "codec")?;
        let codec = buf[pos];
        pos += 1;
        if codec != CODEC_TURBOQUANT {
            return Err(corruption(format!("unsupported vector codec tag {codec}")));
        }

        out.push(read_turboquant_field(
            bytes.clone(),
            buf,
            &mut pos,
            field_id,
            options,
            dimension,
            num_docs,
        )?);
    }
    Ok(out)
}

fn read_turboquant_field(
    raw: OwnedBytes,
    buf: &[u8],
    pos: &mut usize,
    field_id: u32,
    options: VectorOptions,
    dimension: usize,
    num_docs: usize,
) -> crate::Result<LazyVectorField> {
    let bit_width = read_u32_le(buf, pos)? as usize;
    let packed_len = turboquant::packed_len(num_docs, dimension, bit_width);
    check_len(buf, *pos, packed_len, "turboquant packed_codes")?;
    let packed_codes_range = *pos..*pos + packed_len;
    *pos += packed_len;

    let scales_len = num_docs * 4;
    check_len(buf, *pos, scales_len, "turboquant scales")?;
    let scales_range = *pos..*pos + scales_len;
    *pos += scales_len;

    let shift_len = read_u32_le(buf, pos)? as usize * 4;
    check_len(buf, *pos, shift_len, "turboquant tqplus_shift")?;
    let tqplus_shift_range = *pos..*pos + shift_len;
    *pos += shift_len;

    let scale_len = read_u32_le(buf, pos)? as usize * 4;
    check_len(buf, *pos, scale_len, "turboquant tqplus_scale")?;
    let tqplus_scale_range = *pos..*pos + scale_len;
    *pos += scale_len;

    let norms_len = read_u32_le(buf, pos)? as usize * 4;
    check_len(buf, *pos, norms_len, "turboquant norms")?;
    let norms_range = *pos..*pos + norms_len;
    *pos += norms_len;

    Ok(LazyVectorField {
        field_id,
        options,
        num_docs: num_docs as u32,
        dimension,
        raw,
        bit_width,
        packed_codes_range,
        scales_range,
        tqplus_shift_range,
        tqplus_scale_range,
        norms_range,
    })
}

fn f32_slice_from_bytes(bytes: &[u8], dim: usize) -> crate::Result<Vec<f32>> {
    if bytes.len() != dim * 4 {
        return Err(corruption("f32 slice length mismatch"));
    }
    let mut v = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(4) {
        v.push(LittleEndian::read_f32(chunk));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Portable SIMD acceleration (feature = "vector-simd", requires nightly)
// ---------------------------------------------------------------------------

#[cfg(feature = "vector-simd")]
mod simd_impl {
    use std::simd::prelude::*;

    const LANES: usize = 8;
    type F32x = Simd<f32, LANES>;

    #[inline]
    pub(super) fn l2_normalize(v: &mut [f32]) {
        let dim = v.len();
        let full = dim / LANES;
        let mut acc = F32x::splat(0.0);
        for c in 0..full {
            let i = c * LANES;
            let x = F32x::from_slice(&v[i..]);
            acc += x * x;
        }
        let mut norm_sq = acc.reduce_sum();
        for i in (full * LANES)..dim {
            norm_sq += v[i] * v[i];
        }
        if norm_sq > 0.0f32 {
            let inv_scalar = 1.0f32 / norm_sq.sqrt();
            let inv = F32x::splat(inv_scalar);
            for c in 0..full {
                let i = c * LANES;
                let x = F32x::from_slice(&v[i..]);
                (x * inv).copy_to_slice(&mut v[i..i + LANES]);
            }
            for i in (full * LANES)..dim {
                v[i] *= inv_scalar;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Score from distance
// ---------------------------------------------------------------------------

pub(crate) fn distance_to_score(distance: f32) -> crate::Score {
    1.0f32 / (1.0f32 + distance)
}

fn read_u32_le(buf: &[u8], pos: &mut usize) -> crate::Result<u32> {
    check_len(buf, *pos, 4, "u32")?;
    let v = LittleEndian::read_u32(&buf[*pos..*pos + 4]);
    *pos += 4;
    Ok(v)
}

fn check_len(buf: &[u8], pos: usize, need: usize, what: &str) -> crate::Result<()> {
    if buf.len() < pos + need {
        Err(corruption(format!(".vec truncated ({what})")))
    } else {
        Ok(())
    }
}

fn corruption(msg: impl std::fmt::Display) -> TantivyError {
    TantivyError::DataCorruption(crate::error::DataCorruption::comment_only(msg))
}

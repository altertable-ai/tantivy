//! Better-binary-style quantization: 1-bit residuals vs a field centroid + per-vector
//! `(lower, upper)` (Lloyd–Max split). V4 on-disk storage.
//!
//! **Asymmetric search**: `<query, recon>` and `||query - recon||` admit closed forms using only
//! `binary_dot` over packed bits (see [`binary_dot_query_bits`]), matching exact distance on the
//! BBQ-reconstructed vector — same space the HNSW graph is built on.

use crate::schema::VectorDistance;

/// Packed bits per row (ceil(dim / 8)).
#[inline]
pub(crate) fn bbq_bytes_per_row(dim: usize) -> usize {
    dim.div_ceil(8)
}

/// Field centroid = component-wise mean of all row-major `f32` vectors.
pub(crate) fn compute_centroid(flat: &[f32], dim: usize, num_docs: usize) -> Vec<f32> {
    assert_eq!(flat.len(), num_docs * dim);
    if num_docs == 0 {
        return vec![0.0f32; dim];
    }
    let mut c = vec![0.0f32; dim];
    for row in 0..num_docs {
        let base = row * dim;
        for d in 0..dim {
            c[d] += flat[base + d];
        }
    }
    let inv = 1.0f32 / num_docs as f32;
    for x in &mut c {
        *x *= inv;
    }
    c
}

fn quantize_residual_row(
    residual: &[f32],
    dim: usize,
    bits: &mut [u8],
    lower: &mut f32,
    upper: &mut f32,
) {
    debug_assert_eq!(residual.len(), dim);
    debug_assert_eq!(bits.len(), bbq_bytes_per_row(dim));
    bits.fill(0);

    if dim == 0 {
        *lower = 0.0;
        *upper = 0.0;
        return;
    }

    let mut sorted: Vec<f32> = residual.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let threshold = sorted[dim / 2];

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut n0 = 0u32;
    let mut n1 = 0u32;
    for d in 0..dim {
        let v = residual[d];
        let bit = v >= threshold;
        if bit {
            sum1 += v;
            n1 += 1;
            let byte = d / 8;
            let bit_i = d % 8;
            bits[byte] |= 1u8 << bit_i;
        } else {
            sum0 += v;
            n0 += 1;
        }
    }

    if n0 == 0 {
        let mean = sum1 / n1 as f32;
        *lower = mean;
        *upper = mean;
        bits.fill(0xff);
        return;
    }
    if n1 == 0 {
        let mean = sum0 / n0 as f32;
        *lower = mean;
        *upper = mean;
        bits.fill(0);
        return;
    }

    *lower = sum0 / n0 as f32;
    *upper = sum1 / n1 as f32;
}

pub(crate) fn compute_bbq_params(
    flat: &[f32],
    dim: usize,
    num_docs: usize,
) -> (Vec<f32>, Vec<u8>, Vec<f32>, Vec<f32>) {
    let centroid = compute_centroid(flat, dim, num_docs);
    let bpr = bbq_bytes_per_row(dim);
    let mut bits = vec![0u8; num_docs * bpr];
    let mut lowers = vec![0.0f32; num_docs];
    let mut uppers = vec![0.0f32; num_docs];

    let mut residual = vec![0.0f32; dim];
    for row in 0..num_docs {
        let base = row * dim;
        for d in 0..dim {
            residual[d] = flat[base + d] - centroid[d];
        }
        let row_bits = &mut bits[row * bpr..(row + 1) * bpr];
        quantize_residual_row(&residual, dim, row_bits, &mut lowers[row], &mut uppers[row]);
    }

    (centroid, bits, lowers, uppers)
}

pub(crate) fn bbq_dequantize_row(
    centroid: &[f32],
    bits_row: &[u8],
    lower: f32,
    upper: f32,
    out: &mut [f32],
) {
    let dim = out.len();
    debug_assert_eq!(centroid.len(), dim);
    debug_assert_eq!(bits_row.len(), bbq_bytes_per_row(dim));
    for d in 0..dim {
        let bit = (bits_row[d / 8] >> (d % 8)) & 1 != 0;
        let r = if bit { upper } else { lower };
        out[d] = centroid[d] + r;
    }
}

// --- Query contexts (precompute once per search) ---------------------------------------------

pub(crate) struct BbqDotCtx {
    pub qc_dot: f32,
    pub q_sum: f32,
}

impl BbqDotCtx {
    pub fn new(query: &[f32], centroid: &[f32]) -> Self {
        let qc_dot: f32 = query.iter().zip(centroid.iter()).map(|(q, c)| q * c).sum();
        let q_sum: f32 = query.iter().sum();
        Self { qc_dot, q_sum }
    }
}

pub(crate) struct BbqL2Ctx {
    pub qc_dist_sq: f32,
    pub qc_sum: f32,
    pub qc: Vec<f32>,
}

impl BbqL2Ctx {
    pub fn new(query: &[f32], centroid: &[f32]) -> Self {
        let dim = query.len();
        let mut qc = Vec::with_capacity(dim);
        let mut qc_dist_sq = 0.0f32;
        for d in 0..dim {
            let v = query[d] - centroid[d];
            qc_dist_sq += v * v;
            qc.push(v);
        }
        let qc_sum: f32 = qc.iter().sum();
        Self {
            qc_dist_sq,
            qc_sum,
            qc,
        }
    }
}

#[inline]
fn popcount_row(bits: &[u8], dim: usize) -> u32 {
    let full_bytes = dim / 8;
    let mut c = 0u32;
    for b in &bits[..full_bytes] {
        c += b.count_ones();
    }
    let rem = dim % 8;
    if rem > 0 {
        let last = bits[full_bytes];
        let mask = (1u8 << rem) - 1;
        c += (last & mask).count_ones();
    }
    c
}

/// `sum(query[d] where bit[d]=1)` — exact `<query, residual_part>` for 1-bit residual coding.
#[inline]
pub(crate) fn binary_dot_query_bits(query: &[f32], bits: &[u8], dim: usize) -> f32 {
    #[cfg(feature = "vector-simd")]
    return simd_impl::binary_dot_query_bits(query, bits, dim);
    #[cfg(not(feature = "vector-simd"))]
    scalar_binary_dot_query_bits(query, bits, dim)
}

/// Scalar reference path; kept for `#[cfg(test)]` comparisons when `vector-simd` is enabled.
#[inline]
#[cfg_attr(feature = "vector-simd", allow(dead_code))]
fn scalar_binary_dot_query_bits(query: &[f32], bits: &[u8], dim: usize) -> f32 {
    let bpr = bbq_bytes_per_row(dim);
    debug_assert_eq!(bits.len(), bpr);
    let mut s = 0.0f32;
    for b in 0..bpr {
        let byte = bits[b];
        let base = b * 8;
        for k in 0..8 {
            let d = base + k;
            if d >= dim {
                break;
            }
            if (byte >> k) & 1 != 0 {
                s += query[d];
            }
        }
    }
    s
}

/// Same metric as [`crate::vector::io::distance_fn_for`] for Cosine/Dot on recon: `(1 -
/// <q,recon>).max(0)`.
#[inline]
pub(crate) fn bbq_distance_dot(
    ctx: &BbqDotCtx,
    query: &[f32],
    bits: &[u8],
    lower: f32,
    upper: f32,
    dim: usize,
) -> f32 {
    let bd = binary_dot_query_bits(query, bits, dim);
    let approx_sim = ctx.qc_dot + lower * ctx.q_sum + (upper - lower) * bd;
    (1.0f32 - approx_sim).max(0.0f32)
}

#[inline]
pub(crate) fn bbq_distance_l2(
    ctx: &BbqL2Ctx,
    bits: &[u8],
    lower: f32,
    upper: f32,
    dim: usize,
) -> f32 {
    let bd = binary_dot_query_bits(&ctx.qc, bits, dim);
    let qc_r_dot = lower * ctx.qc_sum + (upper - lower) * bd;
    let pop = popcount_row(bits, dim) as f32;
    let r_norm_sq = pop * upper * upper + (dim as f32 - pop) * lower * lower;
    let dist_sq = (ctx.qc_dist_sq - 2.0f32 * qc_r_dot + r_norm_sq).max(0.0f32);
    dist_sq.sqrt()
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bbq_distance(
    dist: VectorDistance,
    query: &[f32],
    dot_ctx: &BbqDotCtx,
    l2_ctx: &BbqL2Ctx,
    bits: &[u8],
    lower: f32,
    upper: f32,
    dim: usize,
) -> f32 {
    match dist {
        VectorDistance::Euclidean => bbq_distance_l2(l2_ctx, bits, lower, upper, dim),
        VectorDistance::Cosine | VectorDistance::DotProduct => {
            bbq_distance_dot(dot_ctx, query, bits, lower, upper, dim)
        }
    }
}

pub(crate) fn gather_bbq_rows(
    bbq_bits: &[u8],
    bytes_per_row: usize,
    doc_ids: &[u32],
    out: &mut [u8],
) {
    let n = doc_ids.len();
    debug_assert_eq!(out.len(), n * bytes_per_row);
    for (i, &id) in doc_ids.iter().enumerate() {
        let start = id as usize * bytes_per_row;
        out[i * bytes_per_row..(i + 1) * bytes_per_row]
            .copy_from_slice(&bbq_bits[start..start + bytes_per_row]);
    }
}

pub(crate) fn bbq_binary_dots_block(
    query: &[f32],
    dim: usize,
    bytes_per_row: usize,
    gathered: &[u8],
    n: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(gathered.len(), n * bytes_per_row);
    debug_assert!(out.len() >= n);
    out[..n].fill(0.0f32);

    #[cfg(feature = "vector-simd")]
    simd_impl::bbq_binary_dots_block(query, dim, bytes_per_row, gathered, n, out);

    #[cfg(not(feature = "vector-simd"))]
    {
        for b in 0..bytes_per_row {
            let byte_base = b * 8;
            for k in 0..8 {
                let d = byte_base + k;
                if d >= dim {
                    break;
                }
                let qd = query[d];
                for i in 0..n {
                    let byte = gathered[i * bytes_per_row + b];
                    if (byte >> k) & 1 != 0 {
                        out[i] += qd;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bbq_l2_distances_from_binary_dots(
    ctx: &BbqL2Ctx,
    lowers: &[f32],
    uppers: &[f32],
    gathered: &[u8],
    dim: usize,
    bytes_per_row: usize,
    n: usize,
    binary_dots: &[f32],
    out: &mut [f32],
) {
    for i in 0..n {
        let row = &gathered[i * bytes_per_row..(i + 1) * bytes_per_row];
        let pop = popcount_row(row, dim) as f32;
        let r_norm_sq = pop * uppers[i] * uppers[i] + (dim as f32 - pop) * lowers[i] * lowers[i];
        let qc_r_dot = lowers[i] * ctx.qc_sum + (uppers[i] - lowers[i]) * binary_dots[i];
        out[i] = (ctx.qc_dist_sq - 2.0f32 * qc_r_dot + r_norm_sq)
            .max(0.0f32)
            .sqrt();
    }
}

pub(crate) fn bbq_dot_distances_block16(
    ctx: &BbqDotCtx,
    lowers: &[f32],
    uppers: &[f32],
    binary_dots: &[f32],
    n: usize,
    out: &mut [f32],
) {
    for i in 0..n {
        let approx_sim =
            ctx.qc_dot + lowers[i] * ctx.q_sum + (uppers[i] - lowers[i]) * binary_dots[i];
        out[i] = (1.0f32 - approx_sim).max(0.0f32);
    }
}

#[cfg(feature = "vector-simd")]
mod simd_impl {
    use std::simd::num::SimdFloat;
    use std::simd::{Mask, Select, Simd};

    const LANES: usize = 8;
    type F32x = Simd<f32, LANES>;
    type I32Mask = Mask<i32, LANES>;

    pub(super) fn binary_dot_query_bits(query: &[f32], bits: &[u8], dim: usize) -> f32 {
        let bytes_per_row = super::bbq_bytes_per_row(dim);
        debug_assert_eq!(bits.len(), bytes_per_row);
        let mut acc = 0.0f32;
        for b in 0..bytes_per_row {
            let base = b * 8;
            let end = (base + 8).min(dim);
            let n = end - base;
            if n == 8 {
                let q = F32x::from_slice(&query[base..base + 8]);
                let byte = bits[b];
                let m: [f32; 8] =
                    std::array::from_fn(|k| if (byte >> k) & 1 != 0 { 1.0f32 } else { 0.0f32 });
                acc += (q * F32x::from_array(m)).reduce_sum();
            } else {
                for k in 0..n {
                    let d = base + k;
                    if (bits[b] >> k) & 1 != 0 {
                        acc += query[d];
                    }
                }
            }
        }
        acc
    }

    pub(super) fn bbq_binary_dots_block(
        query: &[f32],
        dim: usize,
        bytes_per_row: usize,
        gathered: &[u8],
        n: usize,
        out: &mut [f32],
    ) {
        debug_assert!(out[..n].iter().all(|&x| x == 0.0));
        for b in 0..bytes_per_row {
            let base = b * 8;
            for k in 0..8 {
                let d = base + k;
                if d >= dim {
                    break;
                }
                let qd = query[d];
                let qv = F32x::splat(qd);

                let mut i = 0usize;
                while i + LANES <= n {
                    let mask_bits: [bool; LANES] = std::array::from_fn(|j| {
                        let byte = gathered[(i + j) * bytes_per_row + b];
                        (byte >> k) & 1 != 0
                    });
                    let m = I32Mask::from_array(mask_bits);
                    let cur = F32x::from_slice(&out[i..]);
                    (m.select(qv, F32x::splat(0.0)) + cur).copy_to_slice(&mut out[i..i + LANES]);
                    i += LANES;
                }
                while i < n {
                    let byte = gathered[i * bytes_per_row + b];
                    if (byte >> k) & 1 != 0 {
                        out[i] += qd;
                    }
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bbq_round_trip_levels() {
        let dim = 4usize;
        let flat = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let num_docs = 2usize;
        let (centroid, bits, lowers, uppers) = compute_bbq_params(&flat, dim, num_docs);
        assert_eq!(centroid.len(), dim);
        let mut out = vec![0f32; dim];
        let bpr = bbq_bytes_per_row(dim);
        bbq_dequantize_row(&centroid, &bits[0..bpr], lowers[0], uppers[0], &mut out);
        for d in 0..dim {
            let bit = (bits[d / 8] >> (d % 8)) & 1 != 0;
            let r = if bit { uppers[0] } else { lowers[0] };
            let expected = centroid[d] + r;
            assert!(
                (out[d] - expected).abs() < 1e-5,
                "d={d} out={} exp={}",
                out[d],
                expected
            );
        }
    }

    #[test]
    fn bbq_asymmetric_dot_matches_reconstruction_dot() {
        let dim = 32usize;
        let num_docs = 4usize;
        let flat: Vec<f32> = (0..num_docs * dim)
            .map(|i| ((i % 7) as f32) * 0.1 - 0.3)
            .collect();
        let (centroid, bits, lowers, uppers) = compute_bbq_params(&flat, dim, num_docs);
        let bpr = bbq_bytes_per_row(dim);
        let query: Vec<f32> = (0..dim).map(|i| ((i % 5) as f32) * 0.07).collect();
        let ctx = BbqDotCtx::new(&query, &centroid);

        let mut recon = vec![0f32; dim];
        for row in 0..num_docs {
            let row_bits = &bits[row * bpr..(row + 1) * bpr];
            bbq_dequantize_row(&centroid, row_bits, lowers[row], uppers[row], &mut recon);
            let exact_dot: f32 = query.iter().zip(recon.iter()).map(|(a, b)| a * b).sum();
            let d_dot = bbq_distance_dot(&ctx, &query, row_bits, lowers[row], uppers[row], dim);
            let approx_sim = ctx.qc_dot
                + lowers[row] * ctx.q_sum
                + (uppers[row] - lowers[row]) * binary_dot_query_bits(&query, row_bits, dim);
            assert!((approx_sim - exact_dot).abs() < 5e-4);
            let exact_dist = (1.0f32 - exact_dot).max(0.0f32);
            assert!((d_dot - exact_dist).abs() < 5e-4, "row {row}");
        }
    }

    #[test]
    fn bbq_asymmetric_l2_matches_reconstruction() {
        let dim = 24usize;
        let num_docs = 3usize;
        let flat: Vec<f32> = (0..num_docs * dim)
            .map(|i| (i as f32) * 0.02 - 0.2)
            .collect();
        let (centroid, bits, lowers, uppers) = compute_bbq_params(&flat, dim, num_docs);
        let bpr = bbq_bytes_per_row(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.03).collect();
        let l2_ctx = BbqL2Ctx::new(&query, &centroid);
        let mut recon = vec![0f32; dim];
        for row in 0..num_docs {
            let row_bits = &bits[row * bpr..(row + 1) * bpr];
            bbq_dequantize_row(&centroid, row_bits, lowers[row], uppers[row], &mut recon);
            let mut acc = 0.0f32;
            for d in 0..dim {
                let t = query[d] - recon[d];
                acc += t * t;
            }
            let exact_l2 = acc.sqrt();
            let d_l2 = bbq_distance_l2(&l2_ctx, row_bits, lowers[row], uppers[row], dim);
            assert!(
                (d_l2 - exact_l2).abs() < 5e-3,
                "row {row}: d_l2={d_l2} exact={exact_l2}"
            );
        }
    }

    #[test]
    fn block_binary_dots_match_scalar() {
        let dim = 16usize;
        let num_docs = 5usize;
        let flat: Vec<f32> = (0..num_docs * dim).map(|i| (i as f32) * 0.01).collect();
        let (_centroid, bits, _, _) = compute_bbq_params(&flat, dim, num_docs);
        let bpr = bbq_bytes_per_row(dim);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.03).collect();
        let ids = [0u32, 1, 2, 3, 4];
        let mut gathered = vec![0u8; ids.len() * bpr];
        gather_bbq_rows(&bits, bpr, &ids, &mut gathered);
        let mut block_bd = vec![0f32; ids.len()];
        bbq_binary_dots_block(&query, dim, bpr, &gathered, ids.len(), &mut block_bd);
        for (i, &id) in ids.iter().enumerate() {
            let row = &bits[id as usize * bpr..(id as usize + 1) * bpr];
            let s = scalar_binary_dot_query_bits(&query, row, dim);
            assert!((block_bd[i] - s).abs() < 1e-4, "i={i}");
        }
    }
}

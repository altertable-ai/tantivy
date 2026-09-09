//! 4-bit TurboQuant codec for the `.vec` store.
//!
//! Pipeline (ICLR 2026 TurboQuant + TQ+ + RaBitQ-style length renormalization):
//! 1. L2-normalize, pad to the next power of two, random sign flips, FWHT.
//! 2. Per-coordinate TQ+ shift/scale fitted on a strided sample of the segment.
//! 3. Lloyd-Max scalar quantization (16 levels) of the Beta/Gaussian marginal.
//! 4. Pack two 4-bit codes per byte.
//! 5. Store `||v|| / ⟨u, x̂⟩` per row so inner-product scores are unbiased.
//!
//! Search rotates the query once and dots against codebook centroids in the
//! rotated domain — no full inverse-rotate on the HNSW hot path (cosine / dot).

use std::collections::HashMap;
use std::f64::consts::{PI, SQRT_2};
use std::sync::Mutex;

use once_cell::sync::Lazy;

pub const BITS: usize = 4;
pub const N_LEVELS: usize = 1 << BITS;
/// HNSW layer-0 neighbor lists are width 16; FastScan scores that many rows at once.
pub(crate) const FASTSCAN_N: usize = 16;
const CALIBRATION_ROWS: usize = 1024;
const MIN_NORM: f32 = 1e-12;

/// One dimension of the query ADC table. 16-byte aligned for SIMD loads.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct LutRow([f32; N_LEVELS]);

/// Rotation size: next power of two so FWHT is exact. Minimum 4 so 4-bit
/// packing always has an even length.
#[inline]
pub fn padded_dim(dim: usize) -> usize {
    dim.next_power_of_two().max(4)
}

#[inline]
pub fn packed_bytes(padded: usize) -> usize {
    debug_assert!(padded % 2 == 0);
    padded / 2
}

#[derive(Clone, Debug)]
pub struct Codebook {
    pub boundaries: Vec<f32>,
    pub centroids: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct TqPlus {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
}

impl TqPlus {
    fn identity(padded: usize) -> Self {
        Self {
            shift: vec![0.0; padded],
            scale: vec![1.0; padded],
        }
    }
}

#[derive(Clone, Debug)]
pub struct EncodedFlat {
    pub padded_dim: usize,
    pub packed: Vec<u8>,
    pub renorm: Vec<f32>,
    pub tqplus: TqPlus,
}

/// Encode row-major `flat` (`n * dim` f32). `n == 0` yields empty buffers and
/// identity TQ+.
pub fn encode_flat(n: usize, dim: usize, flat: &[f32]) -> EncodedFlat {
    assert_eq!(flat.len(), n * dim);
    let padded = padded_dim(dim);
    if n == 0 {
        return EncodedFlat {
            padded_dim: padded,
            packed: Vec::new(),
            renorm: Vec::new(),
            tqplus: TqPlus::identity(padded),
        };
    }

    let signs = rotation_signs(padded);
    let codebook = codebook_for_dim(padded);

    let mut rotated = vec![0.0f32; n * padded];
    let mut norms = vec![0.0f32; n];
    for row in 0..n {
        let src = &flat[row * dim..row * dim + dim];
        rotate_row(
            src,
            dim,
            padded,
            signs,
            &mut rotated[row * padded..row * padded + padded],
            &mut norms[row],
        );
    }

    let tqplus = fit_tqplus(&rotated, n, padded, &codebook);
    let mut packed = vec![0u8; n * packed_bytes(padded)];
    let mut renorm = vec![0.0f32; n];
    let pb = packed_bytes(padded);
    for row in 0..n {
        encode_rotated_row(
            &rotated[row * padded..row * padded + padded],
            norms[row],
            padded,
            &codebook,
            &tqplus,
            &mut packed[row * pb..row * pb + pb],
            &mut renorm[row],
        );
    }

    EncodedFlat {
        padded_dim: padded,
        packed,
        renorm,
        tqplus,
    }
}

/// Reconstruct one original-`dim` row from packed codes (for merge / `vector()`).
/// `scale` is the per-vector length-renorm factor (≈ original L2 norm).
pub fn reconstruct_row(
    packed_row: &[u8],
    dim: usize,
    padded: usize,
    codebook: &Codebook,
    tqplus: &TqPlus,
    scale: f32,
) -> Vec<f32> {
    let mut rot = unpack_to_rotated(packed_row, padded, codebook, tqplus);
    inverse_rotate_inplace(&mut rot, rotation_signs(padded));
    let mut out = rot[..dim].to_vec();
    if scale != 1.0 {
        for x in &mut out {
            *x *= scale;
        }
    }
    out
}

pub fn reconstruct_flat(
    packed: &[u8],
    n: usize,
    dim: usize,
    padded: usize,
    codebook: &Codebook,
    tqplus: &TqPlus,
    renorm: &[f32],
) -> Vec<f32> {
    let stride = packed_bytes(padded);
    let mut out = vec![0.0f32; n * dim];
    for row in 0..n {
        let rec = reconstruct_row(
            &packed[row * stride..row * stride + stride],
            dim,
            padded,
            codebook,
            tqplus,
            renorm.get(row).copied().unwrap_or(1.0),
        );
        out[row * dim..row * dim + dim].copy_from_slice(&rec);
    }
    out
}

/// Query-side state for scoring packed rows without inverse-rotating them.
pub struct PreparedQuery {
    /// `q_lut[d][code] = (rotated_q[d] / scale[d]) * centroid[code]`.
    /// Candidate scoring is a sum of table lookups (no inner-loop muls).
    q_lut: Vec<LutRow>,
    bias: f32,
    q_orig: Vec<f32>,
}

pub fn prepare_query(
    query: &[f32],
    dim: usize,
    tqplus: &TqPlus,
    codebook: &Codebook,
) -> PreparedQuery {
    let padded = padded_dim(dim);
    let signs = rotation_signs(padded);
    let mut rot = vec![0.0f32; padded];
    let mut norm = 0.0f32;
    rotate_row(query, dim, padded, signs, &mut rot, &mut norm);
    let mut bias = 0.0f32;
    let mut q_lut = vec![LutRow([0.0f32; N_LEVELS]); padded];
    let centroids = codebook_centroids(codebook);
    for i in 0..padded {
        let s = tqplus.scale[i];
        let q = if s > 0.0 { rot[i] / s } else { 0.0 };
        bias -= rot[i] * tqplus.shift[i];
        let row = &mut q_lut[i].0;
        for (code, &c) in centroids.iter().enumerate() {
            row[code] = q * c;
        }
    }
    PreparedQuery {
        q_lut,
        bias,
        q_orig: query.to_vec(),
    }
}

/// Estimated cosine/dot distance `1 − ⟨q, v⟩` using length-renormalized IP.
#[inline]
pub fn dist_dot_packed(prepared: &PreparedQuery, packed_row: &[u8], renorm: f32) -> f32 {
    let ip = estimated_ip(prepared, packed_row, renorm);
    (1.0f32 - ip).max(0.0)
}

/// FastScan: score up to [`FASTSCAN_N`] packed rows. Per-row add order matches
/// [`dist_dot_packed`] (byte 0 lo/hi, byte 1 lo/hi, …). `out.len() == ids.len()`.
pub(crate) fn dist_dot_packed_batch(
    prepared: &PreparedQuery,
    packed: &[u8],
    stride: usize,
    ids: &[u32],
    renorm: &[f32],
    out: &mut [f32],
) {
    debug_assert_eq!(ids.len(), out.len());
    debug_assert!(ids.len() <= FASTSCAN_N);
    if ids.is_empty() {
        return;
    }
    if ids.len() == FASTSCAN_N {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                dist_dot_packed_batch_avx2(prepared, packed, stride, ids, renorm, out);
            }
            return;
        }
    }
    dist_dot_packed_batch_scalar(prepared, packed, stride, ids, renorm, out);
}

/// Euclidean distance via reconstructed original-space vector.
pub fn dist_l2_packed(
    prepared: &PreparedQuery,
    packed_row: &[u8],
    dim: usize,
    padded: usize,
    codebook: &Codebook,
    tqplus: &TqPlus,
    scale: f32,
) -> f32 {
    let rec = reconstruct_row(packed_row, dim, padded, codebook, tqplus, scale);
    let mut acc = 0.0f32;
    for i in 0..dim {
        let d = prepared.q_orig[i] - rec[i];
        acc += d * d;
    }
    acc.sqrt()
}

pub fn codebook_for_dim(padded: usize) -> Codebook {
    static MEMO: Lazy<Mutex<HashMap<usize, Codebook>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    if let Ok(memo) = MEMO.lock() {
        if let Some(hit) = memo.get(&padded) {
            return hit.clone();
        }
    }
    let computed = lloyd_max_gaussian(padded);
    if let Ok(mut memo) = MEMO.lock() {
        memo.insert(padded, computed.clone());
    }
    computed
}

fn codebook_centroids(codebook: &Codebook) -> [f32; N_LEVELS] {
    let mut out = [0.0f32; N_LEVELS];
    let n = codebook.centroids.len().min(N_LEVELS);
    out[..n].copy_from_slice(&codebook.centroids[..n]);
    out
}

#[inline(always)]
fn estimated_ip(prepared: &PreparedQuery, packed_row: &[u8], renorm: f32) -> f32 {
    let lut = prepared.q_lut.as_slice();
    debug_assert_eq!(packed_row.len().saturating_mul(2), lut.len());
    let mut inner = prepared.bias;
    let mut dim = 0usize;
    // SAFETY: `packed_row.len() * 2 == lut.len()`, so `dim`/`dim+1` stay in
    // range. Nibbles are 0..16 and index `[f32; 16]`. Adds are sequential so
    // packed IP matches `q[i] * centroid[code]` associativity.
    unsafe {
        for &byte in packed_row {
            inner += *lut
                .get_unchecked(dim)
                .0
                .get_unchecked((byte & 0x0f) as usize);
            inner += *lut
                .get_unchecked(dim + 1)
                .0
                .get_unchecked((byte >> 4) as usize);
            dim += 2;
        }
    }
    inner * renorm
}

fn dist_dot_packed_batch_scalar(
    prepared: &PreparedQuery,
    packed: &[u8],
    stride: usize,
    ids: &[u32],
    renorm: &[f32],
    out: &mut [f32],
) {
    let lut = prepared.q_lut.as_slice();
    debug_assert_eq!(stride.saturating_mul(2), lut.len());
    let n = ids.len();
    let mut acc = [prepared.bias; FASTSCAN_N];
    unsafe {
        for pos in 0..stride {
            let t0 = &lut.get_unchecked(pos * 2).0;
            let t1 = &lut.get_unchecked(pos * 2 + 1).0;
            for i in 0..n {
                let byte = *packed.get_unchecked(*ids.get_unchecked(i) as usize * stride + pos);
                *acc.get_unchecked_mut(i) += *t0.get_unchecked((byte & 0x0f) as usize);
                *acc.get_unchecked_mut(i) += *t1.get_unchecked((byte >> 4) as usize);
            }
        }
        for i in 0..n {
            let id = *ids.get_unchecked(i) as usize;
            let ip = acc.get_unchecked(i) * *renorm.get_unchecked(id);
            *out.get_unchecked_mut(i) = (1.0f32 - ip).max(0.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_dot_packed_batch_avx2(
    prepared: &PreparedQuery,
    packed: &[u8],
    stride: usize,
    ids: &[u32],
    renorm: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(ids.len(), FASTSCAN_N);
    let lut = prepared.q_lut.as_slice();
    let mut acc0 = _mm256_set1_ps(prepared.bias);
    let mut acc1 = _mm256_set1_ps(prepared.bias);
    let mut bytes = [0u8; FASTSCAN_N];
    let nibble_mask = _mm_set1_epi8(0x0f);
    for pos in 0..stride {
        for (i, slot) in bytes.iter_mut().enumerate() {
            *slot = *packed.get_unchecked(*ids.get_unchecked(i) as usize * stride + pos);
        }
        let codes = _mm_loadu_si128(bytes.as_ptr().cast());
        let lo = _mm_and_si128(codes, nibble_mask);
        let hi = _mm_and_si128(_mm_srli_epi16(codes, 4), nibble_mask);
        let t0 = &lut.get_unchecked(pos * 2).0;
        let t1 = &lut.get_unchecked(pos * 2 + 1).0;
        let lo0 = avx2_lut8(t0, _mm256_cvtepu8_epi32(lo));
        let lo1 = avx2_lut8(t0, _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
        let hi0 = avx2_lut8(t1, _mm256_cvtepu8_epi32(hi));
        let hi1 = avx2_lut8(t1, _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));
        acc0 = _mm256_add_ps(acc0, _mm256_add_ps(lo0, hi0));
        acc1 = _mm256_add_ps(acc1, _mm256_add_ps(lo1, hi1));
    }
    let mut ips = [0.0f32; FASTSCAN_N];
    _mm256_storeu_ps(ips.as_mut_ptr(), acc0);
    _mm256_storeu_ps(ips.as_mut_ptr().add(8), acc1);
    for i in 0..FASTSCAN_N {
        let id = *ids.get_unchecked(i) as usize;
        let ip = ips[i] * *renorm.get_unchecked(id);
        *out.get_unchecked_mut(i) = (1.0f32 - ip).max(0.0);
    }
}

/// 8 nibble indices (0..16) → 8 floats from a 16-entry table via AVX2 permute, not gather.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_lut8(
    table: &[f32; N_LEVELS],
    idx: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let t_lo = _mm256_loadu_ps(table.as_ptr());
    let t_hi = _mm256_loadu_ps(table.as_ptr().add(8));
    let sel_hi = _mm256_cmpgt_epi32(idx, _mm256_set1_epi32(7));
    let idx7 = _mm256_and_si256(idx, _mm256_set1_epi32(7));
    let a = _mm256_permutevar8x32_ps(t_lo, idx7);
    let b = _mm256_permutevar8x32_ps(t_hi, idx7);
    _mm256_blendv_ps(a, b, _mm256_castsi256_ps(sel_hi))
}

fn rotate_row(
    src: &[f32],
    dim: usize,
    padded: usize,
    signs: &[f32],
    dst: &mut [f32],
    norm_out: &mut f32,
) {
    debug_assert_eq!(dst.len(), padded);
    let mut nsq = 0.0f32;
    for i in 0..dim {
        nsq += src[i] * src[i];
    }
    let norm = nsq.sqrt();
    *norm_out = norm;
    let inv = if norm > MIN_NORM { 1.0 / norm } else { 0.0 };
    dst.fill(0.0);
    for i in 0..dim {
        dst[i] = src[i] * inv * signs[i];
    }
    // Padded tail stays 0; still apply sign so inverse-rotate is consistent.
    fwht_inplace(dst);
}

fn inverse_rotate_inplace(v: &mut [f32], signs: &[f32]) {
    fwht_inplace(v);
    for i in 0..v.len() {
        v[i] *= signs[i];
    }
}

/// Normalized FWHT. Self-inverse: applying twice is identity.
fn fwht_inplace(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    while h < n {
        let step = h * 2;
        for i in (0..n).step_by(step) {
            for j in i..i + h {
                let x = a[j];
                let y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
        }
        h = step;
    }
    let s = 1.0f32 / (n as f32).sqrt();
    for x in a.iter_mut() {
        *x *= s;
    }
}

/// Compile-time Rademacher signs for one FWHT width. Seed is mixed with `N` so
/// different padded dims get different flips. SplitMix64, integer-only, so this
/// is a true `const` table — no lazy cache.
const fn rotation_signs_array<const N: usize>() -> [f32; N] {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (N as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut signs = [1.0f32; N];
    let mut i = 0;
    while i < N {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        signs[i] = if z & 1 == 0 { 1.0 } else { -1.0 };
        i += 1;
    }
    signs
}

/// `padded_dim` is always a power of two ≥ 4. Tables cover 4…8192 (OpenAI 3072
/// pads to 4096). Wider fields are a programming error, not a runtime lookup.
fn rotation_signs(padded: usize) -> &'static [f32] {
    macro_rules! table {
        ($n:expr) => {{
            const T: [f32; $n] = rotation_signs_array::<$n>();
            &T
        }};
    }
    match padded {
        4 => table!(4),
        8 => table!(8),
        16 => table!(16),
        32 => table!(32),
        64 => table!(64),
        128 => table!(128),
        256 => table!(256),
        512 => table!(512),
        1024 => table!(1024),
        2048 => table!(2048),
        4096 => table!(4096),
        8192 => table!(8192),
        other => panic!(
            "FWHT padded dim {other} is not a compiled rotation size (need 4..=8192, power of two)"
        ),
    }
}

fn fit_tqplus(rotated: &[f32], n: usize, padded: usize, codebook: &Codebook) -> TqPlus {
    if n < 2 {
        return TqPlus::identity(padded);
    }
    let c_outer = codebook
        .centroids
        .iter()
        .fold(0.0f32, |acc, &c| acc.max(c.abs()));
    // Match turbovec: map empirical quantiles at the codebook-edge CDF onto ±c_outer.
    let sigma = 1.0 / (padded as f64).sqrt();
    let p_hi = norm_cdf(f64::from(c_outer) / sigma).clamp(0.5, 0.999);
    let p_lo = 1.0 - p_hi;

    let sample_n = n.min(CALIBRATION_ROWS);
    let stride = (n / sample_n).max(1);
    let mut tqplus = TqPlus::identity(padded);
    let mut col = Vec::with_capacity(sample_n);
    for d in 0..padded {
        col.clear();
        let mut i = 0usize;
        while i < n && col.len() < sample_n {
            col.push(rotated[i * padded + d]);
            i += stride;
        }
        if col.len() < 2 {
            continue;
        }
        col.sort_by(|a, b| a.total_cmp(b));
        let lo = quantile_sorted(&col, p_lo);
        let hi = quantile_sorted(&col, p_hi);
        if hi - lo < 1e-12 {
            continue;
        }
        let scale = (c_outer - (-c_outer)) / (hi - lo);
        if !scale.is_finite() || scale <= 0.0 {
            continue;
        }
        let shift = (-c_outer) / scale - lo;
        if shift.is_finite() && scale.is_finite() {
            tqplus.shift[d] = shift;
            tqplus.scale[d] = scale;
        }
    }
    tqplus
}

fn quantile_sorted(sorted: &[f32], p: f64) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = ((p * (n.saturating_sub(1) as f64)).round() as usize).min(n - 1);
    sorted[idx]
}

fn encode_rotated_row(
    u_rot: &[f32],
    orig_norm: f32,
    padded: usize,
    codebook: &Codebook,
    tqplus: &TqPlus,
    packed_out: &mut [u8],
    renorm_out: &mut f32,
) {
    let mut codes = vec![0u8; padded];
    let mut x_hat_dot = 0.0f32;
    for d in 0..padded {
        let cal = (u_rot[d] + tqplus.shift[d]) * tqplus.scale[d];
        let code = quantize_scalar(cal, &codebook.boundaries);
        codes[d] = code;
        let centroid = codebook.centroids[code as usize];
        let x_hat = if tqplus.scale[d] > 0.0 {
            centroid / tqplus.scale[d] - tqplus.shift[d]
        } else {
            0.0
        };
        x_hat_dot += u_rot[d] * x_hat;
    }
    *renorm_out = if x_hat_dot.abs() > MIN_NORM {
        orig_norm / x_hat_dot
    } else {
        orig_norm
    };
    pack_nibbles(&codes, packed_out);
}

fn quantize_scalar(x: f32, boundaries: &[f32]) -> u8 {
    let mut c = 0u8;
    for &b in boundaries {
        if x >= b {
            c += 1;
        } else {
            break;
        }
    }
    c
}

fn pack_nibbles(codes: &[u8], out: &mut [u8]) {
    debug_assert_eq!(out.len(), codes.len() / 2);
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = codes[i * 2] & 0x0f;
        let hi = codes[i * 2 + 1] & 0x0f;
        *slot = lo | (hi << 4);
    }
}

fn unpack_to_rotated(
    packed_row: &[u8],
    padded: usize,
    codebook: &Codebook,
    tqplus: &TqPlus,
) -> Vec<f32> {
    let mut rot = vec![0.0f32; padded];
    for (byte_i, &byte) in packed_row.iter().enumerate() {
        let i0 = byte_i * 2;
        for (k, code) in [(i0, byte & 0x0f), (i0 + 1, byte >> 4)] {
            if k >= padded {
                break;
            }
            let centroid = codebook.centroids[code as usize];
            rot[k] = if tqplus.scale[k] > 0.0 {
                centroid / tqplus.scale[k] - tqplus.shift[k]
            } else {
                0.0
            };
        }
    }
    rot
}

/// Lloyd-Max for N(0, 1/d) clipped to [-1, 1] — the high-d limit of the
/// TurboQuant Beta((d-1)/2,(d-1)/2) marginal on the sphere.
fn lloyd_max_gaussian(d: usize) -> Codebook {
    let sigma = 1.0 / (d as f64).sqrt();
    let n_levels = N_LEVELS;
    let spread = (3.0 * sigma).min(1.0);
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..80 {
        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        for i in 0..n_levels - 1 {
            edges.push((centroids[i] + centroids[i + 1]) * 0.5);
        }
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];
        let mut max_change = 0.0f64;
        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];
            new_centroids[i] = gauss_cond_mean(lo, hi, sigma);
            max_change = max_change.max((new_centroids[i] - centroids[i]).abs());
        }
        centroids = new_centroids;
        if max_change < 1e-8 {
            break;
        }
    }

    let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();
    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| (centroids_f32[i] + centroids_f32[i + 1]) * 0.5)
        .collect();
    Codebook {
        boundaries,
        centroids: centroids_f32,
    }
}

fn gauss_cond_mean(lo: f64, hi: f64, sigma: f64) -> f64 {
    let a = lo / sigma;
    let b = hi / sigma;
    let z = norm_cdf(b) - norm_cdf(a);
    if z < 1e-15 {
        return (lo + hi) * 0.5;
    }
    sigma * (norm_pdf(a) - norm_pdf(b)) / z
}

fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_is_self_inverse() {
        let mut v = vec![0.5f32, -0.25, 0.125, 0.0625];
        let orig = v.clone();
        fwht_inplace(&mut v);
        fwht_inplace(&mut v);
        for (a, b) in orig.iter().zip(v.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn rotation_signs_are_const_tables() {
        assert_eq!(rotation_signs(4), rotation_signs_array::<4>().as_slice());
        assert_eq!(
            rotation_signs(256),
            rotation_signs_array::<256>().as_slice()
        );
        assert_eq!(
            rotation_signs(512),
            rotation_signs_array::<512>().as_slice()
        );
        assert!(rotation_signs(256).iter().all(|&s| s == 1.0 || s == -1.0));
        assert_ne!(rotation_signs(256), &rotation_signs(512)[..256]);
    }

    #[test]
    fn pack_unpack_preserves_codes() {
        let codes: Vec<u8> = (0..16).map(|i| i as u8 % 16).collect();
        let mut packed = vec![0u8; 8];
        pack_nibbles(&codes, &mut packed);
        let mut got = Vec::new();
        for &b in &packed {
            got.push(b & 0x0f);
            got.push(b >> 4);
        }
        assert_eq!(got, codes);
    }

    #[test]
    fn encode_decode_keeps_cosine_on_unit_256() {
        let dim = 256usize;
        let n = 32usize;
        let mut flat = vec![0.0f32; n * dim];
        for row in 0..n {
            let mut nsq = 0.0f32;
            for d in 0..dim {
                let x = ((row * 131 + d * 17) % 97) as f32 / 97.0 - 0.5;
                flat[row * dim + d] = x;
                nsq += x * x;
            }
            let inv = 1.0 / nsq.sqrt();
            for d in 0..dim {
                flat[row * dim + d] *= inv;
            }
        }
        let enc = encode_flat(n, dim, &flat);
        let rec = reconstruct_flat(
            &enc.packed,
            n,
            dim,
            enc.padded_dim,
            &codebook_for_dim(enc.padded_dim),
            &enc.tqplus,
            &enc.renorm,
        );
        let mut min_cos = 1.0f32;
        for row in 0..n {
            let a = &flat[row * dim..row * dim + dim];
            let b = &rec[row * dim..row * dim + dim];
            let mut dot = 0.0f32;
            let mut nb = 0.0f32;
            for i in 0..dim {
                dot += a[i] * b[i];
                nb += b[i] * b[i];
            }
            let cos = dot / nb.sqrt();
            min_cos = min_cos.min(cos);
        }
        assert!(
            min_cos > 0.85,
            "4-bit TQ reconstruction cosine {min_cos} too low"
        );
    }

    #[test]
    fn packed_search_ranks_identical_vector_first() {
        let dim = 32usize;
        let n = 8usize;
        let mut flat = vec![0.0f32; n * dim];
        for row in 0..n {
            flat[row * dim + row] = 1.0;
        }
        let enc = encode_flat(n, dim, &flat);
        let cb = codebook_for_dim(enc.padded_dim);
        let query = &flat[3 * dim..4 * dim];
        let prep = prepare_query(query, dim, &enc.tqplus, &cb);
        let stride = packed_bytes(enc.padded_dim);
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for row in 0..n {
            let d = dist_dot_packed(
                &prep,
                &enc.packed[row * stride..row * stride + stride],
                enc.renorm[row],
            );
            if d < best_d {
                best_d = d;
                best = row;
            }
        }
        assert_eq!(best, 3, "self should be nearest, dist={best_d}");
    }

    #[test]
    fn lut_ip_matches_naive_centroid_dot() {
        let dim = 32usize;
        let n = 4usize;
        let mut flat = vec![0.0f32; n * dim];
        for row in 0..n {
            flat[row * dim + row] = 1.0;
        }
        let enc = encode_flat(n, dim, &flat);
        let cb = codebook_for_dim(enc.padded_dim);
        let query = &flat[..dim];
        let prep = prepare_query(query, dim, &enc.tqplus, &cb);
        let stride = packed_bytes(enc.padded_dim);
        let padded = enc.padded_dim;
        let signs = rotation_signs(padded);
        let mut rot = vec![0.0f32; padded];
        let mut norm = 0.0f32;
        rotate_row(query, dim, padded, signs, &mut rot, &mut norm);
        for row in 0..n {
            let packed = &enc.packed[row * stride..row * stride + stride];
            let mut inner = 0.0f32;
            for (byte_i, &byte) in packed.iter().enumerate() {
                let i0 = byte_i * 2;
                let s0 = enc.tqplus.scale[i0];
                let q0 = if s0 > 0.0 { rot[i0] / s0 } else { 0.0 };
                inner += q0 * cb.centroids[(byte & 0x0f) as usize];
                inner -= rot[i0] * enc.tqplus.shift[i0];
                if i0 + 1 < padded {
                    let s1 = enc.tqplus.scale[i0 + 1];
                    let q1 = if s1 > 0.0 { rot[i0 + 1] / s1 } else { 0.0 };
                    inner += q1 * cb.centroids[(byte >> 4) as usize];
                    inner -= rot[i0 + 1] * enc.tqplus.shift[i0 + 1];
                }
            }
            let naive = (1.0f32 - enc.renorm[row] * inner).max(0.0);
            let got = dist_dot_packed(&prep, packed, enc.renorm[row]);
            assert!(
                (naive - got).abs() < 1e-5,
                "row {row}: naive={naive} lut={got}"
            );
        }
    }

    #[test]
    fn fastscan_batch_matches_scalar_rows() {
        let dim = 32usize;
        let n = 16usize;
        let mut flat = vec![0.0f32; n * dim];
        for row in 0..n {
            flat[row * dim + (row % dim)] = 1.0;
        }
        let enc = encode_flat(n, dim, &flat);
        let cb = codebook_for_dim(enc.padded_dim);
        let query = &flat[..dim];
        let prep = prepare_query(query, dim, &enc.tqplus, &cb);
        let stride = packed_bytes(enc.padded_dim);
        let ids: Vec<u32> = (0..n as u32).collect();
        let mut batch = vec![0.0f32; n];
        dist_dot_packed_batch(&prep, &enc.packed, stride, &ids, &enc.renorm, &mut batch);
        for row in 0..n {
            let scalar = dist_dot_packed(
                &prep,
                &enc.packed[row * stride..row * stride + stride],
                enc.renorm[row],
            );
            assert!(
                (scalar - batch[row]).abs() < 1e-5,
                "row {row}: scalar={scalar} batch={}",
                batch[row]
            );
        }
    }
}

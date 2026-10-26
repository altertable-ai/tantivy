//! On-disk layout for the `.vec` segment file.
//!
//! **V4 format** (magic `TNVYVEC4`): 4-bit TurboQuant (TQ+ + length-renorm)
//! packed codes + mmap-resident compact HNSW graph. Older formats are not
//! supported.

use std::io::Write;
use std::ops::Range;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use crate::directory::{FileSlice, OwnedBytes};
use crate::schema::{VectorDistance, VectorOptions};
use crate::vector::turboquant::{
    codebook_for_dim, encode_flat, packed_bytes, reconstruct_flat, reconstruct_row, Codebook,
    TqPlus,
};
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC4";
const VERSION: u32 = 4;

// ---------------------------------------------------------------------------
// Compact HNSW graph — topology stays in the mmap'd blob; no adj copy on open.
// ---------------------------------------------------------------------------

/// Neighbor-id iterator over little-endian u32s in the mmap'd adjacency blob.
pub(crate) struct NeighborIter<'a> {
    bytes: &'a [u8],
    remaining: usize,
}

impl Iterator for NeighborIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }
        if self.bytes.len() < 4 {
            self.remaining = 0;
            return None;
        }
        let id = u32::from_le_bytes(self.bytes[..4].try_into().unwrap());
        self.bytes = &self.bytes[4..];
        self.remaining -= 1;
        Some(id)
    }
}

/// HNSW topology. `raw` is either a slice of the `.vec` mmap or an owned
/// packing of a just-built graph — `neighbors()` never copies the edge list.
pub(crate) struct CompactHnswGraph {
    pub entry_point: u32,
    pub entry_layer: u8,
    pub num_points: u32,
    raw: OwnedBytes,
    layer_counts_off: usize,
    adj_offsets_off: usize,
    adj_data_off: usize,
}

impl CompactHnswGraph {
    pub(crate) fn new(
        entry_point: u32,
        entry_layer: u8,
        num_points: u32,
        adjacency: Vec<Vec<Vec<u32>>>,
    ) -> Self {
        let packed = pack_graph(entry_point, entry_layer, num_points, &adjacency);
        Self::from_bytes(OwnedBytes::new(packed)).expect("freshly packed graph")
    }

    pub(crate) fn from_bytes(raw: OwnedBytes) -> crate::Result<Self> {
        let buf = raw.as_slice();
        let mut pos = 0usize;
        let entry_point = read_u32_le(buf, &mut pos)?;
        if buf.len() <= pos {
            return Err(corruption("compact graph: truncated entry_layer"));
        }
        let entry_layer = buf[pos];
        pos += 1;
        let num_points = read_u32_le(buf, &mut pos)? as usize;
        check_len(buf, pos, num_points, "layer counts")?;
        let layer_counts_off = pos;
        pos += num_points;
        pos = align4(pos);
        let offsets_bytes = num_points.saturating_mul(4);
        check_len(buf, pos, offsets_bytes, "adj offsets")?;
        let adj_offsets_off = pos;
        pos += offsets_bytes;
        let adj_data_off = pos;
        Ok(Self {
            entry_point,
            entry_layer,
            num_points: num_points as u32,
            raw,
            layer_counts_off,
            adj_offsets_off,
            adj_data_off,
        })
    }

    fn layer_counts(&self) -> &[u8] {
        let n = self.num_points as usize;
        &self.raw.as_slice()[self.layer_counts_off..self.layer_counts_off + n]
    }

    fn adj_offset(&self, point: usize) -> Option<u32> {
        let n = self.num_points as usize;
        if point >= n {
            return None;
        }
        let off = self.adj_offsets_off + point * 4;
        let buf = self.raw.as_slice();
        if off + 4 > buf.len() {
            return None;
        }
        Some(LittleEndian::read_u32(&buf[off..off + 4]))
    }

    #[inline]
    pub(crate) fn neighbors(&self, point: u32, layer: usize) -> NeighborIter<'_> {
        let empty = NeighborIter {
            bytes: &[],
            remaining: 0,
        };
        let idx = point as usize;
        let lc = match self.layer_counts().get(idx) {
            Some(&lc) => lc as usize,
            None => return empty,
        };
        if layer >= lc {
            return empty;
        }
        let Some(rel) = self.adj_offset(idx) else {
            return empty;
        };
        let buf = self.raw.as_slice();
        let mut off = self.adj_data_off + rel as usize;
        for _ in 0..layer {
            if off + 4 > buf.len() {
                return empty;
            }
            let count = LittleEndian::read_u32(&buf[off..off + 4]) as usize;
            off += 4 + count * 4;
        }
        if off + 4 > buf.len() {
            return empty;
        }
        let count = LittleEndian::read_u32(&buf[off..off + 4]) as usize;
        let ids_off = off + 4;
        let ids_bytes = count * 4;
        if ids_off + ids_bytes > buf.len() {
            return empty;
        }
        NeighborIter {
            bytes: &buf[ids_off..ids_off + ids_bytes],
            remaining: count,
        }
    }

    pub(crate) fn serialize(&self) -> Vec<u8> {
        self.raw.as_slice().to_vec()
    }
}

fn pack_graph(
    entry_point: u32,
    entry_layer: u8,
    num_points: u32,
    adjacency: &[Vec<Vec<u32>>],
) -> Vec<u8> {
    let n = num_points as usize;
    let mut layer_counts = vec![0u8; n];
    for (i, layers) in adjacency.iter().enumerate().take(n) {
        layer_counts[i] = layers.len() as u8;
    }

    let mut adj_data = Vec::new();
    let mut adj_offsets = vec![0u32; n];
    for i in 0..n {
        adj_offsets[i] = adj_data.len() as u32;
        let layers = adjacency.get(i).map(|l| l.as_slice()).unwrap_or(&[]);
        for neighbors in layers {
            adj_data.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
            for &id in neighbors {
                adj_data.extend_from_slice(&id.to_le_bytes());
            }
        }
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(&entry_point.to_le_bytes());
    buf.push(entry_layer);
    buf.extend_from_slice(&num_points.to_le_bytes());
    buf.extend_from_slice(&layer_counts);
    while buf.len() % 4 != 0 {
        buf.push(0);
    }
    for off in adj_offsets {
        buf.extend_from_slice(&off.to_le_bytes());
    }
    buf.extend_from_slice(&adj_data);
    buf
}

fn align4(pos: usize) -> usize {
    pos.div_ceil(4) * 4
}

// ---------------------------------------------------------------------------
// L2 normalization (cosine → dot-product conversion)
// ---------------------------------------------------------------------------

/// In-place L2-normalize a single vector. Zero-norm vectors are left untouched.
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
    pub flat: Vec<f32>,
    pub graph: CompactHnswGraph,
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

        let num_docs = bundle.num_docs as usize;
        let dim = bundle.options.dimension;
        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(dim as u32)?;

        let encoded = encode_flat(num_docs, dim, &bundle.flat);
        writer.write_u32::<LittleEndian>(encoded.padded_dim as u32)?;
        for &s in &encoded.tqplus.shift {
            writer.write_f32::<LittleEndian>(s)?;
        }
        for &s in &encoded.tqplus.scale {
            writer.write_f32::<LittleEndian>(s)?;
        }
        writer.write_all(&encoded.packed)?;
        for &r in &encoded.renorm {
            writer.write_f32::<LittleEndian>(r)?;
        }

        let graph_raw = bundle.graph.serialize();
        writer.write_u64::<LittleEndian>(graph_raw.len() as u64)?;
        writer.write_all(&graph_raw)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lazy read — mmap-friendly: codes, TQ+, renorm, graph stay as byte ranges.
// ---------------------------------------------------------------------------

pub(crate) struct LazyVectorField {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub dimension: usize,
    pub padded_dim: usize,
    raw: OwnedBytes,
    shift_range: Range<usize>,
    scale_range: Range<usize>,
    codes_range: Range<usize>,
    renorm_range: Range<usize>,
    graph_range: Range<usize>,
}

impl LazyVectorField {
    pub(crate) fn packed_codes(&self) -> &[u8] {
        &self.raw[self.codes_range.clone()]
    }

    pub(crate) fn packed_stride(&self) -> usize {
        packed_bytes(self.padded_dim)
    }

    pub(crate) fn parse_tqplus(&self) -> crate::Result<TqPlus> {
        let padded = self.padded_dim;
        Ok(TqPlus {
            shift: f32_slice_from_bytes(&self.raw[self.shift_range.clone()], padded)?,
            scale: f32_slice_from_bytes(&self.raw[self.scale_range.clone()], padded)?,
        })
    }

    pub(crate) fn parse_renorm(&self) -> crate::Result<Vec<f32>> {
        f32_slice_from_bytes(&self.raw[self.renorm_range.clone()], self.num_docs as usize)
    }

    pub(crate) fn parse_graph(&self) -> crate::Result<CompactHnswGraph> {
        CompactHnswGraph::from_bytes(self.raw.slice(self.graph_range.clone()))
    }

    pub(crate) fn codebook(&self) -> Codebook {
        codebook_for_dim(self.padded_dim)
    }

    pub(crate) fn dequantize_flat(&self) -> crate::Result<Vec<f32>> {
        let tqplus = self.parse_tqplus()?;
        let renorm = self.parse_renorm()?;
        Ok(reconstruct_flat(
            self.packed_codes(),
            self.num_docs as usize,
            self.dimension,
            self.padded_dim,
            &self.codebook(),
            &tqplus,
            &renorm,
        ))
    }

    pub(crate) fn reconstruct_doc(
        &self,
        doc: u32,
        tqplus: &TqPlus,
        codebook: &Codebook,
        scale: f32,
    ) -> Vec<f32> {
        let stride = self.packed_stride();
        let start = doc as usize * stride;
        reconstruct_row(
            &self.packed_codes()[start..start + stride],
            self.dimension,
            self.padded_dim,
            codebook,
            tqplus,
            scale,
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
            "unsupported .vec format (expected TNVYVEC4, got {:?})",
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
        let padded_dim = read_u32_le(buf, &mut pos)? as usize;

        let shift_len = padded_dim * 4;
        check_len(buf, pos, shift_len, "tqplus shift")?;
        let shift_range = pos..pos + shift_len;
        pos += shift_len;

        let scale_len = padded_dim * 4;
        check_len(buf, pos, scale_len, "tqplus scale")?;
        let scale_range = pos..pos + scale_len;
        pos += scale_len;

        let codes_len = num_docs * packed_bytes(padded_dim);
        check_len(buf, pos, codes_len, "packed codes")?;
        let codes_range = pos..pos + codes_len;
        pos += codes_len;

        let renorm_len = num_docs * 4;
        check_len(buf, pos, renorm_len, "renorm")?;
        let renorm_range = pos..pos + renorm_len;
        pos += renorm_len;

        let graph_len = read_u64_le(buf, &mut pos)? as usize;
        check_len(buf, pos, graph_len, "graph")?;
        let graph_range = pos..pos + graph_len;
        pos += graph_len;

        out.push(LazyVectorField {
            field_id,
            options,
            num_docs: num_docs as u32,
            dimension,
            padded_dim,
            raw: bytes.clone(),
            shift_range,
            scale_range,
            codes_range,
            renorm_range,
            graph_range,
        });
    }
    Ok(out)
}

fn f32_slice_from_bytes(bytes: &[u8], n: usize) -> crate::Result<Vec<f32>> {
    if bytes.len() != n * 4 {
        return Err(corruption("f32 slice length mismatch"));
    }
    let mut v = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        v.push(LittleEndian::read_f32(chunk));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Portable SIMD (feature = "vector-simd", requires nightly)
// ---------------------------------------------------------------------------

#[cfg(feature = "vector-simd")]
#[allow(dead_code)]
mod simd_impl {
    use std::simd::prelude::*;

    const LANES: usize = 8;
    type F32x = Simd<f32, LANES>;

    #[inline]
    pub(super) fn dist_l2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dim = a.len();
        let full = dim / LANES;
        let mut acc = F32x::splat(0.0);
        for c in 0..full {
            let i = c * LANES;
            let d = F32x::from_slice(&a[i..]) - F32x::from_slice(&b[i..]);
            acc += d * d;
        }
        let mut sum = acc.reduce_sum();
        for i in (full * LANES)..dim {
            let d = a[i] - b[i];
            sum += d * d;
        }
        sum.sqrt()
    }

    #[inline]
    pub(super) fn dist_dot(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let dim = a.len();
        let full = dim / LANES;
        let mut acc = F32x::splat(0.0);
        for c in 0..full {
            let i = c * LANES;
            acc += F32x::from_slice(&a[i..]) * F32x::from_slice(&b[i..]);
        }
        let mut dot = acc.reduce_sum();
        for i in (full * LANES)..dim {
            dot += a[i] * b[i];
        }
        (1.0f32 - dot).max(0.0f32)
    }

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
// Distance helpers
// ---------------------------------------------------------------------------

pub(crate) fn distance_to_score(distance: f32) -> crate::Score {
    1.0f32 / (1.0f32 + distance)
}

#[allow(dead_code)]
pub(crate) fn distance_fn_for(dist: VectorDistance) -> fn(&[f32], &[f32]) -> f32 {
    match dist {
        #[cfg(feature = "vector-simd")]
        VectorDistance::Euclidean => simd_impl::dist_l2,
        #[cfg(not(feature = "vector-simd"))]
        VectorDistance::Euclidean => dist_l2,

        #[cfg(feature = "vector-simd")]
        VectorDistance::Cosine | VectorDistance::DotProduct => simd_impl::dist_dot,
        #[cfg(not(feature = "vector-simd"))]
        VectorDistance::Cosine | VectorDistance::DotProduct => dist_dot,
    }
}

#[cfg(not(feature = "vector-simd"))]
#[inline]
fn dist_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}

#[cfg(not(feature = "vector-simd"))]
#[inline]
fn dist_dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
    }
    (1.0f32 - dot).max(0.0f32)
}

fn read_u32_le(buf: &[u8], pos: &mut usize) -> crate::Result<u32> {
    check_len(buf, *pos, 4, "u32")?;
    let v = LittleEndian::read_u32(&buf[*pos..*pos + 4]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(buf: &[u8], pos: &mut usize) -> crate::Result<u64> {
    check_len(buf, *pos, 8, "u64")?;
    let v = LittleEndian::read_u64(&buf[*pos..*pos + 8]);
    *pos += 8;
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

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn mmap_graph_neighbors_roundtrip() {
        let adj = vec![
            vec![vec![1, 2], vec![2]],
            vec![vec![0, 2]],
            vec![vec![0, 1], vec![0]],
        ];
        let g = CompactHnswGraph::new(0, 1, 3, adj);
        let n0: Vec<u32> = g.neighbors(0, 0).collect();
        assert_eq!(n0, vec![1, 2]);
        let n0l1: Vec<u32> = g.neighbors(0, 1).collect();
        assert_eq!(n0l1, vec![2]);
        let n1: Vec<u32> = g.neighbors(1, 0).collect();
        assert_eq!(n1, vec![0, 2]);
        let bytes = g.serialize();
        let g2 = CompactHnswGraph::from_bytes(OwnedBytes::new(bytes)).unwrap();
        let n0b: Vec<u32> = g2.neighbors(0, 0).collect();
        assert_eq!(n0b, vec![1, 2]);
        assert!(g2.neighbors(0, 9).next().is_none());
    }
}

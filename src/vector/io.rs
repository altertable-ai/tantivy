//! On-disk layout for the `.vec` segment file.
//!
//! **V3 format** (magic `TNVYVEC3`): per-dimension scalar quantization (SQ8) + mmap-friendly
//! flat `u8` vectors + uncompressed compact HNSW graph. No zstd. Older formats are not
//! supported.

use std::io::Write;
use std::ops::Range;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use crate::directory::{FileSlice, OwnedBytes};
use crate::schema::{VectorDistance, VectorOptions};
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC3";
const VERSION: u32 = 3;

// ---------------------------------------------------------------------------
// Compact HNSW graph
// ---------------------------------------------------------------------------

/// In-memory HNSW graph for search.  Stores only topology (neighbor IDs), not
/// vectors or distances — those are recomputed from the flat store at query time.
pub(crate) struct CompactHnswGraph {
    pub entry_point: u32,
    pub entry_layer: u8,
    pub num_points: u32,
    layer_counts: Vec<u8>,
    adj_offsets: Vec<u32>,
    adj_data: Vec<u32>,
}

impl CompactHnswGraph {
    pub(crate) fn new(
        entry_point: u32,
        entry_layer: u8,
        num_points: u32,
        adjacency: Vec<Vec<Vec<u32>>>,
    ) -> Self {
        let n = adjacency.len();
        let mut layer_counts = Vec::with_capacity(n);
        let mut adj_offsets = Vec::with_capacity(n);
        let total: usize = adjacency
            .iter()
            .map(|layers| layers.iter().map(|nb| 1 + nb.len()).sum::<usize>())
            .sum();
        let mut adj_data = Vec::with_capacity(total);

        for layers in &adjacency {
            layer_counts.push(layers.len() as u8);
            adj_offsets.push(adj_data.len() as u32);
            for neighbors in layers {
                adj_data.push(neighbors.len() as u32);
                adj_data.extend_from_slice(neighbors);
            }
        }

        Self {
            entry_point,
            entry_layer,
            num_points,
            layer_counts,
            adj_offsets,
            adj_data,
        }
    }

    #[inline]
    pub(crate) fn neighbors(&self, point: u32, layer: usize) -> &[u32] {
        let idx = point as usize;
        let lc = match self.layer_counts.get(idx) {
            Some(&lc) => lc as usize,
            None => return &[],
        };
        if layer >= lc {
            return &[];
        }
        let mut off = self.adj_offsets[idx] as usize;
        for _ in 0..layer {
            let count = self.adj_data[off] as usize;
            off += 1 + count;
        }
        let count = self.adj_data[off] as usize;
        &self.adj_data[off + 1..off + 1 + count]
    }

    pub(crate) fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.entry_point.to_le_bytes());
        buf.push(self.entry_layer);
        buf.extend_from_slice(&self.num_points.to_le_bytes());

        buf.extend_from_slice(&self.layer_counts);

        let n = self.num_points as usize;
        for i in 0..n {
            let lc = self.layer_counts[i] as usize;
            let mut off = self.adj_offsets[i] as usize;
            for _ in 0..lc {
                let count = self.adj_data[off] as usize;
                let count16 = count.min(u16::MAX as usize) as u16;
                buf.extend_from_slice(&count16.to_le_bytes());
                for &id in &self.adj_data[off + 1..off + 1 + count16 as usize] {
                    buf.extend_from_slice(&id.to_le_bytes());
                }
                off += 1 + count;
            }
        }
        buf
    }

    pub(crate) fn deserialize(buf: &[u8]) -> crate::Result<Self> {
        let mut pos = 0usize;
        let entry_point = read_u32_le(buf, &mut pos)?;
        if buf.len() <= pos {
            return Err(corruption("compact graph: truncated entry_layer"));
        }
        let entry_layer = buf[pos];
        pos += 1;
        let num_points = read_u32_le(buf, &mut pos)? as usize;

        if buf.len() < pos + num_points {
            return Err(corruption("compact graph: truncated layer counts"));
        }
        let layer_counts: Vec<u8> = buf[pos..pos + num_points].to_vec();
        pos += num_points;

        let mut adj_offsets = Vec::with_capacity(num_points);
        let mut adj_data = Vec::new();

        for &lc in &layer_counts {
            adj_offsets.push(adj_data.len() as u32);
            for _ in 0..lc {
                let count = read_u16_le(buf, &mut pos)? as usize;
                let needed = count * 4;
                if buf.len() < pos + needed {
                    return Err(corruption("compact graph: truncated neighbor list"));
                }
                adj_data.push(count as u32);
                for _ in 0..count {
                    adj_data.push(read_u32_le(buf, &mut pos)?);
                }
            }
        }

        Ok(Self {
            entry_point,
            entry_layer,
            num_points: num_points as u32,
            layer_counts,
            adj_offsets,
            adj_data,
        })
    }
}

// ---------------------------------------------------------------------------
// SQ8 quantization (shared by writer, reader, merger)
// ---------------------------------------------------------------------------

/// Per-dimension min/max over all rows in `flat` (row-major, `num_docs * dim` floats).
pub(crate) fn compute_sq8_params(
    num_docs: usize,
    dim: usize,
    flat: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(flat.len(), num_docs * dim);
    let mut mins = vec![f32::INFINITY; dim];
    let mut maxs = vec![f32::NEG_INFINITY; dim];
    for row in 0..num_docs {
        let base = row * dim;
        for d in 0..dim {
            let v = flat[base + d];
            mins[d] = mins[d].min(v);
            maxs[d] = maxs[d].max(v);
        }
    }
    let scales: Vec<f32> = mins
        .iter()
        .zip(maxs.iter())
        .map(
            |(&min, &max)| {
                if max > min {
                    (max - min) / 255.0
                } else {
                    0.0
                }
            },
        )
        .collect();
    (mins, scales)
}

/// Quantize row-major `f32` flat to `u8` using `mins` / `scales` from [`compute_sq8_params`].
pub(crate) fn quantize_flat_to_u8(
    flat: &[f32],
    dim: usize,
    mins: &[f32],
    scales: &[f32],
) -> Vec<u8> {
    assert_eq!(mins.len(), dim);
    assert_eq!(scales.len(), dim);
    let n = flat.len() / dim;
    let mut out = vec![0u8; flat.len()];
    for row in 0..n {
        let base = row * dim;
        for d in 0..dim {
            let v = flat[base + d];
            let min = mins[d];
            let scale = scales[d];
            out[base + d] = if scale > 0.0 {
                ((v - min) / scale).round().clamp(0.0, 255.0) as u8
            } else {
                0
            };
        }
    }
    out
}

/// Dequantize one row (`dim` bytes) into `out` (length `dim`).
pub(crate) fn dequantize_row_into(row: &[u8], mins: &[f32], scales: &[f32], out: &mut [f32]) {
    let dim = row.len();
    debug_assert_eq!(mins.len(), dim);
    debug_assert_eq!(scales.len(), dim);
    debug_assert_eq!(out.len(), dim);
    for d in 0..dim {
        out[d] = mins[d] + row[d] as f32 * scales[d];
    }
}

/// Dequantize full flat `u8` to `f32` row-major.
pub(crate) fn dequantize_flat_u8_to_f32(
    flat_u8: &[u8],
    num_docs: usize,
    dim: usize,
    mins: &[f32],
    scales: &[f32],
) -> crate::Result<Vec<f32>> {
    if flat_u8.len() != num_docs * dim {
        return Err(corruption(format!(
            "quantized flat size mismatch: {} bytes for {} docs x {}",
            flat_u8.len(),
            num_docs,
            dim
        )));
    }
    let mut out = vec![0f32; num_docs * dim];
    for row in 0..num_docs {
        let base = row * dim;
        dequantize_row_into(
            &flat_u8[base..base + dim],
            mins,
            scales,
            &mut out[base..base + dim],
        );
    }
    Ok(out)
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

        let (mins, scales) = compute_sq8_params(num_docs, dim, &bundle.flat);
        for &m in &mins {
            writer.write_f32::<LittleEndian>(m)?;
        }
        for &s in &scales {
            writer.write_f32::<LittleEndian>(s)?;
        }

        let quantized = quantize_flat_to_u8(&bundle.flat, dim, &mins, &scales);
        writer.write_all(&quantized)?;

        let graph_raw = bundle.graph.serialize();
        writer.write_u64::<LittleEndian>(graph_raw.len() as u64)?;
        writer.write_all(&graph_raw)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lazy read — mmap-friendly: only metadata ranges; vectors stay as bytes in `OwnedBytes`.
// ---------------------------------------------------------------------------

pub(crate) struct LazyVectorField {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub dimension: usize,
    raw: OwnedBytes,
    mins_range: Range<usize>,
    scales_range: Range<usize>,
    vectors_range: Range<usize>,
    graph_range: Range<usize>,
}

impl LazyVectorField {
    pub(crate) fn mins_bytes(&self) -> &[u8] {
        &self.raw[self.mins_range.clone()]
    }

    pub(crate) fn scales_bytes(&self) -> &[u8] {
        &self.raw[self.scales_range.clone()]
    }

    pub(crate) fn quantized_vectors(&self) -> &[u8] {
        &self.raw[self.vectors_range.clone()]
    }

    pub(crate) fn parse_mins_scales(&self) -> crate::Result<(Vec<f32>, Vec<f32>)> {
        let dim = self.dimension;
        let mins = f32_slice_from_bytes(self.mins_bytes(), dim)?;
        let scales = f32_slice_from_bytes(self.scales_bytes(), dim)?;
        Ok((mins, scales))
    }

    pub(crate) fn parse_graph(&self) -> crate::Result<CompactHnswGraph> {
        CompactHnswGraph::deserialize(&self.raw[self.graph_range.clone()])
    }

    /// Full dequantized flat (merge / `flat_vectors` API).
    pub(crate) fn dequantize_flat(&self) -> crate::Result<Vec<f32>> {
        let (mins, scales) = self.parse_mins_scales()?;
        dequantize_flat_u8_to_f32(
            self.quantized_vectors(),
            self.num_docs as usize,
            self.dimension,
            &mins,
            &scales,
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
            "unsupported .vec format (expected TNVYVEC3, got {:?})",
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

        let mins_len = dimension * 4;
        check_len(buf, pos, mins_len, "mins")?;
        let mins_range = pos..pos + mins_len;
        pos += mins_len;

        let scales_len = dimension * 4;
        check_len(buf, pos, scales_len, "scales")?;
        let scales_range = pos..pos + scales_len;
        pos += scales_len;

        let vec_len = num_docs * dimension;
        check_len(buf, pos, vec_len, "vectors")?;
        let vectors_range = pos..pos + vec_len;
        pos += vec_len;

        let graph_len = read_u64_le(buf, &mut pos)? as usize;
        check_len(buf, pos, graph_len, "graph")?;
        let graph_range = pos..pos + graph_len;
        pos += graph_len;

        out.push(LazyVectorField {
            field_id,
            options,
            num_docs: num_docs as u32,
            dimension,
            raw: bytes.clone(),
            mins_range,
            scales_range,
            vectors_range,
            graph_range,
        });
    }
    Ok(out)
}

fn f32_slice_from_bytes(bytes: &[u8], dim: usize) -> crate::Result<Vec<f32>> {
    if bytes.len() != dim * 4 {
        return Err(corruption("mins/scales length mismatch"));
    }
    let mut v = Vec::with_capacity(dim);
    for chunk in bytes.chunks_exact(4) {
        v.push(LittleEndian::read_f32(chunk));
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// Distance helpers (used by the search module)
// ---------------------------------------------------------------------------

pub(crate) fn distance_to_score(distance: f32) -> crate::Score {
    1.0 / (1.0 + distance)
}

pub(crate) fn distance_fn_for(dist: VectorDistance) -> fn(&[f32], &[f32]) -> f32 {
    match dist {
        VectorDistance::Euclidean => dist_l2,
        VectorDistance::Cosine => dist_cosine,
        VectorDistance::DotProduct => dist_dot,
    }
}

fn dist_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn dist_cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let (xd, yd) = (x as f64, y as f64);
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
    }
    if na > 0.0 && nb > 0.0 {
        (1.0 - dot / (na * nb).sqrt()).max(0.0) as f32
    } else {
        0.0
    }
}

fn dist_dot(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    (1.0 - dot).max(0.0)
}

fn read_u16_le(buf: &[u8], pos: &mut usize) -> crate::Result<u16> {
    check_len(buf, *pos, 2, "u16")?;
    let v = LittleEndian::read_u16(&buf[*pos..*pos + 2]);
    *pos += 2;
    Ok(v)
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

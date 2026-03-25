//! On-disk layout for the `.vec` segment file.
//!
//! **V3 format** (magic `TNVYVEC3`): per-dimension scalar-quantized u8 vectors (mmap-friendly)
//! plus uncompressed compact HNSW graph.
//!
//! Layout per field:
//!   field_id (u32 LE), options JSON (len-prefixed), num_docs (u32 LE), dimension (u32 LE),
//!   sq_mins `[f32 LE; dimension]`, sq_scales `[f32 LE; dimension]`,
//!   quantized `[u8; num_docs * dimension]`, graph_len (u64 LE), graph bytes.

use std::io::{self, Write};

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
///
/// The adjacency data lives in three contiguous vectors instead of
/// `Vec<Vec<Vec<u32>>>`, avoiding O(N × layers) small heap allocations:
///
/// * `layer_counts[point]` — how many layers this point participates in.
/// * `adj_offsets[point]`  — index into `adj_data` where this point's packed neighbor lists begin.
/// * `adj_data`            — packed sequences of `[count_u32, id, id, …]` for each layer of each
///   point (layer 0 first, then layer 1, …).
pub(crate) struct CompactHnswGraph {
    pub entry_point: u32,
    pub entry_layer: u8,
    pub num_points: u32,
    layer_counts: Vec<u8>,
    adj_offsets: Vec<u32>,
    adj_data: Vec<u32>,
}

impl CompactHnswGraph {
    /// Build from per-point adjacency lists (convenience for the hnsw_rs extraction
    /// path which naturally produces `Vec<Vec<Vec<u32>>>`).
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

    /// Neighbor IDs for `point` at `layer`.  O(layer) scan over the packed
    /// counts, but layers are almost always 0 (rarely > 1), so effectively O(1).
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

    /// Serialize to a byte buffer.
    ///
    /// Wire format:
    /// ```text
    /// entry_point: u32 LE
    /// entry_layer: u8
    /// num_points:  u32 LE
    /// layer_counts: [u8; num_points]
    /// per point, per layer: num_neighbors u16 LE, [neighbor_id u32 LE; ...]
    /// ```
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

    /// Deserialize from the raw byte buffer.
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
// Scalar quantization (per-dimension u8)
// ---------------------------------------------------------------------------

/// Per-dimension min/max range for scalar quantization to u8.
///
/// `scale[d] = (max[d] - min[d]) / 255.0`, or `0.0` when the range is zero.
pub(crate) struct SqParams {
    pub mins: Vec<f32>,
    pub scales: Vec<f32>,
}

impl SqParams {
    /// Compute min/max per dimension from row-major `flat` (`num_docs * dim` floats).
    pub(crate) fn from_flat(flat: &[f32], dim: usize) -> Self {
        if dim == 0 || flat.is_empty() {
            return Self {
                mins: Vec::new(),
                scales: Vec::new(),
            };
        }
        let n = flat.len() / dim;
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        for row in 0..n {
            let base = row * dim;
            for d in 0..dim {
                let v = flat[base + d];
                mins[d] = mins[d].min(v);
                maxs[d] = maxs[d].max(v);
            }
        }
        let mut scales = vec![0.0f32; dim];
        for d in 0..dim {
            let range = maxs[d] - mins[d];
            scales[d] = if range > 0.0 && range.is_finite() {
                range / 255.0
            } else {
                0.0
            };
        }
        Self { mins, scales }
    }

    /// Dequantize one vector into `buf` (length `dimension`).
    #[inline]
    pub(crate) fn dequantize_into(&self, quantized: &[u8], buf: &mut [f32]) {
        debug_assert_eq!(quantized.len(), buf.len());
        buf.iter_mut()
            .zip(quantized.iter())
            .zip(self.mins.iter().zip(&self.scales))
            .for_each(|((out, &q), (&min, &scale))| {
                *out = min + q as f32 * scale;
            });
    }

    /// Dequantize all rows into `num_docs * dim` floats (for merge / API).
    pub(crate) fn dequantize_all(&self, quantized: &[u8], dim: usize, num_docs: usize) -> Vec<f32> {
        let mut out = vec![0f32; num_docs * dim];
        for row in 0..num_docs {
            let row_q = &quantized[row * dim..(row + 1) * dim];
            let row_f = &mut out[row * dim..(row + 1) * dim];
            self.dequantize_into(row_q, row_f);
        }
        out
    }
}

fn quantize_flat(flat: &[f32], dim: usize, params: &SqParams) -> Vec<u8> {
    let n = flat.len() / dim;
    let mut out = vec![0u8; flat.len()];
    for row in 0..n {
        for d in 0..dim {
            let v = flat[row * dim + d];
            let min = params.mins[d];
            let scale = params.scales[d];
            let idx = if scale == 0.0 {
                0u8
            } else {
                ((v - min) / scale).round().clamp(0.0, 255.0) as u8
            };
            out[row * dim + d] = idx;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// V3 bundle written during indexing / merge
// ---------------------------------------------------------------------------

/// Payload for one vector field going into the `.vec` file.
pub(crate) struct VectorFieldBundle {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub flat: Vec<f32>,
    pub graph: CompactHnswGraph,
}

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

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

        let dim = bundle.options.dimension;
        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(dim as u32)?;

        let sq = SqParams::from_flat(&bundle.flat, dim);
        let expected = bundle.num_docs as usize * dim;
        if bundle.flat.len() != expected {
            return Err(TantivyError::InternalError(format!(
                "vector flat len {} expected {}",
                bundle.flat.len(),
                expected
            )));
        }
        write_f32_slice(writer, &sq.mins)
            .map_err(|e| TantivyError::InternalError(format!("vector write sq_mins: {e}")))?;
        write_f32_slice(writer, &sq.scales)
            .map_err(|e| TantivyError::InternalError(format!("vector write sq_scales: {e}")))?;

        let quantized = quantize_flat(&bundle.flat, dim, &sq);
        writer.write_all(&quantized)?;

        let graph_raw = bundle.graph.serialize();
        writer.write_u64::<LittleEndian>(graph_raw.len() as u64)?;
        writer.write_all(&graph_raw)?;
    }
    Ok(())
}

fn write_f32_slice(writer: &mut dyn Write, slice: &[f32]) -> io::Result<()> {
    for &f in slice {
        writer.write_all(&f.to_le_bytes())?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Read — mmap-friendly slices + graph deserialized once
// ---------------------------------------------------------------------------

/// One vector field parsed from the `.vec` file: quantized bytes in mmap + SQ params + graph.
pub(crate) struct MmapVectorField {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub dimension: usize,
    pub sq: SqParams,
    pub(crate) quantized_bytes: OwnedBytes,
    pub graph: CompactHnswGraph,
}

pub(crate) fn read_vec_file(data: FileSlice) -> crate::Result<Vec<MmapVectorField>> {
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

        let num_docs = read_u32_le(buf, &mut pos)?;
        let dimension = read_u32_le(buf, &mut pos)? as usize;

        let mins = read_f32_slice(buf, &mut pos, dimension)?;
        let scales = read_f32_slice(buf, &mut pos, dimension)?;
        let sq = SqParams { mins, scales };

        let q_len = num_docs as usize * dimension;
        check_len(buf, pos, q_len, "quantized vectors")?;
        let q_start = pos;
        pos += q_len;

        let graph_len = read_u64_le(buf, &mut pos)? as usize;
        check_len(buf, pos, graph_len, "graph")?;
        let graph = CompactHnswGraph::deserialize(&buf[pos..pos + graph_len])?;
        pos += graph_len;

        let quantized_bytes = bytes.slice(q_start..q_start + q_len);

        out.push(MmapVectorField {
            field_id,
            options,
            num_docs,
            dimension,
            sq,
            quantized_bytes,
            graph,
        });
    }
    Ok(out)
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

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------

fn read_f32_slice(buf: &[u8], pos: &mut usize, len: usize) -> crate::Result<Vec<f32>> {
    let need = len * 4;
    check_len(buf, *pos, need, "f32 slice")?;
    let mut v = Vec::with_capacity(len);
    for _ in 0..len {
        v.push(LittleEndian::read_f32(&buf[*pos..*pos + 4]));
        *pos += 4;
    }
    Ok(v)
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

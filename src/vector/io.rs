//! On-disk layout for the `.vec` segment file.
//!
//! **V2 format** (magic `TNVYVEC2`): compact graph + zstd-compressed flat vectors.
//!
//! Layout per field:
//!   field_id (u32 LE), options JSON (len-prefixed), num_docs (u32 LE), dimension (u32 LE),
//!   zstd-compressed flat vectors (len-prefixed u64 LE blob),
//!   zstd-compressed compact HNSW graph (len-prefixed u64 LE blob).
//!
//! Reading is **lazy**: [`read_vec_file_lazy`] parses the header/offsets but does
//! *not* decompress the blobs.  Decompression is deferred to query time via
//! [`LazyVectorField::decompress_flat`] / [`LazyVectorField::decompress_graph`].

use std::io::Write;
use std::ops::Range;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use crate::directory::{FileSlice, OwnedBytes};
use crate::schema::{VectorDistance, VectorOptions};
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC2";
const VERSION: u32 = 2;

const ZSTD_COMPRESSION_LEVEL: i32 = 3;

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

    /// Serialize to a byte buffer (before zstd compression).
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

    /// Deserialize from the raw (decompressed) byte buffer.
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
// V2 bundle written during indexing / merge
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

        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(bundle.options.dimension as u32)?;

        // --- flat vectors: zstd-compressed ---
        let flat_bytes = flat_to_le_bytes(&bundle.flat);
        let flat_compressed = zstd_compress(&flat_bytes)?;
        writer.write_u64::<LittleEndian>(flat_compressed.len() as u64)?;
        writer.write_all(&flat_compressed)?;

        // --- compact graph: zstd-compressed ---
        let graph_raw = bundle.graph.serialize();
        let graph_compressed = zstd_compress(&graph_raw)?;
        writer.write_u64::<LittleEndian>(graph_compressed.len() as u64)?;
        writer.write_all(&graph_compressed)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Lazy read — parse metadata/offsets only, defer decompression to query time
// ---------------------------------------------------------------------------

/// One vector field parsed from the `.vec` file but **not yet decompressed**.
/// The compressed flat-vector and graph blobs are retained as byte-range
/// references into the underlying (typically mmap'd) `OwnedBytes`.
pub(crate) struct LazyVectorField {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub dimension: usize,
    raw: OwnedBytes,
    flat_range: Range<usize>,
    graph_range: Range<usize>,
}

impl LazyVectorField {
    pub(crate) fn decompress_flat(&self) -> crate::Result<Vec<f32>> {
        let compressed = &self.raw[self.flat_range.clone()];
        let decompressed = zstd_decompress(compressed)?;
        le_bytes_to_flat(&decompressed, self.num_docs as usize * self.dimension)
    }

    pub(crate) fn decompress_graph(&self) -> crate::Result<CompactHnswGraph> {
        let compressed = &self.raw[self.graph_range.clone()];
        let decompressed = zstd_decompress(compressed)?;
        CompactHnswGraph::deserialize(&decompressed)
    }
}

pub(crate) fn read_vec_file_lazy(data: FileSlice) -> crate::Result<Vec<LazyVectorField>> {
    let bytes = data.read_bytes()?;
    let buf = bytes.as_slice();
    let mut pos = 0usize;

    check_len(buf, 0, 8, "magic")?;
    if &buf[..8] != MAGIC {
        return Err(corruption(format!(
            "unsupported .vec format (expected TNVYVEC2, got {:?})",
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

        let flat_clen = read_u64_le(buf, &mut pos)? as usize;
        check_len(buf, pos, flat_clen, "flat compressed")?;
        let flat_range = pos..pos + flat_clen;
        pos += flat_clen;

        let graph_clen = read_u64_le(buf, &mut pos)? as usize;
        check_len(buf, pos, graph_clen, "graph compressed")?;
        let graph_range = pos..pos + graph_clen;
        pos += graph_clen;

        out.push(LazyVectorField {
            field_id,
            options,
            num_docs,
            dimension,
            raw: bytes.clone(),
            flat_range,
            graph_range,
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
// Compression helpers
// ---------------------------------------------------------------------------

fn zstd_compress(data: &[u8]) -> crate::Result<Vec<u8>> {
    zstd::bulk::compress(data, ZSTD_COMPRESSION_LEVEL)
        .map_err(|e| TantivyError::InternalError(format!("zstd compress: {e}")))
}

fn zstd_decompress(data: &[u8]) -> crate::Result<Vec<u8>> {
    // Allow up to 2 GiB decompressed; real payloads are much smaller.
    zstd::bulk::decompress(data, 2 << 30)
        .map_err(|e| TantivyError::InternalError(format!("zstd decompress: {e}")))
}

// ---------------------------------------------------------------------------
// Byte-level helpers
// ---------------------------------------------------------------------------

fn flat_to_le_bytes(flat: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; flat.len() * 4];
    for (i, &f) in flat.iter().enumerate() {
        LittleEndian::write_f32(&mut out[i * 4..(i + 1) * 4], f);
    }
    out
}

fn le_bytes_to_flat(bytes: &[u8], expected_floats: usize) -> crate::Result<Vec<f32>> {
    if bytes.len() != expected_floats * 4 {
        return Err(corruption(format!(
            "flat size mismatch: {} bytes for {} floats",
            bytes.len(),
            expected_floats
        )));
    }
    let mut v = Vec::with_capacity(expected_floats);
    for chunk in bytes.chunks_exact(4) {
        v.push(LittleEndian::read_f32(chunk));
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

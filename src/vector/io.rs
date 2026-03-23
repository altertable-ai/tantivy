//! On-disk layout for the `.vec` segment file (HNSW dump + dense vectors for merge).

use std::io::Write;
use std::ops::Range;
use std::sync::Arc;

#[cfg(feature = "mmap")]
use tempfile::TempDir;

use crate::directory::{FileSlice, OwnedBytes};
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use crate::schema::VectorOptions;
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC1";
pub(crate) const FLAT_MAGIC: &[u8; 8] = b"TNVYFLT1";
const VERSION: u32 = 1;

/// Graph or data bytes from the writer, or a sub-range of a mmap-backed segment file.
pub(crate) enum BytesMaybeMmap {
    Owned(Vec<u8>),
    Slice {
        backing: Arc<OwnedBytes>,
        range: Range<usize>,
    },
}

impl BytesMaybeMmap {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Slice { backing, range } => backing.as_slice().get(range.clone()).unwrap_or(&[]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Owned(v) => v.len(),
            Self::Slice { range, .. } => range.len(),
        }
    }
}

/// Keeps the temp directory and mmap-backed [`OwnedBytes`] alive while copying HNSW bytes into
/// `.vec` (see [`VectorFieldBundle::hnsw_dump_keepalive`]).
#[cfg(feature = "mmap")]
pub(crate) struct HnswDumpKeepalive {
    pub(crate) _dir: TempDir,
    pub(crate) graph: OwnedBytes,
    pub(crate) data: OwnedBytes,
}

#[cfg(feature = "mmap")]
impl HnswDumpKeepalive {
    pub(crate) fn as_bundle_slices(self: &Arc<Self>) -> (BytesMaybeMmap, BytesMaybeMmap) {
        let glen = self.graph.len();
        let dlen = self.data.len();
        (
            BytesMaybeMmap::Slice {
                backing: Arc::new(self.graph.clone()),
                range: 0..glen,
            },
            BytesMaybeMmap::Slice {
                backing: Arc::new(self.data.clone()),
                range: 0..dlen,
            },
        )
    }
}

/// When `mmap` is disabled, [`VectorFieldBundle::hnsw_dump_keepalive`] is always `None`.
#[cfg(not(feature = "mmap"))]
pub(crate) struct HnswDumpKeepalive(());

/// Row-major `f32` (`num_docs * dimension`) from the writer, or a mmap view of the `.vec` file.
pub(crate) enum FlatStorage {
    Owned(Vec<f32>),
    Mmap {
        backing: Arc<OwnedBytes>,
        range: Range<usize>,
    },
}

/// Serialized payload for one vector field inside the `.vec` file.
pub(crate) struct VectorFieldBundle {
    pub field_id: u32,
    pub options: VectorOptions,
    pub graph: BytesMaybeMmap,
    pub data: BytesMaybeMmap,
    pub flat: FlatStorage,
    pub num_docs: u32,
    /// Present only when graph/data were mmap'd from a temp `file_dump`; keeps the temp dir alive
    /// until [`write_vec_file`] finishes. Always `None` when loaded via [`read_vec_file`].
    #[allow(dead_code)] // Retained for drop order; not accessed otherwise.
    pub(crate) hnsw_dump_keepalive: Option<Arc<HnswDumpKeepalive>>,
}

/// Writes all vector fields for a segment into `writer`.
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
        let graph = bundle.graph.as_slice();
        writer.write_u64::<LittleEndian>(bundle.graph.len() as u64)?;
        writer.write_all(graph)?;
        let data = bundle.data.as_slice();
        writer.write_u64::<LittleEndian>(bundle.data.len() as u64)?;
        writer.write_all(data)?;
        writer.write_all(FLAT_MAGIC)?;
        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(bundle.options.dimension as u32)?;
        match &bundle.flat {
            FlatStorage::Owned(v) => {
                // `.vec` stores little-endian `f32`; on LE hosts write the buffer in one shot.
                #[cfg(target_endian = "little")]
                {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            v.as_ptr() as *const u8,
                            v.len() * std::mem::size_of::<f32>(),
                        )
                    };
                    writer.write_all(bytes)?;
                }
                #[cfg(not(target_endian = "little"))]
                {
                    for &f in v {
                        writer.write_f32::<LittleEndian>(f)?;
                    }
                }
            }
            FlatStorage::Mmap { .. } => {
                return Err(TantivyError::InternalError(
                    "write_vec_file: flat vectors must be owned".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn read_u32_le(buf: &[u8], pos: &mut usize) -> crate::Result<u32> {
    if buf.len() < *pos + 4 {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(".vec truncated (u32)"),
        ));
    }
    let v = LittleEndian::read_u32(&buf[*pos..*pos + 4]);
    *pos += 4;
    Ok(v)
}

fn read_u64_le(buf: &[u8], pos: &mut usize) -> crate::Result<u64> {
    if buf.len() < *pos + 8 {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only(".vec truncated (u64)"),
        ));
    }
    let v = LittleEndian::read_u64(&buf[*pos..*pos + 8]);
    *pos += 8;
    Ok(v)
}

/// Reads the `.vec` file into bundles. The returned [`FlatStorage::Mmap`] views borrow the
/// underlying [`OwnedBytes`] (typically mmap-backed when using [`crate::directory::MmapDirectory`]),
/// so vector rows are not copied out of the segment file.
pub(crate) fn read_vec_file(data: FileSlice) -> crate::Result<Vec<VectorFieldBundle>> {
    let backing = Arc::new(data.read_bytes()?);
    let buf = backing.as_slice();
    let mut pos = 0usize;

    if buf.len() < 8 || buf[..8] != *MAGIC {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only("invalid .vec magic"),
        ));
    }
    pos += 8;
    let _version = read_u32_le(buf, &mut pos)?;
    let n_fields = read_u32_le(buf, &mut pos)? as usize;

    let mut out = Vec::with_capacity(n_fields);
    for _ in 0..n_fields {
        let field_id = read_u32_le(buf, &mut pos)?;
        let opt_len = read_u32_le(buf, &mut pos)? as usize;
        if buf.len() < pos + opt_len {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(".vec truncated (options)"),
            ));
        }
        let options: VectorOptions = serde_json::from_slice(&buf[pos..pos + opt_len]).map_err(
            |e| {
                TantivyError::DataCorruption(crate::error::DataCorruption::comment_only(format!(
                    "vector options: {e}"
                )))
            },
        )?;
        pos += opt_len;

        let graph_len = read_u64_le(buf, &mut pos)? as usize;
        if buf.len() < pos + graph_len {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(".vec truncated (graph)"),
            ));
        }
        let graph = BytesMaybeMmap::Slice {
            backing: Arc::clone(&backing),
            range: pos..pos + graph_len,
        };
        pos += graph_len;

        let data_len = read_u64_le(buf, &mut pos)? as usize;
        if buf.len() < pos + data_len {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(".vec truncated (data)"),
            ));
        }
        let data = BytesMaybeMmap::Slice {
            backing: Arc::clone(&backing),
            range: pos..pos + data_len,
        };
        pos += data_len;

        if buf.len() < pos + 8 || buf[pos..pos + 8] != *FLAT_MAGIC {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only("invalid flat magic in .vec"),
            ));
        }
        pos += 8;

        let num_docs = read_u32_le(buf, &mut pos)?;
        let dim = read_u32_le(buf, &mut pos)? as usize;
        let flat_byte_len = num_docs as usize * dim * 4;
        if buf.len() < pos + flat_byte_len {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only(".vec truncated (flat)"),
            ));
        }
        let flat = FlatStorage::Mmap {
            backing: Arc::clone(&backing),
            range: pos..pos + flat_byte_len,
        };
        pos += flat_byte_len;

        out.push(VectorFieldBundle {
            field_id,
            options,
            graph,
            data,
            flat,
            num_docs,
            hnsw_dump_keepalive: None,
        });
    }

    debug_assert_eq!(pos, buf.len());
    Ok(out)
}

pub(crate) fn distance_to_score(distance: f32) -> crate::Score {
    // Higher is better for collectors; distance is lower-is-better.
    1.0 / (1.0 + distance)
}

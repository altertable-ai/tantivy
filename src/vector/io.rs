//! On-disk layout for the `.vec` segment file (HNSW dump + dense vectors for merge).

use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crate::directory::FileSlice;

use crate::schema::VectorOptions;
use crate::TantivyError;

pub(crate) const MAGIC: &[u8; 8] = b"TNVYVEC1";
pub(crate) const FLAT_MAGIC: &[u8; 8] = b"TNVYFLT1";
const VERSION: u32 = 1;

/// Serialized payload for one vector field inside the `.vec` file.
pub(crate) struct VectorFieldBundle {
    pub field_id: u32,
    pub options: VectorOptions,
    pub graph: Vec<u8>,
    pub data: Vec<u8>,
    /// Row-major `f32`, length `num_docs * dimension`.
    pub flat: Vec<f32>,
    pub num_docs: u32,
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
        let opts_json = serde_json::to_vec(&bundle.options).map_err(|e| {
            TantivyError::InternalError(format!("vector options json: {e}"))
        })?;
        writer.write_u32::<LittleEndian>(opts_json.len() as u32)?;
        writer.write_all(&opts_json)?;
        writer.write_u64::<LittleEndian>(bundle.graph.len() as u64)?;
        writer.write_all(&bundle.graph)?;
        writer.write_u64::<LittleEndian>(bundle.data.len() as u64)?;
        writer.write_all(&bundle.data)?;
        writer.write_all(FLAT_MAGIC)?;
        writer.write_u32::<LittleEndian>(bundle.num_docs)?;
        writer.write_u32::<LittleEndian>(bundle.options.dimension as u32)?;
        for &f in &bundle.flat {
            writer.write_f32::<LittleEndian>(f)?;
        }
    }
    Ok(())
}

/// Reads the `.vec` file into bundles.
pub(crate) fn read_vec_file(data: FileSlice) -> crate::Result<Vec<VectorFieldBundle>> {
    let bytes = data.read_bytes()?;
    let mut r: &[u8] = bytes.as_ref();
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(TantivyError::DataCorruption(
            crate::error::DataCorruption::comment_only("invalid .vec magic"),
        ));
    }
    let _version = r.read_u32::<LittleEndian>()?;
    let n_fields = r.read_u32::<LittleEndian>()?;
    let mut out = Vec::with_capacity(n_fields as usize);
    for _ in 0..n_fields {
        let field_id = r.read_u32::<LittleEndian>()?;
        let opt_len = r.read_u32::<LittleEndian>()? as usize;
        let mut opts_buf = vec![0u8; opt_len];
        r.read_exact(&mut opts_buf)?;
        let options: VectorOptions = serde_json::from_slice(&opts_buf).map_err(|e| {
            TantivyError::DataCorruption(crate::error::DataCorruption::comment_only(format!(
                "vector options: {e}"
            )))
        })?;
        let graph_len = r.read_u64::<LittleEndian>()? as usize;
        let mut graph = vec![0u8; graph_len];
        r.read_exact(&mut graph)?;
        let data_len = r.read_u64::<LittleEndian>()? as usize;
        let mut data = vec![0u8; data_len];
        r.read_exact(&mut data)?;
        let mut fm = [0u8; 8];
        r.read_exact(&mut fm)?;
        if &fm != FLAT_MAGIC {
            return Err(TantivyError::DataCorruption(
                crate::error::DataCorruption::comment_only("invalid flat magic in .vec"),
            ));
        }
        let num_docs = r.read_u32::<LittleEndian>()?;
        let dim = r.read_u32::<LittleEndian>()? as usize;
        let flat_len = num_docs as usize * dim;
        let mut flat = Vec::with_capacity(flat_len);
        for _ in 0..flat_len {
            flat.push(r.read_f32::<LittleEndian>()?);
        }
        out.push(VectorFieldBundle {
            field_id,
            options,
            graph,
            data,
            flat,
            num_docs,
        });
    }
    Ok(out)
}

pub(crate) fn distance_to_score(distance: f32) -> crate::Score {
    // Higher is better for collectors; distance is lower-is-better.
    1.0 / (1.0 + distance)
}

pub(crate) fn max_layer_for_n(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    ((n as f32).ln().trunc() as usize).clamp(1, 16)
}

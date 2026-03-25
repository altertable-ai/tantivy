//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open parses the V3 file: scalar-quantized vectors stay as mmap-backed
//! `OwnedBytes`; the HNSW graph is deserialized once. Search dequantizes vectors
//! on demand into a reusable buffer.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::Field;
use crate::vector::io::{distance_fn_for, distance_to_score, read_vec_file, MmapVectorField};
use crate::vector::mmaped_hnsw;
use crate::{DocId, Score};

/// Readers for all vector fields in a segment.
#[derive(Clone)]
pub struct VectorFieldReaders {
    readers: Arc<HashMap<u32, Arc<VectorFieldReader>>>,
}

impl VectorFieldReaders {
    pub(crate) fn open(data: FileSlice) -> crate::Result<Self> {
        let fields = read_vec_file(data)?;
        let mut map = HashMap::with_capacity(fields.len());
        for field in fields {
            let field_id = field.field_id;
            let reader = VectorFieldReader::new(field);
            map.insert(field_id, Arc::new(reader));
        }
        Ok(VectorFieldReaders {
            readers: Arc::new(map),
        })
    }

    pub(crate) fn empty() -> Self {
        VectorFieldReaders {
            readers: Arc::new(HashMap::new()),
        }
    }

    /// Returns the reader for a field, if present in this segment.
    pub fn get(&self, field: Field) -> Option<Arc<VectorFieldReader>> {
        self.readers.get(&field.field_id()).cloned()
    }
}

/// One vector field backed by mmap-friendly quantized storage.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, …).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    dim: usize,
    sq: crate::vector::io::SqParams,
    quantized_bytes: crate::directory::OwnedBytes,
    graph: crate::vector::io::CompactHnswGraph,
}

impl VectorFieldReader {
    fn new(field: MmapVectorField) -> Self {
        Self {
            field_id: field.field_id,
            options: field.options.clone(),
            num_docs: field.num_docs,
            dim: field.dimension,
            sq: field.sq,
            quantized_bytes: field.quantized_bytes,
            graph: field.graph,
        }
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> crate::Result<Vec<(DocId, Score)>> {
        let dim = self.dim;
        if query.len() != dim {
            return Err(crate::TantivyError::InvalidArgument(format!(
                "query dimension {} does not match field dimension {}",
                query.len(),
                dim
            )));
        }
        let dist = distance_fn_for(self.options.distance);
        let q = self.quantized_bytes.as_slice();
        let mut buf = vec![0f32; dim];
        let mut distance_to = |id: u32| -> f32 {
            let start = id as usize * dim;
            self.sq.dequantize_into(&q[start..start + dim], &mut buf);
            dist(query, &buf)
        };
        Ok(mmaped_hnsw::search(&self.graph, &mut distance_to, k, ef)
            .into_iter()
            .map(|r| (r.doc_id as DocId, distance_to_score(r.distance)))
            .collect())
    }

    /// All flat vectors (row-major, `num_docs * dimension` floats), dequantized for merge.
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        Ok(self.sq.dequantize_all(
            self.quantized_bytes.as_slice(),
            self.dim,
            self.num_docs as usize,
        ))
    }

    /// Dequantized vector for `doc` in this segment, if in range.
    pub fn vector(&self, doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        let dim = self.dim;
        if doc >= self.num_docs {
            return Ok(None);
        }
        let start = doc as usize * dim;
        let mut out = vec![0f32; dim];
        self.sq
            .dequantize_into(&self.quantized_bytes[start..start + dim], &mut out);
        Ok(Some(out))
    }
}

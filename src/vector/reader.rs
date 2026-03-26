//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open is **instant** — only metadata/offsets are parsed.
//! Decompression of flat vectors and the HNSW graph happens on each
//! `search()` / `vector()` / `flat_vectors()` call.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::Field;
use crate::vector::io::{distance_to_score, read_vec_file_lazy, LazyVectorField};
use crate::vector::mmaped_hnsw;
use crate::{DocId, Score};

/// Readers for all vector fields in a segment.
#[derive(Clone)]
pub struct VectorFieldReaders {
    readers: Arc<HashMap<u32, Arc<VectorFieldReader>>>,
}

impl VectorFieldReaders {
    pub(crate) fn open(data: FileSlice) -> crate::Result<Self> {
        let fields = read_vec_file_lazy(data)?;
        let mut map = HashMap::with_capacity(fields.len());
        for lazy in fields {
            let field_id = lazy.field_id;
            let reader = VectorFieldReader::new(lazy);
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

/// One vector field backed by compressed on-disk data.
/// Decompression happens lazily on each query.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, …).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    lazy: LazyVectorField,
}

impl VectorFieldReader {
    fn new(lazy: LazyVectorField) -> Self {
        Self {
            field_id: lazy.field_id,
            options: lazy.options.clone(),
            num_docs: lazy.num_docs,
            lazy,
        }
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    ///
    /// Decompresses flat vectors + HNSW graph on every call.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> crate::Result<Vec<(DocId, Score)>> {
        let flat = self.lazy.decompress_flat()?;
        let graph = self.lazy.decompress_graph()?;
        Ok(mmaped_hnsw::search(
            &graph,
            &flat,
            self.options.dimension,
            self.options.distance,
            query,
            k,
            ef,
        )
        .into_iter()
        .map(|r| (r.doc_id as DocId, distance_to_score(r.distance)))
        .collect())
    }

    /// Decompress and return all flat vectors (row-major, `num_docs * dimension` floats).
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        self.lazy.decompress_flat()
    }

    /// Decompress flat vectors and return the slice for a single document.
    pub fn vector(&self, doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        let dim = self.options.dimension;
        if doc >= self.num_docs {
            return Ok(None);
        }
        let flat = self.lazy.decompress_flat()?;
        let start = doc as usize * dim;
        Ok(flat.get(start..start + dim).map(|s| s.to_vec()))
    }
}

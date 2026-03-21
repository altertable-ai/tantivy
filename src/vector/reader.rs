//! Loads `.vec` segment data and runs k-NN queries.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::Field;
use crate::vector::hnsw::build_hnsw_in_memory;
use crate::vector::io::{distance_to_score, read_vec_file, VectorFieldBundle};
use crate::vector::VectorIndexInner;
use crate::{DocId, Score};

/// Readers for all vector fields in a segment.
#[derive(Clone)]
pub struct VectorFieldReaders {
    readers: Arc<HashMap<u32, Arc<VectorFieldReader>>>,
}

impl VectorFieldReaders {
    pub(crate) fn open(data: FileSlice) -> crate::Result<Self> {
        let bundles = read_vec_file(data)?;
        let mut map = HashMap::with_capacity(bundles.len());
        for bundle in bundles {
            let reader = VectorFieldReader::open(bundle)?;
            map.insert(reader.field_id, Arc::new(reader));
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

/// One vector field: HNSW index + dense rows (for merge / inspection).
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, ...).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    flat: Vec<f32>,
    inner: VectorIndexInner,
}

impl VectorFieldReader {
    fn open(bundle: VectorFieldBundle) -> crate::Result<Self> {
        let inner = build_hnsw_in_memory(&bundle.options, bundle.num_docs, &bundle.flat)?;
        Ok(VectorFieldReader {
            field_id: bundle.field_id,
            options: bundle.options,
            num_docs: bundle.num_docs,
            flat: bundle.flat,
            inner,
        })
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(DocId, Score)> {
        let neighbors = match &self.inner {
            VectorIndexInner::L2(h) => h.search(query, k, ef),
            VectorIndexInner::Cosine(h) => h.search(query, k, ef),
            VectorIndexInner::Dot(h) => h.search(query, k, ef),
        };
        neighbors
            .into_iter()
            .map(|n| (n.d_id as DocId, distance_to_score(n.distance)))
            .collect()
    }

    /// Dense vector for `doc` in this segment, if in range.
    pub fn vector(&self, doc: DocId) -> Option<&[f32]> {
        let dim = self.options.dimension;
        if doc >= self.num_docs {
            return None;
        }
        let start = doc as usize * dim;
        self.flat.get(start..start + dim)
    }

    /// Full flat storage (row-major), for segment merge.
    pub fn flat_vectors(&self) -> &[f32] {
        &self.flat
    }
}

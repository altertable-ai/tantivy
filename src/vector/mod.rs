//! Dense vector fields and HNSW approximate nearest neighbor search.
//!
//! Enabled by the `vector` crate feature (on by default).

#[cfg(feature = "vector")]
pub(crate) mod hnsw;
#[cfg(feature = "vector")]
mod io;
#[cfg(feature = "vector")]
mod mmaped_hnsw;
#[cfg(feature = "vector")]
pub(crate) mod reader;
#[cfg(feature = "vector")]
mod turboquant;
#[cfg(feature = "vector")]
pub(crate) mod writer;

#[cfg(feature = "vector")]
pub(crate) use io::{normalize_flat_for_cosine, write_vec_file, VectorFieldBundle};
#[cfg(feature = "vector")]
pub use reader::{VectorFieldReader, VectorFieldReaders};
#[cfg(feature = "vector")]
pub(crate) use writer::{build_compact_graph_from_flat, VectorFieldsWriter};

#[cfg(all(test, feature = "vector"))]
mod tests;

#[cfg(not(feature = "vector"))]
use std::sync::Arc;

#[cfg(not(feature = "vector"))]
use crate::directory::{FileSlice, WritePtr};
#[cfg(not(feature = "vector"))]
use crate::schema::{document::Document, Field, VectorOptions};
#[cfg(not(feature = "vector"))]
use crate::{DocId, Score};

#[cfg(not(feature = "vector"))]
#[derive(Default)]
pub(crate) struct CompactHnswGraph;

#[cfg(not(feature = "vector"))]
#[allow(dead_code)]
pub(crate) struct VectorFieldBundle {
    pub field_id: u32,
    pub options: VectorOptions,
    pub num_docs: u32,
    pub flat: Vec<f32>,
    pub graph: CompactHnswGraph,
}

#[cfg(not(feature = "vector"))]
pub(crate) fn write_vec_file(
    _writer: &mut dyn std::io::Write,
    _fields: &[VectorFieldBundle],
) -> crate::Result<()> {
    Ok(())
}

#[cfg(not(feature = "vector"))]
#[allow(dead_code)] // Merger uses the real implementations only with `feature = "vector"`.
pub(crate) fn build_compact_graph_from_flat(
    _options: &VectorOptions,
    _max_doc: DocId,
    _flat: &[f32],
) -> crate::Result<CompactHnswGraph> {
    Ok(CompactHnswGraph)
}

#[cfg(not(feature = "vector"))]
#[allow(dead_code)]
pub(crate) fn normalize_flat_for_cosine(
    _flat: &mut [f32],
    _dim: usize,
    _dist: crate::schema::VectorDistance,
) {
}

#[cfg(not(feature = "vector"))]
/// Readers for all vector fields in a segment when vector support is disabled.
#[derive(Clone)]
pub struct VectorFieldReaders;

#[cfg(not(feature = "vector"))]
impl VectorFieldReaders {
    pub(crate) fn open(_data: FileSlice) -> crate::Result<Self> {
        Ok(Self)
    }

    pub(crate) fn empty() -> Self {
        Self
    }

    /// Returns the reader for a field, if present in this segment.
    pub fn get(&self, _field: Field) -> Option<Arc<VectorFieldReader>> {
        None
    }
}

#[cfg(not(feature = "vector"))]
/// Placeholder reader type when vector support is disabled.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, ...).
    pub options: VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
}

#[cfg(not(feature = "vector"))]
impl VectorFieldReader {
    /// Approximate k-nearest neighbors for `query` (always empty without feature).
    pub fn search(
        &self,
        _query: &[f32],
        _k: usize,
        _ef: usize,
    ) -> crate::Result<Vec<(DocId, Score)>> {
        Ok(Vec::new())
    }

    /// Dense vector for `doc` in this segment, if in range.
    pub fn vector(&self, _doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Full flat storage (row-major), for segment merge.
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        Ok(Vec::new())
    }
}

#[cfg(not(feature = "vector"))]
#[derive(Default)]
pub(crate) struct VectorFieldsWriter;

#[cfg(not(feature = "vector"))]
impl VectorFieldsWriter {
    pub(crate) fn for_schema(_schema: &crate::schema::Schema) -> Self {
        Self
    }

    pub(crate) fn add_document<D: Document>(
        &mut self,
        _doc: &D,
        _doc_id: DocId,
    ) -> crate::Result<()> {
        Ok(())
    }

    pub(crate) fn mem_usage(&self) -> usize {
        0
    }

    pub(crate) fn serialize(&self, _writer: WritePtr, _max_doc: DocId) -> crate::Result<()> {
        Ok(())
    }
}

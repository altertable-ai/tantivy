//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open parses the V4 header and materializes TQ+ params, per-vector
//! length-renorm scales, and a mmap-backed compact HNSW graph. Packed 4-bit
//! codes stay as a byte slice into the mmap region.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::{Field, VectorDistance};
use crate::vector::io::{
    distance_to_score, l2_normalize, read_vec_file_lazy, CompactHnswGraph, LazyVectorField,
};
use crate::vector::mmaped_hnsw;
use crate::vector::turboquant::{Codebook, TqPlus};
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
            let reader = VectorFieldReader::new(lazy)?;
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

/// One vector field backed by mmap'd TurboQuant codes and an mmap-resident graph.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, …).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    lazy: LazyVectorField,
    graph: CompactHnswGraph,
    tqplus: TqPlus,
    renorm: Vec<f32>,
    codebook: Codebook,
}

impl VectorFieldReader {
    fn new(lazy: LazyVectorField) -> crate::Result<Self> {
        let graph = lazy.parse_graph()?;
        let tqplus = lazy.parse_tqplus()?;
        let renorm = lazy.parse_renorm()?;
        let codebook = lazy.codebook();
        Ok(Self {
            field_id: lazy.field_id,
            options: lazy.options.clone(),
            num_docs: lazy.num_docs,
            lazy,
            graph,
            tqplus,
            renorm,
            codebook,
        })
    }

    /// Packed 4-bit codes (`num_docs * padded_dim / 2` bytes), mmap-backed.
    pub fn quantized_vectors(&self) -> &[u8] {
        self.lazy.packed_codes()
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> crate::Result<Vec<(DocId, Score)>> {
        let dim = self.options.dimension;
        let query = if self.options.distance == VectorDistance::Cosine {
            let mut q = query.to_vec();
            l2_normalize(&mut q);
            q
        } else {
            query.to_vec()
        };
        Ok(mmaped_hnsw::search(
            &self.graph,
            self.lazy.packed_codes(),
            &self.renorm,
            &self.tqplus,
            &self.codebook,
            dim,
            self.options.distance,
            &query,
            k,
            ef,
        )
        .into_iter()
        .map(|r| (r.doc_id as DocId, distance_to_score(r.distance)))
        .collect())
    }

    /// All flat vectors (row-major, `num_docs * dimension` floats), reconstructed from TQ4.
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        self.lazy.dequantize_flat()
    }

    /// Reconstructed vector for `doc`, if in range.
    pub fn vector(&self, doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        if doc >= self.num_docs {
            return Ok(None);
        }
        Ok(Some(self.lazy.reconstruct_doc(
            doc,
            &self.tqplus,
            &self.codebook,
            self.renorm.get(doc as usize).copied().unwrap_or(1.0),
        )))
    }
}

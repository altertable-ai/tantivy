//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open parses the V4 header and materializes the compact HNSW graph plus
//! field centroid and per-document BBQ lower/upper. Packed bits stay mmap-backed;
//! [`VectorFieldReader::search`] uses asymmetric BBQ distances. [`VectorFieldReader::flat_vectors`]
//! and [`VectorFieldReader::vector`] lossily reconstruct f32 rows for merge / retrieval.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::{Field, VectorDistance};
use crate::vector::bbq::{bbq_bytes_per_row, bbq_dequantize_row};
use crate::vector::io::{
    distance_to_score, l2_normalize, read_vec_file_lazy, CompactHnswGraph, LazyVectorField,
};
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

/// One vector field backed by mmap'd BBQ data and an in-memory compact graph.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, …).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    lazy: LazyVectorField,
    graph: CompactHnswGraph,
    centroid: Vec<f32>,
    bbq_lower: Vec<f32>,
    bbq_upper: Vec<f32>,
}

impl VectorFieldReader {
    fn new(lazy: LazyVectorField) -> crate::Result<Self> {
        let graph = lazy.parse_graph()?;
        let centroid = lazy.parse_centroid()?;
        let bbq_lower = lazy.parse_lowers()?;
        let bbq_upper = lazy.parse_uppers()?;
        Ok(Self {
            field_id: lazy.field_id,
            options: lazy.options.clone(),
            num_docs: lazy.num_docs,
            lazy,
            graph,
            centroid,
            bbq_lower,
            bbq_upper,
        })
    }

    /// Field centroid (component-wise mean), length = dimension.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Per-document lower residual level after BBQ (length = num_docs).
    pub fn bbq_lower(&self) -> &[f32] {
        &self.bbq_lower
    }

    /// Per-document upper residual level after BBQ (length = num_docs).
    pub fn bbq_upper(&self) -> &[f32] {
        &self.bbq_upper
    }

    /// Packed 1-bit BBQ residuals: `num_docs * ceil(dimension / 8)` bytes, mmap-backed.
    pub fn bbq_bits(&self) -> &[u8] {
        self.lazy.bbq_bits_bytes()
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
            self.lazy.bbq_bits_bytes(),
            &self.bbq_lower,
            &self.bbq_upper,
            &self.centroid,
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

    /// All flat vectors (row-major, `num_docs * dimension` floats), lossy BBQ reconstruction.
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        self.lazy.dequantize_flat()
    }

    /// Lossily reconstructed vector for `doc`, if in range.
    pub fn vector(&self, doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        let dim = self.options.dimension;
        if doc >= self.num_docs {
            return Ok(None);
        }
        let bpr = bbq_bytes_per_row(dim);
        let bits_all = self.lazy.bbq_bits_bytes();
        let start = doc as usize * bpr;
        let row = &bits_all[start..start + bpr];
        let mut out = vec![0f32; dim];
        bbq_dequantize_row(
            &self.centroid,
            row,
            self.bbq_lower[doc as usize],
            self.bbq_upper[doc as usize],
            &mut out,
        );
        Ok(Some(out))
    }
}

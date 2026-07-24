//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open parses the V3 header and materializes the compact HNSW graph plus
//! quantization parameters. Quantized vectors stay as a byte slice into the mmap
//! region; [`VectorFieldReader::search`] uses a scratch buffer only (no full flat
//! allocation). [`VectorFieldReader::flat_vectors`] and [`VectorFieldReader::vector`]
//! dequantize on demand for the public retrieval API.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::schema::{Field, VectorDistance};
use crate::vector::io::{
    dequantize_row_into, distance_to_score, l2_normalize, read_vec_file_lazy, CompactHnswGraph,
    LazyVectorField,
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

/// One vector field backed by mmap'd SQ8 data and an in-memory compact graph.
pub struct VectorFieldReader {
    /// Schema field id.
    pub field_id: u32,
    /// Field options (dimension, distance, …).
    pub options: crate::schema::VectorOptions,
    /// Number of documents indexed in this segment.
    pub num_docs: u32,
    lazy: LazyVectorField,
    graph: CompactHnswGraph,
    mins: Vec<f32>,
    scales: Vec<f32>,
}

impl VectorFieldReader {
    fn new(lazy: LazyVectorField) -> crate::Result<Self> {
        let graph = lazy.parse_graph()?;
        let (mins, scales) = lazy.parse_mins_scales()?;
        Ok(Self {
            field_id: lazy.field_id,
            options: lazy.options.clone(),
            num_docs: lazy.num_docs,
            lazy,
            graph,
            mins,
            scales,
        })
    }

    /// Row-major quantized vectors (`num_docs * dimension` bytes), mmap-backed.
    pub fn quantized_vectors(&self) -> &[u8] {
        self.lazy.quantized_vectors()
    }

    /// Per-dimension minimum and `(max-min)/255` scale for SQ8 dequantization.
    pub fn quantization_params(&self) -> (&[f32], &[f32]) {
        (&self.mins, &self.scales)
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
            self.lazy.quantized_vectors(),
            &self.mins,
            &self.scales,
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

    /// All flat vectors (row-major, `num_docs * dimension` floats), dequantized from SQ8.
    pub fn flat_vectors(&self) -> crate::Result<Vec<f32>> {
        self.lazy.dequantize_flat()
    }

    /// Dequantized vector for `doc`, if in range.
    pub fn vector(&self, doc: DocId) -> crate::Result<Option<Vec<f32>>> {
        let dim = self.options.dimension;
        if doc >= self.num_docs {
            return Ok(None);
        }
        let flat_u8 = self.lazy.quantized_vectors();
        let start = doc as usize * dim;
        let row = &flat_u8[start..start + dim];
        let mut out = vec![0f32; dim];
        dequantize_row_into(row, &self.mins, &self.scales, &mut out);
        Ok(Some(out))
    }
}

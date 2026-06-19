//! Loads `.vec` segment data and runs k-NN queries.
//!
//! Reader open parses the V5 header and materializes turbovec's blocked search cache.
//! Persisted bit-plane codes stay mmap-backed.

use std::collections::HashMap;
use std::sync::Arc;

use crate::directory::FileSlice;
use crate::fastfield::AliveBitSet;
use crate::schema::{Field, VectorDistance};
use crate::vector::io::{distance_to_score, l2_normalize, read_vec_file_lazy, LazyVectorField};
use crate::vector::turboquant::{self, TurboQuantCache};
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
    bit_width: usize,
    scales: Vec<f32>,
    tqplus_shift: Vec<f32>,
    tqplus_scale: Vec<f32>,
    cache: TurboQuantCache,
}

impl VectorFieldReader {
    fn new(lazy: LazyVectorField) -> crate::Result<Self> {
        let bit_width = lazy.turbo_bit_width();
        let cache = turboquant::prepare_cache(
            bit_width,
            lazy.dimension,
            lazy.num_docs as usize,
            lazy.turbo_packed_codes(),
        )?;
        let scales = lazy.parse_turbo_scales()?;
        let tqplus_shift = lazy.parse_tqplus_shift()?;
        let tqplus_scale = lazy.parse_tqplus_scale()?;
        Ok(Self {
            field_id: lazy.field_id,
            options: lazy.options.clone(),
            num_docs: lazy.num_docs,
            lazy,
            bit_width,
            scales,
            tqplus_shift,
            tqplus_scale,
            cache,
        })
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        _ef: usize,
        alive: Option<&AliveBitSet>,
    ) -> crate::Result<Vec<(DocId, Score)>> {
        let dim = self.options.dimension;
        let query = if self.options.distance == VectorDistance::Cosine {
            let mut q = query.to_vec();
            l2_normalize(&mut q);
            q
        } else {
            query.to_vec()
        };
        let mask = alive
            .map(|alive| turboquant::build_alive_mask(self.num_docs as usize, alive.iter_alive()));
        Ok(turboquant::search(
            &query,
            &self.cache,
            &self.scales,
            &self.tqplus_shift,
            &self.tqplus_scale,
            self.bit_width,
            dim,
            self.num_docs as usize,
            k,
            mask.as_deref(),
        )?
        .into_iter()
        .map(|(doc, similarity)| {
            let distance = (1.0f32 - similarity).max(0.0);
            (doc, distance_to_score(distance))
        })
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
        let flat = self.flat_vectors()?;
        let start = doc as usize * dim;
        Ok(Some(flat[start..start + dim].to_vec()))
    }
}

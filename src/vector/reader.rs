//! Loads `.vec` segment data and runs k-NN queries.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use crate::directory::{FileSlice, OwnedBytes};
use crate::schema::Field;
use crate::vector::io::{distance_to_score, read_vec_file, FlatStorage, VectorFieldBundle};
use crate::vector::mmaped_hnsw::open_vector_index;
use crate::vector::VectorIndexInner;
use crate::{DocId, Score};

/// Dense vector storage: either an owned buffer (merge / RAM) or a 4-byte-aligned view into the
/// segment file (typically mmap-backed).
enum VectorFlatInner {
    Owned(Vec<f32>),
    MmapAligned {
        backing: Arc<OwnedBytes>,
        range: Range<usize>,
    },
}

impl VectorFlatInner {
    fn from_storage(flat: FlatStorage) -> crate::Result<Self> {
        match flat {
            FlatStorage::Owned(v) => Ok(Self::Owned(v)),
            FlatStorage::Mmap { backing, range } => {
                let bytes = backing.as_slice().get(range.clone()).ok_or_else(|| {
                    crate::TantivyError::DataCorruption(
                        crate::error::DataCorruption::comment_only("vector flat range"),
                    )
                })?;
                if bytes.len() % 4 != 0 {
                    return Err(crate::TantivyError::DataCorruption(
                        crate::error::DataCorruption::comment_only(
                            "vector flat payload not multiple of 4",
                        ),
                    ));
                }
                let (prefix, f32s, suffix) = unsafe { bytes.align_to::<f32>() };
                if prefix.is_empty() && suffix.is_empty() && f32s.len() * 4 == bytes.len() {
                    Ok(Self::MmapAligned { backing, range })
                } else {
                    let v: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                    Ok(Self::Owned(v))
                }
            }
        }
    }

    fn as_f32_slice(&self) -> &[f32] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::MmapAligned { backing, range } => {
                let bytes = &backing.as_slice()[range.clone()];
                let (prefix, f32s, suffix) = unsafe { bytes.align_to::<f32>() };
                debug_assert!(prefix.is_empty() && suffix.is_empty());
                f32s
            }
        }
    }
}

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
    flat: VectorFlatInner,
    inner: VectorIndexInner,
}

impl VectorFieldReader {
    fn open(bundle: VectorFieldBundle) -> crate::Result<Self> {
        let flat = VectorFlatInner::from_storage(bundle.flat)?;
        let inner = open_vector_index(
            &bundle.options,
            bundle.graph.as_slice(),
            bundle.data.as_slice(),
            bundle.num_docs,
            flat.as_f32_slice(),
        )?;
        Ok(VectorFieldReader {
            field_id: bundle.field_id,
            options: bundle.options,
            num_docs: bundle.num_docs,
            flat,
            inner,
        })
    }

    /// Approximate k-nearest neighbors for `query` (same dimension as the field).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(DocId, Score)> {
        self.inner
            .search(query, k, ef)
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
        let flat = self.flat.as_f32_slice();
        flat.get(start..start + dim)
    }

    /// Full flat storage (row-major), for segment merge.
    pub fn flat_vectors(&self) -> &[f32] {
        self.flat.as_f32_slice()
    }
}

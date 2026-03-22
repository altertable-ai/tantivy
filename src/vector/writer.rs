//! Collects per-document vectors during segment indexing and serializes HNSW graphs.

use std::fs;
use std::io::Write;

use common::TerminatingWrite;
use hnsw_rs::api::AnnT;

use crate::directory::WritePtr;
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::hnsw::{build_hnsw_in_memory, VectorIndexInner};
use crate::vector::io::{write_vec_file, BytesMaybeMmap, FlatStorage, VectorFieldBundle};
use crate::{DocId, TantivyError};

/// Collects dense vectors for all vector fields in a segment.
pub(crate) struct VectorFieldsWriter {
    /// For each field index (schema order): optional per-field writer.
    per_field: Vec<Option<PerVectorFieldWriter>>,
}

struct PerVectorFieldWriter {
    field: Field,
    options: VectorOptions,
    /// One slot per document id in this segment.
    vectors: Vec<Option<Vec<f32>>>,
}

impl VectorFieldsWriter {
    pub(crate) fn for_schema(schema: &Schema) -> Self {
        let n = schema.num_fields();
        let mut per_field: Vec<Option<PerVectorFieldWriter>> = Vec::with_capacity(n);
        per_field.resize_with(n, || None);
        for (field, field_entry) in schema.fields() {
            if let FieldType::Vector(opts) = field_entry.field_type() {
                per_field[field.field_id() as usize] = Some(PerVectorFieldWriter {
                    field,
                    options: opts.clone(),
                    vectors: Vec::new(),
                });
            }
        }
        Self { per_field }
    }

    pub(crate) fn add_document<D: Document>(
        &mut self,
        doc: &D,
        doc_id: DocId,
    ) -> crate::Result<()> {
        for (field, value) in doc.iter_fields_and_values() {
            if let Some(writer) = self
                .per_field
                .get_mut(field.field_id() as usize)
                .and_then(|w| w.as_mut())
            {
                let slice = value
                    .as_value()
                    .as_leaf()
                    .and_then(|l| l.as_vector())
                    .ok_or_else(|| {
                        TantivyError::SchemaError(format!(
                            "Expected vector slice for field {:?}",
                            field.field_id()
                        ))
                    })?;
                if slice.len() != writer.options.dimension {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Vector dimension mismatch for field: expected {}, got {}",
                        writer.options.dimension,
                        slice.len()
                    )));
                }
                let doc_idx = doc_id as usize;
                if doc_idx >= writer.vectors.len() {
                    writer.vectors.resize(doc_idx + 1, None);
                }
                writer.vectors[doc_idx] = Some(slice.to_vec());
            }
        }
        Ok(())
    }

    pub(crate) fn mem_usage(&self) -> usize {
        self.per_field
            .iter()
            .filter_map(|w| w.as_ref())
            .map(|w| {
                w.vectors
                    .iter()
                    .filter_map(|v| v.as_ref())
                    .map(|v| v.capacity() * std::mem::size_of::<f32>())
                    .sum::<usize>()
            })
            .sum()
    }

    pub(crate) fn serialize(&self, mut writer: WritePtr, max_doc: DocId) -> crate::Result<()> {
        let mut bundles: Vec<VectorFieldBundle> = Vec::new();
        for slot in &self.per_field {
            let Some(field_writer) = slot else {
                continue;
            };
            let mut flat: Vec<f32> =
                Vec::with_capacity(max_doc as usize * field_writer.options.dimension);
            for doc in 0..max_doc {
                let Some(vec) = field_writer
                    .vectors
                    .get(doc as usize)
                    .and_then(|v| v.as_ref())
                else {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Missing vector for doc {doc} in field {:?}",
                        field_writer.field
                    )));
                };
                flat.extend_from_slice(vec);
            }
            let (graph, data) = build_and_dump_hnsw(&field_writer.options, max_doc, &flat)?;
            bundles.push(VectorFieldBundle {
                field_id: field_writer.field.field_id(),
                options: field_writer.options.clone(),
                graph: BytesMaybeMmap::Owned(graph),
                data: BytesMaybeMmap::Owned(data),
                flat: FlatStorage::Owned(flat),
                num_docs: max_doc,
            });
        }
        write_vec_file(&mut writer, &bundles)?;
        writer.flush()?;
        writer.terminate()?;
        Ok(())
    }
}

fn build_and_dump_hnsw(
    options: &VectorOptions,
    max_doc: DocId,
    flat: &[f32],
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    if max_doc == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let inner = build_hnsw_in_memory(options, max_doc, flat)?;
    let dir = tempfile::tempdir().map_err(|e| {
        TantivyError::InternalError(format!("temp dir for hnsw dump: {e}"))
    })?;
    let basename = match &inner {
        VectorIndexInner::L2(h) => h.file_dump(dir.path(), "tntv"),
        VectorIndexInner::Cosine(h) => h.file_dump(dir.path(), "tntv"),
        VectorIndexInner::Dot(h) => h.file_dump(dir.path(), "tntv"),
    }
    .map_err(|e| TantivyError::InternalError(format!("hnsw file_dump: {e}")))?;
    let graph_path = dir.path().join(format!("{basename}.hnsw.graph"));
    let data_path = dir.path().join(format!("{basename}.hnsw.data"));
    let graph = fs::read(&graph_path).map_err(|e| {
        TantivyError::InternalError(format!("read hnsw graph dump: {e}"))
    })?;
    let data = fs::read(&data_path).map_err(|e| {
        TantivyError::InternalError(format!("read hnsw data dump: {e}"))
    })?;
    Ok((graph, data))
}

/// Rebuilds HNSW graph bytes after a segment merge.
pub(crate) fn build_hnsw_from_flat(
    options: &VectorOptions,
    max_doc: DocId,
    flat: &[f32],
) -> crate::Result<(Vec<u8>, Vec<u8>)> {
    build_and_dump_hnsw(options, max_doc, flat)
}

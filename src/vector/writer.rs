//! Collects per-document vectors during segment indexing and serializes the compact
//! HNSW graph + scalar-quantized vectors into the `.vec` file (V3 format).

use std::io::Write;

use common::TerminatingWrite;

use crate::directory::WritePtr;
use crate::schema::document::{Document, Value};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::hnsw::extract_compact_graph;
use crate::vector::io::{write_vec_file, VectorFieldBundle};
use crate::{DocId, TantivyError};

/// Collects dense vectors for all vector fields in a segment.
pub(crate) struct VectorFieldsWriter {
    per_field: Vec<Option<PerVectorFieldWriter>>,
}

struct PerVectorFieldWriter {
    field: Field,
    options: VectorOptions,
    /// Row-major contiguous buffer: `flat[doc * dim .. (doc+1) * dim]`.
    flat: Vec<f32>,
    /// One bit per doc slot to detect gaps (docs that were never added).
    populated: Vec<bool>,
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
                    flat: Vec::new(),
                    populated: Vec::new(),
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
                let dim = writer.options.dimension;
                if slice.len() != dim {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Vector dimension mismatch for field: expected {}, got {}",
                        dim,
                        slice.len()
                    )));
                }
                let doc_idx = doc_id as usize;
                if doc_idx >= writer.populated.len() {
                    writer.flat.resize((doc_idx + 1) * dim, 0.0);
                    writer.populated.resize(doc_idx + 1, false);
                }
                let offset = doc_idx * dim;
                writer.flat[offset..offset + dim].copy_from_slice(slice);
                writer.populated[doc_idx] = true;
            }
        }
        Ok(())
    }

    pub(crate) fn mem_usage(&self) -> usize {
        self.per_field
            .iter()
            .filter_map(|w| w.as_ref())
            .map(|w| w.flat.capacity() * std::mem::size_of::<f32>())
            .sum()
    }

    pub(crate) fn serialize(self, mut writer: WritePtr, max_doc: DocId) -> crate::Result<()> {
        let mut bundles: Vec<VectorFieldBundle> = Vec::new();
        for slot in self.per_field {
            let Some(mut field_writer) = slot else {
                continue;
            };
            let dim = field_writer.options.dimension;
            for doc in 0..max_doc {
                if !field_writer
                    .populated
                    .get(doc as usize)
                    .copied()
                    .unwrap_or(false)
                {
                    return Err(TantivyError::InvalidArgument(format!(
                        "Missing vector for doc {doc} in field {:?}",
                        field_writer.field
                    )));
                }
            }
            field_writer.flat.truncate(max_doc as usize * dim);
            let graph =
                extract_compact_graph(&field_writer.options, max_doc, &field_writer.flat)?;
            bundles.push(VectorFieldBundle {
                field_id: field_writer.field.field_id(),
                options: field_writer.options,
                num_docs: max_doc,
                flat: field_writer.flat,
                graph,
            });
        }
        write_vec_file(&mut writer, &bundles)?;
        writer.flush()?;
        writer.terminate()?;
        Ok(())
    }
}

/// Build a compact HNSW graph from flat vectors (used during merge).
pub(crate) fn build_compact_graph_from_flat(
    options: &VectorOptions,
    max_doc: DocId,
    flat: &[f32],
) -> crate::Result<crate::vector::io::CompactHnswGraph> {
    extract_compact_graph(options, max_doc, flat)
}

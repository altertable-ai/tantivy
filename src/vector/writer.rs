//! Collects per-document vectors during segment indexing and serializes the compact
//! HNSW graph + 4-bit TurboQuant codes into the `.vec` file (V4 `TNVYVEC4` format).

use std::io::Write;

use crate::schema::document::{ErasedDocument, ErasedValue};
use crate::schema::{Field, FieldType, Schema, VectorOptions};
use crate::vector::hnsw::extract_compact_graph;
use crate::vector::io::{normalize_flat_for_cosine, write_vec_file, VectorFieldBundle};
use crate::{DocId, TantivyError};

/// Collects dense vectors for all vector fields in a segment.
pub(crate) struct VectorFieldsWriter {
    per_field: Vec<Option<PerVectorFieldWriter>>,
    max_doc: DocId,
}

struct PerVectorFieldWriter {
    field: Field,
    options: VectorOptions,
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
        Self {
            per_field,
            max_doc: 0,
        }
    }

    pub(crate) fn add_document(
        &mut self,
        doc: &dyn ErasedDocument,
        doc_id: DocId,
    ) -> crate::Result<()> {
        self.max_doc = self.max_doc.max(doc_id + 1);
        for (field, value) in doc.erased_fields() {
            if let Some(writer) = self
                .per_field
                .get_mut(field.field_id() as usize)
                .and_then(|w| w.as_mut())
            {
                let slice = match &value {
                    ErasedValue::Leaf(leaf) => leaf.as_vector(),
                    _ => None,
                }
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

    /// `old_doc_ids` is the pre-remap doc id for each output row. `None` keeps insertion order.
    pub(crate) fn serialize(
        &self,
        writer: &mut dyn Write,
        old_doc_ids: Option<&[DocId]>,
    ) -> crate::Result<()> {
        let num_out = old_doc_ids
            .map(|ids| ids.len())
            .unwrap_or(self.max_doc as usize) as DocId;
        let mut bundles: Vec<VectorFieldBundle> = Vec::new();
        for slot in &self.per_field {
            let Some(field_writer) = slot else {
                continue;
            };
            let mut flat: Vec<f32> =
                Vec::with_capacity(num_out as usize * field_writer.options.dimension);
            for out_doc in 0..num_out {
                let doc = old_doc_ids
                    .map(|ids| ids[out_doc as usize])
                    .unwrap_or(out_doc);
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

            normalize_flat_for_cosine(
                &mut flat,
                field_writer.options.dimension,
                field_writer.options.distance,
            );

            let graph = extract_compact_graph(&field_writer.options, num_out, &flat)?;
            bundles.push(VectorFieldBundle {
                field_id: field_writer.field.field_id(),
                options: field_writer.options.clone(),
                num_docs: num_out,
                flat,
                graph,
            });
        }
        write_vec_file(writer, &bundles)?;
        writer.flush()?;
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

//! Vector index as a built-in [`SegmentPlugin`].
//!
//! Indexing goes through [`VectorFieldsWriter`]. Merge rebuilds the HNSW graph from
//! TurboQuant-reconstructed vectors in the merged document order.

use std::any::Any;
use std::collections::BTreeMap;
use std::io::Write;

use common::{HasLen, TerminatingWrite};

use crate::directory::error::OpenReadError;
use crate::index::{SegmentComponent, SegmentReader};
use crate::plugin::{PluginMergeContext, PluginWriter, PluginWriterContext, SegmentPlugin};
use crate::schema::document::ErasedDocument;
use crate::schema::{FieldType, Schema};
use crate::space_usage::ComponentSpaceUsage;
use crate::vector::writer::VectorFieldsWriter;
use crate::vector::{
    build_compact_graph_from_flat, normalize_flat_for_cosine, write_vec_file, VectorFieldBundle,
};
use crate::{DocId, Segment};

pub(crate) const VECTOR_EXT: &str = "vec";

pub(crate) fn vector_component() -> SegmentComponent {
    SegmentComponent::Custom(VECTOR_EXT.to_string())
}

/// Built-in segment plugin that writes and merges `.vec` HNSW indexes.
pub struct VectorPlugin;

impl SegmentPlugin for VectorPlugin {
    fn extensions(&self) -> &[&str] {
        &[VECTOR_EXT]
    }

    fn create_writer(&self, ctx: &PluginWriterContext) -> crate::Result<Box<dyn PluginWriter>> {
        Ok(Box::new(VectorFieldsWriter::for_schema(
            &ctx.segment.schema(),
        )))
    }

    fn merge(&self, ctx: PluginMergeContext) -> crate::Result<()> {
        let mut vec_write = ctx.target_segment.open_write(vector_component())?;
        write_merged_vectors(&ctx, &mut vec_write)?;
        vec_write.terminate()?;
        Ok(())
    }

    fn space_usage(
        &self,
        segment_reader: &SegmentReader,
    ) -> crate::Result<BTreeMap<String, ComponentSpaceUsage>> {
        let bytes = match segment_reader.open_read(vector_component()) {
            Ok(file) => file.len().into(),
            Err(OpenReadError::FileDoesNotExist(_)) => 0u64.into(),
            Err(err) => return Err(err.into()),
        };
        let mut usage = BTreeMap::new();
        usage.insert(VECTOR_EXT.to_string(), ComponentSpaceUsage::Basic(bytes));
        Ok(usage)
    }
}

impl PluginWriter for VectorFieldsWriter {
    fn add_document(
        &mut self,
        doc_id: DocId,
        doc: &dyn ErasedDocument,
        _schema: &Schema,
    ) -> crate::Result<()> {
        VectorFieldsWriter::add_document(self, doc, doc_id)
    }

    fn serialize(
        self: Box<Self>,
        segment: &Segment,
        doc_id_map: Option<&crate::indexer::doc_id_mapping::DocIdMapping>,
    ) -> crate::Result<()> {
        let old_doc_ids = doc_id_map.map(|mapping| mapping.iter_old_doc_ids().collect::<Vec<_>>());
        let mut writer = segment.open_write(vector_component())?;
        VectorFieldsWriter::serialize(&self, &mut writer, old_doc_ids.as_deref())?;
        writer.terminate()?;
        Ok(())
    }

    fn mem_usage(&self) -> usize {
        VectorFieldsWriter::mem_usage(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

fn write_merged_vectors(ctx: &PluginMergeContext, vec_write: &mut dyn Write) -> crate::Result<()> {
    let old_addrs: Vec<_> = ctx.doc_id_mapping.iter_old_doc_addrs().collect();
    let max_doc = old_addrs.len() as u32;
    let mut bundles: Vec<VectorFieldBundle> = Vec::new();

    for (field, field_entry) in ctx.schema.fields() {
        let FieldType::Vector(options) = field_entry.field_type() else {
            continue;
        };
        let dim = options.dimension;
        let segment_flats: Vec<Option<Vec<f32>>> = ctx
            .readers
            .iter()
            .map(|reader| {
                reader
                    .vector_readers()
                    .get(field)
                    .map(|vfr| vfr.flat_vectors())
                    .transpose()
            })
            .collect::<crate::Result<Vec<_>>>()?;

        let mut flat: Vec<f32> = Vec::with_capacity(max_doc as usize * dim);
        for old_addr in &old_addrs {
            let seg_flat = segment_flats[old_addr.segment_ord as usize]
                .as_ref()
                .ok_or_else(|| {
                    crate::TantivyError::DataCorruption(crate::error::DataCorruption::comment_only(
                        format!(
                            "Missing vector reader for field {:?} during merge",
                            field_entry.name()
                        ),
                    ))
                })?;
            let start = old_addr.doc_id as usize * dim;
            let end = start + dim;
            if end > seg_flat.len() {
                return Err(crate::TantivyError::DataCorruption(
                    crate::error::DataCorruption::comment_only(format!(
                        "Missing vector for field {:?} during merge (doc {:?})",
                        field_entry.name(),
                        old_addr
                    )),
                ));
            }
            flat.extend_from_slice(&seg_flat[start..end]);
        }
        normalize_flat_for_cosine(&mut flat, dim, options.distance);
        let graph = build_compact_graph_from_flat(options, max_doc, &flat)?;
        bundles.push(VectorFieldBundle {
            field_id: field.field_id(),
            options: options.clone(),
            num_docs: max_doc,
            flat,
            graph,
        });
    }

    write_vec_file(vec_write, &bundles)?;
    vec_write.flush()?;
    Ok(())
}

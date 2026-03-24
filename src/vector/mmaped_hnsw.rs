//! Vector search index: always [`HnswIo::load_hnsw`] with [`ReloadOptions::new(true)`] so the
//! `hnsw_rs` data file is mmap’d — the same “segment bytes → mmap” model as the rest of Tantivy.
//!
//! When the `.vec` bundle has no graph/data (legacy), we build in memory, `file_dump` into a temp
//! directory, then load through this path so behavior stays one code path.
#![allow(dead_code)] // `io` is only referenced via ouroboros `inner` borrows

use std::fs;
use std::path::Path;

use hnsw_rs::api::AnnT;
use hnsw_rs::hnswio::{HnswIo, ReloadOptions};
use hnsw_rs::prelude::*;
use ouroboros::self_referencing;
use tempfile::TempDir;

use crate::schema::{VectorDistance, VectorOptions};
use crate::vector::hnsw::{build_hnsw_for_flat, BuiltHnsw};
use crate::TantivyError;

#[self_referencing]
pub(crate) struct MmapedHnswL2 {
    dump_dir: TempDir,
    io: HnswIo,
    #[borrows(mut io)]
    #[not_covariant]
    inner: Hnsw<'this, f32, DistL2>,
}

#[self_referencing]
pub(crate) struct MmapedHnswCosine {
    dump_dir: TempDir,
    io: HnswIo,
    #[borrows(mut io)]
    #[not_covariant]
    inner: Hnsw<'this, f32, DistCosine>,
}

#[self_referencing]
pub(crate) struct MmapedHnswDot {
    dump_dir: TempDir,
    io: HnswIo,
    #[borrows(mut io)]
    #[not_covariant]
    inner: Hnsw<'this, f32, DistDot>,
}

/// Search index: mmap-backed reload from an `hnsw_rs` dump on disk (temp dir under the segment
/// reader).
pub(crate) enum VectorIndexInner {
    L2(MmapedHnswL2),
    Cosine(MmapedHnswCosine),
    Dot(MmapedHnswDot),
}

impl VectorIndexInner {
    pub(crate) fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbour> {
        match self {
            Self::L2(l) => l.search(query, k, ef),
            Self::Cosine(l) => l.search(query, k, ef),
            Self::Dot(l) => l.search(query, k, ef),
        }
    }
}

impl MmapedHnswL2 {
    pub(crate) fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbour> {
        self.with_inner(|h| h.search(query, k, ef))
    }
}

impl MmapedHnswCosine {
    pub(crate) fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbour> {
        self.with_inner(|h| h.search(query, k, ef))
    }
}

impl MmapedHnswDot {
    pub(crate) fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<Neighbour> {
        self.with_inner(|h| h.search(query, k, ef))
    }
}

fn write_dump_files(dir: &Path, basename: &str, graph: &[u8], data: &[u8]) -> crate::Result<()> {
    fs::write(dir.join(format!("{basename}.hnsw.graph")), graph)
        .map_err(|e| TantivyError::InternalError(format!("write hnsw graph temp: {e}")))?;
    fs::write(dir.join(format!("{basename}.hnsw.data")), data)
        .map_err(|e| TantivyError::InternalError(format!("write hnsw data temp: {e}")))?;
    Ok(())
}

fn load_mmap_after_dump_on_disk(
    options: &VectorOptions,
    dir: TempDir,
    basename: &str,
) -> crate::Result<VectorIndexInner> {
    let io = HnswIo::new_with_options(dir.path(), basename, ReloadOptions::new(true));
    match options.distance {
        VectorDistance::Euclidean => {
            let mmaped = MmapedHnswL2::try_new(dir, io, |io: &mut HnswIo| {
                let mut h = io
                    .load_hnsw::<f32, DistL2>()
                    .map_err(|e| TantivyError::InternalError(format!("hnsw load (L2): {e}")))?;
                h.set_searching_mode(true);
                Ok::<_, TantivyError>(h)
            })?;
            Ok(VectorIndexInner::L2(mmaped))
        }
        VectorDistance::Cosine => {
            let mmaped = MmapedHnswCosine::try_new(dir, io, |io: &mut HnswIo| {
                let mut h = io
                    .load_hnsw::<f32, DistCosine>()
                    .map_err(|e| TantivyError::InternalError(format!("hnsw load (Cosine): {e}")))?;
                h.set_searching_mode(true);
                Ok::<_, TantivyError>(h)
            })?;
            Ok(VectorIndexInner::Cosine(mmaped))
        }
        VectorDistance::DotProduct => {
            let mmaped = MmapedHnswDot::try_new(dir, io, |io: &mut HnswIo| {
                let mut h = io
                    .load_hnsw::<f32, DistDot>()
                    .map_err(|e| TantivyError::InternalError(format!("hnsw load (Dot): {e}")))?;
                h.set_searching_mode(true);
                Ok::<_, TantivyError>(h)
            })?;
            Ok(VectorIndexInner::Dot(mmaped))
        }
    }
}

/// Open the search index: graph/data bytes from the segment, or build+dump from `flat` (legacy).
pub(crate) fn open_vector_index(
    options: &VectorOptions,
    graph: &[u8],
    data: &[u8],
    max_doc: u32,
    flat: &[f32],
) -> crate::Result<VectorIndexInner> {
    if !graph.is_empty() && !data.is_empty() {
        let dir = tempfile::tempdir().map_err(|e| {
            TantivyError::InternalError(format!("temp dir for hnsw mmap load: {e}"))
        })?;
        let basename = "tntv";
        write_dump_files(dir.path(), basename, graph, data)?;
        return load_mmap_after_dump_on_disk(options, dir, basename);
    }

    // Legacy or missing dump: build from flat, dump to temp, then same mmap reload path.
    if max_doc == 0 || flat.is_empty() {
        return Err(TantivyError::InvalidArgument(
            "vector index: missing graph/data and no vectors to rebuild from".to_string(),
        ));
    }
    let built = build_hnsw_for_flat(options, max_doc, flat)?;
    let dir = tempfile::tempdir()
        .map_err(|e| TantivyError::InternalError(format!("temp dir for hnsw rebuild dump: {e}")))?;
    let basename = match &built {
        BuiltHnsw::L2(h) => h.file_dump(dir.path(), "tntv"),
        BuiltHnsw::Cosine(h) => h.file_dump(dir.path(), "tntv"),
        BuiltHnsw::Dot(h) => h.file_dump(dir.path(), "tntv"),
    }
    .map_err(|e| TantivyError::InternalError(format!("hnsw file_dump (rebuild): {e}")))?;
    load_mmap_after_dump_on_disk(options, dir, &basename)
}

//! K-NN quality regression tests on the real wiki embedding fixture
//! (`benches/wiki_embedded.f32.bin`, `benches/wiki_embedded.meta.json`).
//!
//! These compare [`KnnQuery`] results to brute-force cosine-distance ranking so future
//! storage/compression changes (e.g. quantization) can be checked against a baseline.
//!
//! **Brute-force ground truth** uses the **original full-precision `f32` embeddings** from
//! [`load_fixture_vectors`] (the wiki fixture bytes), **not** the HNSW graph and **not** any
//! lossy representation. After indexing + merge, reconstructed rows from the index are checked
//! for cosine agreement with the fixture (4-bit TurboQuant, not bit-exact).
//!
//! **Note:** With `harness = false`, `benches/vector_wiki.rs` is not built as a test target, so
//! quality checks live here instead of inside the Criterion bench file.

#![cfg(feature = "vector")]

use std::collections::HashSet;

use tantivy::collector::TopDocs;
use tantivy::query::KnnQuery;
use tantivy::schema::{Schema, VectorDistance, VectorOptions};
use tantivy::{doc, Index, IndexWriter, TantivyDocument};

/// Same cosine distance as [`tantivy::vector::io`] (1 − cosine similarity on normalized vectors).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        let dot: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x / norm_a) * (y / norm_b))
            .sum();
        (1.0 - dot).max(0.0)
    } else {
        0.0
    }
}

/// Brute-force top-`k` doc ids by ascending cosine distance (tie-break by doc id).
fn brute_force_top_k_ids(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut scored: Vec<(f32, u32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (cosine_distance(query, v), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn recall_at_k(truth: &[u32], retrieved: &[u32]) -> f32 {
    let ts: HashSet<_> = truth.iter().copied().collect();
    let rs: HashSet<_> = retrieved.iter().copied().collect();
    let inter = ts.intersection(&rs).count();
    inter as f32 / truth.len() as f32
}

const EMBEDDING_DIM: usize = 384;
const NUM_DOCS: usize = 1000;

fn load_fixture_vectors() -> Vec<Vec<f32>> {
    let blob = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/benches/wiki_embedded.f32.bin"
    ));
    assert_eq!(blob.len(), EMBEDDING_DIM * NUM_DOCS * 4);
    let floats: Vec<f32> = blob
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    floats.chunks(EMBEDDING_DIM).map(|c| c.to_vec()).collect()
}

fn schema_and_field() -> (Schema, tantivy::schema::Field) {
    let mut sb = Schema::builder();
    let emb = sb.add_vector_field(
        "emb",
        VectorOptions::new(EMBEDDING_DIM).set_distance(VectorDistance::Cosine),
    );
    (sb.build(), emb)
}

fn build_wiki_index() -> tantivy::Result<(Index, tantivy::schema::Field)> {
    let (schema, field) = schema_and_field();
    let vectors = load_fixture_vectors();
    let index = Index::create_in_ram(schema);
    {
        // Single thread => deterministic DocId assignment and fewer surprises vs multi-threaded
        // writer (see `Index::writer_for_tests` in tantivy).
        let mut w: IndexWriter = index.writer_with_num_threads(1, 100_000_000)?;
        for v in &vectors {
            w.add_document(doc!(field => v.clone()))?;
        }
        w.commit()?;
        w.wait_merging_threads()?;
    }
    // Large commits can flush to multiple segments; merge once so we have one segment and
    // brute-force baselines use the same contiguous `flat_vectors()` table as HNSW (see merge
    // tests).
    {
        let mut w = index.writer_with_num_threads::<TantivyDocument>(1, 100_000_000)?;
        let mut seg_ids = index.searchable_segment_ids()?;
        seg_ids.sort();
        if seg_ids.len() > 1 {
            w.merge(&seg_ids).wait()?;
        }
        w.commit()?;
        w.wait_merging_threads()?;
    }
    Ok((index, field))
}

/// Dequantized `f32` flat from the vector index (TQ4 reconstruction).
fn corpus_rows_from_reader(
    field: tantivy::schema::Field,
    index: &Index,
) -> tantivy::Result<Vec<Vec<f32>>> {
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(
        searcher.segment_readers().len(),
        1,
        "fixture tests assume a single segment; adjust if merge policy changes"
    );
    let seg = searcher.segment_reader(0);
    assert_eq!(
        seg.num_docs(),
        NUM_DOCS as u32,
        "expected all wiki docs in one segment"
    );
    let vread = seg
        .vector_readers()
        .get(field)
        .expect("vector field reader");
    let flat = vread.flat_vectors()?;
    assert_eq!(flat.len(), NUM_DOCS * EMBEDDING_DIM);
    Ok(flat
        .chunks_exact(EMBEDDING_DIM)
        .map(|row| row.to_vec())
        .collect())
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

/// One test builds the index once to avoid running two heavy indexes in parallel (default
/// `cargo test` runs `#[test]` fns concurrently), which was flaky under `--all-features`.
#[test]
fn wiki_knn_quality_against_brute_force_fixture() -> tantivy::Result<()> {
    let original_fixture = load_fixture_vectors();
    let (index, field) = build_wiki_index()?;
    let rows = corpus_rows_from_reader(field, &index)?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg = searcher.segment_reader(0);
    let vread = seg
        .vector_readers()
        .get(field)
        .expect("vector field reader");

    assert_eq!(rows.len(), original_fixture.len());
    for (doc_idx, (row, orig_row)) in rows.iter().zip(original_fixture.iter()).enumerate() {
        let cos = cosine_sim(row, orig_row);
        assert!(
            cos > 0.99,
            "doc {doc_idx}: TQ4 reconstruction cosine {cos} too low"
        );
    }

    for qi in [0usize, 42, 256, 500, 999] {
        let v = vread.vector(qi as u32)?.expect("vector");
        let cos = cosine_sim(&v, &original_fixture[qi]);
        assert!(
            cos > 0.99,
            "vector({qi}): TQ4 reconstruction cosine {cos} too low"
        );
    }

    let k = 10usize;
    let ef = 1024usize;
    let query_indices = [42usize, 128, 333, 500, 750];

    for &qi in &query_indices {
        let query = &original_fixture[qi];
        let truth = brute_force_top_k_ids(query, &original_fixture, k);
        let q = KnnQuery::new(field, query.clone(), k).with_ef_search(ef);
        let top_docs = searcher.search(&q, &TopDocs::with_limit(k).order_by_score())?;
        assert_eq!(top_docs.len(), k, "query index {qi}");
        let retrieved: Vec<u32> = top_docs.iter().map(|(_, addr)| addr.doc_id).collect();
        let recall = recall_at_k(&truth, &retrieved);
        assert!(
            recall >= 1.0,
            "query index {qi}: recall@{k} was {recall} (expected 1.0). truth={truth:?} \
             retrieved={retrieved:?}",
        );
    }
    Ok(())
}

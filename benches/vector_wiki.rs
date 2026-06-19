//! Index and search dense embeddings for [`benches/wiki.json`](wiki.json).
//!
//! When you run this bench, stderr includes the serialized **`.vec` segment file** size
//! for the configured schema, via [`Searcher::space_usage`].
//!
//! **Artifacts** (committed): `wiki_embedded.f32.bin`, `wiki_embedded.meta.json`.
//!
//! **Regenerate** after editing `wiki.json` or changing the model:
//!
//! ```text
//! hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir benches/hf_models/all-MiniLM-L6-v2
//! cd benches/scripts && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
//! python embed_wiki.py
//! ```

use std::sync::Once;

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tantivy::collector::TopDocs;
use tantivy::query::KnnQuery;
use tantivy::schema::{Schema, VectorDistance, VectorOptions};
use tantivy::{doc, ByteCount, Index, IndexWriter};

/// Prints once: total bytes of the `.vec` file(s) for [`schema_and_field`] + [`load_vectors`].
static REPORT_VEC_FILE_SIZE: Once = Once::new();

fn report_vec_file_size_once() {
    REPORT_VEC_FILE_SIZE.call_once(|| {
        let bytes = wiki_vector_vec_file_bytes();
        eprintln!(
            "[vector_wiki] .vec size ({} docs × {} dims, cosine): {} ({} bytes)",
            NUM_DOCS,
            EMBEDDING_DIM,
            bytes,
            bytes.get_bytes(),
        );
    });
}

/// Total size of the vector index (on-disk `.vec` segment component) for the wiki embedding run.
fn wiki_vector_vec_file_bytes() -> ByteCount {
    let (schema, field) = schema_and_field();
    let vectors = load_vectors();
    let index = Index::create_in_ram(schema);
    let mut w = index.writer(100_000_000).unwrap();
    for v in &vectors {
        w.add_document(doc!(field => v.clone())).unwrap();
    }
    w.commit().unwrap();
    w.wait_merging_threads().unwrap();
    let reader = index.reader().unwrap();
    let usage = reader.searcher().space_usage().unwrap();
    usage.segments().iter().map(|s| s.vector_index()).sum()
}

/// Must match [`wiki_embedded.meta.json`](wiki_embedded.meta.json) and `wiki_embedded.f32.bin`.
const EMBEDDING_DIM: usize = 384;
const NUM_DOCS: usize = 1000;

fn embed_blob() -> &'static [u8] {
    include_bytes!("wiki_embedded.f32.bin")
}

fn load_vectors() -> Vec<Vec<f32>> {
    let blob = embed_blob();
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

fn index_wiki_vectors(c: &mut Criterion) {
    report_vec_file_size_once();
    let (schema, field) = schema_and_field();
    let vectors = load_vectors();
    let bytes = embed_blob().len() as u64;

    let mut group = c.benchmark_group("index-wiki-embeddings");
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function("index-no-commit", |b| {
        b.iter(|| {
            let index = Index::create_in_ram(schema.clone());
            let w: IndexWriter = index.writer(100_000_000).unwrap();
            for v in &vectors {
                w.add_document(doc!(field => v.clone())).unwrap();
            }
            black_box(w);
        });
    });

    group.bench_function("index-with-commit", |b| {
        b.iter(|| {
            let index = Index::create_in_ram(schema.clone());
            let mut w: IndexWriter = index.writer(100_000_000).unwrap();
            for v in &vectors {
                w.add_document(doc!(field => v.clone())).unwrap();
            }
            w.commit().unwrap();
        });
    });
}

fn knn_wiki_vectors(c: &mut Criterion) {
    report_vec_file_size_once();
    let (schema, field) = schema_and_field();
    let vectors = load_vectors();
    let index = Index::create_in_ram(schema);
    let mut w = index.writer(100_000_000).unwrap();
    for v in &vectors {
        w.add_document(doc!(field => v.clone())).unwrap();
    }
    w.commit().unwrap();
    w.wait_merging_threads().unwrap();
    let reader = index.reader().unwrap();
    let searcher = reader.searcher();
    let query_vec = vectors[42].clone();
    let q = KnnQuery::new(field, query_vec, 10).with_ef_search(64);

    c.bench_function("knn-top-10-cosine", |b| {
        b.iter(|| {
            black_box(
                searcher
                    .search(&q, &TopDocs::with_limit(10).order_by_score())
                    .unwrap(),
            );
        });
    });
}

criterion_group!(benches, index_wiki_vectors, knn_wiki_vectors);
criterion_main!(benches);

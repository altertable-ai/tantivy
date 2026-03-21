//! Index and search dense embeddings for [`benches/wiki.json`](wiki.json).
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

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tantivy::collector::TopDocs;
use tantivy::query::KnnQuery;
use tantivy::schema::{Schema, VectorDistance, VectorOptions};
use tantivy::{doc, Index, IndexWriter};

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

use crate::collector::TopDocs;
use crate::error::TantivyError;
use crate::query::KnnQuery;
use crate::schema::{Schema, VectorOptions};
use crate::{Index, IndexWriter};

#[test]
fn knn_query_orders_by_similarity() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let emb = schema_builder.add_vector_field("emb", VectorOptions::new(3));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer(15_000_000)?;
    writer.add_document(doc!(emb => vec![1.0f32, 0.0, 0.0]))?;
    writer.add_document(doc!(emb => vec![0.0f32, 1.0, 0.0]))?;
    writer.add_document(doc!(emb => vec![0.0f32, 0.0, 1.0]))?;
    writer.commit()?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query = KnnQuery::new(emb, vec![0.0f32, 1.0, 0.0], 2);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(2).order_by_score())?;
    assert_eq!(top_docs.len(), 2);
    // Closest to [0,1,0] is doc 1, then doc 0 or 2 depending on metric; first must be 1.
    assert_eq!(top_docs[0].1.doc_id, 1);
    Ok(())
}

#[test]
fn merge_segments_rebuilds_vector_index() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let emb = schema_builder.add_vector_field("emb", VectorOptions::new(3));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        writer.add_document(doc!(emb => vec![1.0f32, 0.0, 0.0]))?;
        writer.add_document(doc!(emb => vec![0.9f32, 0.1, 0.0]))?;
        writer.add_document(doc!(emb => vec![0.8f32, 0.2, 0.0]))?;
        writer.commit()?;
    }
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        writer.add_document(doc!(emb => vec![0.0f32, 1.0, 0.0]))?;
        writer.add_document(doc!(emb => vec![0.0f32, 0.9, 0.1]))?;
        writer.add_document(doc!(emb => vec![0.0f32, 0.8, 0.2]))?;
        writer.commit()?;
    }
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        let mut seg_ids = index.searchable_segment_ids()?;
        seg_ids.sort();
        writer.merge(&seg_ids).wait()?;
        writer.commit()?;
        writer.wait_merging_threads()?;
    }
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let seg_reader = searcher.segment_reader(0);
    let vread = seg_reader
        .vector_readers()
        .get(emb)
        .expect("merged segment should load vector index");
    let query = [0.0f32, 1.0, 0.0];
    let mut best_doc = 0u32;
    let mut best_sq = f32::INFINITY;
    for d in 0..seg_reader.num_docs() {
        let v = vread.vector(d)?.expect("vector row");
        let sq: f32 = v
            .iter()
            .zip(query.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if sq < best_sq {
            best_sq = sq;
            best_doc = d;
        }
    }
    let knn = KnnQuery::new(emb, query.to_vec(), 1).with_ef_search(512);
    let top_docs = searcher.search(&knn, &TopDocs::with_limit(1).order_by_score())?;
    assert_eq!(top_docs.len(), 1);
    assert_eq!(
        top_docs[0].1.doc_id, best_doc,
        "k-NN top doc should match brute-force nearest neighbor after merge"
    );
    Ok(())
}

/// HNSW can miss unreachable points; when `k >= n` we brute-force so every doc is returned.
#[test]
fn knn_query_returns_all_docs_when_k_covers_the_segment() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let emb = schema_builder.add_vector_field("emb", VectorOptions::new(3));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer = index.writer_for_tests()?;
    writer.add_document(doc!(emb => vec![1.0f32, 0.0, 0.0]))?;
    writer.add_document(doc!(emb => vec![0.0f32, 1.0, 0.0]))?;
    writer.add_document(doc!(emb => vec![0.0f32, 0.0, 1.0]))?;
    writer.commit()?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.segment_readers().len(), 1);
    assert_eq!(searcher.segment_reader(0).num_docs(), 3);

    let query = KnnQuery::new(emb, vec![0.0f32, 1.0, 0.0], 10).with_ef_search(1);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10).order_by_score())?;
    assert_eq!(top_docs.len(), 3);
    assert_eq!(top_docs[0].1.doc_id, 1);
    let mut ids: Vec<u32> = top_docs.iter().map(|(_, addr)| addr.doc_id).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1, 2]);
    Ok(())
}

#[test]
fn knn_query_rejects_wrong_query_dimension() -> crate::Result<()> {
    let mut schema_builder = Schema::builder();
    let emb = schema_builder.add_vector_field("emb", VectorOptions::new(3));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer(15_000_000)?;
    writer.add_document(doc!(emb => vec![1.0f32, 0.0, 0.0]))?;
    writer.commit()?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query = KnnQuery::new(emb, vec![0.0f32, 1.0], 1);
    let err = searcher
        .search(&query, &TopDocs::with_limit(1).order_by_score())
        .unwrap_err();
    match err {
        TantivyError::InvalidArgument(msg) => {
            assert!(msg.contains("dimension"), "{msg}");
        }
        e => panic!("unexpected error: {e:?}"),
    }
    Ok(())
}

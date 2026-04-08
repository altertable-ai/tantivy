use crate::collector::TopDocs;
use crate::error::TantivyError;
use crate::query::KnnQuery;
use crate::schema::{Schema, VectorOptions};
use crate::vector::io::l2_normalize;
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
    let seg = searcher.segment_reader(0);
    let vread = seg.vector_readers().get(emb).expect("vector reader");
    let query_vec = vec![0.0f32, 1.0, 0.0];
    let mut qn = query_vec.clone();
    l2_normalize(&mut qn);
    let mut best_doc = 0u32;
    let mut best_dist = f32::INFINITY;
    for d in 0..3u32 {
        let v = vread.vector(d)?.expect("row");
        let dot: f32 = qn.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let dist = (1.0f32 - dot).max(0.0f32);
        if dist < best_dist {
            best_dist = dist;
            best_doc = d;
        }
    }
    let query = KnnQuery::new(emb, query_vec, 2).with_ef_search(64);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(2).order_by_score())?;
    assert_eq!(top_docs.len(), 2);
    assert_eq!(
        top_docs[0].1.doc_id, best_doc,
        "top k-NN doc should match brute force on BBQ-reconstructed vectors"
    );
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
    let mut q = query.to_vec();
    l2_normalize(&mut q);
    let mut best_doc = 0u32;
    let mut best_dist = f32::INFINITY;
    for d in 0..seg_reader.num_docs() {
        let v = vread.vector(d)?.expect("vector row");
        let dot: f32 = q.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        let dist = (1.0f32 - dot).max(0.0f32);
        if dist < best_dist || (dist == best_dist && d < best_doc) {
            best_dist = dist;
            best_doc = d;
        }
    }
    let knn = KnnQuery::new(emb, query.to_vec(), 1).with_ef_search(2048);
    let top_docs = searcher.search(&knn, &TopDocs::with_limit(1).order_by_score())?;
    assert_eq!(top_docs.len(), 1);
    let top_id = top_docs[0].1.doc_id;
    let v_top = vread.vector(top_id)?.expect("top row");
    let dot_top: f32 = q.iter().zip(v_top.iter()).map(|(a, b)| a * b).sum();
    let dist_top = (1.0f32 - dot_top).max(0.0f32);
    assert!(
        dist_top <= best_dist + 1e-3,
        "k-NN distance {dist_top} should be near brute best {best_dist} (best_doc={best_doc}, \
         top_id={top_id})"
    );
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

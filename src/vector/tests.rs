use crate::collector::TopDocs;
use crate::directory::FileSlice;
use crate::error::TantivyError;
use crate::query::KnnQuery;
use crate::schema::{Schema, VectorOptions};
use crate::vector::io::{read_vec_file, write_vec_file, CompactHnswGraph, VectorFieldBundle};
use crate::{Index, IndexWriter};

#[test]
fn vec_file_v3_write_read_roundtrip() -> crate::Result<()> {
    let graph = CompactHnswGraph::new(0, 0, 2, vec![vec![vec![]], vec![vec![]]]);
    let bundle = VectorFieldBundle {
        field_id: 7,
        options: VectorOptions::new(2),
        num_docs: 2,
        flat: vec![0.0f32, 0.0, 1.0, 1.0],
        graph,
    };
    let mut buf = Vec::new();
    write_vec_file(&mut buf, &[bundle])?;
    let fields = read_vec_file(FileSlice::from(buf))?;
    assert_eq!(fields.len(), 1);
    assert_eq!(fields[0].field_id, 7);
    assert_eq!(fields[0].num_docs, 2);
    assert_eq!(fields[0].dimension, 2);
    let flat = fields[0]
        .sq
        .dequantize_all(fields[0].quantized_bytes.as_slice(), 2, 2);
    assert_eq!(flat.len(), 4);
    for (a, b) in flat.iter().zip([0.0f32, 0.0, 1.0, 1.0].iter()) {
        assert!((a - b).abs() < 1e-3);
    }
    Ok(())
}

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
    let emb = schema_builder.add_vector_field("emb", VectorOptions::new(2));
    let schema = schema_builder.build();
    let index = Index::create_in_ram(schema);
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        writer.add_document(doc!(emb => vec![1.0f32, 0.0]))?;
        writer.commit()?;
    }
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        writer.add_document(doc!(emb => vec![0.0f32, 1.0]))?;
        writer.commit()?;
    }
    {
        let mut writer: IndexWriter = index.writer(15_000_000)?;
        // Deterministic merge order (API order of segment ids can vary between runs).
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
    let query = [0.0f32, 1.0];
    // Brute-force nearest doc id (squared L2); independent of merge stacking order.
    let mut best_doc = 0u32;
    let mut best_sq = f32::INFINITY;
    for d in 0..seg_reader.num_docs() {
        let v = vread.vector(d).expect("vector").expect("vector row");
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
    // High ef reduces approximate-search variance on tiny graphs.
    let knn = KnnQuery::new(emb, query.to_vec(), 1).with_ef_search(512);
    let top_docs = searcher.search(&knn, &TopDocs::with_limit(1).order_by_score())?;
    assert_eq!(top_docs.len(), 1);
    assert_eq!(
        top_docs[0].1.doc_id, best_doc,
        "k-NN top doc should match brute-force nearest neighbor after merge"
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

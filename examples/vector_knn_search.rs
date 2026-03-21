//! Index small dense vectors and run a k-NN query (`KnnQuery`) against a vector field.

use tantivy::collector::TopDocs;
use tantivy::query::KnnQuery;
use tantivy::schema::{Schema, VectorOptions};
use tantivy::{doc, Index, IndexWriter};

fn main() -> tantivy::Result<()> {
    let mut schema_builder = Schema::builder();
    let vector_field = schema_builder.add_vector_field("vec", VectorOptions::new(4));
    let schema = schema_builder.build();

    let index = Index::create_in_ram(schema);
    let mut writer: IndexWriter = index.writer(15_000_000)?;

    writer.add_document(doc!(vector_field => vec![1.0f32, 0.0, 0.0, 0.0]))?;
    writer.add_document(doc!(vector_field => vec![0.0f32, 1.0, 0.0, 0.0]))?;
    writer.commit()?;

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let query = KnnQuery::new(vector_field, vec![0.0f32, 1.0, 0.0, 0.0], 1);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(1).order_by_score())?;

    println!("Top match doc_id: {}", top_docs[0].1.doc_id);
    Ok(())
}

//! Approximate k-nearest neighbor query over a dense vector field.

use std::fmt;

use crate::docset::TERMINATED;
use crate::index::SegmentReader;
use crate::query::empty_query::EmptyScorer;
use crate::query::{EnableScoring, Explanation, Query, Scorer, Weight};
use crate::schema::Field;
use crate::{DocId, DocSet, Score, TantivyError};

/// k-nearest neighbor search against a [`crate::schema::Field`] of type vector.
///
/// Scores are derived from the distance metric (higher is more similar).
#[derive(Clone)]
pub struct KnnQuery {
    field: Field,
    query_vector: Vec<f32>,
    k: usize,
    ef_search: usize,
}

impl fmt::Debug for KnnQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KnnQuery")
            .field("field", &self.field)
            .field("k", &self.k)
            .field("ef_search", &self.ef_search)
            .field("query_dim", &self.query_vector.len())
            .finish()
    }
}

impl KnnQuery {
    /// Creates a k-NN query. `ef_search` defaults to a sensible value; tune with
    /// [`Self::with_ef_search`].
    pub fn new(field: Field, query_vector: Vec<f32>, k: usize) -> Self {
        Self {
            field,
            query_vector,
            k,
            // Kept for API compatibility; TurboQuant flat-scan search ignores this parameter.
            ef_search: k.max(32) * 2,
        }
    }

    /// Sets the historical graph-search `ef` parameter.
    ///
    /// TurboQuant flat-scan search ignores this parameter.
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Field searched.
    pub fn field(&self) -> Field {
        self.field
    }

    /// Requested neighbor count.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Query for KnnQuery {
    fn weight(&self, enable_scoring: EnableScoring<'_>) -> crate::Result<Box<dyn Weight>> {
        let schema = enable_scoring.schema();
        let field_entry = schema.get_field_entry(self.field);
        if !field_entry.field_type().is_vector() {
            return Err(TantivyError::SchemaError(format!(
                "KnnQuery requires a vector field, got {:?}",
                field_entry.field_type()
            )));
        }
        Ok(Box::new(KnnWeight {
            field: self.field,
            query_vector: self.query_vector.clone(),
            k: self.k,
            ef_search: self.ef_search,
        }))
    }
}

struct KnnWeight {
    field: Field,
    query_vector: Vec<f32>,
    k: usize,
    ef_search: usize,
}

impl Weight for KnnWeight {
    fn scorer(&self, reader: &SegmentReader, boost: Score) -> crate::Result<Box<dyn Scorer>> {
        let Some(vector_reader) = reader.vector_readers().get(self.field) else {
            return Ok(Box::new(EmptyScorer));
        };
        let dim = vector_reader.options.dimension;
        if self.query_vector.len() != dim {
            return Err(TantivyError::InvalidArgument(format!(
                "Query vector dimension {} does not match field dimension {}",
                self.query_vector.len(),
                dim
            )));
        }
        let mut hits = vector_reader.search(
            &self.query_vector,
            self.k,
            self.ef_search,
            reader.alive_bitset(),
        )?;
        hits.sort_by_key(|(d, _)| *d);
        let docs: Vec<DocId> = hits.iter().map(|(d, _)| *d).collect();
        let scores: Vec<Score> = hits.iter().map(|(_, s)| *s * boost).collect();
        Ok(Box::new(KnnScorer::new(docs, scores)))
    }

    fn explain(&self, reader: &SegmentReader, doc: DocId) -> crate::Result<Explanation> {
        let mut scorer = self.scorer(reader, 1.0)?;
        if scorer.seek(doc) != doc {
            return Err(TantivyError::InvalidArgument(format!(
                "Document #{doc} is not among the k-NN results in this segment"
            )));
        }
        Ok(Explanation::new("knn", scorer.score()))
    }
}

struct KnnScorer {
    docs: Vec<DocId>,
    scores: Vec<Score>,
    cursor: usize,
}

impl KnnScorer {
    fn new(docs: Vec<DocId>, scores: Vec<Score>) -> Self {
        debug_assert_eq!(docs.len(), scores.len());
        Self {
            docs,
            scores,
            cursor: 0,
        }
    }
}

impl DocSet for KnnScorer {
    fn advance(&mut self) -> DocId {
        self.cursor += 1;
        if self.cursor >= self.docs.len() {
            self.cursor = self.docs.len();
            return TERMINATED;
        }
        self.doc()
    }

    fn doc(&self) -> DocId {
        if self.cursor >= self.docs.len() {
            return TERMINATED;
        }
        self.docs[self.cursor]
    }

    fn size_hint(&self) -> u32 {
        self.docs.len() as u32
    }
}

impl Scorer for KnnScorer {
    fn score(&mut self) -> Score {
        if self.cursor >= self.scores.len() {
            return 0.0;
        }
        self.scores[self.cursor]
    }
}

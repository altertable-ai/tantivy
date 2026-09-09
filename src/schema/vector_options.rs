//! Options for dense vector fields used with approximate nearest neighbor search.

use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum VectorDistance {
    /// Cosine distance (1 - cosine similarity), as defined in `anndists`.
    #[default]
    Cosine,
    /// Squared L2 (Euclidean) distance.
    Euclidean,
    /// Dot product distance (for normalized vectors, related to cosine).
    DotProduct,
}

/// Configuration for a vector field indexed with HNSW.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorOptions {
    /// Embedding dimension (number of `f32` components per document).
    pub dimension: usize,
    /// Distance metric used for search and index construction.
    #[serde(default)]
    pub distance: VectorDistance,
    /// Store the vector in the doc store for retrieval.
    #[serde(default)]
    pub stored: bool,
    /// HNSW `ef_construction` parameter (insertion search width).
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    /// Maximum number of neighbors per layer in the HNSW graph.
    #[serde(default = "default_max_nb_connection")]
    pub max_nb_connection: usize,
}

fn default_ef_construction() -> usize {
    200
}

fn default_max_nb_connection() -> usize {
    16
}

impl VectorOptions {
    /// Creates options for a vector field with the given fixed dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            distance: VectorDistance::default(),
            stored: false,
            ef_construction: default_ef_construction(),
            max_nb_connection: default_max_nb_connection(),
        }
    }

    /// Sets the distance metric.
    pub fn set_distance(mut self, distance: VectorDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Sets whether the vector is stored in the doc store.
    pub fn set_stored(mut self, stored: bool) -> Self {
        self.stored = stored;
        self
    }

    /// Sets HNSW `ef_construction`.
    pub fn set_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Sets the maximum number of HNSW connections per node.
    pub fn set_max_nb_connection(mut self, m: usize) -> Self {
        self.max_nb_connection = m;
        self
    }

    /// Vector fields are always indexed for k-NN search.
    pub fn is_indexed(&self) -> bool {
        true
    }

    /// Stored in the row-oriented doc store.
    pub fn is_stored(&self) -> bool {
        self.stored
    }
}

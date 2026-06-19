//! Options for dense vector fields used with approximate nearest neighbor search.

use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity search.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum VectorDistance {
    /// Cosine distance (1 - cosine similarity).
    #[default]
    Cosine,
    /// Dot product distance (for normalized vectors, related to cosine).
    DotProduct,
}

/// Configuration for a vector field indexed for k-NN search.
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
    /// TurboQuant bits per coordinate for cosine/dot fields.
    #[serde(default = "default_vector_bit_width")]
    pub bit_width: usize,
}

fn default_vector_bit_width() -> usize {
    4
}

impl VectorOptions {
    /// Creates options for a vector field with the given fixed dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            distance: VectorDistance::default(),
            stored: false,
            bit_width: default_vector_bit_width(),
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

    /// Sets TurboQuant bits per coordinate for cosine/dot fields.
    pub fn set_bit_width(mut self, bit_width: usize) -> Self {
        self.bit_width = bit_width;
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

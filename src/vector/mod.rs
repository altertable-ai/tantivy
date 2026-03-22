//! Dense vector fields and HNSW approximate nearest neighbor search.
//!
//! Enabled by the `vector` crate feature (on by default).

pub(crate) mod hnsw;
mod io;
pub(crate) mod reader;
pub(crate) mod writer;

pub use reader::{VectorFieldReader, VectorFieldReaders};

pub(crate) use hnsw::VectorIndexInner;
pub(crate) use io::{write_vec_file, BytesMaybeMmap, FlatStorage, VectorFieldBundle};
pub(crate) use writer::{build_hnsw_from_flat, VectorFieldsWriter};

#[cfg(test)]
mod tests;

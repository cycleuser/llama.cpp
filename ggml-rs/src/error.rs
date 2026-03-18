//! Error types for ggml-rs

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),

    #[error("Data type mismatch: expected {expected:?}, got {actual:?}")]
    DataTypeMismatch {
        expected: String,
        actual: String,
    },

    #[error("Out of memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory {
        requested: usize,
        available: usize,
    },

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Tensor not contiguous")]
    NotContiguous,

    #[error("Invalid tensor view")]
    InvalidView,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("Context has been freed")]
    ContextFreed,
}

impl Error {
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Error::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    pub fn backend(msg: impl Into<String>) -> Self {
        Error::Backend(msg.into())
    }

    pub fn unsupported(msg: impl Into<String>) -> Self {
        Error::Unsupported(msg.into())
    }
}
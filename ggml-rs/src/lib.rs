//! ggml-rs: A pure Rust implementation of the ggml tensor library
//!
//! This crate provides a safe, idiomatic Rust interface for tensor operations
//! commonly used in machine learning inference, particularly for LLMs.
//!
//! # Features
//! - Pure Rust implementation with zero unsafe in user-facing API
//! - Multiple backend support (CPU, Metal, CUDA, Vulkan)
//! - Quantization support (Q4, Q5, Q8, etc.)
//! - Thread-safe tensor operations
//!
//! # Example
//! ```rust
//! use ggml_rs::{Context, Tensor, DataType};
//!
//! let ctx = Context::new();
//! let a = ctx.new_tensor_2d(DataType::F32, 4, 3)?;
//! let b = ctx.new_tensor_2d(DataType::F32, 3, 5)?;
//! let c = a.matmul(&b)?;
//! ```

pub mod core;
pub mod ops;
pub mod quant;
pub mod backends;
pub mod graph;
pub mod error;

pub use core::{Context, Tensor, DataType, TensorView};
pub use error::{Error, Result};
pub use graph::{Graph, Node};

#[cfg(feature = "metal")]
pub use backends::metal::MetalBackend;

#[cfg(feature = "cuda")]
pub use backends::cuda::CudaBackend;

pub mod prelude {
    //! Common imports
    pub use crate::core::{Context, Tensor, DataType, TensorView, Shape};
    pub use crate::ops::{MatMul, Add, Mul, Scale, Norm};
    pub use crate::quant::{QuantizationType, QuantizedTensor};
    pub use crate::error::{Error, Result};
}

/// Version of the ggml-rs library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimum supported ggml format version
pub const GGML_MIN_FORMAT_VERSION: u32 = 1;
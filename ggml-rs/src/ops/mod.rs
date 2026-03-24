//! Tensor operations for ggml-rs

mod binary;
mod unary;
mod reduce;
mod matmul;
mod activate;

pub use binary::{Add, Sub, Mul, Div};
pub use unary::{Neg, Scale, Abs, Sqrt, Exp, Log};
pub use reduce::{Sum, Mean, Max, Min, SoftMax};
pub use matmul::MatMul;
pub use activate::{GELU, SILU, ReLU, Tanh};
//! Core types and operations

mod tensor;
mod context;
mod data_type;
mod shape;
mod buffer;

pub use tensor::{Tensor, TensorView, TensorMutView};
pub use context::Context;
pub use data_type::{DataType, ElementType};
pub use shape::{Shape, Strides};
pub use buffer::{Buffer, BufferMut};
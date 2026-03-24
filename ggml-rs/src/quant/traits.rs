//! Quantization traits

use crate::core::Tensor;
use crate::error::Result;

pub trait QuantizationType: Clone + Copy + std::fmt::Debug {
    fn block_size() -> usize;
    fn block_bytes() -> usize;
}

pub trait QuantizedTensor {
    fn dtype(&self) -> crate::core::DataType;
    fn block_count(&self) -> usize;
    fn as_bytes(&self) -> &[u8];
}

pub trait Quantize {
    fn quantize(data: &[f32]) -> Result<Vec<u8>>;
}

pub trait Dequantize {
    fn dequantize(bytes: &[u8], nblocks: usize) -> Result<Vec<f32>>;
}
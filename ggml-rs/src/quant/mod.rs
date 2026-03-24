//! Quantization types and operations

mod q4_0;
mod q4_1;
mod q8_0;
mod traits;

pub use traits::{QuantizationType, QuantizedTensor, Dequantize, Quantize};
pub use q4_0::BlockQ4_0;
pub use q4_1::BlockQ4_1;
pub use q8_0::BlockQ8_0;

use crate::core::DataType;
use crate::error::Result;

pub fn block_size(dtype: DataType) -> usize {
    match dtype {
        DataType::Q4_0 => q4_0::QK4_0,
        DataType::Q4_1 => q4_1::QK4_1,
        DataType::Q8_0 => q8_0::QK8_0,
        _ => 0,
    }
}

pub fn block_size_bytes(dtype: DataType) -> usize {
    match dtype {
        DataType::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
        DataType::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
        DataType::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
        _ => 0,
    }
}

pub fn quantized_size(nelements: usize, dtype: DataType) -> usize {
    let bs = block_size(dtype);
    if bs == 0 {
        return nelements * dtype.size_in_bytes();
    }
    let nblocks = (nelements + bs - 1) / bs;
    nblocks * block_size_bytes(dtype)
}
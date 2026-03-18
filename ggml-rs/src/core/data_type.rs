//! Data types supported by ggml

use std::fmt;
use half::{f16, bf16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    I8,
    I16,
    I32,
    I64,
    F64,
}

impl DataType {
    #[inline]
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 => 2,
            DataType::BF16 => 2,
            DataType::Q4_0 => 0,
            DataType::Q4_1 => 0,
            DataType::Q5_0 => 0,
            DataType::Q5_1 => 0,
            DataType::Q8_0 => 0,
            DataType::Q8_1 => 0,
            DataType::Q2_K => 0,
            DataType::Q3_K => 0,
            DataType::Q4_K => 0,
            DataType::Q5_K => 0,
            DataType::Q6_K => 0,
            DataType::Q8_K => 0,
            DataType::I8 => 1,
            DataType::I16 => 2,
            DataType::I32 => 4,
            DataType::I64 => 8,
            DataType::F64 => 8,
        }
    }

    #[inline]
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            DataType::Q4_0
                | DataType::Q4_1
                | DataType::Q5_0
                | DataType::Q5_1
                | DataType::Q8_0
                | DataType::Q8_1
                | DataType::Q2_K
                | DataType::Q3_K
                | DataType::Q4_K
                | DataType::Q5_K
                | DataType::Q6_K
                | DataType::Q8_K
        )
    }

    #[inline]
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64
        )
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64
        )
    }

    pub fn name(&self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::Q4_0 => "q4_0",
            DataType::Q4_1 => "q4_1",
            DataType::Q5_0 => "q5_0",
            DataType::Q5_1 => "q5_1",
            DataType::Q8_0 => "q8_0",
            DataType::Q8_1 => "q8_1",
            DataType::Q2_K => "q2_k",
            DataType::Q3_K => "q3_k",
            DataType::Q4_K => "q4_k",
            DataType::Q5_K => "q5_k",
            DataType::Q6_K => "q6_k",
            DataType::Q8_K => "q8_k",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::F64 => "f64",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ElementType {
    F32(f32),
    F16(f16),
    BF16(bf16),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F64(f64),
}

impl From<f32> for ElementType {
    fn from(v: f32) -> Self {
        ElementType::F32(v)
    }
}

impl From<i32> for ElementType {
    fn from(v: i32) -> Self {
        ElementType::I32(v)
    }
}
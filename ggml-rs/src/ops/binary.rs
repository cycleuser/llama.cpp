//! Binary operations

use crate::core::{DataType, Shape, Tensor};
use crate::error::{Error, Result};

pub struct Add;
pub struct Sub;
pub struct Mul;
pub struct Div;

trait BinaryOp {
    fn name() -> &'static str;
    fn compute(a: f32, b: f32) -> f32;
}

impl BinaryOp for Add {
    #[inline] fn name() -> &'static str { "add" }
    #[inline] fn compute(a: f32, b: f32) -> f32 { a + b }
}

impl BinaryOp for Sub {
    #[inline] fn name() -> &'static str { "sub" }
    #[inline] fn compute(a: f32, b: f32) -> f32 { a - b }
}

impl BinaryOp for Mul {
    #[inline] fn name() -> &'static str { "mul" }
    #[inline] fn compute(a: f32, b: f32) -> f32 { a * b }
}

impl BinaryOp for Div {
    #[inline] fn name() -> &'static str { "div" }
    #[inline] fn compute(a: f32, b: f32) -> f32 { a / b }
}

fn binary_op<Op: BinaryOp>(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() != b.dtype() {
        return Err(Error::DataTypeMismatch {
            expected: a.dtype().name().to_string(),
            actual: b.dtype().name().to_string(),
        });
    }

    if !a.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "Binary ops require float types, got {}",
            a.dtype()
        )));
    }

    let shape = a.shape().broadcast_shape(b.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape().dims(), b.shape().dims()))?;

    let result = Tensor::new(shape.clone(), a.dtype())?;

    match a.dtype() {
        DataType::F32 => {
            let a_data = a.as_slice::<f32>()?;
            let b_data = b.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            if a.shape() == &shape && b.shape() == &shape {
                for (r, (&av, &bv)) in r_slice.iter_mut().zip(a_data.iter().zip(b_data.iter())) {
                    *r = Op::compute(av, bv);
                }
            } else if a.shape() == &shape {
                for (i, r) in r_slice.iter_mut().enumerate() {
                    let bi = i % b.nelements();
                    *r = Op::compute(a_data[i], b_data[bi]);
                }
            } else if b.shape() == &shape {
                for (i, r) in r_slice.iter_mut().enumerate() {
                    let ai = i % a.nelements();
                    *r = Op::compute(a_data[ai], b_data[i]);
                }
            } else {
                for (i, r) in r_slice.iter_mut().enumerate() {
                    let ai = i % a.nelements();
                    let bi = i % b.nelements();
                    *r = Op::compute(a_data[ai], b_data[bi]);
                }
            }

            Tensor::from_data(shape, a.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", a.dtype()))),
    }
}

pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op::<Add>(a, b)
}

pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op::<Sub>(a, b)
}

pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op::<Mul>(a, b)
}

pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op::<Div>(a, b)
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        add(self, other)
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        sub(self, other)
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        mul(self, other)
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        div(self, other)
    }
}
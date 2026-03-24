//! Unary operations

use crate::core::{DataType, Tensor};
use crate::error::{Error, Result};

pub struct Neg;
pub struct Scale;
pub struct Abs;
pub struct Sqrt;
pub struct Exp;
pub struct Log;

trait UnaryOp {
    fn compute(x: f32) -> f32;
}

impl UnaryOp for Neg {
    #[inline] fn compute(x: f32) -> f32 { -x }
}

impl UnaryOp for Abs {
    #[inline] fn compute(x: f32) -> f32 { x.abs() }
}

impl UnaryOp for Sqrt {
    #[inline] fn compute(x: f32) -> f32 { x.sqrt() }
}

impl UnaryOp for Exp {
    #[inline] fn compute(x: f32) -> f32 { x.exp() }
}

impl UnaryOp for Log {
    #[inline] fn compute(x: f32) -> f32 { x.ln() }
}

fn unary_op<Op: UnaryOp>(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "Unary ops require float types, got {}",
            tensor.dtype()
        )));
    }

    let result = Tensor::new(tensor.shape().clone(), tensor.dtype())?;

    match tensor.dtype() {
        DataType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            for (r, &v) in r_slice.iter_mut().zip(data.iter()) {
                *r = Op::compute(v);
            }

            Tensor::from_data(tensor.shape().clone(), tensor.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", tensor.dtype()))),
    }
}

pub fn neg(tensor: &Tensor) -> Result<Tensor> {
    unary_op::<Neg>(tensor)
}

pub fn abs(tensor: &Tensor) -> Result<Tensor> {
    unary_op::<Abs>(tensor)
}

pub fn sqrt(tensor: &Tensor) -> Result<Tensor> {
    unary_op::<Sqrt>(tensor)
}

pub fn exp(tensor: &Tensor) -> Result<Tensor> {
    unary_op::<Exp>(tensor)
}

pub fn log(tensor: &Tensor) -> Result<Tensor> {
    unary_op::<Log>(tensor)
}

pub fn scale(tensor: &Tensor, factor: f32) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "Scale requires float types, got {}",
            tensor.dtype()
        )));
    }

    let result = Tensor::new(tensor.shape().clone(), tensor.dtype())?;

    match tensor.dtype() {
        DataType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            for (r, &v) in r_slice.iter_mut().zip(data.iter()) {
                *r = v * factor;
            }

            Tensor::from_data(tensor.shape().clone(), tensor.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", tensor.dtype()))),
    }
}

impl Tensor {
    pub fn neg(&self) -> Result<Tensor> {
        neg(self)
    }

    pub fn abs(&self) -> Result<Tensor> {
        abs(self)
    }

    pub fn sqrt(&self) -> Result<Tensor> {
        sqrt(self)
    }

    pub fn exp(&self) -> Result<Tensor> {
        exp(self)
    }

    pub fn log(&self) -> Result<Tensor> {
        log(self)
    }

    pub fn scale(&self, factor: f32) -> Result<Tensor> {
        scale(self, factor)
    }
}
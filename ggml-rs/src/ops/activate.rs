//! Activation functions

use crate::core::{DataType, Tensor};
use crate::error::{Error, Result};

pub struct GELU;
pub struct SILU;
pub struct ReLU;
pub struct Tanh;

fn gelu(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    let x3 = x * x * x;
    x * 0.5 * (1.0 + ((SQRT_2_OVER_PI * (x + 0.044715 * x3)).tanh()))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

fn tanh_fn(x: f32) -> f32 {
    x.tanh()
}

trait ActivationFn {
    fn compute(x: f32) -> f32;
    fn name() -> &'static str;
}

impl ActivationFn for GELU {
    #[inline] fn compute(x: f32) -> f32 { gelu(x) }
    #[inline] fn name() -> &'static str { "gelu" }
}

impl ActivationFn for SILU {
    #[inline] fn compute(x: f32) -> f32 { silu(x) }
    #[inline] fn name() -> &'static str { "silu" }
}

impl ActivationFn for ReLU {
    #[inline] fn compute(x: f32) -> f32 { relu(x) }
    #[inline] fn name() -> &'static str { "relu" }
}

impl ActivationFn for Tanh {
    #[inline] fn compute(x: f32) -> f32 { tanh_fn(x) }
    #[inline] fn name() -> &'static str { "tanh" }
}

fn activation<Op: ActivationFn>(tensor: &Tensor) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "{} requires float types, got {}",
            Op::name(),
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

pub fn gelu(tensor: &Tensor) -> Result<Tensor> {
    activation::<GELU>(tensor)
}

pub fn silu(tensor: &Tensor) -> Result<Tensor> {
    activation::<SILU>(tensor)
}

pub fn relu(tensor: &Tensor) -> Result<Tensor> {
    activation::<ReLU>(tensor)
}

pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    activation::<Tanh>(tensor)
}

impl Tensor {
    pub fn gelu(&self) -> Result<Tensor> {
        gelu(self)
    }

    pub fn silu(&self) -> Result<Tensor> {
        silu(self)
    }

    pub fn relu(&self) -> Result<Tensor> {
        relu(self)
    }

    pub fn tanh(&self) -> Result<Tensor> {
        tanh(self)
    }
}
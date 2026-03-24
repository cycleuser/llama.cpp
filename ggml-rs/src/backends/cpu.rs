//! CPU backend implementation

use crate::core::Tensor;
use crate::error::{Error, Result};
use crate::ops;
use crate::backends::{Backend, ComputeBackend};

pub struct CpuBackend {
    num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend {
            num_threads: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
        }
    }

    pub fn with_threads(num_threads: usize) -> Self {
        CpuBackend { num_threads }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn device_count(&self) -> usize {
        1
    }

    fn memory_available(&self, _device: usize) -> usize {
        usize::MAX
    }

    fn memory_used(&self, _device: usize) -> usize {
        0
    }
}

impl ComputeBackend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Result<Tensor> {
        ops::matmul(a, b, transpose_a, transpose_b)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::add(a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::mul(a, b)
    }

    fn scale(&self, tensor: &Tensor, factor: f32) -> Result<Tensor> {
        ops::scale(tensor, factor)
    }

    fn softmax(&self, tensor: &Tensor, axis: usize) -> Result<Tensor> {
        ops::softmax(tensor, axis)
    }

    fn gelu(&self, tensor: &Tensor) -> Result<Tensor> {
        ops::gelu(tensor)
    }

    fn silu(&self, tensor: &Tensor) -> Result<Tensor> {
        ops::silu(tensor)
    }

    fn norm(&self, tensor: &Tensor, eps: f32) -> Result<Tensor> {
        if tensor.ndim() < 1 {
            return Err(Error::InvalidDimension("Norm requires at least 1D tensor".into()));
        }

        let last_dim = tensor.shape()[tensor.ndim() - 1];
        let mut result = Tensor::new(tensor.shape().clone(), tensor.dtype())?;

        match tensor.dtype() {
            crate::core::DataType::F32 => {
                let data = tensor.as_slice::<f32>()?;
                let r_data = result.as_bytes();
                let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data.to_vec().into_boxed_slice());

                let num_rows = tensor.nelements() / last_dim;
                for row in 0..num_rows {
                    let row_start = row * last_dim;

                    let mean: f32 = data[row_start..row_start + last_dim].iter().sum::<f32>() / last_dim as f32;
                    let var: f32 = data[row_start..row_start + last_dim]
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>() / last_dim as f32;

                    let std = (var + eps).sqrt();
                    for (i, &v) in data[row_start..row_start + last_dim].iter().enumerate() {
                        r_slice[row_start + i] = (v - mean) / std;
                    }
                }
            }
            _ => return Err(Error::unsupported(format!("Norm not implemented for {:?}", tensor.dtype()))),
        }

        Ok(result)
    }

    fn rms_norm(&self, tensor: &Tensor, eps: f32) -> Result<Tensor> {
        if tensor.ndim() < 1 {
            return Err(Error::InvalidDimension("RMSNorm requires at least 1D tensor".into()));
        }

        let last_dim = tensor.shape()[tensor.ndim() - 1];
        let mut result = Tensor::new(tensor.shape().clone(), tensor.dtype())?;

        match tensor.dtype() {
            crate::core::DataType::F32 => {
                let data = tensor.as_slice::<f32>()?;
                let r_data = result.as_bytes();
                let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data.to_vec().into_boxed_slice());

                let num_rows = tensor.nelements() / last_dim;
                for row in 0..num_rows {
                    let row_start = row * last_dim;

                    let ss: f32 = data[row_start..row_start + last_dim]
                        .iter()
                        .map(|&x| x * x)
                        .sum();

                    let rms = (ss / last_dim as f32 + eps).sqrt();
                    let inv_rms = 1.0 / rms;

                    for (i, &v) in data[row_start..row_start + last_dim].iter().enumerate() {
                        r_slice[row_start + i] = v * inv_rms;
                    }
                }
            }
            _ => return Err(Error::unsupported(format!("RMSNorm not implemented for {:?}", tensor.dtype()))),
        }

        Ok(result)
    }
}
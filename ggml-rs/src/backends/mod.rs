//! Backend implementations for tensor operations

mod cpu;

pub use cpu::CpuBackend;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

use crate::core::Tensor;
use crate::error::Result;

pub trait Backend {
    fn name(&self) -> &'static str;
    fn device_count(&self) -> usize;
    fn memory_available(&self, device: usize) -> usize;
    fn memory_used(&self, device: usize) -> usize;
}

pub trait ComputeBackend: Backend {
    fn matmul(&self, a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Result<Tensor>;
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;
    fn scale(&self, tensor: &Tensor, factor: f32) -> Result<Tensor>;
    fn softmax(&self, tensor: &Tensor, axis: usize) -> Result<Tensor>;
    fn gelu(&self, tensor: &Tensor) -> Result<Tensor>;
    fn silu(&self, tensor: &Tensor) -> Result<Tensor>;
    fn norm(&self, tensor: &Tensor, eps: f32) -> Result<Tensor>;
    fn rms_norm(&self, tensor: &Tensor, eps: f32) -> Result<Tensor>;
}

pub fn get_best_backend() -> Box<dyn Backend> {
    #[cfg(feature = "metal")]
    {
        if let Ok(backend) = metal::MetalBackend::new() {
            return Box::new(backend);
        }
    }

    #[cfg(feature = "cuda")]
    {
        if let Ok(backend) = cuda::CudaBackend::new() {
            return Box::new(backend);
        }
    }

    Box::new(CpuBackend::new())
}
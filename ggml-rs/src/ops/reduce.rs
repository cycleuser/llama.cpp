//! Reduction operations

use crate::core::{DataType, Shape, Tensor};
use crate::error::{Error, Result};

pub struct Sum;
pub struct Mean;
pub struct Max;
pub struct Min;
pub struct SoftMax;

pub fn sum(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    reduce_op::<Sum>(tensor, axes)
}

pub fn mean(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    reduce_op::<Mean>(tensor, axes)
}

pub fn max(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    reduce_op::<Max>(tensor, axes)
}

pub fn min(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    reduce_op::<Min>(tensor, axes)
}

pub fn softmax(tensor: &Tensor, axis: usize) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "Softmax requires float types, got {}",
            tensor.dtype()
        )));
    }

    let axis = if axis < 0 { tensor.ndim() + axis as usize } else { axis };
    if axis >= tensor.ndim() {
        return Err(Error::InvalidDimension(format!("Axis {} out of bounds", axis)));
    }

    let result = Tensor::new(tensor.shape().clone(), tensor.dtype())?;

    match tensor.dtype() {
        DataType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            let axis_dim = tensor.shape()[axis];
            let outer_size: usize = tensor.shape().dims()[..axis].iter().product();
            let inner_size: usize = tensor.shape().dims()[(axis + 1)..].iter().product();
            let axis_stride = inner_size;
            let chunk_size = axis_dim * inner_size;

            for outer in 0..outer_size {
                let base = outer * chunk_size;
                for inner in 0..inner_size {
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..axis_dim {
                        let idx = base + i * axis_stride + inner;
                        max_val = max_val.max(data[idx]);
                    }

                    let mut sum_exp = 0.0f32;
                    for i in 0..axis_dim {
                        let idx = base + i * axis_stride + inner;
                        let exp_val = (data[idx] - max_val).exp();
                        sum_exp += exp_val;
                        r_slice[idx] = exp_val;
                    }

                    for i in 0..axis_dim {
                        let idx = base + i * axis_stride + inner;
                        r_slice[idx] /= sum_exp;
                    }
                }
            }

            Tensor::from_data(tensor.shape().clone(), tensor.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", tensor.dtype()))),
    }
}

trait ReduceOp {
    fn init() -> f32;
    fn accumulate(acc: f32, val: f32) -> f32;
    fn finalize(acc: f32, count: usize) -> f32;
}

impl ReduceOp for Sum {
    #[inline] fn init() -> f32 { 0.0 }
    #[inline] fn accumulate(acc: f32, val: f32) -> f32 { acc + val }
    #[inline] fn finalize(acc: f32, _count: usize) -> f32 { acc }
}

impl ReduceOp for Mean {
    #[inline] fn init() -> f32 { 0.0 }
    #[inline] fn accumulate(acc: f32, val: f32) -> f32 { acc + val }
    #[inline] fn finalize(acc: f32, count: usize) -> f32 { acc / count as f32 }
}

impl ReduceOp for Max {
    #[inline] fn init() -> f32 { f32::NEG_INFINITY }
    #[inline] fn accumulate(acc: f32, val: f32) -> f32 { acc.max(val) }
    #[inline] fn finalize(acc: f32, _count: usize) -> f32 { acc }
}

impl ReduceOp for Min {
    #[inline] fn init() -> f32 { f32::INFINITY }
    #[inline] fn accumulate(acc: f32, val: f32) -> f32 { acc.min(val) }
    #[inline] fn finalize(acc: f32, _count: usize) -> f32 { acc }
}

fn reduce_op<Op: ReduceOp>(tensor: &Tensor, axes: Option<&[usize]>) -> Result<Tensor> {
    if !tensor.dtype().is_float() {
        return Err(Error::unsupported(format!(
            "Reduce ops require float types, got {}",
            tensor.dtype()
        )));
    }

    let axes: Vec<usize> = axes
        .map(|a| a.to_vec())
        .unwrap_or_else(|| (0..tensor.ndim()).collect());

    let mut out_dims: Vec<usize> = tensor.shape().dims().to_vec();
    for &axis in &axes {
        out_dims[axis] = 1;
    }
    let out_shape = Shape::new(&out_dims);

    let result = Tensor::new(out_shape.clone(), tensor.dtype())?;

    match tensor.dtype() {
        DataType::F32 => {
            let data = tensor.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            let mut out_idx = 0;
            for i in 0..tensor.nelements() {
                let mut in_coords = vec![0usize; tensor.ndim()];
                let mut temp = i;
                for d in (0..tensor.ndim()).rev() {
                    in_coords[d] = temp % tensor.shape()[d];
                    temp /= tensor.shape()[d];
                }

                let mut out_coords = in_coords.clone();
                for &axis in &axes {
                    out_coords[axis] = 0;
                }

                let out_flat = coords_to_flat(&out_coords, &out_shape);

                if i == 0 || out_flat != out_idx {
                    if i > 0 {
                        r_slice[out_idx] = Op::finalize(r_slice[out_idx], tensor.nelements() / r_slice.len());
                    }
                    out_idx = out_flat;
                    r_slice[out_idx] = Op::init();
                }
                r_slice[out_idx] = Op::accumulate(r_slice[out_idx], data[i]);
            }
            if r_slice.len() > 0 {
                r_slice[out_idx] = Op::finalize(r_slice[out_idx], tensor.nelements() / r_slice.len());
            }

            Tensor::from_data(out_shape, tensor.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", tensor.dtype()))),
    }
}

fn coords_to_flat(coords: &[usize], shape: &Shape) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for i in (0..coords.len()).rev() {
        idx += coords[i] * stride;
        stride *= shape[i];
    }
    idx
}

impl Tensor {
    pub fn sum(&self, axes: Option<&[usize]>) -> Result<Tensor> {
        sum(self, axes)
    }

    pub fn mean(&self, axes: Option<&[usize]>) -> Result<Tensor> {
        mean(self, axes)
    }

    pub fn max(&self, axes: Option<&[usize]>) -> Result<Tensor> {
        max(self, axes)
    }

    pub fn min(&self, axes: Option<&[usize]>) -> Result<Tensor> {
        min(self, axes)
    }

    pub fn softmax(&self, axis: usize) -> Result<Tensor> {
        softmax(self, axis)
    }
}
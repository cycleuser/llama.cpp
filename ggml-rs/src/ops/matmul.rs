//! Matrix multiplication operations

use crate::core::{DataType, Shape, Tensor};
use crate::error::{Error, Result};

pub struct MatMul;

pub fn matmul(a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Result<Tensor> {
    if !a.dtype().is_float() || !b.dtype().is_float() {
        return Err(Error::unsupported("MatMul requires float types"));
    }

    if a.dtype() != b.dtype() {
        return Err(Error::DataTypeMismatch {
            expected: a.dtype().name().to_string(),
            actual: b.dtype().name().to_string(),
        });
    }

    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    if a_ndim < 1 || b_ndim < 1 {
        return Err(Error::InvalidDimension("MatMul requires at least 1D tensors".into()));
    }

    let (a_rows, a_cols) = if transpose_a {
        (a.shape()[a_ndim - 1], if a_ndim > 1 { a.shape()[a_ndim - 2] } else { 1 })
    } else {
        (if a_ndim > 1 { a.shape()[a_ndim - 2] } else { 1 }, a.shape()[a_ndim - 1])
    };

    let (b_rows, b_cols) = if transpose_b {
        (b.shape()[b_ndim - 1], if b_ndim > 1 { b.shape()[b_ndim - 2] } else { 1 })
    } else {
        (if b_ndim > 1 { b.shape()[b_ndim - 2] } else { 1 }, b.shape()[b_ndim - 1])
    };

    if a_cols != b_rows {
        return Err(Error::shape_mismatch(&[a_cols, b_rows], &[a.shape().dims().to_vec(), b.shape().dims().to_vec()].concat()));
    }

    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    let mut out_dims: Vec<usize> = Vec::new();
    
    let max_batch_ndim = a_ndim.saturating_sub(2).max(b_ndim.saturating_sub(2));
    for i in 0..max_batch_ndim {
        let a_dim = if i < a_ndim.saturating_sub(2) {
            a.shape()[i]
        } else {
            1
        };
        let b_dim = if i < b_ndim.saturating_sub(2) {
            b.shape()[i]
        } else {
            1
        };
        out_dims.push(a_dim.max(b_dim));
    }

    if a_ndim > 1 {
        out_dims.push(m);
    }
    out_dims.push(n);

    if out_dims.is_empty() {
        out_dims.push(1);
    }

    let out_shape = Shape::new(&out_dims);
    let batch_size: usize = out_dims[..out_dims.len().saturating_sub(2)].iter().product().max(1);

    let result = Tensor::new(out_shape.clone(), a.dtype())?;

    match a.dtype() {
        DataType::F32 => {
            let a_data = a.as_slice::<f32>()?;
            let b_data = b.as_slice::<f32>()?;
            let mut r_data = result.as_bytes().to_vec();
            let r_slice = bytemuck::cast_slice_mut::<u8, f32>(&mut r_data);

            let a_batch_stride = if a_ndim > 2 { a.shape()[a_ndim - 2] * a.shape()[a_ndim - 1] } else { 0 };
            let b_batch_stride = if b_ndim > 2 { b.shape()[b_ndim - 2] * b.shape()[b_ndim - 1] } else { 0 };

            for batch in 0..batch_size {
                let a_offset = batch * a_batch_stride;
                let b_offset = batch * b_batch_stride;
                let r_offset = batch * m * n;

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            let a_val = if transpose_a {
                                a_data[a_offset + l * a_rows + i]
                            } else {
                                a_data[a_offset + i * k + l]
                            };
                            let b_val = if transpose_b {
                                b_data[b_offset + j * b_rows + l]
                            } else {
                                b_data[b_offset + l * n + j]
                            };
                            sum += a_val * b_val;
                        }
                        r_slice[r_offset + i * n + j] = sum;
                    }
                }
            }

            Tensor::from_data(out_shape, a.dtype(), r_data)
        }
        _ => Err(Error::unsupported(format!("Unsupported dtype: {}", a.dtype()))),
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        matmul(self, other, false, false)
    }

    pub fn matmul_t(&self, other: &Tensor, transpose_a: bool, transpose_b: bool) -> Result<Tensor> {
        matmul(self, other, transpose_a, transpose_b)
    }
}
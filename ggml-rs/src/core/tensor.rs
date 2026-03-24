//! Tensor types for ggml-rs

use crate::core::{BufferRef, DataType, Shape, Strides};
use crate::error::{Error, Result};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Tensor {
    data: BufferRef,
    shape: Shape,
    strides: Strides,
    dtype: DataType,
    name: Option<Arc<str>>,
}

impl Tensor {
    pub fn new(shape: Shape, dtype: DataType) -> Result<Self> {
        let element_size = if dtype.is_quantized() {
            return Err(Error::unsupported(format!(
                "Quantized type {} requires special construction",
                dtype
            )));
        } else {
            dtype.size_in_bytes()
        };

        let size = shape.nelements() * element_size;
        let data = BufferRef::from_slice(&vec![0u8; size]);
        let strides = Strides::from_shape(&shape);

        Ok(Tensor {
            data,
            shape,
            strides,
            dtype,
            name: None,
        })
    }

    pub fn from_data(shape: Shape, dtype: DataType, data: Vec<u8>) -> Result<Self> {
        let element_size = dtype.size_in_bytes();
        let expected_size = shape.nelements() * element_size;

        if data.len() != expected_size && !dtype.is_quantized() {
            return Err(Error::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }

        let strides = Strides::from_shape(&shape);
        let data = BufferRef::from_slice(&data);

        Ok(Tensor {
            data,
            shape,
            strides,
            dtype,
            name: None,
        })
    }

    pub fn from_slice<T: bytemuck::Pod + Clone>(data: &[T], shape: Shape) -> Result<Self> {
        let expected_len = shape.nelements();
        if data.len() != expected_len {
            return Err(Error::ShapeMismatch {
                expected: vec![expected_len],
                actual: vec![data.len()],
            });
        }

        let dtype = match std::mem::size_of::<T>() {
            1 => DataType::I8,
            2 => DataType::F16,
            4 => DataType::F32,
            8 => DataType::F64,
            _ => return Err(Error::unsupported("Unsupported element type")),
        };

        let bytes = bytemuck::cast_slice(data);
        let strides = Strides::from_shape(&shape);

        Ok(Tensor {
            data: BufferRef::from_slice(bytes),
            shape,
            strides,
            dtype,
            name: None,
        })
    }

    #[inline]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    #[inline]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn set_name(&mut self, name: impl Into<Arc<str>>) {
        self.name = Some(name.into());
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    #[inline]
    pub fn nelements(&self) -> usize {
        self.shape.nelements()
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.data.size()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.strides.is_contiguous(&self.shape)
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    #[inline]
    pub fn is_vector(&self) -> bool {
        self.shape.is_vector()
    }

    #[inline]
    pub fn is_matrix(&self) -> bool {
        self.shape.is_matrix()
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    #[inline]
    pub fn data_size(&self) -> usize {
        self.data.size()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_slice()
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> Result<&[T]> {
        if std::mem::size_of::<T>() != self.dtype.size_in_bytes() {
            return Err(Error::DataTypeMismatch {
                expected: self.dtype.name().to_string(),
                actual: format!("{} bytes", std::mem::size_of::<T>()),
            });
        }
        Ok(self.data.as_slice_t())
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView {
            data: self.data.as_ptr(),
            shape: &self.shape,
            strides: &self.strides,
            dtype: self.dtype,
        }
    }

    pub fn reshape(&self, shape: Shape) -> Result<Self> {
        if shape.nelements() != self.shape.nelements() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape.nelements()],
                actual: vec![shape.nelements()],
            });
        }

        let strides = Strides::from_shape(&shape);
        Ok(Tensor {
            data: self.data.clone(),
            shape,
            strides,
            dtype: self.dtype,
            name: self.name.clone(),
        })
    }

    pub fn broadcast_to(&self, shape: Shape) -> Result<Self> {
        let broadcast_shape = self
            .shape
            .broadcast_shape(&shape)
            .ok_or_else(|| Error::InvalidOperation("Cannot broadcast shapes".into()))?;

        if broadcast_shape != shape {
            return Err(Error::ShapeMismatch {
                expected: shape.dims().to_vec(),
                actual: broadcast_shape.dims().to_vec(),
            });
        }

        let strides = Strides::from_shape(&self.shape);
        Ok(Tensor {
            data: self.data.clone(),
            shape: broadcast_shape,
            strides,
            dtype: self.dtype,
            name: self.name.clone(),
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    data: *const u8,
    shape: &'a Shape,
    strides: &'a Strides,
    dtype: DataType,
}

impl<'a> TensorView<'a> {
    #[inline]
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &Strides {
        self.strides
    }

    #[inline]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data
    }

    #[inline]
    pub fn nelements(&self) -> usize {
        self.shape.nelements()
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.shape.nelements() * self.dtype.size_in_bytes()
    }
}

pub struct TensorMutView<'a> {
    data: *mut u8,
    shape: &'a Shape,
    strides: &'a Strides,
    dtype: DataType,
}

impl<'a> TensorMutView<'a> {
    #[inline]
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    #[inline]
    pub fn strides(&self) -> &Strides {
        self.strides
    }

    #[inline]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data
    }

    #[inline]
    pub fn data_ptr_mut(&mut self) -> *mut u8 {
        self.data
    }

    #[inline]
    pub fn nelements(&self) -> usize {
        self.shape.nelements()
    }

    #[inline]
    pub fn nbytes(&self) -> usize {
        self.shape.nelements() * self.dtype.size_in_bytes()
    }

    pub fn fill_zero(&mut self) {
        if self.data.is_null() {
            return;
        }
        let size = self.nbytes();
        if size > 0 {
            unsafe {
                std::ptr::write_bytes(self.data, 0, size);
            }
        }
    }
}
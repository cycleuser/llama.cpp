//! Tensor shape and strides

use std::fmt;
use std::ops::{Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: [usize; 4],
    ndim: usize,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        let mut shape = Shape {
            dims: [1; 4],
            ndim: dims.len().min(4),
        };
        for (i, &d) in dims.iter().take(4).enumerate() {
            shape.dims[i] = d;
        }
        shape
    }

    pub fn from_dims(dims: [usize; 4], ndim: usize) -> Self {
        Shape { dims, ndim: ndim.min(4) }
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.ndim]
    }

    #[inline]
    pub fn dim(&self, i: usize) -> usize {
        if i < self.ndim {
            self.dims[i]
        } else {
            1
        }
    }

    #[inline]
    pub fn nelements(&self) -> usize {
        self.dims[..self.ndim].iter().product()
    }

    #[inline]
    pub fn nbytes(&self, element_size: usize) -> usize {
        self.nelements() * element_size
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.ndim == 0 || (self.ndim == 1 && self.dims[0] == 1)
    }

    #[inline]
    pub fn is_vector(&self) -> bool {
        self.ndim == 1
    }

    #[inline]
    pub fn is_matrix(&self) -> bool {
        self.ndim == 2
    }

    pub fn broadcast_shape(&self, other: &Shape) -> Option<Shape> {
        let max_ndim = self.ndim.max(other.ndim);
        let mut result = [1usize; 4];

        for i in 0..max_ndim {
            let d1 = if i < self.ndim { self.dims[self.ndim - 1 - i] } else { 1 };
            let d2 = if i < other.ndim { other.dims[other.ndim - 1 - i] } else { 1 };

            result[max_ndim - 1 - i] = if d1 == d2 {
                d1
            } else if d1 == 1 {
                d2
            } else if d2 == 1 {
                d1
            } else {
                return None;
            };
        }

        Some(Shape::from_dims(result, max_ndim))
    }

    pub fn transpose(&self) -> Shape {
        match self.ndim {
            0 | 1 => self.clone(),
            2 => Shape::new(&[self.dims[1], self.dims[0]]),
            _ => {
                let mut dims = self.dims;
                dims[..self.ndim].reverse();
                Shape::from_dims(dims, self.ndim)
            }
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, i: usize) -> &Self::Output {
        &self.dims[i]
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.dims[i]
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &d) in self.dims().iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides {
    strides: [usize; 4],
    ndim: usize,
}

impl Strides {
    pub fn from_shape(shape: &Shape) -> Self {
        let mut strides = [1usize; 4];
        for i in (0..shape.ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape.dims[i + 1];
        }
        Strides {
            strides,
            ndim: shape.ndim,
        }
    }

    pub fn from_slice(strides: &[usize]) -> Self {
        let mut s = Strides {
            strides: [1; 4],
            ndim: strides.len().min(4),
        };
        for (i, &st) in strides.iter().take(4).enumerate() {
            s.strides[i] = st;
        }
        s
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides[..self.ndim]
    }

    #[inline]
    pub fn stride(&self, i: usize) -> usize {
        if i < 4 { self.strides[i] } else { 1 }
    }

    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        if self.ndim != shape.ndim {
            return false;
        }
        let expected = Strides::from_shape(shape);
        self.strides[..self.ndim] == expected.strides[..expected.ndim]
    }
}

impl Index<usize> for Strides {
    type Output = usize;

    fn index(&self, i: usize) -> &Self::Output {
        &self.strides[i]
    }
}
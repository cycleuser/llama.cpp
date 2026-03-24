//! Raw buffer abstraction for tensor data

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

#[derive(Debug)]
pub struct Buffer {
    ptr: NonNull<u8>,
    layout: Layout,
    size: usize,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(size: usize) -> Self {
        if size == 0 {
            return Buffer {
                ptr: NonNull::dangling(),
                layout: Layout::from_size_align(0, 1).unwrap(),
                size: 0,
            };
        }

        let layout = Layout::from_size_align(size, 64).expect("Invalid layout");
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).expect("Allocation failed");

        Buffer { ptr, layout, size }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        let mut buf = Buffer::new(data.len());
        buf.as_mut_slice().copy_from_slice(data);
        buf
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        if self.size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.size == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
        }
    }

    pub fn as_slice_t<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(self.as_slice())
    }

    pub fn as_mut_slice_t<T: bytemuck::Pod + bytemuck::NoUninit>(&mut self) -> &mut [T] {
        bytemuck::cast_slice_mut(self.as_mut_slice())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                dealloc(self.ptr.as_ptr(), self.layout);
            }
        }
    }
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Buffer::from_slice(self.as_slice())
    }
}

pub struct BufferMut {
    inner: Option<Buffer>,
}

impl BufferMut {
    pub fn new(size: usize) -> Self {
        BufferMut {
            inner: Some(Buffer::new(size)),
        }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        BufferMut {
            inner: Some(Buffer::from_slice(data)),
        }
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ref().map(|b| b.as_ptr()).unwrap_or(std::ptr::null())
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.inner.as_mut().map(|b| b.as_mut_ptr()).unwrap_or(std::ptr::null_mut())
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.inner.as_ref().map(|b| b.size()).unwrap_or(0)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_ref().map(|b| b.as_slice()).unwrap_or(&[])
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.inner.as_mut().map(|b| b.as_mut_slice()).unwrap_or(&mut [])
    }

    pub fn freeze(self) -> Buffer {
        self.inner.unwrap_or_else(|| Buffer::new(0))
    }
}

#[derive(Clone)]
pub struct BufferRef {
    inner: Arc<Buffer>,
}

impl BufferRef {
    pub fn new(buffer: Buffer) -> Self {
        BufferRef {
            inner: Arc::new(buffer),
        }
    }

    pub fn from_slice(data: &[u8]) -> Self {
        BufferRef::new(Buffer::from_slice(data))
    }

    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.inner.as_ptr()
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.inner.size()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.inner.as_slice()
    }

    #[inline]
    pub fn as_slice_t<T: bytemuck::Pod>(&self) -> &[T] {
        self.inner.as_slice_t()
    }
}

impl std::fmt::Debug for BufferRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferRef")
            .field("size", &self.size())
            .finish()
    }
}
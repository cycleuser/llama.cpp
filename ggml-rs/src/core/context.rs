//! Memory context for tensor allocation

use crate::core::{DataType, Shape, Tensor};
use crate::error::{Error, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

static CONTEXT_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextId(usize);

impl ContextId {
    fn new() -> Self {
        ContextId(CONTEXT_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

struct ContextInner {
    id: ContextId,
    memory_used: AtomicUsize,
    memory_limit: usize,
    tensors: Mutex<HashMap<Arc<str>, Arc<Tensor>>>,
}

#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    pub fn new() -> Self {
        Context::with_limit(usize::MAX)
    }

    pub fn with_limit(memory_limit: usize) -> Self {
        Context {
            inner: Arc::new(ContextInner {
                id: ContextId::new(),
                memory_used: AtomicUsize::new(0),
                memory_limit,
                tensors: Mutex::new(HashMap::new()),
            }),
        }
    }

    #[inline]
    pub fn id(&self) -> ContextId {
        self.inner.id
    }

    #[inline]
    pub fn memory_used(&self) -> usize {
        self.inner.memory_used.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn memory_limit(&self) -> usize {
        self.inner.memory_limit
    }

    #[inline]
    pub fn available_memory(&self) -> usize {
        self.inner.memory_limit.saturating_sub(self.memory_used())
    }

    pub fn new_tensor(&self, shape: Shape, dtype: DataType) -> Result<Tensor> {
        let element_size = dtype.size_in_bytes();
        let size = shape.nelements() * element_size;

        if size > self.available_memory() {
            return Err(Error::OutOfMemory {
                requested: size,
                available: self.available_memory(),
            });
        }

        let tensor = Tensor::new(shape, dtype)?;
        self.inner.memory_used.fetch_add(size, Ordering::Relaxed);
        Ok(tensor)
    }

    pub fn new_tensor_1d(&self, dtype: DataType, ne0: usize) -> Result<Tensor> {
        self.new_tensor(Shape::new(&[ne0]), dtype)
    }

    pub fn new_tensor_2d(&self, dtype: DataType, ne0: usize, ne1: usize) -> Result<Tensor> {
        self.new_tensor(Shape::new(&[ne0, ne1]), dtype)
    }

    pub fn new_tensor_3d(&self, dtype: DataType, ne0: usize, ne1: usize, ne2: usize) -> Result<Tensor> {
        self.new_tensor(Shape::new(&[ne0, ne1, ne2]), dtype)
    }

    pub fn new_tensor_4d(&self, dtype: DataType, ne0: usize, ne1: usize, ne2: usize, ne3: usize) -> Result<Tensor> {
        self.new_tensor(Shape::new(&[ne0, ne1, ne2, ne3]), dtype)
    }

    pub fn new_f32(&self, ne0: usize) -> Result<Tensor> {
        self.new_tensor_1d(DataType::F32, ne0)
    }

    pub fn new_f16(&self, ne0: usize) -> Result<Tensor> {
        self.new_tensor_1d(DataType::F16, ne0)
    }

    pub fn new_i32(&self, ne0: usize) -> Result<Tensor> {
        self.new_tensor_1d(DataType::I32, ne0)
    }

    pub fn set_tensor(&self, name: impl Into<Arc<str>>, tensor: Tensor) {
        let name = name.into();
        self.inner.tensors.lock().insert(name, Arc::new(tensor));
    }

    pub fn get_tensor(&self, name: &str) -> Option<Arc<Tensor>> {
        self.inner.tensors.lock().get(name).cloned()
    }

    pub fn remove_tensor(&self, name: &str) -> Option<Arc<Tensor>> {
        self.inner.tensors.lock().remove(name)
    }

    pub fn tensor_count(&self) -> usize {
        self.inner.tensors.lock().len()
    }

    pub fn tensor_names(&self) -> Vec<Arc<str>> {
        self.inner.tensors.lock().keys().cloned().collect()
    }

    pub fn free(&self, size: usize) {
        self.inner.memory_used.fetch_sub(size, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.inner.tensors.lock().clear();
        self.inner.memory_used.store(0, Ordering::Relaxed);
    }
}

impl Default for Context {
    fn default() -> Self {
        Context::new()
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.inner.id)
            .field("memory_used", &self.memory_used())
            .field("memory_limit", &self.inner.memory_limit)
            .field("tensor_count", &self.tensor_count())
            .finish()
    }
}

#[derive(Clone)]
pub struct ContextBuilder {
    memory_limit: usize,
}

impl ContextBuilder {
    pub fn new() -> Self {
        ContextBuilder {
            memory_limit: usize::MAX,
        }
    }

    pub fn memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    pub fn build(self) -> Context {
        Context::with_limit(self.memory_limit)
    }
}

impl Default for ContextBuilder {
    fn default() -> Self {
        ContextBuilder::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = Context::new();
        assert!(ctx.memory_used() == 0);
    }

    #[test]
    fn test_tensor_allocation() {
        let ctx = Context::new();
        let tensor = ctx.new_f32(100).unwrap();
        assert_eq!(tensor.nelements(), 100);
        assert_eq!(tensor.dtype(), DataType::F32);
    }

    #[test]
    fn test_memory_limit() {
        let ctx = Context::with_limit(1000);
        let tensor = ctx.new_f32(100).unwrap();
        assert_eq!(ctx.memory_used(), 400);

        let result = ctx.new_f32(1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_named_tensors() {
        let ctx = Context::new();
        let tensor = ctx.new_f32(10).unwrap();
        ctx.set_tensor("test", tensor);
        
        let retrieved = ctx.get_tensor("test");
        assert!(retrieved.is_some());
    }
}
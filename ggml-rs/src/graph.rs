//! Computation graph for lazy evaluation

use crate::core::{Context, DataType, Shape, Tensor};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Input {
        name: String,
        shape: Shape,
        dtype: DataType,
    },
    Constant {
        tensor: Arc<Tensor>,
    },
    Add {
        a: NodeId,
        b: NodeId,
    },
    Sub {
        a: NodeId,
        b: NodeId,
    },
    Mul {
        a: NodeId,
        b: NodeId,
    },
    Div {
        a: NodeId,
        b: NodeId,
    },
    MatMul {
        a: NodeId,
        b: NodeId,
        transpose_a: bool,
        transpose_b: bool,
    },
    Scale {
        input: NodeId,
        factor: f32,
    },
    SoftMax {
        input: NodeId,
        axis: usize,
    },
    GELU {
        input: NodeId,
    },
    SILU {
        input: NodeId,
    },
    ReLU {
        input: NodeId,
    },
    Norm {
        input: NodeId,
        eps: f32,
    },
    RMSNorm {
        input: NodeId,
        eps: f32,
    },
    Reshape {
        input: NodeId,
        shape: Shape,
    },
    Transpose {
        input: NodeId,
    },
}

pub struct Node {
    id: NodeId,
    op: Operation,
    shape: Shape,
    dtype: DataType,
    computed: Option<Arc<Tensor>>,
}

impl Node {
    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    pub fn is_computed(&self) -> bool {
        self.computed.is_some()
    }

    pub fn get_tensor(&self) -> Option<&Tensor> {
        self.computed.as_deref()
    }
}

pub struct Graph {
    nodes: HashMap<NodeId, Node>,
    inputs: HashMap<String, NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            inputs: HashMap::new(),
            outputs: Vec::new(),
        }
    }

    pub fn input(&mut self, name: &str, shape: Shape, dtype: DataType) -> NodeId {
        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::Input {
                name: name.to_string(),
                shape: shape.clone(),
                dtype,
            },
            shape,
            dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        self.inputs.insert(name.to_string(), id);
        id
    }

    pub fn constant(&mut self, tensor: Tensor) -> NodeId {
        let id = NodeId::new();
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype();

        let node = Node {
            id,
            op: Operation::Constant {
                tensor: Arc::new(tensor),
            },
            shape,
            dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        let a_node = self.get_node(a)?;
        let b_node = self.get_node(b)?;

        let shape = a_node.shape.broadcast_shape(b_node.shape())
            .ok_or_else(|| Error::shape_mismatch(a_node.shape().dims(), b_node.shape().dims()))?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::Add { a, b },
            shape,
            dtype: a_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> Result<NodeId> {
        let a_node = self.get_node(a)?;
        let b_node = self.get_node(b)?;

        let shape = a_node.shape.broadcast_shape(b_node.shape())
            .ok_or_else(|| Error::shape_mismatch(a_node.shape().dims(), b_node.shape().dims()))?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::Mul { a, b },
            shape,
            dtype: a_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn matmul(&mut self, a: NodeId, b: NodeId, transpose_a: bool, transpose_b: bool) -> Result<NodeId> {
        let a_node = self.get_node(a)?;
        let b_node = self.get_node(b)?;

        let a_ndim = a_node.shape.ndim();
        let b_ndim = b_node.shape.ndim();

        let (a_rows, a_cols) = if transpose_a {
            (a_node.shape[a_ndim - 1], if a_ndim > 1 { a_node.shape[a_ndim - 2] } else { 1 })
        } else {
            (if a_ndim > 1 { a_node.shape[a_ndim - 2] } else { 1 }, a_node.shape[a_ndim - 1])
        };

        let (b_rows, b_cols) = if transpose_b {
            (b_node.shape[b_ndim - 1], if b_ndim > 1 { b_node.shape[b_ndim - 2] } else { 1 })
        } else {
            (if b_ndim > 1 { b_node.shape[b_ndim - 2] } else { 1 }, b_node.shape[b_ndim - 1])
        };

        if a_cols != b_rows {
            return Err(Error::shape_mismatch(&[a_cols], &[b_rows]));
        }

        let mut out_dims = vec![];
        if a_ndim > 1 { out_dims.push(a_rows); }
        out_dims.push(b_cols);
        let shape = Shape::new(&out_dims);

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::MatMul { a, b, transpose_a, transpose_b },
            shape,
            dtype: a_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn softmax(&mut self, input: NodeId, axis: usize) -> Result<NodeId> {
        let input_node = self.get_node(input)?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::SoftMax { input, axis },
            shape: input_node.shape.clone(),
            dtype: input_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn gelu(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.get_node(input)?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::GELU { input },
            shape: input_node.shape.clone(),
            dtype: input_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn silu(&mut self, input: NodeId) -> Result<NodeId> {
        let input_node = self.get_node(input)?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::SILU { input },
            shape: input_node.shape.clone(),
            dtype: input_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn rms_norm(&mut self, input: NodeId, eps: f32) -> Result<NodeId> {
        let input_node = self.get_node(input)?;

        let id = NodeId::new();
        let node = Node {
            id,
            op: Operation::RMSNorm { input, eps },
            shape: input_node.shape.clone(),
            dtype: input_node.dtype,
            computed: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn set_output(&mut self, node: NodeId) {
        if !self.outputs.contains(&node) {
            self.outputs.push(node);
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    fn get_node(&self, id: NodeId) -> Result<&Node> {
        self.nodes.get(&id).ok_or(Error::InvalidOperation("Invalid node id".into()))
    }

    fn get_node_mut(&mut self, id: NodeId) -> Result<&mut Node> {
        self.nodes.get_mut(&id).ok_or(Error::InvalidOperation("Invalid node id".into()))
    }

    pub fn compute(&mut self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Arc<Tensor>>> {
        for (name, tensor) in inputs {
            if let Some(&id) = self.inputs.get(name) {
                let node = self.get_node_mut(id)?;
                node.computed = Some(Arc::new(tensor.clone()));
            }
        }

        for &output_id in &self.outputs {
            self.compute_node(output_id)?;
        }

        let mut results = Vec::with_capacity(self.outputs.len());
        for &output_id in &self.outputs {
            let node = self.get_node(output_id)?;
            results.push(node.computed.clone().ok_or(Error::InvalidOperation("Output not computed".into()))?);
        }

        Ok(results)
    }

    fn compute_node(&mut self, id: NodeId) -> Result<()> {
        {
            let node = self.get_node(id)?;
            if node.computed.is_some() {
                return Ok(());
            }
        }

        let op = {
            let node = self.get_node(id)?;
            node.op.clone()
        };

        let result = match op {
            Operation::Input { .. } => {
                return Err(Error::InvalidOperation("Input not provided".into()));
            }
            Operation::Constant { tensor } => tensor,
            Operation::Add { a, b } => {
                self.compute_node(a)?;
                self.compute_node(b)?;
                let a_tensor = self.get_node(a)?.computed.clone().ok_or(Error::InvalidOperation("A not computed".into()))?;
                let b_tensor = self.get_node(b)?.computed.clone().ok_or(Error::InvalidOperation("B not computed".into()))?;
                Arc::new(crate::ops::add(&a_tensor, &b_tensor)?)
            }
            Operation::Mul { a, b } => {
                self.compute_node(a)?;
                self.compute_node(b)?;
                let a_tensor = self.get_node(a)?.computed.clone().ok_or(Error::InvalidOperation("A not computed".into()))?;
                let b_tensor = self.get_node(b)?.computed.clone().ok_or(Error::InvalidOperation("B not computed".into()))?;
                Arc::new(crate::ops::mul(&a_tensor, &b_tensor)?)
            }
            Operation::MatMul { a, b, transpose_a, transpose_b } => {
                self.compute_node(a)?;
                self.compute_node(b)?;
                let a_tensor = self.get_node(a)?.computed.clone().ok_or(Error::InvalidOperation("A not computed".into()))?;
                let b_tensor = self.get_node(b)?.computed.clone().ok_or(Error::InvalidOperation("B not computed".into()))?;
                Arc::new(crate::ops::matmul(&a_tensor, &b_tensor, transpose_a, transpose_b)?)
            }
            Operation::SoftMax { input, axis } => {
                self.compute_node(input)?;
                let tensor = self.get_node(input)?.computed.clone().ok_or(Error::InvalidOperation("Input not computed".into()))?;
                Arc::new(crate::ops::softmax(&tensor, axis)?)
            }
            Operation::GELU { input } => {
                self.compute_node(input)?;
                let tensor = self.get_node(input)?.computed.clone().ok_or(Error::InvalidOperation("Input not computed".into()))?;
                Arc::new(crate::ops::gelu(&tensor)?)
            }
            Operation::SILU { input } => {
                self.compute_node(input)?;
                let tensor = self.get_node(input)?.computed.clone().ok_or(Error::InvalidOperation("Input not computed".into()))?;
                Arc::new(crate::ops::silu(&tensor)?)
            }
            Operation::Scale { input, factor } => {
                self.compute_node(input)?;
                let tensor = self.get_node(input)?.computed.clone().ok_or(Error::InvalidOperation("Input not computed".into()))?;
                Arc::new(crate::ops::scale(&tensor, factor)?)
            }
            _ => return Err(Error::unsupported("Operation not implemented")),
        };

        let node = self.get_node_mut(id)?;
        node.computed = Some(result);

        Ok(())
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic() {
        let mut graph = Graph::new();

        let a = graph.input("a", Shape::new(&[3, 4]), DataType::F32);
        let b = graph.input("b", Shape::new(&[3, 4]), DataType::F32);
        let c = graph.add(a, b).unwrap();
        graph.set_output(c);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.input_count(), 2);
        assert_eq!(graph.output_count(), 1);
    }

    #[test]
    fn test_graph_compute() {
        let mut graph = Graph::new();

        let a = graph.input("a", Shape::new(&[2, 2]), DataType::F32);
        let b = graph.input("b", Shape::new(&[2, 2]), DataType::F32);
        let c = graph.add(a, b).unwrap();
        graph.set_output(c);

        let tensor_a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::new(&[2, 2])).unwrap();
        let tensor_b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::new(&[2, 2])).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("a".to_string(), tensor_a);
        inputs.insert("b".to_string(), tensor_b);

        let results = graph.compute(&inputs).unwrap();
        let output = results[0].as_slice::<f32>().unwrap();

        assert_eq!(output, &[6.0f32, 8.0, 10.0, 12.0]);
    }
}
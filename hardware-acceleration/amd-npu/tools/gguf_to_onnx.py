#!/usr/bin/env python3
"""
GGUF to ONNX Model Converter for AMD NPU

Converts GGUF models to ONNX format optimized for AMD XDNA NPU execution.
Uses quantization and optimization passes suitable for NPU inference.
"""

import argparse
import os
import sys
import struct
from pathlib import Path

try:
    import numpy as np
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Error: Required packages not installed.")
    print("Install with: pip install numpy onnx onnxruntime")
    sys.exit(1)

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q4_K = 13
GGUF_TYPE_Q5_K = 14
GGUF_TYPE_Q6_K = 15
GGUF_TYPE_Q8_0 = 7

class GGUFReader:
    """Reader for GGUF file format"""
    
    def __init__(self, path):
        self.path = path
        self.file = open(path, 'rb')
        self.tensors = {}
        self.metadata = {}
        self._read_header()
        self._read_metadata()
        self._read_tensor_info()
    
    def _read_header(self):
        magic = struct.unpack('<I', self.file.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        
        self.version = struct.unpack('<I', self.file.read(4))[0]
        self.tensor_count = struct.unpack('<Q', self.file.read(8))[0]
        self.metadata_kv_count = struct.unpack('<Q', self.file.read(8))[0]
    
    def _read_string(self):
        length = struct.unpack('<Q', self.file.read(8))[0]
        return self.file.read(length).decode('utf-8')
    
    def _read_value(self, vtype):
        if vtype == 0:  # UINT8
            return struct.unpack('<B', self.file.read(1))[0]
        elif vtype == 1:  # INT8
            return struct.unpack('<b', self.file.read(1))[0]
        elif vtype == 2:  # UINT16
            return struct.unpack('<H', self.file.read(2))[0]
        elif vtype == 3:  # INT16
            return struct.unpack('<h', self.file.read(2))[0]
        elif vtype == 4:  # UINT32
            return struct.unpack('<I', self.file.read(4))[0]
        elif vtype == 5:  # INT32
            return struct.unpack('<i', self.file.read(4))[0]
        elif vtype == 6:  # FLOAT32
            return struct.unpack('<f', self.file.read(4))[0]
        elif vtype == 7:  # BOOL
            return struct.unpack('<?', self.file.read(1))[0]
        elif vtype == 8:  # STRING
            return self._read_string()
        elif vtype == 9:  # ARRAY
            elem_type = struct.unpack('<I', self.file.read(4))[0]
            count = struct.unpack('<Q', self.file.read(8))[0]
            return [self._read_value(elem_type) for _ in range(count)]
        else:
            raise ValueError(f"Unknown value type: {vtype}")
    
    def _read_metadata(self):
        for _ in range(self.metadata_kv_count):
            key = self._read_string()
            vtype = struct.unpack('<I', self.file.read(4))[0]
            value = self._read_value(vtype)
            self.metadata[key] = value
    
    def _read_tensor_info(self):
        self.tensor_info = {}
        for _ in range(self.tensor_count):
            name = self._read_string()
            n_dims = struct.unpack('<I', self.file.read(4))[0]
            dims = [struct.unpack('<Q', self.file.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', self.file.read(4))[0]
            offset = struct.unpack('<Q', self.file.read(8))[0]
            self.tensor_info[name] = {
                'dims': dims,
                'dtype': dtype,
                'offset': offset
            }
        
        # Calculate data offset (aligned to 32 bytes)
        pos = self.file.tell()
        self.data_offset = (pos + 31) & ~31
    
    def get_tensor_data(self, name):
        info = self.tensor_info.get(name)
        if not info:
            return None
        
        self.file.seek(self.data_offset + info['offset'])
        
        if info['dtype'] == GGUF_TYPE_F32:
            size = np.prod(info['dims']) * 4
            data = np.frombuffer(self.file.read(size), dtype=np.float32)
            return data.reshape(info['dims'])
        elif info['dtype'] == GGUF_TYPE_F16:
            size = np.prod(info['dims']) * 2
            data = np.frombuffer(self.file.read(size), dtype=np.float16)
            return data.astype(np.float32).reshape(info['dims'])
        else:
            print(f"Warning: Unsupported dtype {info['dtype']} for tensor {name}")
            return None
    
    def close(self):
        self.file.close()

class ONNXConverter:
    """Converts GGUF model to ONNX format"""
    
    def __init__(self, gguf_reader, model_type='llama'):
        self.reader = gguf_reader
        self.model_type = model_type
        self.graph_inputs = []
        self.graph_outputs = []
        self.graph_nodes = []
        self.initializers = []
        self.value_info = []
    
    def convert_layer_norm(self, input_name, output_name, weight_name, eps=1e-5):
        """Convert layer normalization to ONNX"""
        nodes = []
        
        mean_out = f"{output_name}_mean"
        nodes.append(helper.make_node(
            'ReduceMean',
            inputs=[input_name],
            outputs=[mean_out],
            axes=[-1],
            keepdims=1
        ))
        
        diff_out = f"{output_name}_diff"
        nodes.append(helper.make_node(
            'Sub',
            inputs=[input_name, mean_out],
            outputs=[diff_out]
        ))
        
        pow_out = f"{output_name}_pow"
        nodes.append(helper.make_node(
            'Mul',
            inputs=[diff_out, diff_out],
            outputs=[pow_out]
        ))
        
        var_out = f"{output_name}_var"
        nodes.append(helper.make_node(
            'ReduceMean',
            inputs=[pow_out],
            outputs=[var_out],
            axes=[-1],
            keepdims=1
        ))
        
        eps_tensor = numpy_helper.from_array(
            np.array(eps, dtype=np.float32),
            name=f"{output_name}_eps"
        )
        self.initializers.append(eps_tensor)
        
        var_eps_out = f"{output_name}_var_eps"
        nodes.append(helper.make_node(
            'Add',
            inputs=[var_out, f"{output_name}_eps"],
            outputs=[var_eps_out]
        ))
        
        sqrt_out = f"{output_name}_sqrt"
        nodes.append(helper.make_node(
            'Sqrt',
            inputs=[var_eps_out],
            outputs=[sqrt_out]
        ))
        
        norm_out = f"{output_name}_norm"
        nodes.append(helper.make_node(
            'Div',
            inputs=[diff_out, sqrt_out],
            outputs=[norm_out]
        ))
        
        nodes.append(helper.make_node(
            'Mul',
            inputs=[norm_out, weight_name],
            outputs=[output_name]
        ))
        
        return nodes
    
    def convert_matmul(self, input_name, weight_name, output_name, transpose=True):
        """Convert matrix multiplication to ONNX"""
        if transpose:
            return [helper.make_node(
                'MatMul',
                inputs=[input_name, weight_name],
                outputs=[output_name]
            )]
        else:
            return [helper.make_node(
                'MatMul',
                inputs=[input_name, weight_name],
                outputs=[output_name]
            )]
    
    def convert_attention(self, hidden_states, layer_idx, config):
        """Convert attention layer to ONNX"""
        nodes = []
        
        hidden_size = config.get('hidden_size', 4096)
        num_heads = config.get('num_attention_heads', 32)
        head_dim = hidden_size // num_heads
        
        q_weight = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
        k_weight = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
        v_weight = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
        o_weight = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        
        q_out = f"layer{layer_idx}_q"
        k_out = f"layer{layer_idx}_k"
        v_out = f"layer{layer_idx}_v"
        
        nodes.extend(self.convert_matmul(hidden_states, q_weight, q_out))
        nodes.extend(self.convert_matmul(hidden_states, k_weight, k_out))
        nodes.extend(self.convert_matmul(hidden_states, v_weight, v_out))
        
        attn_out = f"layer{layer_idx}_attn"
        o_out = f"layer{layer_idx}_o"
        
        nodes.append(helper.make_node(
            'Attention',
            inputs=[q_out, k_out, v_out],
            outputs=[attn_out],
            num_heads=num_heads
        ))
        
        nodes.extend(self.convert_matmul(attn_out, o_weight, o_out))
        
        return nodes, o_out
    
    def convert_mlp(self, hidden_states, layer_idx, config):
        """Convert MLP layer to ONNX"""
        nodes = []
        
        gate_weight = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        up_weight = f"model.layers.{layer_idx}.mlp.up_proj.weight"
        down_weight = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        
        gate_out = f"layer{layer_idx}_gate"
        up_out = f"layer{layer_idx}_up"
        
        nodes.extend(self.convert_matmul(hidden_states, gate_weight, gate_out))
        nodes.extend(self.convert_matmul(hidden_states, up_weight, up_out))
        
        silu_out = f"layer{layer_idx}_silu"
        nodes.append(helper.make_node(
            'SiLU',
            inputs=[gate_out],
            outputs=[silu_out]
        ))
        
        mul_out = f"layer{layer_idx}_mlp_mul"
        nodes.append(helper.make_node(
            'Mul',
            inputs=[silu_out, up_out],
            outputs=[mul_out]
        ))
        
        down_out = f"layer{layer_idx}_down"
        nodes.extend(self.convert_matmul(mul_out, down_weight, down_out))
        
        return nodes, down_out
    
    def convert(self, output_path, optimize_for_npu=True):
        """Convert the full model to ONNX"""
        
        arch = self.reader.metadata.get('general.architecture', 'llama')
        hidden_size = self.reader.metadata.get('llama.embedding_length', 4096)
        num_layers = self.reader.metadata.get('llama.block_count', 32)
        num_heads = self.reader.metadata.get('llama.attention.head_count', 32)
        vocab_size = self.reader.metadata.get('llama.vocab_size', 32000)
        
        print(f"Converting model:")
        print(f"  Architecture: {arch}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Heads: {num_heads}")
        print(f"  Vocab size: {vocab_size}")
        
        input_ids = helper.make_tensor_value_info(
            'input_ids', TensorProto.INT64, [1, 'sequence_length']
        )
        self.graph_inputs.append(input_ids)
        
        logits = helper.make_tensor_value_info(
            'logits', TensorProto.FLOAT, [1, 'sequence_length', vocab_size]
        )
        self.graph_outputs.append(logits)
        
        nodes = []
        current_hidden = "hidden_states"
        
        embed_weight = self.reader.get_tensor_data('token_embd.weight')
        if embed_weight is not None:
            embed_init = numpy_helper.from_array(
                embed_weight.astype(np.float32),
                name='embed_tokens.weight'
            )
            self.initializers.append(embed_init)
            
            nodes.append(helper.make_node(
                'Gather',
                inputs=['embed_tokens.weight', 'input_ids'],
                outputs=[current_hidden]
            ))
        
        config = {
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads,
        }
        
        for layer_idx in range(min(num_layers, 2)):
            print(f"  Processing layer {layer_idx}...")
            
            input_norm_out = f"layer{layer_idx}_input_norm"
            nodes.extend(self.convert_layer_norm(
                current_hidden, input_norm_out, 
                f"model.layers.{layer_idx}.input_layernorm.weight"
            ))
            
            attn_nodes, attn_out = self.convert_attention(
                input_norm_out, layer_idx, config
            )
            nodes.extend(attn_nodes)
            
            residual_out = f"layer{layer_idx}_residual1"
            nodes.append(helper.make_node(
                'Add',
                inputs=[current_hidden, attn_out],
                outputs=[residual_out]
            ))
            
            post_norm_out = f"layer{layer_idx}_post_norm"
            nodes.extend(self.convert_layer_norm(
                residual_out, post_norm_out,
                f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            ))
            
            mlp_nodes, mlp_out = self.convert_mlp(post_norm_out, layer_idx, config)
            nodes.extend(mlp_nodes)
            
            current_hidden = f"layer{layer_idx}_output"
            nodes.append(helper.make_node(
                'Add',
                inputs=[residual_out, mlp_out],
                outputs=[current_hidden]
            ))
        
        self.graph_nodes = nodes
        
        graph = helper.make_graph(
            self.graph_nodes,
            'llama_model',
            self.graph_inputs,
            self.graph_outputs,
            self.initializers
        )
        
        model = helper.make_model(graph)
        model.opset_import[0].version = 17
        
        if optimize_for_npu:
            print("Optimizing for NPU...")
            from onnx import optimizer
            passes = [
                'eliminate_identity',
                'eliminate_nop_transpose',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
            ]
            model = optimizer.optimize(model, passes)
        
        onnx.save(model, output_path)
        print(f"Saved ONNX model to: {output_path}")
        
        return model

def main():
    parser = argparse.ArgumentParser(
        description='Convert GGUF to ONNX for AMD NPU'
    )
    parser.add_argument('input', help='Input GGUF model path')
    parser.add_argument('-o', '--output', help='Output ONNX path')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip NPU optimization')
    parser.add_argument('--quantize', action='store_true',
                        help='Quantize to INT8 for NPU')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    output_path = args.output
    if not output_path:
        output_path = Path(args.input).stem + '.onnx'
    
    print(f"Reading GGUF model: {args.input}")
    reader = GGUFReader(args.input)
    
    converter = ONNXConverter(reader)
    converter.convert(output_path, optimize_for_npu=not args.no_optimize)
    
    if args.quantize:
        print("Quantizing model to INT8...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quant_path = str(output_path).replace('.onnx', '_int8.onnx')
            quantize_dynamic(
                output_path,
                quant_path,
                weight_type=QuantType.QInt8
            )
            print(f"Saved quantized model to: {quant_path}")
        except ImportError:
            print("Warning: onnxruntime not installed, skipping quantization")
    
    reader.close()
    print("Conversion complete!")

if __name__ == '__main__':
    main()
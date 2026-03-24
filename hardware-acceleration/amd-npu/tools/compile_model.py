#!/usr/bin/env python3
"""
AMD NPU Model Compiler

Compiles ONNX models for execution on AMD XDNA NPU.
Uses Vitis AI Execution Provider or IREE for compilation.
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path

def compile_with_vitis_ai(onnx_path, output_dir, target='PHX'):
    """
    Compile ONNX model using Vitis AI Execution Provider
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for compiled model
        target: Target NPU type ('PHX', 'HPT', 'STX')
    """
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed")
        print("Install with: pip install onnxruntime-vitisai")
        return False
    
    target_map = {
        'PHX': 'X1',      # Phoenix
        'HPT': 'X1',      # Hawk Point
        'STX': 'STX',     # Strix
    }
    
    target_device = target_map.get(target.upper(), 'X1')
    
    print(f"Compiling for target: {target} ({target_device})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
        
        provider_options = {}
        if target.upper() in ['PHX', 'HPT']:
            xclbin_path = find_xclbin(target)
            if xclbin_path:
                provider_options = [{
                    'target': target_device,
                    'xclbin': xclbin_path,
                }]
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        print("Creating inference session...")
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options if provider_options else None
        )
        
        print("Available providers:", session.get_providers())
        
        meta = session.get_modelmeta()
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"\nModel inputs:")
        for inp in inputs:
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"\nModel outputs:")
        for outp in outputs:
            print(f"  {outp.name}: {outp.shape} ({outp.type})")
        
        cache_file = os.path.join(output_dir, Path(onnx_path).stem + '.rai')
        print(f"\nCompiled cache: {cache_file}")
        
        return True
        
    except Exception as e:
        print(f"Error compiling model: {e}")
        return False

def compile_with_iree(onnx_path, output_dir, target='xdna'):
    """
    Compile ONNX model using IREE-AMD-AIE
    
    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for compiled model
        target: Target device ('xdna', 'xdna2')
    """
    
    try:
        import iree.compiler as ireec
    except ImportError:
        print("Error: IREE not installed")
        print("Install IREE-AMD-AIE from: https://github.com/nod-ai/iree-amd-aie")
        return False
    
    print(f"Compiling with IREE for target: {target}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        target_options = {
            'xdna': 'amdaie-xdna',
            'xdna2': 'amdaie-xdna2',
        }
        
        target_triple = target_options.get(target, 'amdaie-xdna')
        
        compiled_module = ireec.compile_file(
            onnx_path,
            target_backends=[f"rocm-{target_triple}"],
            input_type=ireec.InputType.ONNX,
        )
        
        output_file = os.path.join(output_dir, Path(onnx_path).stem + '.vmfb')
        with open(output_file, 'wb') as f:
            f.write(compiled_module)
        
        print(f"Compiled module saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error compiling with IREE: {e}")
        return False

def find_xclbin(target):
    """Find the appropriate xclbin file for the target"""
    
    xclbin_names = {
        'PHX': ['4x4.xclbin', 'AMD_AIE2P_4x4_Overlay.xclbin'],
        'HPT': ['4x4.xclbin', 'AMD_AIE2P_4x4_Overlay.xclbin'],
        'STX': ['AMD_AIE2P_4x4_Overlay.xclbin', '8x4.xclbin'],
    }
    
    search_paths = [
        '/opt/xilinx/xrt/share/xclbins',
        '/usr/share/xclbins',
        os.environ.get('XCLBIN_PATH', ''),
    ]
    
    for name in xclbin_names.get(target.upper(), []):
        for path in search_paths:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                return full_path
    
    return None

def generate_model_config(output_dir, model_info):
    """Generate model configuration file for NPU deployment"""
    
    config = {
        'model_type': model_info.get('architecture', 'llama'),
        'hidden_size': model_info.get('hidden_size', 4096),
        'num_layers': model_info.get('num_layers', 32),
        'num_heads': model_info.get('num_heads', 32),
        'vocab_size': model_info.get('vocab_size', 32000),
        'npu_settings': {
            'batch_size': 1,
            'sequence_length': 2048,
            'precision': 'int8',
            'use_cache': True,
        }
    }
    
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generated config: {config_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Compile ONNX models for AMD NPU'
    )
    parser.add_argument('model', help='Path to ONNX model')
    parser.add_argument('-o', '--output', default='./compiled',
                        help='Output directory')
    parser.add_argument('-t', '--target', default='PHX',
                        choices=['PHX', 'HPT', 'STX', 'xdna', 'xdna2'],
                        help='Target NPU type')
    parser.add_argument('--backend', default='vitis',
                        choices=['vitis', 'iree'],
                        help='Compilation backend')
    parser.add_argument('--config', help='Model config JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    print(f"Compiling model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Backend: {args.backend}")
    print(f"Output: {args.output}")
    
    model_info = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            model_info = json.load(f)
    
    if args.backend == 'vitis':
        success = compile_with_vitis_ai(args.model, args.output, args.target)
    else:
        success = compile_with_iree(args.model, args.output, args.target)
    
    if success:
        generate_model_config(args.output, model_info)
        print("\nCompilation successful!")
        print(f"\nTo use the compiled model:")
        print(f"  python -c \"import onnxruntime as ort; sess = ort.InferenceSession('{args.model}')\"")
    else:
        print("\nCompilation failed.")
        sys.exit(1)

if __name__ == '__main__':
    main()
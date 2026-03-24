#!/usr/bin/env python3
"""
Quick start script for AMD NPU support
Demonstrates basic usage of NPU for inference
"""

import os
import sys
import time
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    try:
        import onnxruntime as ort
    except ImportError:
        missing.append("onnxruntime")
    
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def check_npu_availability():
    """Check if NPU is available"""
    try:
        import onnxruntime as ort
        
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
        
        if "VitisAIExecutionProvider" in providers:
            print("✓ Vitis AI Execution Provider available (NPU support)")
            return True
        else:
            print("✗ Vitis AI EP not available")
            print("  NPU may not be detected or drivers not installed")
            return False
    except Exception as e:
        print(f"Error checking NPU: {e}")
        return False

def create_test_model(output_path):
    """Create a simple test model for NPU"""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
        import numpy as np
        
        # Create a simple matmul model
        input1 = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 128])
        output1 = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64])
        
        # Weight tensor
        weight_data = np.random.randn(128, 64).astype(np.float32)
        weight = numpy_helper.from_array(weight_data, name='weight')
        
        # MatMul node
        matmul_node = helper.make_node(
            'MatMul',
            inputs=['input', 'weight'],
            outputs=['output']
        )
        
        # Create graph
        graph = helper.make_graph(
            [matmul_node],
            'test_model',
            [input1],
            [output1],
            [weight]
        )
        
        # Create model
        model = helper.make_model(graph)
        model.opset_import[0].version = 17
        
        # Save model
        onnx.save(model, output_path)
        print(f"Created test model: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating model: {e}")
        return False

def run_inference(model_path, device='NPU', iterations=10):
    """Run inference on the model"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Select provider
        if device == 'NPU':
            providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        print(f"\nRunning inference on {device}...")
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        print(f"Active provider: {session.get_providers()[0]}")
        
        # Prepare input
        input_data = np.random.randn(1, 128).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(3):
            session.run(None, {input_name: input_data})
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            output = session.run(None, {input_name: input_data})
        end = time.time()
        
        avg_time = (end - start) / iterations * 1000
        print(f"Average latency: {avg_time:.2f} ms")
        print(f"Throughput: {1000/avg_time:.1f} iter/s")
        
        return True
        
    except Exception as e:
        print(f"Error running inference: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='AMD NPU Quick Start'
    )
    parser.add_argument('--check', action='store_true',
                        help='Check NPU availability')
    parser.add_argument('--test', action='store_true',
                        help='Run test inference')
    parser.add_argument('--device', default='NPU',
                        choices=['NPU', 'CPU'],
                        help='Target device')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations')
    parser.add_argument('--model', help='Path to ONNX model')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print(" AMD NPU Quick Start")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check NPU
    if args.check or args.test:
        print("\n[1] Checking NPU availability...")
        npu_available = check_npu_availability()
    
    if args.test:
        print("\n[2] Running test inference...")
        
        # Create test model if not provided
        if args.model:
            model_path = args.model
        else:
            model_path = "/tmp/test_model.onnx"
            if not create_test_model(model_path):
                return 1
        
        # Run inference
        device = args.device if npu_available else 'CPU'
        if not run_inference(model_path, device=device, iterations=args.iterations):
            return 1
    
    if not args.check and not args.test:
        # Default: just check
        print("\nChecking NPU availability...")
        check_npu_availability()
        print("\nUsage:")
        print("  python quickstart.py --check      # Check NPU status")
        print("  python quickstart.py --test       # Run test inference")
        print("  python quickstart.py --test --device CPU  # Force CPU")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
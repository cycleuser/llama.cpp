#!/usr/bin/env python3
"""
AMD NPU Benchmark Tool

Benchmarks model inference performance on AMD XDNA NPU.
"""

import argparse
import time
import sys
import os
import json
import numpy as np
from pathlib import Path

def benchmark_onnx(model_path, device='NPU', iterations=100, warmup=10):
    """
    Benchmark ONNX model inference
    
    Args:
        model_path: Path to ONNX model
        device: Target device ('NPU', 'CPU', 'GPU')
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
    """
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("Error: onnxruntime not installed")
        return None
    
    providers = {
        'NPU': ['VitisAIExecutionProvider', 'CPUExecutionProvider'],
        'CPU': ['CPUExecutionProvider'],
        'GPU': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    }
    
    provider_list = providers.get(device, ['CPUExecutionProvider'])
    
    print(f"\nBenchmark Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print(f"  Providers: {provider_list}")
    print(f"  Iterations: {iterations}")
    print(f"  Warmup: {warmup}")
    
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=provider_list
        )
        
        actual_providers = session.get_providers()
        print(f"  Active providers: {actual_providers}")
        
        inputs = session.get_inputs()
        input_shapes = {inp.name: inp.shape for inp in inputs}
        
        print(f"\nGenerating input data...")
        input_data = {}
        for inp in inputs:
            shape = list(inp.shape)
            for i, dim in enumerate(shape):
                if isinstance(dim, str) or dim <= 0:
                    shape[i] = 1 if 'batch' in dim.lower() else 128
            
            print(f"  {inp.name}: {shape} ({inp.type})")
            
            if inp.type == 'tensor(int64)':
                input_data[inp.name] = np.random.randint(0, 1000, shape).astype(np.int64)
            else:
                input_data[inp.name] = np.random.randn(*shape).astype(np.float32)
        
        print(f"\nWarmup ({warmup} iterations)...")
        for i in range(warmup):
            outputs = session.run(None, input_data)
        
        print(f"\nBenchmarking ({iterations} iterations)...")
        latencies = []
        
        for i in range(iterations):
            start = time.perf_counter()
            outputs = session.run(None, input_data)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
            
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{iterations}")
        
        latencies = np.array(latencies)
        
        results = {
            'device': device,
            'provider': actual_providers[0],
            'iterations': iterations,
            'latency': {
                'mean_ms': float(np.mean(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'p50_ms': float(np.percentile(latencies, 50)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
            },
            'throughput': {
                'iterations_per_sec': float(1000 / np.mean(latencies)),
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_llama_cpp(model_path, prompt="Hello, world!", tokens=100, device='auto'):
    """
    Benchmark llama.cpp inference
    
    Args:
        model_path: Path to GGUF model
        prompt: Input prompt
        tokens: Number of tokens to generate
        device: Target device
    """
    
    print(f"\nBenchmarking llama.cpp:")
    print(f"  Model: {model_path}")
    print(f"  Prompt: {prompt}")
    print(f"  Tokens: {tokens}")
    
    llama_cli = './build/bin/llama-cli'
    if not os.path.exists(llama_cli):
        llama_cli = 'llama-cli'
    
    cmd = [
        llama_cli,
        '-m', model_path,
        '-p', prompt,
        '-n', str(tokens),
        '--timing-info',
        '-no-cnv',
    ]
    
    if device == 'npu':
        cmd.extend(['-sm', 'layer'])
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    output = result.stdout
    
    lines = output.split('\n')
    results = {
        'total_time_sec': end - start,
        'model': model_path,
    }
    
    for line in lines:
        if 'tokens per second' in line.lower():
            parts = line.split()
            for i, p in enumerate(parts):
                if 'tokens' in p.lower():
                    try:
                        results['tokens_per_sec'] = float(parts[i-1])
                    except:
                        pass
        if 'prompt processing' in line.lower():
            try:
                results['prompt_time'] = float(line.split()[3])
            except:
                pass
        if 'generation' in line.lower():
            try:
                results['generation_time'] = float(line.split()[2])
            except:
                pass
    
    return results

def compare_devices(model_path, iterations=50):
    """Compare performance across devices"""
    
    print("\n" + "="*60)
    print("Comparing device performance")
    print("="*60)
    
    devices = ['CPU', 'NPU']
    results = {}
    
    for device in devices:
        print(f"\n--- {device} ---")
        result = benchmark_onnx(model_path, device=device, iterations=iterations)
        if result:
            results[device] = result
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    for device, result in results.items():
        print(f"\n{device}:")
        print(f"  Latency: {result['latency']['mean_ms']:.2f} ms (±{result['latency']['std_ms']:.2f})")
        print(f"  Throughput: {result['throughput']['iterations_per_sec']:.2f} iter/s")
    
    if len(results) > 1:
        cpu_latency = results.get('CPU', {}).get('latency', {}).get('mean_ms', 1)
        npu_latency = results.get('NPU', {}).get('latency', {}).get('mean_ms', 1)
        if cpu_latency and npu_latency:
            speedup = cpu_latency / npu_latency
            print(f"\nNPU speedup vs CPU: {speedup:.2f}x")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark AMD NPU performance'
    )
    parser.add_argument('model', help='Path to model (ONNX or GGUF)')
    parser.add_argument('-d', '--device', default='NPU',
                        choices=['NPU', 'CPU', 'GPU', 'auto'],
                        help='Target device')
    parser.add_argument('-i', '--iterations', type=int, default=100,
                        help='Number of iterations')
    parser.add_argument('-w', '--warmup', type=int, default=10,
                        help='Warmup iterations')
    parser.add_argument('--compare', action='store_true',
                        help='Compare all devices')
    parser.add_argument('-o', '--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)
    
    if args.compare:
        results = compare_devices(args.model, args.iterations)
    else:
        if args.model.endswith('.onnx'):
            results = benchmark_onnx(
                args.model, 
                device=args.device,
                iterations=args.iterations,
                warmup=args.warmup
            )
        else:
            results = benchmark_llama_cpp(args.model)
    
    if results:
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        print("\nBenchmark failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
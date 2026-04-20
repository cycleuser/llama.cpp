#!/usr/bin/env python3
"""
Compare GGUF (ollama) vs HF (converted) model:
1. Extract tokenizer from GGUF and build HF tokenizer
2. Compare weight values
3. Run benchmark on ollama model

Usage:
    python3 compare_models.py --gguf <path> --hf <path> [--weights] [--benchmark] [--extract-tokenizer]
"""

import argparse
import json
import sys
import subprocess
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open


def extract_tokenizer_from_gguf(gguf_path, output_dir):
    """Extract full tokenizer from GGUF and create HF-compatible tokenizer files."""
    import gguf

    reader = gguf.GGUFReader(gguf_path, "r")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_field = reader.fields.get("tokenizer.ggml.tokens")
    if tokens_field is None:
        raise ValueError("No tokenizer.ggml.tokens field found")

    tokens = tokens_field.contents()
    print(f"  Tokens: {len(tokens)}")

    merges_field = reader.fields.get("tokenizer.ggml.merges")
    merges = []
    if merges_field is not None:
        merges = merges_field.contents()
        print(f"  Merges: {len(merges)}")

    scores_field = reader.fields.get("tokenizer.ggml.scores")
    scores = None
    if scores_field is not None:
        scores = scores_field.contents()

    types_field = reader.fields.get("tokenizer.ggml.token_type")
    token_types = None
    if types_field is not None:
        token_types = types_field.contents()

    bos_field = reader.fields.get("tokenizer.ggml.bos_token_id")
    eos_field = reader.fields.get("tokenizer.ggml.eos_token_id")
    unk_field = reader.fields.get("tokenizer.ggml.unknown_token_id")
    pad_field = reader.fields.get("tokenizer.ggml.padding_token_id")

    bos_token_id = int(bos_field.contents()) if bos_field is not None else None
    eos_token_id = int(eos_field.contents()) if eos_field is not None else None
    unk_token_id = int(unk_field.contents()) if unk_field is not None else None
    pad_token_id = int(pad_field.contents()) if pad_field is not None else None

    template_field = reader.fields.get("tokenizer.chat_template")
    chat_template = None
    if template_field is not None:
        chat_template = template_field.contents()

    print(f"  BOS: {bos_token_id}, EOS: {eos_token_id}, UNK: {unk_token_id}, PAD: {pad_token_id}")

    # tokenizer_config.json
    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": True,
        "model_max_length": 262144,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    if chat_template:
        tokenizer_config["chat_template"] = chat_template
    if bos_token_id is not None:
        tokenizer_config["bos_token"] = tokens[bos_token_id] if isinstance(tokens, list) else str(bos_token_id)
    if eos_token_id is not None:
        tokenizer_config["eos_token"] = tokens[eos_token_id] if isinstance(tokens, list) else str(eos_token_id)
    if unk_token_id is not None:
        tokenizer_config["unk_token"] = tokens[unk_token_id] if isinstance(tokens, list) else str(unk_token_id)
    if pad_token_id is not None:
        tokenizer_config["pad_token"] = tokens[pad_token_id] if isinstance(tokens, list) else str(pad_token_id)

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    # special_tokens_map.json
    special_tokens = {}
    if bos_token_id is not None:
        special_tokens["bos_token"] = {"content": tokens[bos_token_id] if isinstance(tokens, list) else "", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if eos_token_id is not None:
        special_tokens["eos_token"] = {"content": tokens[eos_token_id] if isinstance(tokens, list) else "", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if unk_token_id is not None:
        special_tokens["unk_token"] = {"content": tokens[unk_token_id] if isinstance(tokens, list) else "", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    if pad_token_id is not None:
        special_tokens["pad_token"] = {"content": tokens[pad_token_id] if isinstance(tokens, list) else "", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False}

    with open(output_dir / "special_tokens_map.json", "w") as f:
        json.dump(special_tokens, f, indent=2, ensure_ascii=False)

    # Build vocab
    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    # Build merges list
    merges_list = []
    if isinstance(merges, list):
        for m in merges:
            if isinstance(m, str) and " " in m:
                merges_list.append(m)

    # tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "vocab": vocab,
            "merges": merges_list,
        },
    }

    with open(output_dir / "tokenizer.json", "w") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)

    print(f"  Tokenizer files saved to {output_dir}")
    return output_dir


def dequantize_q8_0(raw_data, shape):
    """Dequantize Q8_0 tensor. GGUF Q8_0: block_size=32, block_bytes=34 (2 f16 scale + 32 int8)."""
    import struct
    block_size = 32
    block_bytes = 34

    total_elements = int(np.prod(shape))
    n_blocks = total_elements // block_size

    raw = np.frombuffer(raw_data.tobytes(), dtype=np.uint8)
    expected_bytes = n_blocks * block_bytes
    raw = raw[:expected_bytes]

    result = np.zeros(total_elements, dtype=np.float32)
    for i in range(n_blocks):
        offset = i * block_bytes
        d = struct.unpack('<e', bytes(raw[offset:offset+2]))[0]
        qs = raw[offset+2:offset+34].view(np.int8).astype(np.float32)
        result[i*block_size:(i+1)*block_size] = d * qs

    return result.reshape(shape[::-1]).copy()


def dequantize_tensor(tensor):
    """Dequantize a GGUF tensor to float32."""
    import gguf
    qtype = gguf.GGMLQuantizationType(tensor.tensor_type)

    if qtype == gguf.GGMLQuantizationType.F32:
        data = np.array(tensor.data, dtype=np.float32)
    elif qtype == gguf.GGMLQuantizationType.F16:
        data = np.array(tensor.data, dtype=np.float16).astype(np.float32)
    elif qtype == gguf.GGMLQuantizationType.Q8_0:
        data = dequantize_q8_0(tensor.data, tensor.shape)
    else:
        data = np.array(tensor.data, dtype=np.float32)

    return data


def load_gguf_weights(gguf_path):
    """Load weights from GGUF file."""
    try:
        import gguf
    except ImportError:
        print("ERROR: gguf package not installed. Install with: pip install gguf")
        sys.exit(1)

    reader = gguf.GGUFReader(gguf_path, "r")
    weights = {}
    for tensor in reader.tensors:
        name = tensor.name
        data = dequantize_tensor(tensor)
        weights[name] = data

    return weights


def load_hf_weights(hf_path):
    """Load weights from HF safetensors directory."""
    hf_path = Path(hf_path)
    safetensors_files = list(hf_path.glob("*.safetensors"))
    if not safetensors_files:
        print(f"ERROR: No safetensors files found in {hf_path}")
        sys.exit(1)

    weights = {}
    for st_file in safetensors_files:
        print(f"  Loading {st_file.name}...")
        with safe_open(st_file, framework="np") as f:
            for key in f.keys():
                hf_key = key
                if hf_key.startswith("model."):
                    hf_key = hf_key[6:]
                weights[hf_key] = f.get_tensor(key).astype(np.float32)

    return weights


def map_hf_to_gguf_name(hf_name):
    """Map HF tensor name to GGUF tensor name."""
    if hf_name.startswith("vision_tower."):
        return hf_name

    layer_match = re.match(r"layers\.(\d+)\.(.+)", hf_name)
    if layer_match:
        layer_id = int(layer_match.group(1))
        sub_name = layer_match.group(2)

        mapping = {
            "input_layernorm.weight": f"blk.{layer_id}.attn_norm.weight",
            "post_attention_layernorm.weight": f"blk.{layer_id}.post_attention_norm.weight",
            "self_attn.qkv_proj.weight": f"blk.{layer_id}.attn_qkv.weight",
            "self_attn.gate_proj.weight": f"blk.{layer_id}.attn_gate.weight",
            "mlp.gate_proj.weight": f"blk.{layer_id}.ffn_gate.weight",
            "mlp.up_proj.weight": f"blk.{layer_id}.ffn_up.weight",
            "mlp.down_proj.weight": f"blk.{layer_id}.ffn_down.weight",
            "mamba.norm.weight": f"blk.{layer_id}.ssm_norm.weight",
            "mamba.alpha.weight": f"blk.{layer_id}.ssm_alpha.weight",
            "mamba.beta.weight": f"blk.{layer_id}.ssm_beta.weight",
            "mamba.conv1d.weight": f"blk.{layer_id}.ssm_conv1d.weight",
            "mamba.out_proj.weight": f"blk.{layer_id}.ssm_out.weight",
            "mamba.A_log": f"blk.{layer_id}.ssm_A",
            "mamba.dt_proj.bias": f"blk.{layer_id}.ssm_dt",
            "mamba.dt_proj.weight": f"blk.{layer_id}.ssm_dt_proj",
        }

        if sub_name in mapping:
            return mapping[sub_name]

    if hf_name == "embed_tokens.weight":
        return "token_embd.weight"

    if hf_name == "lm_head.weight":
        return "output.weight"

    return None


def compare_weights(gguf_weights, hf_weights, tolerance=1e-3):
    """Compare weights between GGUF and HF models."""
    print("\n" + "=" * 80)
    print("WEIGHT COMPARISON RESULTS")
    print("=" * 80)

    matched = 0
    mismatched = 0
    shape_mismatch = 0
    gguf_only = 0
    hf_only = 0
    max_diff = 0.0
    max_diff_name = ""
    diff_details = []

    gguf_keys = set(gguf_weights.keys())
    hf_keys = set(hf_weights.keys())

    matched_pairs = {}

    for gguf_name in sorted(gguf_keys):
        if gguf_name in hf_keys:
            hf_name = gguf_name
        else:
            hf_name = None
            for hkey in hf_keys:
                mapped = map_hf_to_gguf_name(hkey)
                if mapped == gguf_name:
                    hf_name = hkey
                    break

        if hf_name is None:
            gguf_only += 1
            continue

        matched_pairs[gguf_name] = hf_name
        gguf_data = gguf_weights[gguf_name]
        hf_data = hf_weights[hf_name]

        if gguf_data.shape != hf_data.shape:
            shape_mismatch += 1
            diff_details.append({
                "tensor": gguf_name,
                "issue": "shape_mismatch",
                "gguf_shape": list(gguf_data.shape),
                "hf_shape": list(hf_data.shape),
            })
            if shape_mismatch <= 5:
                print(f"  SHAPE MISMATCH: {gguf_name}")
                print(f"    GGUF: {gguf_data.shape}, HF: {hf_data.shape}")
            continue

        diff = np.abs(gguf_data - hf_data)
        max_d = float(np.max(diff))
        mean_d = float(np.mean(diff))

        if max_d > tolerance:
            mismatched += 1
            if max_d > max_diff:
                max_diff = max_d
                max_diff_name = gguf_name
            diff_details.append({
                "tensor": gguf_name,
                "issue": "value_mismatch",
                "max_diff": max_d,
                "mean_diff": mean_d,
            })
            if mismatched <= 5:
                print(f"  DIFF: {gguf_name} (max={max_d:.6e}, mean={mean_d:.6e})")
        else:
            matched += 1

    matched_hf = set(matched_pairs.values())
    for hf_name in hf_keys:
        if hf_name not in matched_hf:
            hf_only += 1

    print(f"\n  Matched (within tolerance):  {matched}")
    print(f"  Value mismatched:            {mismatched}")
    print(f"  Shape mismatched:            {shape_mismatch}")
    print(f"  GGUF-only tensors:           {gguf_only}")
    print(f"  HF-only tensors:             {hf_only}")

    if max_diff_name:
        print(f"\n  Max difference: {max_diff_name} (max_diff={max_diff:.6e})")

    total_compared = matched + mismatched + shape_mismatch
    if total_compared > 0:
        accuracy = matched / total_compared * 100
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: {accuracy:.1f}% tensors match ({matched}/{total_compared})")
        print(f"{'=' * 80}")

    return {
        "matched": matched,
        "mismatched": mismatched,
        "shape_mismatch": shape_mismatch,
        "gguf_only": gguf_only,
        "hf_only": hf_only,
        "max_diff": max_diff,
        "max_diff_name": max_diff_name,
        "accuracy": accuracy if total_compared > 0 else 0,
        "diff_details": diff_details,
    }


def run_ollama_inference(model_name, prompts, max_tokens=256):
    """Run inference using ollama."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] Prompt: {prompt[:60]}...")

        try:
            result = subprocess.run(
                ["ollama", "run", model_name, "--nowordwrap"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120
            )
            generated = result.stdout.strip()

            print(f"    Output: {generated[:200]}...")
            results.append({"prompt": prompt, "generated": generated, "status": "ok"})

        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT (>120s)")
            results.append({"prompt": prompt, "generated": "", "status": "timeout"})
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({"prompt": prompt, "generated": "", "status": "error", "error": str(e)})

    return results


def run_benchmark(model_name, benchmark_name="mmlu"):
    """Run a standardized benchmark using ollama."""
    mmlu_questions = [
        {"question": "What is the capital of France? A. London B. Paris C. Berlin D. Madrid. Return only the letter of the correct answer.", "answer": "B"},
        {"question": "Which planet is known as the Red Planet? A. Venus B. Mars C. Jupiter D. Saturn. Return only the letter of the correct answer.", "answer": "B"},
        {"question": "What is the chemical symbol for gold? A. Go B. Gd C. Au D. Ag. Return only the letter of the correct answer.", "answer": "C"},
        {"question": "In Python, which keyword is used to define a function? A. func B. def C. function D. define. Return only the letter.", "answer": "B"},
        {"question": "What is the square root of 144? A. 10 B. 11 C. 12 D. 13. Return only the letter.", "answer": "C"},
        {"question": "Which data structure uses FIFO? A. Stack B. Queue C. Tree D. Graph. Return only the letter.", "answer": "B"},
        {"question": "What is the largest ocean on Earth? A. Atlantic B. Indian C. Arctic D. Pacific. Return only the letter.", "answer": "D"},
        {"question": "Who wrote Romeo and Juliet? A. Charles Dickens B. William Shakespeare C. Jane Austen D. Mark Twain. Return only the letter.", "answer": "B"},
        {"question": "What is the binary representation of decimal 10? A. 1010 B. 1100 C. 1001 D. 1110. Return only the letter.", "answer": "A"},
        {"question": "Which is NOT a programming language? A. Python B. Java C. HTML D. C++. Return only the letter.", "answer": "C"},
    ]

    print(f"\n  Running {benchmark_name.upper()}-style benchmark ({len(mmlu_questions)} questions)...")

    correct = 0
    results = []

    for i, q in enumerate(mmlu_questions):
        prompt = q["question"]
        expected = q["answer"]

        try:
            result = subprocess.run(
                ["ollama", "run", model_name, "--nowordwrap"],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=90
            )
            generated = result.stdout.strip()

            # Extract answer: look for patterns like "Answer: X" or just the last letter
            # First try to find "Answer: X" pattern
            answer_match = re.search(r'[Aa]nswer[:\s]*([A-D])', generated)
            if not answer_match:
                # Try to find the last standalone A-D letter at end of response
                lines = generated.split('\n')
                for line in reversed(lines):
                    line_match = re.search(r'\b([A-D])\b', line.strip())
                    if line_match:
                        answer_match = line_match
                        break

            if not answer_match:
                # Fallback: find any A-D letter
                answer_match = re.search(r'\b([A-D])\b', generated)

            predicted = answer_match.group(1) if answer_match else ""

            is_correct = predicted == expected
            if is_correct:
                correct += 1

            status = 'PASS' if is_correct else 'FAIL'
            print(f"    Q{i+1}: Expected={expected}, Predicted={predicted}, {status}")
            results.append({
                "question": prompt[:50],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
                "raw_output": generated[:150],
            })

        except subprocess.TimeoutExpired:
            print(f"    Q{i+1}: TIMEOUT")
            results.append({
                "question": prompt[:50],
                "expected": expected,
                "predicted": "",
                "correct": False,
                "error": "timeout",
            })
        except Exception as e:
            print(f"    Q{i+1}: ERROR - {e}")
            results.append({
                "question": prompt[:50],
                "expected": expected,
                "predicted": "",
                "correct": False,
                "error": str(e),
            })

    answered = len([r for r in results if r.get("predicted")])
    accuracy = correct / answered * 100 if answered > 0 else 0
    print(f"\n  Benchmark accuracy: {accuracy:.1f}% ({correct}/{answered} answered, {len(mmlu_questions)} total)")

    return {
        "benchmark": benchmark_name,
        "accuracy": accuracy,
        "correct": correct,
        "answered": answered,
        "total": len(mmlu_questions),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare GGUF vs HF model")
    parser.add_argument("--gguf", required=True, help="Path to GGUF file")
    parser.add_argument("--hf", required=True, help="Path to HF model directory")
    parser.add_argument("--weights", action="store_true", help="Compare weights")
    parser.add_argument("--infer", action="store_true", help="Run inference with ollama")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on ollama")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Weight comparison tolerance")
    parser.add_argument("--extract-tokenizer", action="store_true", help="Extract tokenizer from GGUF to HF dir")
    parser.add_argument("--ollama-model", default="huihui_ai/qwen3.5-abliterated:0.8B", help="Ollama model name")
    args = parser.parse_args()

    gguf_path = Path(args.gguf)
    hf_path = Path(args.hf)

    if not gguf_path.exists():
        print(f"ERROR: GGUF file not found: {gguf_path}")
        sys.exit(1)

    if not hf_path.exists():
        print(f"ERROR: HF directory not found: {hf_path}")
        sys.exit(1)

    if args.extract_tokenizer:
        print("Extracting tokenizer from GGUF...")
        extract_tokenizer_from_gguf(gguf_path, hf_path)

    weight_results = None
    if args.weights:
        print("\nLoading GGUF weights...")
        gguf_weights = load_gguf_weights(gguf_path)
        print(f"  Loaded {len(gguf_weights)} tensors")

        print("Loading HF weights...")
        hf_weights = load_hf_weights(hf_path)
        print(f"  Loaded {len(hf_weights)} tensors")

        weight_results = compare_weights(gguf_weights, hf_weights, args.tolerance)

    benchmark_results = None
    if args.benchmark:
        print("\n" + "=" * 80)
        print("BENCHMARK (ollama)")
        print("=" * 80)
        benchmark_results = run_benchmark(args.ollama_model)

    infer_results = None
    if args.infer:
        test_prompts = [
            "What is the capital of France?",
            "What is 2 + 2?",
            "Write a Python function to calculate factorial:",
            "用中文介绍一下北京",
        ]

        print("\n" + "=" * 80)
        print("INFERENCE (ollama)")
        print("=" * 80)
        infer_results = run_ollama_inference(args.ollama_model, test_prompts)

    output = {
        "gguf_path": str(gguf_path),
        "hf_path": str(hf_path),
        "ollama_model": args.ollama_model,
    }
    if weight_results:
        output["weight_comparison"] = weight_results
    if benchmark_results:
        output["benchmark"] = benchmark_results
    if infer_results:
        output["inference"] = infer_results

    output_file = Path("model_comparison_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    main()

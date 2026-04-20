#!/usr/bin/env python3
"""
Test script for convert_gguf_to_hf.py

Creates a small test GGUF model and verifies the round-trip conversion.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add gguf-py to path
sys.path.insert(0, str(Path(__file__).parent / "gguf-py"))
import gguf
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType


def create_test_gguf(output_path: Path):
    """Create a small test GGUF model for testing."""
    print(f"Creating test GGUF model at {output_path}")

    writer = GGUFWriter(path=None, arch="llama")

    # Add architecture metadata
    writer.add_context_length(2048)
    writer.add_embedding_length(64)
    writer.add_feed_forward_length(128)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(16)
    writer.add_rope_freq_base(10000)
    writer.add_vocab_size(32)

    # Add tokenizer metadata
    writer.add_tokenizer_model("llama")
    tokens = [f"tok{i}" for i in range(32)]
    writer.add_token_list(tokens)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)

    # Create test tensors with known values
    # Embedding: (vocab_size, hidden_size) = (32, 64)
    embed = np.random.randn(32, 64).astype(np.float16)
    writer.add_tensor("token_embd.weight", embed, raw_dtype=GGMLQuantizationType.F16)

    # Layer 0 tensors
    # Attn norm: (hidden_size,) = (64,)
    attn_norm_0 = np.ones(64, dtype=np.float32)
    writer.add_tensor("blk.0.attn_norm.weight", attn_norm_0, raw_dtype=GGMLQuantizationType.F32)

    # Q, K, V: (hidden_size, hidden_size) = (64, 64)
    q_0 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.0.attn_q.weight", q_0, raw_dtype=GGMLQuantizationType.F16)

    k_0 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.0.attn_k.weight", k_0, raw_dtype=GGMLQuantizationType.F16)

    v_0 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.0.attn_v.weight", v_0, raw_dtype=GGMLQuantizationType.F16)

    # Attn out: (hidden_size, hidden_size) = (64, 64)
    attn_out_0 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.0.attn_out.weight", attn_out_0, raw_dtype=GGMLQuantizationType.F16)

    # FFN norm: (hidden_size,) = (64,)
    ffn_norm_0 = np.ones(64, dtype=np.float32)
    writer.add_tensor("blk.0.ffn_norm.weight", ffn_norm_0, raw_dtype=GGMLQuantizationType.F32)

    # FFN gate, down, up: (ff_size, hidden_size) = (128, 64), (hidden_size, ff_size) = (64, 128)
    ffn_gate_0 = np.random.randn(128, 64).astype(np.float16)
    writer.add_tensor("blk.0.ffn_gate.weight", ffn_gate_0, raw_dtype=GGMLQuantizationType.F16)

    ffn_down_0 = np.random.randn(64, 128).astype(np.float16)
    writer.add_tensor("blk.0.ffn_down.weight", ffn_down_0, raw_dtype=GGMLQuantizationType.F16)

    ffn_up_0 = np.random.randn(128, 64).astype(np.float16)
    writer.add_tensor("blk.0.ffn_up.weight", ffn_up_0, raw_dtype=GGMLQuantizationType.F16)

    # Layer 1 tensors (same shapes)
    attn_norm_1 = np.ones(64, dtype=np.float32)
    writer.add_tensor("blk.1.attn_norm.weight", attn_norm_1, raw_dtype=GGMLQuantizationType.F32)

    q_1 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.1.attn_q.weight", q_1, raw_dtype=GGMLQuantizationType.F16)

    k_1 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.1.attn_k.weight", k_1, raw_dtype=GGMLQuantizationType.F16)

    v_1 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.1.attn_v.weight", v_1, raw_dtype=GGMLQuantizationType.F16)

    attn_out_1 = np.random.randn(64, 64).astype(np.float16)
    writer.add_tensor("blk.1.attn_out.weight", attn_out_1, raw_dtype=GGMLQuantizationType.F16)

    ffn_norm_1 = np.ones(64, dtype=np.float32)
    writer.add_tensor("blk.1.ffn_norm.weight", ffn_norm_1, raw_dtype=GGMLQuantizationType.F32)

    ffn_gate_1 = np.random.randn(128, 64).astype(np.float16)
    writer.add_tensor("blk.1.ffn_gate.weight", ffn_gate_1, raw_dtype=GGMLQuantizationType.F16)

    ffn_down_1 = np.random.randn(64, 128).astype(np.float16)
    writer.add_tensor("blk.1.ffn_down.weight", ffn_down_1, raw_dtype=GGMLQuantizationType.F16)

    ffn_up_1 = np.random.randn(128, 64).astype(np.float16)
    writer.add_tensor("blk.1.ffn_up.weight", ffn_up_1, raw_dtype=GGMLQuantizationType.F16)

    # Output norm: (hidden_size,) = (64,)
    output_norm = np.ones(64, dtype=np.float32)
    writer.add_tensor("output_norm.weight", output_norm, raw_dtype=GGMLQuantizationType.F32)

    # LM head: (vocab_size, hidden_size) = (32, 64)
    lm_head = np.random.randn(32, 64).astype(np.float16)
    writer.add_tensor("output.weight", lm_head, raw_dtype=GGMLQuantizationType.F16)

    # Write to file
    writer.write_header_to_file(path=output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Test GGUF created successfully ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def test_conversion(gguf_path: Path, output_dir: Path):
    """Test the GGUF to HF conversion."""
    print(f"\nTesting conversion of {gguf_path}")
    print(f"Output directory: {output_dir}")

    # Run conversion
    sys.path.insert(0, str(Path(__file__).parent))
    from convert_gguf_to_hf import convert_gguf_to_hf

    convert_gguf_to_hf(
        input_path=gguf_path,
        output_dir=output_dir,
        outtype="f32",
        split_experts=False,  # No experts in test model
        undo_permute=True,
        use_safetensors=False,  # Use PyTorch format for simpler testing
    )

    # Verify output files
    print("\nVerifying output files...")
    assert (output_dir / "config.json").exists(), "config.json not found"
    assert (output_dir / "pytorch_model.bin").exists(), "pytorch_model.bin not found"

    # Load and verify config
    with open(output_dir / "config.json") as f:
        config = json.load(f)

    print(f"\nConfig contents:")
    print(json.dumps(config, indent=2))

    # Verify config values
    assert config["hidden_size"] == 64, f"Expected hidden_size=64, got {config['hidden_size']}"
    assert config["num_hidden_layers"] == 2, f"Expected num_hidden_layers=2, got {config['num_hidden_layers']}"
    assert config["num_attention_heads"] == 4, f"Expected num_attention_heads=4, got {config['num_attention_heads']}"
    assert config["intermediate_size"] == 128, f"Expected intermediate_size=128, got {config['intermediate_size']}"

    # Load and verify model weights
    import torch
    state_dict = torch.load(output_dir / "pytorch_model.bin", weights_only=True, map_location="cpu")

    print(f"\nModel state dict keys ({len(state_dict)} tensors):")
    for name, tensor in sorted(state_dict.items()):
        print(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")

    # Verify tensor names and shapes
    expected_tensors = {
        "model.embed_tokens.weight": (32, 64),
        "model.norm.weight": (64,),
        "lm_head.weight": (32, 64),
        "model.layers.0.input_layernorm.weight": (64,),
        "model.layers.0.self_attn.q_proj.weight": (64, 64),
        "model.layers.0.self_attn.k_proj.weight": (64, 64),
        "model.layers.0.self_attn.v_proj.weight": (64, 64),
        "model.layers.0.self_attn.o_proj.weight": (64, 64),
        "model.layers.0.post_attention_layernorm.weight": (64,),
        "model.layers.0.mlp.gate_proj.weight": (128, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 128),
        "model.layers.0.mlp.up_proj.weight": (128, 64),
        "model.layers.1.input_layernorm.weight": (64,),
        "model.layers.1.self_attn.q_proj.weight": (64, 64),
        "model.layers.1.self_attn.k_proj.weight": (64, 64),
        "model.layers.1.self_attn.v_proj.weight": (64, 64),
        "model.layers.1.self_attn.o_proj.weight": (64, 64),
        "model.layers.1.post_attention_layernorm.weight": (64,),
        "model.layers.1.mlp.gate_proj.weight": (128, 64),
        "model.layers.1.mlp.down_proj.weight": (64, 128),
        "model.layers.1.mlp.up_proj.weight": (128, 64),
    }

    print("\nVerifying tensor shapes...")
    for name, expected_shape in expected_tensors.items():
        if name not in state_dict:
            print(f"  MISSING: {name}")
            continue
        actual_shape = tuple(state_dict[name].shape)
        if actual_shape == expected_shape:
            print(f"  OK: {name} {actual_shape}")
        else:
            print(f"  FAIL: {name} expected {expected_shape}, got {actual_shape}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        gguf_path = tmpdir / "test_model.gguf"
        output_dir = tmpdir / "hf_model"

        # Create test GGUF
        create_test_gguf(gguf_path)

        # Test conversion
        test_conversion(gguf_path, output_dir)


if __name__ == "__main__":
    main()

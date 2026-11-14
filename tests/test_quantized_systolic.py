"""Tests for native INT8/FP8 per-channel systolic quantization, TP=2 sharding, and block-tree parity."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from benchmarks.block_diffusion_jax import load_qwen_block_tree_runner
from benchmarks.quantization_jax import (
    batched_expert_systolic_linear,
    normalize_weight_quant,
    pytree_nbytes,
    quant_np_dtype,
    quantize_np_per_channel,
    scale_spec_for_weight,
    systolic_linear,
)
from benchmarks.qwen_jax import load_qwen_runner


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Serialize numpy arrays into a Hugging Face .safetensors checkpoint file."""
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, arr in tensors.items():
        c_arr = np.ascontiguousarray(arr)
        if c_arr.dtype == np.float32:
            dtype_str = "F32"
        elif c_arr.dtype == np.float16:
            dtype_str = "F16"
        else:
            raise ValueError(f"Unsupported test dtype {c_arr.dtype}")
        raw = c_arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(c_arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for chunk in chunks:
            f.write(chunk)


def _write_word_tokenizer(vocab_size: int, save_path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {f"tok{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = 0
    vocab["<|im_start|>"] = 1
    vocab["<|im_end|>"] = 2
    vocab["<think>"] = 3
    vocab["</think>"] = 4
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(save_path))


def _write_cyclic_qwen_dir(model_dir: Path) -> Path:
    """Write a minimal cyclic Qwen safetensors checkpoint for quantization parity tests."""
    model_dir.mkdir(parents=True, exist_ok=True)
    hidden = 128
    num_layers = 2
    num_q = 4
    num_kv = 2
    head_dim = 32
    inter = 128
    vocab = 288

    _write_word_tokenizer(vocab, model_dir / "tokenizer.json")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_q,
                "num_key_value_heads": num_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "eos_token_id": 2,
            }
        ),
        encoding="utf-8",
    )

    embed = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(24):
        embed[tok, tok] = 4.0
    embed[2, 64] = 1.0

    lm_head = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(24):
        nxt = 10 + ((tok - 10 + 1) % 4) if 10 <= tok <= 13 else 10
        lm_head[nxt, tok] = 5.0

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": lm_head,
        "model.norm.weight": np.ones((hidden,), dtype=np.float32),
    }
    rng = np.random.default_rng(17)
    for i in range(num_layers):
        tensors[f"model.layers.{i}.input_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        # Small structured non-zero matrices keep QKV + residual paths well conditioned under INT8/FP8.
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = (
            rng.normal(0.0, 0.02, size=(num_q * head_dim, hidden)).astype(np.float32)
        )
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = (
            rng.normal(0.0, 0.02, size=(num_kv * head_dim, hidden)).astype(np.float32)
        )
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = np.zeros(
            (num_kv * head_dim, hidden), dtype=np.float32
        )
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = np.zeros(
            (hidden, num_q * head_dim), dtype=np.float32
        )
        tensors[f"model.layers.{i}.self_attn.q_proj.bias"] = np.zeros((num_q * head_dim,), dtype=np.float32)
        tensors[f"model.layers.{i}.self_attn.k_proj.bias"] = np.zeros((num_kv * head_dim,), dtype=np.float32)
        tensors[f"model.layers.{i}.self_attn.v_proj.bias"] = np.zeros((num_kv * head_dim,), dtype=np.float32)
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = (
            rng.normal(0.0, 0.01, size=(inter, hidden)).astype(np.float32)
        )
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = (
            rng.normal(0.0, 0.01, size=(inter, hidden)).astype(np.float32)
        )
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = np.zeros((hidden, inter), dtype=np.float32)

    _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir


def _write_qwen3_and_block_drafter_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Create a deterministic Qwen3 (q/k norm, no bias) + block-diffusion drafter pair."""
    target_dir = tmp_path / "qwen3_target"
    draft_dir = tmp_path / "qwen3_drafter"
    target_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)

    hidden = 128
    num_t_layers = 4
    num_d_layers = 2
    num_q = 8
    num_kv = 2
    head_dim = 64
    inter = 128
    vocab = 288
    block_size = 8
    eos_id = 286
    mask_id = 287
    rope_theta = 10.0

    _write_word_tokenizer(vocab, target_dir / "tokenizer.json")
    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_hidden_layers": num_t_layers,
                "num_attention_heads": num_q,
                "num_key_value_heads": num_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-6,
                "rope_theta": rope_theta,
                "tie_word_embeddings": False,
                "attention_bias": False,
                "eos_token_id": eos_id,
            }
        ),
        encoding="utf-8",
    )
    (draft_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_hidden_layers": num_d_layers,
                "num_attention_heads": num_q,
                "num_key_value_heads": num_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "head_dim": head_dim,
                "rms_norm_eps": 1e-6,
                "rope_theta": rope_theta,
                "dflash_config": {
                    "block_size": block_size,
                    "mask_token_id": mask_id,
                    "target_layer_ids": [1, 2],
                },
            }
        ),
        encoding="utf-8",
    )

    embed = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(24):
        embed[tok, tok] = 4.0
    embed[eos_id, 64] = 1.0
    embed[mask_id, 31] = 1.0
    embed[mask_id, 127] = 1.0

    lm_head = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(24):
        nxt = 8 + ((tok - 8 + 1) % 16) if 8 <= tok <= 23 else 8
        lm_head[nxt, tok] = 5.0

    t_tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": lm_head,
        "model.norm.weight": np.ones((hidden,), dtype=np.float32),
    }
    for i in range(num_t_layers):
        t_tensors[f"model.layers.{i}.input_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        t_tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        t_tensors[f"model.layers.{i}.self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        t_tensors[f"model.layers.{i}.self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        wq_t = np.zeros((num_q * head_dim, hidden), dtype=np.float32)
        wq_t[0, 127] = 0.01
        wk_t = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
        wk_t[0, 127] = 0.01
        t_tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = wq_t
        t_tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = wk_t
        t_tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
        t_tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = np.zeros((hidden, num_q * head_dim), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.up_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.down_proj.weight"] = np.zeros((hidden, inter), dtype=np.float32)
    _write_safetensors(target_dir / "model.safetensors", t_tensors)

    half = head_dim // 2
    freq_band = 1
    mask_band = 31
    inv_freq = 1.0 / (rope_theta ** (freq_band / float(half)))

    fc_w = np.zeros((hidden, 2 * hidden), dtype=np.float32)
    fc_w[32:64, :32] = 0.02

    d_tensors: dict[str, np.ndarray] = {
        "fc.weight": fc_w,
        "hidden_norm.weight": np.ones((hidden,), dtype=np.float32),
        "norm.weight": np.ones((hidden,), dtype=np.float32),
    }

    wq0 = np.zeros((num_q * head_dim, hidden), dtype=np.float32)
    wk0 = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
    wv0 = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
    wo0 = np.zeros((hidden, num_q * head_dim), dtype=np.float32)

    for kv_h in range(num_kv):
        base = kv_h * head_dim
        for tok in range(8, 24):
            wk0[base + freq_band, tok] = 1.0
            wv0[base + (tok - 8), tok] = 2.0
        wk0[base + mask_band, 31] = 1.0
        wk0[base + 10, 32] = 1.0

    for step_s in range(1, block_size):
        q_base = step_s * head_dim
        angle = -float(step_s) * inv_freq
        wq0[q_base + freq_band, 31] = 0.80 * math.cos(angle)
        wq0[q_base + half + freq_band, 31] = 0.80 * math.sin(angle)
        wq0[q_base + mask_band, 31] = 0.60
        for j in range(16):
            target_dim = 8 + ((j + step_s - 1) % 16)
            wo0[target_dim, q_base + j] = 1.0

    for l_idx in range(num_d_layers):
        d_tensors[f"layers.{l_idx}.input_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.post_attention_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        if l_idx == 0:
            d_tensors[f"layers.{l_idx}.self_attn.q_proj.weight"] = wq0
            d_tensors[f"layers.{l_idx}.self_attn.k_proj.weight"] = wk0
            d_tensors[f"layers.{l_idx}.self_attn.v_proj.weight"] = wv0
            d_tensors[f"layers.{l_idx}.self_attn.o_proj.weight"] = wo0 * 2.0
        else:
            d_tensors[f"layers.{l_idx}.self_attn.q_proj.weight"] = np.zeros((num_q * head_dim, hidden), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.k_proj.weight"] = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.v_proj.weight"] = np.zeros((num_kv * head_dim, hidden), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.o_proj.weight"] = np.zeros((hidden, num_q * head_dim), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.gate_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.up_proj.weight"] = np.zeros((inter, hidden), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.down_proj.weight"] = np.zeros((hidden, inter), dtype=np.float32)

    _write_safetensors(draft_dir / "model.safetensors", d_tensors)
    return target_dir, draft_dir


def test_per_channel_int8_and_fp8_quantizer_accuracy_and_batched_experts() -> None:
    """Verify INT8/FP8 normalization, per-channel scales, systolic_linear, and batched MoE kernel."""
    assert normalize_weight_quant(None) == "none"
    assert normalize_weight_quant("INT8_Weight_Only") == "int8"
    assert normalize_weight_quant("float8_e4m3fn") == "fp8"
    assert quant_np_dtype("int8") == np.int8
    assert quant_np_dtype("fp8") == ml_dtypes.float8_e4m3fn
    assert scale_spec_for_weight(P(None, None, "tp")) == P(None, "tp")
    assert scale_spec_for_weight(P(None, "tp", None)) == P()
    with pytest.raises(ValueError, match="bool"):
        normalize_weight_quant(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str or None"):
        normalize_weight_quant(123)  # type: ignore[arg-type]

    rng = np.random.default_rng(123)
    w = rng.normal(0.0, 1.0, size=(64, 48)).astype(np.float32)
    w *= np.linspace(0.1, 6.0, 48, dtype=np.float32)[None, :]
    w[:, 7] = 0.0

    q_int8, s_int8 = quantize_np_per_channel(w, "int8")
    assert q_int8.dtype == np.int8 and q_int8.shape == (64, 48)
    assert s_int8.dtype == np.float32 and s_int8.shape == (48,)
    assert s_int8[7] == 1.0 and np.all(q_int8[:, 7] == 0)
    recon_int8 = q_int8.astype(np.float32) * s_int8[None, :]
    assert np.linalg.norm(recon_int8 - w) / np.linalg.norm(w) < 0.015

    q_fp8, s_fp8 = quantize_np_per_channel(w, "fp8")
    assert q_fp8.dtype == ml_dtypes.float8_e4m3fn and q_fp8.shape == (64, 48)
    assert s_fp8.dtype == np.float32 and s_fp8.shape == (48,)
    assert np.all(q_fp8[:, 7].astype(np.float32) == 0.0)
    recon_fp8 = q_fp8.astype(np.float32) * s_fp8[None, :]
    assert np.linalg.norm(recon_fp8 - w) / np.linalg.norm(w) < 0.08

    x_np = rng.normal(0.0, 0.5, size=(16, 64)).astype(np.float32)
    bias_np = rng.normal(0.0, 0.1, size=(48,)).astype(np.float32)
    ref_out = x_np @ w + bias_np[None, :]

    x_bf16 = jax.device_put(jnp.asarray(x_np, dtype=jnp.bfloat16))
    b_bf16 = jax.device_put(jnp.asarray(bias_np, dtype=jnp.bfloat16))
    out_int8 = np.asarray(
        systolic_linear(x_bf16, jax.device_put(jnp.asarray(q_int8)), b=b_bf16, w_scale=jax.device_put(jnp.asarray(s_int8))),
        dtype=np.float32,
    )
    out_fp8 = np.asarray(
        systolic_linear(x_bf16, jax.device_put(jnp.asarray(q_fp8)), b=b_bf16, w_scale=jax.device_put(jnp.asarray(s_fp8))),
        dtype=np.float32,
    )
    assert np.linalg.norm(out_int8 - ref_out) / np.linalg.norm(ref_out) < 0.02
    assert np.linalg.norm(out_fp8 - ref_out) / np.linalg.norm(ref_out) < 0.08

    # Batched MoE expert systolic kernel [E, K, in_dim] x [E, in_dim, out_dim]
    x_exp_np = rng.normal(0.0, 0.5, size=(4, 12, 64)).astype(np.float32)
    w_exp_np = rng.normal(0.0, 0.5, size=(4, 64, 32)).astype(np.float32)
    ref_exp = np.einsum("eki,eio->eko", x_exp_np, w_exp_np)

    q_exp_int8, s_exp_int8 = quantize_np_per_channel(w_exp_np, "int8")
    q_exp_fp8, s_exp_fp8 = quantize_np_per_channel(w_exp_np, "fp8")
    assert q_exp_int8.shape == (4, 64, 32) and s_exp_int8.shape == (4, 32)
    assert pytree_nbytes((q_exp_int8, s_exp_int8)) == (4 * 64 * 32) + (4 * 32 * 4)

    x_exp_bf16 = jax.device_put(jnp.asarray(x_exp_np, dtype=jnp.bfloat16))
    out_exp_int8 = np.asarray(
        batched_expert_systolic_linear(
            x_exp_bf16,
            jax.device_put(jnp.asarray(q_exp_int8)),
            w_experts_scale=jax.device_put(jnp.asarray(s_exp_int8)),
        ),
        dtype=np.float32,
    )
    out_exp_fp8 = np.asarray(
        batched_expert_systolic_linear(
            x_exp_bf16,
            jax.device_put(jnp.asarray(q_exp_fp8)),
            w_experts_scale=jax.device_put(jnp.asarray(s_exp_fp8)),
        ),
        dtype=np.float32,
    )
    assert out_exp_int8.shape == (4, 12, 32)
    assert np.linalg.norm(out_exp_int8 - ref_exp) / np.linalg.norm(ref_exp) < 0.02
    assert np.linalg.norm(out_exp_fp8 - ref_exp) / np.linalg.norm(ref_exp) < 0.08


def test_qwen_int8_and_fp8_streaming_memory_compression_and_3mode_parity(tmp_path: Path) -> None:
    """Verify streaming INT8/FP8 checkpoint loading shrinks linear HBM ~2x with 3-mode parity."""
    ckpt_dir = _write_cyclic_qwen_dir(tmp_path / "qwen_ckpt")
    r_none, _ = load_qwen_runner(
        str(ckpt_dir), max_seq_len=512, dtype_name="bfloat16", tile_size=128, tp_size=1, enable_xla_cache=False, weight_quant="none"
    )
    r_int8, _ = load_qwen_runner(
        str(ckpt_dir), max_seq_len=512, dtype_name="bfloat16", tile_size=128, tp_size=1, enable_xla_cache=False, weight_quant="int8"
    )
    r_fp8, _ = load_qwen_runner(
        str(ckpt_dir), max_seq_len=512, dtype_name="bfloat16", tile_size=128, tp_size=1, enable_xla_cache=False, weight_quant="fp8"
    )

    assert r_none.weight_quant == "none" and r_none.weights.wq.dtype == jnp.bfloat16
    assert r_int8.weight_quant == "int8" and r_int8.weights.wq.dtype == jnp.int8
    assert r_int8.weights.lm_head.dtype == jnp.int8
    assert r_int8.weights.embed.dtype == jnp.bfloat16
    assert r_int8.weights.wq_scale is not None and r_int8.weights.wq_scale.dtype == jnp.float32
    assert r_fp8.weight_quant == "fp8" and r_fp8.weights.wq.dtype == jnp.float8_e4m3fn
    assert r_fp8.weights.lm_head.dtype == jnp.float8_e4m3fn

    base_linear_bytes = r_none.weights.linear_weight_bytes()
    assert r_int8.weights.linear_weight_bytes() < base_linear_bytes * 0.65
    assert r_fp8.weights.linear_weight_bytes() < base_linear_bytes * 0.65
    assert r_int8.model_hbm_bytes < r_none.model_hbm_bytes
    assert r_fp8.model_hbm_bytes < r_none.model_hbm_bytes

    prompt = [10, 11, 12, 13] * 12 + [10]
    expected = [11, 12, 13, 10] * 4
    for mode_name, runner in (("int8", r_int8), ("fp8", r_fp8)):
        mv = runner.run_vanilla_jax(prompt, max_new_tokens=16)
        ms = runner.run_static_arena(prompt, max_new_tokens=16)
        mt = runner.run_tpuforge_tree(prompt, max_new_tokens=16)
        assert mv.token_ids == expected
        assert ms.token_ids == expected
        assert mt.token_ids == expected
        assert mt.compile_count == 0
        assert mt.mean_tokens_per_decode_pass > 2.0
        assert mt.extra["weight_quant"] == mode_name


def test_qwen3_block_tree_with_int8_target_exact_parity(tmp_path: Path) -> None:
    """Verify Qwen3 INT8 target + block-diffusion tree drafter keeps exact greedy parity and 0 compiles."""
    target_dir, draft_dir = _write_qwen3_and_block_drafter_pair(tmp_path)
    runner, _, _ = load_qwen_block_tree_runner(
        str(target_dir),
        str(draft_dir),
        max_seq_len=512,
        dtype_name="bfloat16",
        tile_size=128,
        tp_size=1,
        top_k=4,
        enable_xla_cache=False,
        weight_quant="int8",
        draft_weight_quant="none",
    )
    assert runner.target.weights.wq.dtype == jnp.int8
    assert runner.draft_weights.wq.dtype == jnp.bfloat16

    runner.warmup()
    cold_prompt = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mv = runner.target.run_vanilla_jax(cold_prompt, max_new_tokens=21)
    ms = runner.target.run_static_arena(cold_prompt, max_new_tokens=21)
    mbt = runner.run_block_tree(cold_prompt, max_new_tokens=21)

    expected = [8 + ((i + 1) % 16) for i in range(21)]
    assert mv.token_ids == expected
    assert ms.token_ids == expected
    assert mbt.token_ids == expected
    assert mbt.mean_tokens_per_decode_pass >= 5.0
    assert mbt.compile_count == 0
    assert mbt.extra["weight_quant"] == "int8"
    assert mbt.extra["draft_weight_quant"] == "none"

    runner_qdraft, _, _ = load_qwen_block_tree_runner(
        str(target_dir),
        str(draft_dir),
        max_seq_len=512,
        dtype_name="bfloat16",
        tile_size=128,
        tp_size=1,
        top_k=4,
        enable_xla_cache=False,
        weight_quant="int8",
        draft_weight_quant="int8",
    )
    assert runner_qdraft.draft_weights.fc.dtype == jnp.int8
    assert runner_qdraft.draft_weights.wq.dtype == jnp.int8
    runner_qdraft.warmup()
    mq = runner_qdraft.run_block_tree(cold_prompt, max_new_tokens=14)
    assert mq.token_ids == expected[:14]
    assert mq.compile_count == 0
    assert mq.extra["draft_weight_quant"] == "int8"


def test_int8_tp2_megatron_sharding_and_zero_captured_constants_subprocess(tmp_path: Path) -> None:
    """Verify INT8 weights + scales shard cleanly under TP=2 with all-reduce HLO and zero captured constants."""
    ckpt_dir = _write_cyclic_qwen_dir(tmp_path / "tp2_int8_ckpt")
    repo_root = Path(__file__).resolve().parent.parent
    script = f"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from benchmarks.qwen_jax import load_qwen_runner

assert len(jax.devices()) == 2, jax.devices()
r1, _ = load_qwen_runner({str(ckpt_dir)!r}, max_seq_len=512, dtype_name="bfloat16", tile_size=128, tp_size=1, enable_xla_cache=False, weight_quant="int8")
r2, _ = load_qwen_runner({str(ckpt_dir)!r}, max_seq_len=512, dtype_name="bfloat16", tile_size=128, tp_size=2, enable_xla_cache=False, weight_quant="int8")
assert r2.weights.wq.dtype == jnp.int8
assert r2.weights.wq.sharding.spec == P(None, None, "tp")
assert r2.weights.wq_scale.sharding.spec == P(None, "tp")
assert r2.weights.wo.sharding.spec == P(None, "tp", None)
assert r2.weights.wo_scale.sharding.spec == P()
np.testing.assert_allclose(np.asarray(r1.weights.wo_scale), np.asarray(r2.weights.wo_scale), rtol=0.0, atol=0.0)

kc2, vc2 = r2._empty_kv_cache()
hlo = r2._static_systolic_tile_step_jit.lower(
    r2._weights_tuple,
    r2._put_rep(0, jnp.int32),
    r2._put_rep([0] * 128, jnp.int32),
    r2._put_rep([0] * 128, jnp.int32),
    r2._put_rep(jnp.zeros((128, 128), jnp.float32)),
    kc2,
    vc2,
).compile().as_text()
assert "all-reduce(" in hlo, "Expected Megatron TP all-reduce in INT8 HLO"

prompt = [10, 11, 12, 13] * 12 + [10]
mv = r1.run_vanilla_jax(prompt, max_new_tokens=16)
ms = r2.run_static_arena(prompt, max_new_tokens=16)
mt = r2.run_tpuforge_tree(prompt, max_new_tokens=16)
assert mv.token_ids == ms.token_ids == mt.token_ids == [11, 12, 13, 10] * 4
assert mt.extra["tp_size"] == 2 and mt.extra["weight_quant"] == "int8"
"""
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "1024"
    env["TPUFORGE_XLA_CACHE_DIR"] = "0"
    py_paths = [str(repo_root), str(repo_root / "build"), str(repo_root / "tests")]
    venv_site = repo_root / ".venv" / "Lib" / "site-packages"
    if venv_site.is_dir():
        py_paths.append(str(venv_site))
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, f"INT8 TP=2 subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "constants were captured during lowering" not in proc.stderr, proc.stderr

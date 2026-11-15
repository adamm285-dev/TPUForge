"""Tests for Google Gemma 4 (dense, MoE 3D stacked, k_eq_v, proportional RoPE, DFlash, TP, mmap)."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.gemma4_jax import (
    Gemma4Config,
    encode_gemma4_chat_messages,
    is_gemma4_model_dir,
    load_gemma4_block_tree_runner,
    load_gemma4_moe_stream_runner,
    load_gemma4_runner,
)
from benchmarks.run_qwen_bench import run_benchmark_suite


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


def _write_gemma4_tokenizer(vocab_size: int, save_path: Path) -> None:
    """Create a minimal WordLevel tokenizer with Gemma 4 turn/channel special tokens."""
    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    id_to_tok = [f"tok{i}" for i in range(vocab_size)]
    id_to_tok[0] = "<unk>"
    id_to_tok[1] = "<eos>"
    id_to_tok[2] = "<bos>"
    extra_specials = [
        "<|turn>",
        "<turn|>",
        "<|think|>",
        "<|channel>",
        "<channel|>",
        "<|mask|>",
    ]
    for offset, name in enumerate(extra_specials):
        id_to_tok[vocab_size - 7 + offset] = name
    vocab = {tok_str: idx for idx, tok_str in enumerate(id_to_tok)}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.add_special_tokens(
        [AddedToken(t, special=True) for t in ["<bos>", "<eos>", *extra_specials]]
    )
    tok.save(str(save_path))


def _create_synthetic_gemma4_dir(
    root: Path,
    *,
    seed: int = 17,
    enable_moe: bool = False,
    attention_k_eq_v: bool = True,
    num_layers: int = 6,
    layer_types: list[str] | None = None,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    num_global_kv_heads: int = 1,
    head_dim: int = 16,
    global_head_dim: int = 32,
    vocab_size: int = 320,
    sliding_window: int = 4,
    num_experts: int = 6,
    top_k_experts: int = 2,
    moe_intermediate_size: int = 16,
    tie_word_embeddings: bool = True,
    final_logit_softcapping: float | None = 30.0,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Build a synthetic Gemma 4 safetensors checkpoint (multimodal wrapper + vision tensor)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def w_mat(rows: int, cols: int, scale: float = 0.25) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    def w_vec(dim: int, base: float = 1.0) -> np.ndarray:
        return (base + rng.standard_normal((dim,)) * 0.1).astype(np.float32)

    eff_types = layer_types or [
        "full_attention" if (i % 3 == 2) else "sliding_attention" for i in range(num_layers)
    ]
    tensors: dict[str, np.ndarray] = {}
    prefix = "model.language_model."
    tensors[f"{prefix}embed_tokens.weight"] = w_mat(vocab_size, hidden_size, scale=0.35)
    tensors[f"{prefix}norm.weight"] = w_vec(hidden_size)
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = w_mat(vocab_size, hidden_size, scale=0.3)

    # Extra vision weights that Gemma4Weights.from_model_dir must ignore cleanly.
    tensors["vision_tower.patch_embed.weight"] = w_mat(8, 8, scale=0.1)

    for i in range(num_layers):
        lp = f"{prefix}layers.{i}."
        is_full = eff_types[i] == "full_attention"
        kvh = num_global_kv_heads if is_full else num_kv_heads
        hd = global_head_dim if is_full else head_dim

        tensors[f"{lp}input_layernorm.weight"] = w_vec(hidden_size)
        tensors[f"{lp}post_attention_layernorm.weight"] = w_vec(hidden_size)
        tensors[f"{lp}pre_feedforward_layernorm.weight"] = w_vec(hidden_size)
        tensors[f"{lp}post_feedforward_layernorm.weight"] = w_vec(hidden_size)
        tensors[f"{lp}layer_scalar"] = np.array([0.85 + 0.05 * i], dtype=np.float32)

        tensors[f"{lp}self_attn.q_proj.weight"] = w_mat(num_heads * hd, hidden_size)
        tensors[f"{lp}self_attn.q_norm.weight"] = w_vec(hd)
        tensors[f"{lp}self_attn.k_proj.weight"] = w_mat(kvh * hd, hidden_size)
        tensors[f"{lp}self_attn.k_norm.weight"] = w_vec(hd)
        if not (is_full and attention_k_eq_v):
            tensors[f"{lp}self_attn.v_proj.weight"] = w_mat(kvh * hd, hidden_size)
        tensors[f"{lp}self_attn.o_proj.weight"] = w_mat(hidden_size, num_heads * hd)

        tensors[f"{lp}mlp.gate_proj.weight"] = w_mat(intermediate_size, hidden_size)
        tensors[f"{lp}mlp.up_proj.weight"] = w_mat(intermediate_size, hidden_size)
        tensors[f"{lp}mlp.down_proj.weight"] = w_mat(hidden_size, intermediate_size)

        if enable_moe:
            tensors[f"{lp}post_feedforward_layernorm_1.weight"] = w_vec(hidden_size)
            tensors[f"{lp}pre_feedforward_layernorm_2.weight"] = w_vec(hidden_size)
            tensors[f"{lp}post_feedforward_layernorm_2.weight"] = w_vec(hidden_size)
            tensors[f"{lp}router.proj.weight"] = w_mat(num_experts, hidden_size, scale=0.4)
            tensors[f"{lp}router.scale"] = w_vec(hidden_size, base=0.7)
            tensors[f"{lp}router.per_expert_scale"] = w_vec(num_experts, base=1.1)
            tensors[f"{lp}experts.gate_up_proj"] = (
                rng.standard_normal((num_experts, 2 * moe_intermediate_size, hidden_size)) * 0.25
            ).astype(np.float32)
            tensors[f"{lp}experts.down_proj"] = (
                rng.standard_normal((num_experts, hidden_size, moe_intermediate_size)) * 0.25
            ).astype(np.float32)

    cfg = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "tie_word_embeddings": tie_word_embeddings,
        "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "num_global_key_value_heads": num_global_kv_heads,
            "head_dim": head_dim,
            "global_head_dim": global_head_dim,
            "rms_norm_eps": 1e-6,
            "vocab_size": vocab_size,
            "sliding_window": sliding_window,
            "layer_types": eff_types,
            "attention_k_eq_v": attention_k_eq_v,
            "tie_word_embeddings": tie_word_embeddings,
            "final_logit_softcapping": final_logit_softcapping,
            "enable_moe_block": enable_moe,
            "num_experts": num_experts if enable_moe else None,
            "top_k_experts": top_k_experts if enable_moe else None,
            "moe_intermediate_size": moe_intermediate_size if enable_moe else None,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "rope_parameters": {
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
                "full_attention": {
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                },
            },
        },
    }
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (root / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [1, vocab_size - 7]}, indent=2), encoding="utf-8"
    )
    _write_gemma4_tokenizer(vocab_size, root / "tokenizer.json")
    _write_safetensors(root / "model.safetensors", tensors)
    return root, tensors


def _create_synthetic_gemma4_dflash_dir(
    root: Path,
    *,
    seed: int = 99,
    hidden_size: int = 32,
    num_target_layers: int = 6,
    target_layer_ids: list[int] | None = None,
    vocab_size: int = 320,
) -> Path:
    """Create a DFlash drafter matching Gemma 4 DFlash schema (4 sliding + 1 full, softcap=30)."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    def w_mat(rows: int, cols: int, scale: float = 0.2) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    tids = target_layer_ids or [0, 2, 4, 5]
    num_layers = 5
    num_heads = 8
    num_kv_heads = 4
    head_dim = 16
    inter = 64
    ctx_in = len(tids) * hidden_size
    mask_id = vocab_size - 2

    tensors: dict[str, np.ndarray] = {
        "fc.weight": w_mat(hidden_size, ctx_in),
        "hidden_norm.weight": np.ones((hidden_size,), dtype=np.float32),
        "norm.weight": np.ones((hidden_size,), dtype=np.float32),
    }
    for i in range(num_layers):
        lp = f"layers.{i}."
        tensors[f"{lp}input_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        tensors[f"{lp}post_attention_layernorm.weight"] = np.ones((hidden_size,), dtype=np.float32)
        tensors[f"{lp}self_attn.q_proj.weight"] = w_mat(num_heads * head_dim, hidden_size)
        tensors[f"{lp}self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        tensors[f"{lp}self_attn.k_proj.weight"] = w_mat(num_kv_heads * head_dim, hidden_size)
        tensors[f"{lp}self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        tensors[f"{lp}self_attn.v_proj.weight"] = w_mat(num_kv_heads * head_dim, hidden_size)
        tensors[f"{lp}self_attn.o_proj.weight"] = w_mat(hidden_size, num_heads * head_dim)
        tensors[f"{lp}mlp.gate_proj.weight"] = w_mat(inter, hidden_size)
        tensors[f"{lp}mlp.up_proj.weight"] = w_mat(inter, hidden_size)
        tensors[f"{lp}mlp.down_proj.weight"] = w_mat(hidden_size, inter)

    dcfg = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "intermediate_size": inter,
        "vocab_size": vocab_size,
        "head_dim": head_dim,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "sliding_window": 32,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "dflash_config": {
            "block_size": 4,
            "mask_token_id": mask_id,
            "target_layer_ids": tids,
            "num_target_layers": num_target_layers,
            "final_logit_softcapping": 30.0,
            "input_embedding_scale": 1.0,
            "output_multiplier": 1.0,
        },
    }
    (root / "config.json").write_text(json.dumps(dcfg, indent=2), encoding="utf-8")
    _write_safetensors(root / "model.safetensors", tensors)
    return root


def _np_rms(x: np.ndarray, w: np.ndarray | None, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    out = xf * inv
    if w is not None:
        out = out * w.astype(np.float32)
    return out


def _np_rope_prop(
    x: np.ndarray, pos: np.ndarray, head_dim: int, theta: float, partial_factor: float
) -> np.ndarray:
    half = head_dim // 2
    angles_cnt = int(partial_factor * head_dim // 2)
    exps = (2.0 * np.arange(angles_cnt, dtype=np.float32)) / float(head_dim)
    inv_rot = 1.0 / (theta**exps)
    inv = np.concatenate([inv_rot, np.zeros((half - angles_cnt,), dtype=np.float32)], axis=0)
    freqs = pos.astype(np.float32)[:, None] * inv[None, :]
    cos = np.cos(freqs)[:, None, :]
    sin = np.sin(freqs)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _np_gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * (x**3))))


def _np_softmax(x: np.ndarray) -> np.ndarray:
    xm = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - xm)
    return e / np.sum(e, axis=-1, keepdims=True)


def _np_gemma4_prefill_logits(
    cfg: Gemma4Config, tensors: dict[str, np.ndarray], prompt_tokens: list[int]
) -> np.ndarray:
    """Independent pure-NumPy FP32 reference for Gemma 4 forward pass."""
    p = "model.language_model."
    seq = len(prompt_tokens)
    pos = np.arange(seq, dtype=np.int32)
    x = tensors[f"{p}embed_tokens.weight"][np.asarray(prompt_tokens)] * math.sqrt(cfg.hidden_size)

    for i, lt in enumerate(cfg.layer_types):
        lp = f"{p}layers.{i}."
        is_full = lt == "full_attention"
        kvh = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
        hd = cfg.global_head_dim if is_full else cfg.head_dim
        theta = cfg.full_rope_theta if is_full else cfg.sliding_rope_theta
        pf = cfg.full_partial_rotary_factor if is_full else cfg.sliding_partial_rotary_factor

        h = _np_rms(x, tensors[f"{lp}input_layernorm.weight"], cfg.rms_norm_eps)
        q = (h @ tensors[f"{lp}self_attn.q_proj.weight"].T).reshape(seq, cfg.num_attention_heads, hd)
        k = (h @ tensors[f"{lp}self_attn.k_proj.weight"].T).reshape(seq, kvh, hd)
        if f"{lp}self_attn.v_proj.weight" in tensors:
            v = (h @ tensors[f"{lp}self_attn.v_proj.weight"].T).reshape(seq, kvh, hd)
        else:
            v = k.copy()

        q = _np_rope_prop(_np_rms(q, tensors[f"{lp}self_attn.q_norm.weight"], cfg.rms_norm_eps), pos, hd, theta, pf)
        k = _np_rope_prop(_np_rms(k, tensors[f"{lp}self_attn.k_norm.weight"], cfg.rms_norm_eps), pos, hd, theta, pf)
        v = _np_rms(v, None, cfg.rms_norm_eps)

        rep = cfg.num_attention_heads // kvh
        k_e = np.repeat(k, rep, axis=1)
        v_e = np.repeat(v, rep, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k_e)  # scaling = 1.0
        diff = pos[:, None] - pos[None, :]
        vis = diff >= 0
        if not is_full:
            vis = vis & (diff < cfg.sliding_window)
        scores = scores + np.where(vis, 0.0, -1e9)[None, :, :]
        probs = _np_softmax(scores)
        ctx = np.einsum("hqk,khd->qhd", probs, v_e).reshape(seq, cfg.num_attention_heads * hd)
        attn_out = ctx @ tensors[f"{lp}self_attn.o_proj.weight"].T
        x_mid = x + _np_rms(attn_out, tensors[f"{lp}post_attention_layernorm.weight"], cfg.rms_norm_eps)

        h_ff = _np_rms(x_mid, tensors[f"{lp}pre_feedforward_layernorm.weight"], cfg.rms_norm_eps)
        gate = h_ff @ tensors[f"{lp}mlp.gate_proj.weight"].T
        up = h_ff @ tensors[f"{lp}mlp.up_proj.weight"].T
        mlp_out = (_np_gelu_tanh(gate) * up) @ tensors[f"{lp}mlp.down_proj.weight"].T

        if cfg.enable_moe_block:
            h1 = _np_rms(mlp_out, tensors[f"{lp}post_feedforward_layernorm_1.weight"], cfg.rms_norm_eps)
            r_in = _np_rms(x_mid, None, cfg.rms_norm_eps) * tensors[f"{lp}router.scale"] * (cfg.hidden_size**-0.5)
            r_probs = _np_softmax(r_in @ tensors[f"{lp}router.proj.weight"].T)
            top_idx = np.argsort(-r_probs, axis=-1, kind="stable")[:, : cfg.num_experts_per_tok]
            top_vals = np.take_along_axis(r_probs, top_idx, axis=-1)
            top_vals = (top_vals / np.sum(top_vals, axis=-1, keepdims=True)) * tensors[
                f"{lp}router.per_expert_scale"
            ][top_idx]

            x_exp = _np_rms(x_mid, tensors[f"{lp}pre_feedforward_layernorm_2.weight"], cfg.rms_norm_eps)
            gu = tensors[f"{lp}experts.gate_up_proj"]
            dp = tensors[f"{lp}experts.down_proj"]
            moe_out = np.zeros_like(x_exp)
            mi = cfg.moe_intermediate_size
            for t in range(seq):
                for s in range(cfg.num_experts_per_tok):
                    eid = int(top_idx[t, s])
                    w = float(top_vals[t, s])
                    eg = x_exp[t] @ gu[eid, :mi, :].T
                    eu = x_exp[t] @ gu[eid, mi:, :].T
                    eo = (_np_gelu_tanh(eg) * eu) @ dp[eid].T
                    moe_out[t] += w * eo
            h2 = _np_rms(moe_out, tensors[f"{lp}post_feedforward_layernorm_2.weight"], cfg.rms_norm_eps)
            ffn_in = h1 + h2
        else:
            ffn_in = mlp_out

        x = (x_mid + _np_rms(ffn_in, tensors[f"{lp}post_feedforward_layernorm.weight"], cfg.rms_norm_eps)) * tensors[
            f"{lp}layer_scalar"
        ]

    final_h = _np_rms(x[-1], tensors[f"{p}norm.weight"], cfg.rms_norm_eps)
    logits = final_h @ tensors[f"{p}embed_tokens.weight"].T
    if cfg.final_logit_softcapping:
        cap = float(cfg.final_logit_softcapping)
        logits = np.tanh(logits / cap) * cap
    return logits.astype(np.float32)


def test_gemma4_math_parity_periodic_moe_vs_numpy_ref(tmp_path: Path) -> None:
    ckpt_dir, tensors = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_moe_ref",
        seed=42,
        enable_moe=True,
        attention_k_eq_v=True,
        num_layers=6,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        sliding_window=3,
    )
    assert is_gemma4_model_dir(ckpt_dir)
    runner, _ = load_gemma4_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        enable_xla_cache=False,
    )
    assert runner.config.periodic_block_params() == (2, 2)
    assert runner.weights.lm_head is None  # Tied embed reused without duplicate matrix.

    # Prompt longer than sliding_window=3 tests both sliding cache eviction and N=1 vs N=128 paths.
    prompt = [2, 11, 23, 37, 41, 53]
    jax_logits = runner.compute_prefill_last_logits(prompt)
    np_logits = _np_gemma4_prefill_logits(runner.config, tensors, prompt)
    assert np.max(np.abs(jax_logits - np_logits)) < 1e-4

    # Verify N=1 (vanilla decode) and N=128 (static_arena all-expert batched dot) produce exact steps.
    safe_eos = [1]
    res_v = runner.run_vanilla_jax(prompt, max_new_tokens=6, eos_token_ids=safe_eos)
    res_s = runner.run_static_arena(prompt, max_new_tokens=6, eos_token_ids=safe_eos)
    assert res_v.token_ids == res_s.token_ids
    assert len(res_v.token_ids) == 6

    seq = list(prompt)
    for tok in res_v.token_ids:
        ref_step_logits = _np_gemma4_prefill_logits(runner.config, tensors, seq)
        assert int(np.argmax(ref_step_logits)) == tok
        seq.append(tok)


@pytest.mark.parametrize("weight_quant", ["none", "int8", "fp8"])
def test_gemma4_four_modes_quant_sweep_and_ple_guard(tmp_path: Path, weight_quant: str) -> None:
    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / f"gemma4_dense_{weight_quant}",
        seed=101,
        enable_moe=False,
        num_layers=6,
    )
    draft_dir = _create_synthetic_gemma4_dflash_dir(
        tmp_path / f"gemma4_draft_{weight_quant}",
        seed=202,
        num_target_layers=6,
    )

    block_runner, _, _ = load_gemma4_block_tree_runner(
        str(ckpt_dir),
        str(draft_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        top_k=4,
        enable_xla_cache=False,
        weight_quant=weight_quant,
    )
    block_runner.warmup()
    target = block_runner.target

    # Guard against unsupported E2B/E4B Per-Layer Embeddings and KV sharing.
    bad_dir = tmp_path / f"bad_ple_{weight_quant}"
    shutil.copytree(ckpt_dir, bad_dir)
    bad_cfg = json.loads((bad_dir / "config.json").read_text(encoding="utf-8"))
    bad_cfg["text_config"]["hidden_size_per_layer_input"] = 256
    (bad_dir / "config.json").write_text(json.dumps(bad_cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Per-Layer Embeddings"):
        Gemma4Config.from_dir(bad_dir)

    # Zero-recompile sweep across unaligned prompt lengths.
    s_compiles = []
    v_compiles = []
    for p in ([2, 8, 9, 10, 11], [2, 12, 13, 14, 15, 16, 17, 18, 19], [2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]):
        ms = target.run_static_arena(p, max_new_tokens=3, eos_token_ids=[1])
        mv = target.run_vanilla_jax(p, max_new_tokens=3, eos_token_ids=[1])
        s_compiles.append(ms.compile_count)
        v_compiles.append(mv.compile_count)
    assert sum(s_compiles) == 1
    assert sum(v_compiles) >= 3

    prompt = [2, 12, 14, 16, 12, 14, 16, 18, 12, 14, 16]
    mv = target.run_vanilla_jax(prompt, max_new_tokens=10, eos_token_ids=[1])
    ms = target.run_static_arena(prompt, max_new_tokens=10, eos_token_ids=[1])
    mt = target.run_tpuforge_tree(prompt, max_new_tokens=10, eos_token_ids=[1])
    mbt = block_runner.run_block_tree(prompt, max_new_tokens=10, eos_token_ids=[1])
    assert mv.token_ids == ms.token_ids == mt.token_ids == mbt.token_ids
    assert len(mbt.token_ids) == 10


def test_gemma4_moe_mmap_streaming_parity_and_hbm_isolation(tmp_path: Path) -> None:
    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_moe_stream",
        seed=77,
        enable_moe=True,
        num_layers=6,
        hidden_size=32,
        intermediate_size=64,
        num_experts=24,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    hbm_runner, _ = load_gemma4_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        enable_xla_cache=False,
        weight_quant="int8",
    )
    stream_runner, _ = load_gemma4_moe_stream_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        weight_quant="int8",
        host_store_mode="mmap",
        stage_experts=3,  # Smaller than total unique active experts -> exercises multi-chunk staging.
        enable_xla_cache=False,
    )
    try:
        assert stream_runner.trunk_hbm_bytes < 0.70 * hbm_runner.model_hbm_bytes
        prompt = [2, 15, 22, 29, 36, 15, 22, 29, 36, 15, 22, 29, 36]
        m_hbm = hbm_runner.run_vanilla_jax(prompt, max_new_tokens=10, eos_token_ids=[1])
        m_sv = stream_runner.run_vanilla_jax(prompt, max_new_tokens=10, eos_token_ids=[1])
        m_ss = stream_runner.run_static_arena(prompt, max_new_tokens=10, eos_token_ids=[1])
        m_st = stream_runner.run_tpuforge_tree(prompt, max_new_tokens=10, eos_token_ids=[1])
        assert m_hbm.token_ids == m_sv.token_ids == m_ss.token_ids == m_st.token_ids
        # Verify zero steady-state recompiles on static tree after initial static_arena compile.
        assert m_st.compile_count == 0
        assert m_st.extra["staged_chunks_total"] >= 1
    finally:
        stream_runner.close()

    # Confirm Windows mmap handles are completely released.
    shutil.rmtree(ckpt_dir)
    assert not ckpt_dir.exists()


def test_gemma4_tp2_replicated_global_kv_subprocess(tmp_path: Path) -> None:
    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_tp2",
        seed=313,
        enable_moe=True,
        num_layers=6,
        hidden_size=32,
        intermediate_size=64,
        num_heads=8,
        num_kv_heads=4,
        num_global_kv_heads=1,  # 1 < tp_size=2 exercises replicated global KV + sharded sliding KV.
        moe_intermediate_size=16,
    )
    r_tp1, _ = load_gemma4_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        enable_xla_cache=False,
    )
    prompt = [2, 10, 14, 18, 22, 10, 14, 18]
    ref_tokens = r_tp1.run_vanilla_jax(prompt, max_new_tokens=6, eos_token_ids=[1]).token_ids

    script = f"""
import json, jax, numpy as np
from benchmarks.gemma4_jax import load_gemma4_runner
assert len(jax.devices()) == 2
runner, _ = load_gemma4_runner(
    {str(ckpt_dir)!r},
    max_seq_len=256,
    dtype_name="float32",
    tile_size=128,
    tp_size=2,
    enable_xla_cache=False,
)
k_c, v_c = runner._empty_kv_caches()
lowered = runner._static_systolic_tile_step_jit.lower(
    runner._weights_tuple,
    runner._put_rep(0, dtype=jax.numpy.int32),
    runner._put_rep(np.zeros((128,), dtype=np.int32)),
    runner._put_rep(np.arange(128, dtype=np.int32)),
    runner._put_rep(np.zeros((128, 128), dtype=np.float32)),
    k_c,
    v_c,
)
mlir_bytes = len(lowered.as_text().encode("utf-8"))
compiled_hlo = lowered.compile().as_text()
prompt = {prompt!r}
mv = runner.run_vanilla_jax(prompt, max_new_tokens=6, eos_token_ids=[1]).token_ids
ms = runner.run_static_arena(prompt, max_new_tokens=6, eos_token_ids=[1]).token_ids
mt = runner.run_tpuforge_tree(prompt, max_new_tokens=6, eos_token_ids=[1]).token_ids
print(json.dumps({{
    "has_all_reduce": "all-reduce(" in compiled_hlo,
    "mlir_bytes": mlir_bytes,
    "mv": mv,
    "ms": ms,
    "mt": mt,
}}))
"""
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    py_paths = [str(repo_root), str(repo_root / "build")]
    venv_site = repo_root / ".venv" / "Lib" / "site-packages"
    if venv_site.is_dir():
        py_paths.append(str(venv_site))
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["has_all_reduce"] is True
    assert payload["mlir_bytes"] < 200_000
    assert payload["mv"] == payload["ms"] == payload["mt"] == ref_tokens


def test_encode_gemma4_chat_messages_and_cli_dispatch(tmp_path: Path) -> None:
    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_cli",
        seed=55,
        enable_moe=False,
        num_layers=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    )
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ckpt_dir / "tokenizer.json"))
    msgs = [
        {"role": "system", "content": "tok10"},
        {"role": "user", "content": "tok11 tok12"},
    ]
    ids_no_think = encode_gemma4_chat_messages(tok, msgs, enable_thinking=False)
    ids_think = encode_gemma4_chat_messages(tok, msgs, enable_thinking=True)
    bos_id = tok.token_to_id("<bos>")
    turn_open = tok.token_to_id("<|turn>")
    turn_close = tok.token_to_id("<turn|>")
    think_tok = tok.token_to_id("<|think|>")
    chan_open = tok.token_to_id("<|channel>")
    chan_close = tok.token_to_id("<channel|>")

    assert ids_no_think[0] == bos_id
    assert turn_open in ids_no_think and turn_close in ids_no_think
    assert chan_open in ids_no_think and chan_close in ids_no_think
    assert think_tok not in ids_no_think
    assert think_tok in ids_think and chan_open in ids_think and chan_close not in ids_think

    report = run_benchmark_suite(
        model_id=str(ckpt_dir),
        max_seq_len=512,
        max_new_tokens=4,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
    )
    assert report["arch"] == "gemma4"
    assert report["workload_1_prompt_sweep"]["static_arena_compiles"] == 1
    assert report["workload_2_code_refactor"]["static_tree_exact_parity"] is True
    assert report["workload_3_cold_codegen"]["vanilla_jax"]["generated_len"] >= 1

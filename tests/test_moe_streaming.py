"""Tests for M-008: host mmap/DRAM sparse MoE expert streaming and systolic tree amortization."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.moe_streaming_jax import (
    MoeConfig,
    load_moe_streaming_runner,
    run_moe_benchmark_suite,
)


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
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(save_path))


def _np_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x_f = x.astype(np.float32)
    var = np.mean(x_f * x_f, axis=-1, keepdims=True)
    return (x_f / np.sqrt(var + eps)) * weight.astype(np.float32)


def _np_rope(x: np.ndarray, positions: np.ndarray, head_dim: int, theta: float) -> np.ndarray:
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / float(head_dim)))
    freqs = positions.astype(np.float32)[:, None] * inv_freq[None, :]
    cos = np.cos(freqs)[:, None, :]
    sin = np.sin(freqs)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(np.float32)


def _np_silu(x: np.ndarray) -> np.ndarray:
    x_f = x.astype(np.float32)
    return x_f / (1.0 + np.exp(-x_f))


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_f = x.astype(np.float32)
    x_max = np.max(x_f, axis=axis, keepdims=True)
    exps = np.exp(x_f - x_max)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def _np_reference_prefill_last_logits(
    ckpt_dir: Path,
    cfg: MoeConfig,
    tensors: dict[str, np.ndarray],
    prompt_tokens: list[int],
) -> np.ndarray:
    """Pure-NumPy FP32 dense-loop MoE reference across OLMoE and Qwen3-MoE checkpoints."""
    del ckpt_dir
    seq_len = len(prompt_tokens)
    positions = np.arange(seq_len, dtype=np.int32)
    causal_mask = np.where(
        positions[None, :] <= positions[:, None],
        0.0,
        -1e30,
    ).astype(np.float32)

    embed = tensors["model.embed_tokens.weight"].astype(np.float32)
    x = embed[np.asarray(prompt_tokens, dtype=np.int32)].copy()

    def linear(inp: np.ndarray, w_key: str, b_key: str | None = None) -> np.ndarray:
        out = inp @ tensors[w_key].astype(np.float32).T
        if b_key is not None and b_key in tensors:
            out = out + tensors[b_key].astype(np.float32)
        return out

    def apply_qk_norm(
        proj: np.ndarray, key: str, num_heads: int
    ) -> np.ndarray:
        if key not in tensors:
            return proj.reshape(seq_len, num_heads, cfg.head_dim)
        nw = tensors[key].astype(np.float32)
        if nw.shape[-1] == num_heads * cfg.head_dim:
            return _np_rms_norm(proj, nw, cfg.rms_norm_eps).reshape(
                seq_len, num_heads, cfg.head_dim
            )
        return _np_rms_norm(
            proj.reshape(seq_len, num_heads, cfg.head_dim), nw, cfg.rms_norm_eps
        )

    rep = cfg.num_attention_heads // cfg.num_key_value_heads
    scale = 1.0 / np.sqrt(float(cfg.head_dim))

    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        h = _np_rms_norm(x, tensors[f"{p}.input_layernorm.weight"], cfg.rms_norm_eps)

        q = apply_qk_norm(
            linear(h, f"{p}.self_attn.q_proj.weight", f"{p}.self_attn.q_proj.bias"),
            f"{p}.self_attn.q_norm.weight",
            cfg.num_attention_heads,
        )
        k = apply_qk_norm(
            linear(h, f"{p}.self_attn.k_proj.weight", f"{p}.self_attn.k_proj.bias"),
            f"{p}.self_attn.k_norm.weight",
            cfg.num_key_value_heads,
        )
        v = linear(
            h, f"{p}.self_attn.v_proj.weight", f"{p}.self_attn.v_proj.bias"
        ).reshape(seq_len, cfg.num_key_value_heads, cfg.head_dim)

        q = _np_rope(q, positions, cfg.head_dim, cfg.rope_theta)
        k = _np_rope(k, positions, cfg.head_dim, cfg.rope_theta)

        k_exp = np.repeat(k, rep, axis=1)
        v_exp = np.repeat(v, rep, axis=1)
        scores = np.einsum("qhd,khd->hqk", q, k_exp) * scale + causal_mask[None, :, :]
        attn_weights = _np_softmax(scores, axis=-1)
        ctx = np.einsum("hqk,khd->qhd", attn_weights, v_exp).reshape(
            seq_len, cfg.num_attention_heads * cfg.head_dim
        )

        x = x + linear(ctx, f"{p}.self_attn.o_proj.weight")
        h_mlp = _np_rms_norm(x, tensors[f"{p}.post_attention_layernorm.weight"], cfg.rms_norm_eps)

        if not cfg.is_moe_layer(i):
            gate = linear(h_mlp, f"{p}.mlp.gate_proj.weight")
            up = linear(h_mlp, f"{p}.mlp.up_proj.weight")
            x = x + (
                (_np_silu(gate) * up) @ tensors[f"{p}.mlp.down_proj.weight"].astype(np.float32).T
            )
            continue

        x_out = x.copy()
        if f"{p}.mlp.shared_expert.gate_proj.weight" in tensors:
            sg = linear(h_mlp, f"{p}.mlp.shared_expert.gate_proj.weight")
            su = linear(h_mlp, f"{p}.mlp.shared_expert.up_proj.weight")
            shared = (_np_silu(sg) * su) @ tensors[
                f"{p}.mlp.shared_expert.down_proj.weight"
            ].astype(np.float32).T
            if f"{p}.mlp.shared_expert_gate.weight" in tensors:
                sr = tensors[f"{p}.mlp.shared_expert_gate.weight"].astype(np.float32)
                s_gate = 1.0 / (1.0 + np.exp(-(h_mlp @ sr.T)))
                shared = shared * s_gate
            x_out = x_out + shared

        router_logits = h_mlp @ tensors[f"{p}.mlp.gate.weight"].astype(np.float32).T
        probs = _np_softmax(router_logits, axis=-1)
        top_idx = np.argsort(-probs, axis=-1, kind="stable")[:, : cfg.num_experts_per_tok]
        top_vals = np.take_along_axis(probs, top_idx, axis=-1)
        if cfg.norm_topk_prob:
            top_vals = top_vals / np.maximum(np.sum(top_vals, axis=-1, keepdims=True), 1e-12)

        stacked_gu_key = f"{p}.mlp.experts.gate_up_proj"
        stacked_d_key = f"{p}.mlp.experts.down_proj"
        is_stacked = stacked_gu_key in tensors

        moe_acc = np.zeros_like(x_out, dtype=np.float32)
        for t_idx in range(seq_len):
            row_h = h_mlp[t_idx]
            for k_idx in range(cfg.num_experts_per_tok):
                e_id = int(top_idx[t_idx, k_idx])
                w_gating = float(top_vals[t_idx, k_idx])
                if is_stacked:
                    gu = tensors[stacked_gu_key][e_id].astype(np.float32)
                    wg = gu[: cfg.moe_intermediate_size, :]
                    wu = gu[cfg.moe_intermediate_size :, :]
                    wd = tensors[stacked_d_key][e_id].astype(np.float32)
                else:
                    ep = f"{p}.mlp.experts.{e_id}"
                    wg = tensors[f"{ep}.gate_proj.weight"].astype(np.float32)
                    wu = tensors[f"{ep}.up_proj.weight"].astype(np.float32)
                    wd = tensors[f"{ep}.down_proj.weight"].astype(np.float32)
                g_val = row_h @ wg.T
                u_val = row_h @ wu.T
                e_out = (_np_silu(g_val) * u_val) @ wd.T
                moe_acc[t_idx] += w_gating * e_out
        x = x_out + moe_acc

    h_final = _np_rms_norm(x[-1:], tensors["model.norm.weight"], cfg.rms_norm_eps)
    lm_w = (
        tensors["lm_head.weight"]
        if "lm_head.weight" in tensors
        else tensors["model.embed_tokens.weight"]
    )
    logits = h_final @ lm_w.astype(np.float32).T
    return logits[0]


def _write_random_moe_checkpoint(
    model_dir: Path,
    *,
    model_type: str,
    stacked_experts: bool,
    flattened_qk_norm: bool,
    decoder_sparse_step: int = 1,
    include_shared_expert: bool = False,
    tie_word_embeddings: bool = False,
    seed: int = 7,
) -> tuple[Path, MoeConfig, dict[str, np.ndarray]]:
    """Create a realistic FP32 HF OLMoE or Qwen3-MoE checkpoint for numerical reference parity."""
    model_dir.mkdir(parents=True, exist_ok=True)
    hidden = 64
    num_layers = 2
    num_q = 4
    num_kv = 4 if flattened_qk_norm else 2
    head_dim = 16
    inter = 96
    moe_inter = 48
    num_experts = 6
    top_k = 2
    vocab = 288
    eos_id = 286

    _write_word_tokenizer(vocab, model_dir / "tokenizer.json")
    cfg_dict: dict[str, Any] = {
        "model_type": model_type,
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "intermediate_size": inter,
        "moe_intermediate_size": moe_inter,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "decoder_sparse_step": decoder_sparse_step,
        "shared_expert_intermediate_size": moe_inter if include_shared_expert else 0,
        "rms_norm_eps": 1e-5 if "olmoe" in model_type else 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tie_word_embeddings,
        "eos_token_id": eos_id,
    }
    if "olmoe" not in model_type:
        cfg_dict["norm_topk_prob"] = True

    (model_dir / "config.json").write_text(json.dumps(cfg_dict), encoding="utf-8")
    cfg = MoeConfig.from_dir(model_dir)

    rng = np.random.default_rng(seed)

    def randn(shape: tuple[int, ...], std: float = 0.08) -> np.ndarray:
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": randn((vocab, hidden), 0.12),
        "model.norm.weight": rng.uniform(0.8, 1.2, size=(hidden,)).astype(np.float32),
    }
    if not tie_word_embeddings:
        tensors["lm_head.weight"] = randn((vocab, hidden), 0.08)

    for i in range(num_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = rng.uniform(0.8, 1.2, size=(hidden,)).astype(
            np.float32
        )
        tensors[f"{p}.post_attention_layernorm.weight"] = rng.uniform(
            0.8, 1.2, size=(hidden,)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = randn((num_q * head_dim, hidden))
        tensors[f"{p}.self_attn.k_proj.weight"] = randn((num_kv * head_dim, hidden))
        tensors[f"{p}.self_attn.v_proj.weight"] = randn((num_kv * head_dim, hidden))
        tensors[f"{p}.self_attn.o_proj.weight"] = randn((hidden, num_q * head_dim))

        q_nw_dim = (num_q * head_dim) if flattened_qk_norm else head_dim
        k_nw_dim = (num_kv * head_dim) if flattened_qk_norm else head_dim
        tensors[f"{p}.self_attn.q_norm.weight"] = rng.uniform(0.8, 1.2, size=(q_nw_dim,)).astype(
            np.float32
        )
        tensors[f"{p}.self_attn.k_norm.weight"] = rng.uniform(0.8, 1.2, size=(k_nw_dim,)).astype(
            np.float32
        )

        if not cfg.is_moe_layer(i):
            tensors[f"{p}.mlp.gate_proj.weight"] = randn((inter, hidden))
            tensors[f"{p}.mlp.up_proj.weight"] = randn((inter, hidden))
            tensors[f"{p}.mlp.down_proj.weight"] = randn((hidden, inter))
            continue

        tensors[f"{p}.mlp.gate.weight"] = randn((num_experts, hidden), 0.25)
        if include_shared_expert:
            tensors[f"{p}.mlp.shared_expert.gate_proj.weight"] = randn((moe_inter, hidden))
            tensors[f"{p}.mlp.shared_expert.up_proj.weight"] = randn((moe_inter, hidden))
            tensors[f"{p}.mlp.shared_expert.down_proj.weight"] = randn((hidden, moe_inter))
            tensors[f"{p}.mlp.shared_expert_gate.weight"] = randn((1, hidden), 0.2)

        if stacked_experts:
            tensors[f"{p}.mlp.experts.gate_up_proj"] = randn((num_experts, 2 * moe_inter, hidden))
            tensors[f"{p}.mlp.experts.down_proj"] = randn((num_experts, hidden, moe_inter))
        else:
            for e_id in range(num_experts):
                ep = f"{p}.mlp.experts.{e_id}"
                tensors[f"{ep}.gate_proj.weight"] = randn((moe_inter, hidden))
                tensors[f"{ep}.up_proj.weight"] = randn((moe_inter, hidden))
                tensors[f"{ep}.down_proj.weight"] = randn((hidden, moe_inter))

    _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir, cfg, tensors


def _write_cyclic_moe_checkpoint(
    model_dir: Path,
    *,
    num_layers: int = 4,
    hidden: int = 128,
    moe_inter: int = 128,
    num_experts: int = 16,
    top_k: int = 2,
    route_token_mod: int = 16,
    stacked_experts: bool = False,
    seed: int = 41,
) -> Path:
    """Create a deterministic cyclic MoE checkpoint that preserves cycle accuracy across quant."""
    model_dir.mkdir(parents=True, exist_ok=True)
    num_q = 4
    num_kv = 2
    head_dim = 32
    vocab = 288
    eos_id = 286

    _write_word_tokenizer(vocab, model_dir / "tokenizer.json")
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": hidden,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_q,
                "num_key_value_heads": num_kv,
                "head_dim": head_dim,
                "vocab_size": vocab,
                "intermediate_size": moe_inter,
                "moe_intermediate_size": moe_inter,
                "num_experts": num_experts,
                "num_experts_per_tok": top_k,
                "norm_topk_prob": True,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "eos_token_id": eos_id,
            }
        ),
        encoding="utf-8",
    )

    embed = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(32):
        embed[tok, tok] = 4.0

    lm_head = np.zeros((vocab, hidden), dtype=np.float32)
    for tok in range(32):
        nxt = 10 + ((tok - 10 + 1) % 4) if 10 <= tok <= 13 else 10
        lm_head[nxt, tok] = 6.0

    # Distinct tokens 10..25 each route to distinct primary experts 0..15 with small runner-up (e+1).
    router_w = np.zeros((num_experts, hidden), dtype=np.float32)
    for idx in range(num_experts):
        tok_feature = 10 + (idx % route_token_mod)
        router_w[idx, tok_feature] += 8.0
        prev_feature = 10 + ((idx - 1) % route_token_mod)
        router_w[idx, prev_feature] += 3.0

    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": lm_head,
        "model.norm.weight": np.ones((hidden,), dtype=np.float32),
    }

    rng = np.random.default_rng(seed)
    for i in range(num_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones((hidden,), dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = rng.normal(
            0.0, 0.015, size=(num_q * head_dim, hidden)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.k_proj.weight"] = rng.normal(
            0.0, 0.015, size=(num_kv * head_dim, hidden)
        ).astype(np.float32)
        tensors[f"{p}.self_attn.v_proj.weight"] = np.zeros(
            (num_kv * head_dim, hidden), dtype=np.float32
        )
        tensors[f"{p}.self_attn.o_proj.weight"] = np.zeros(
            (hidden, num_q * head_dim), dtype=np.float32
        )
        tensors[f"{p}.self_attn.q_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        tensors[f"{p}.self_attn.k_norm.weight"] = np.ones((head_dim,), dtype=np.float32)
        tensors[f"{p}.mlp.gate.weight"] = router_w.copy()

        # Non-zero bounded SwiGLU weights trigger full systolic execution while preserving cycles.
        if stacked_experts:
            tensors[f"{p}.mlp.experts.gate_up_proj"] = rng.normal(
                0.0, 0.015, size=(num_experts, 2 * moe_inter, hidden)
            ).astype(np.float32)
            tensors[f"{p}.mlp.experts.down_proj"] = rng.normal(
                0.0, 0.008, size=(num_experts, hidden, moe_inter)
            ).astype(np.float32)
        else:
            for e_id in range(num_experts):
                ep = f"{p}.mlp.experts.{e_id}"
                tensors[f"{ep}.gate_proj.weight"] = rng.normal(
                    0.0, 0.015, size=(moe_inter, hidden)
                ).astype(np.float32)
                tensors[f"{ep}.up_proj.weight"] = rng.normal(
                    0.0, 0.015, size=(moe_inter, hidden)
                ).astype(np.float32)
                tensors[f"{ep}.down_proj.weight"] = rng.normal(
                    0.0, 0.008, size=(hidden, moe_inter)
                ).astype(np.float32)

    _write_safetensors(model_dir / "model.safetensors", tensors)
    return model_dir


def test_olmoe_and_qwen3_moe_config_and_full_hbm_reference_parity(tmp_path: Path) -> None:
    """Verify OLMoE (flattened qk_norm, 2D experts) and Qwen3-MoE (per-head qk_norm, 3D + dense)."""
    prompt = [5, 19, 3, 27, 42, 11, 88, 17, 64]

    olmoe_dir, olmoe_cfg, olmoe_tensors = _write_random_moe_checkpoint(
        tmp_path / "olmoe_ckpt",
        model_type="olmoe",
        stacked_experts=False,
        flattened_qk_norm=True,
        decoder_sparse_step=1,
        include_shared_expert=False,
        tie_word_embeddings=True,
        seed=11,
    )
    assert olmoe_cfg.norm_topk_prob is False
    assert olmoe_cfg.is_moe_layer(0) is True
    assert olmoe_cfg.is_moe_layer(1) is True

    ref_olmoe = _np_reference_prefill_last_logits(olmoe_dir, olmoe_cfg, olmoe_tensors, prompt)
    with load_moe_streaming_runner(
        str(olmoe_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size="auto",
        host_store_mode="mmap",
        stage_experts=4,
        enable_xla_cache=False,
    )[0] as runner:
        got_olmoe = runner.compute_prefill_last_logits(prompt)
        max_err = float(np.max(np.abs(ref_olmoe - got_olmoe)))
        assert max_err < 1e-4, f"OLMoE logit mismatch vs FP32 NumPy reference: {max_err}"

        runner.warmup()
        m_van = runner.run_vanilla_moe(prompt, max_new_tokens=8)
        m_stat = runner.run_static_arena_moe(prompt, max_new_tokens=8)
        m_tree = runner.run_tpuforge_tree_moe(prompt, max_new_tokens=8)
        assert m_van.token_ids == m_stat.token_ids == m_tree.token_ids
        assert len(m_van.token_ids) == 8

    qwen3_dir, qwen3_cfg, qwen3_tensors = _write_random_moe_checkpoint(
        tmp_path / "qwen3_moe_ckpt",
        model_type="qwen3_moe",
        stacked_experts=True,
        flattened_qk_norm=False,
        decoder_sparse_step=2,
        include_shared_expert=True,
        tie_word_embeddings=False,
        seed=23,
    )
    assert qwen3_cfg.norm_topk_prob is True
    assert qwen3_cfg.is_moe_layer(0) is False
    assert qwen3_cfg.is_moe_layer(1) is True

    ref_qwen3 = _np_reference_prefill_last_logits(qwen3_dir, qwen3_cfg, qwen3_tensors, prompt)
    with load_moe_streaming_runner(
        str(qwen3_dir),
        max_seq_len=256,
        dtype_name="float32",
        tile_size="auto",
        host_store_mode="mmap",
        stage_experts=4,
        enable_xla_cache=False,
    )[0] as runner:
        got_qwen3 = runner.compute_prefill_last_logits(prompt)
        max_err = float(np.max(np.abs(ref_qwen3 - got_qwen3)))
        assert max_err < 1e-4, f"Qwen3-MoE logit mismatch vs FP32 NumPy reference: {max_err}"

        runner.warmup()
        m_van = runner.run_vanilla_moe(prompt, max_new_tokens=8)
        m_stat = runner.run_static_arena_moe(prompt, max_new_tokens=8)
        m_tree = runner.run_tpuforge_tree_moe(prompt, max_new_tokens=8)
        assert m_van.token_ids == m_stat.token_ids == m_tree.token_ids
        assert len(m_van.token_ids) == 8


def test_moe_mmap_vs_dram_zero_recompile_and_hbm_footprint_isolation(tmp_path: Path) -> None:
    """Verify HBM isolation (<45% total size), multi-chunk streaming, zero recompile, quant parity."""
    ckpt_dir = _write_cyclic_moe_checkpoint(
        tmp_path / "cyclic_16e",
        num_layers=4,
        hidden=128,
        moe_inter=128,
        num_experts=16,
        top_k=2,
        route_token_mod=16,
        stacked_experts=False,
    )

    # Prompt uses 16 distinct routing tokens [10..25], guaranteeing >4 unique experts per tile.
    prompt_multi = [10 + (i % 16) for i in range(19)]
    prompts = [
        prompt_multi,
        [10 + (i % 16) for i in range(53)],
        [10 + (i % 16) for i in range(131)],
    ]

    reference_tokens: list[int] | None = None
    configs = [
        ("mmap", "none"),
        ("dram", "none"),
        ("mmap", "int8"),
        ("dram", "fp8"),
    ]
    for store_mode, quant_mode in configs:
        with load_moe_streaming_runner(
            str(ckpt_dir),
            max_seq_len=384,
            dtype_name="bfloat16",
            tile_size="auto",
            host_store_mode=store_mode,
            stage_experts=4,
            weight_quant=quant_mode,
            lru_cache_experts=8,
            enable_xla_cache=False,
        )[0] as runner:
            pinned_hbm = runner.trunk_hbm_bytes + runner.staging_hbm_bytes
            full_hbm_equiv = runner.trunk_hbm_bytes + runner.total_expert_source_bytes
            assert pinned_hbm < 0.45 * full_hbm_equiv, (
                f"Expected HBM footprint < 45% full equivalent, got {pinned_hbm} / {full_hbm_equiv}"
            )

            runner.warmup()
            for idx, p in enumerate(prompts):
                m_static = runner.run_static_arena_moe(p, max_new_tokens=6)
                m_tree = runner.run_tpuforge_tree_moe(p, max_new_tokens=6)
                assert m_static.compile_count == 0
                assert m_tree.compile_count == 0
                assert m_static.token_ids == m_tree.token_ids
                if idx == 0:
                    # 19 tokens across 4 layers with stage_experts=4 requires >1 chunk per layer.
                    assert m_static.extra["h2d_chunk_transfers"] > 4
                    if reference_tokens is None:
                        reference_tokens = m_static.token_ids
                    else:
                        assert m_static.token_ids == reference_tokens


def test_moe_systolic_tree_pcie_expert_transfer_amortization_and_3mode_parity(
    tmp_path: Path,
) -> None:
    """Verify exact 3-mode parity and PCIe expert H2D amortization via 128-node systolic tree."""
    ckpt_dir = _write_cyclic_moe_checkpoint(
        tmp_path / "cyclic_8e",
        num_layers=2,
        hidden=128,
        moe_inter=128,
        num_experts=8,
        top_k=2,
        route_token_mod=4,
        stacked_experts=True,
    )
    prompt = [10, 11, 12, 13] * 14 + [10]

    with load_moe_streaming_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="bfloat16",
        tile_size="auto",
        host_store_mode="mmap",
        stage_experts=8,
        weight_quant="none",
        enable_xla_cache=False,
    )[0] as runner:
        runner.warmup()
        m_vanilla = runner.run_vanilla_moe(prompt, max_new_tokens=20)
        m_static = runner.run_static_arena_moe(prompt, max_new_tokens=20)
        m_tree = runner.run_tpuforge_tree_moe(prompt, max_new_tokens=20)

    expected = [11, 12, 13, 10] * 5
    assert m_vanilla.token_ids == expected
    assert m_static.token_ids == expected
    assert m_tree.token_ids == expected
    assert m_tree.compile_count == 0
    assert m_tree.mean_tokens_per_decode_pass >= 3.0

    assert m_tree.extra["decode_h2d_chunk_transfers"] < m_vanilla.extra["decode_h2d_chunk_transfers"]
    assert m_tree.extra["decode_experts_per_token"] < m_vanilla.extra["decode_experts_per_token"]
    assert m_tree.extra["decode_routed_expert_bytes"] < m_vanilla.extra["decode_routed_expert_bytes"]


def test_moe_mmap_handle_release_and_cli_report(tmp_path: Path) -> None:
    """Verify mmap handles close cleanly on Windows and benchmark report schema is complete."""
    ckpt_dir = _write_cyclic_moe_checkpoint(
        tmp_path / "cyclic_release",
        num_layers=2,
        hidden=128,
        moe_inter=64,
        num_experts=8,
        top_k=2,
        route_token_mod=4,
        stacked_experts=False,
    )
    report_path = tmp_path / "moe_report.json"

    with load_moe_streaming_runner(
        str(ckpt_dir),
        max_seq_len=256,
        dtype_name="bfloat16",
        tile_size="auto",
        host_store_mode="mmap",
        stage_experts=4,
        lru_cache_experts=0,
        enable_xla_cache=False,
    )[0] as runner:
        metrics = runner.run_tpuforge_tree_moe([10, 11, 12, 13] * 6 + [10], max_new_tokens=8)
        assert metrics.token_ids == [11, 12, 13, 10] * 2

    report = run_moe_benchmark_suite(
        model_id=str(ckpt_dir),
        max_seq_len=256,
        max_new_tokens=8,
        dtype_name="bfloat16",
        tile_size="auto",
        host_store_mode="mmap",
        stage_experts=4,
        weight_quant="none",
        lru_cache_experts=0,
        prompt_tokens=[10, 11, 12, 13] * 6 + [10],
        output_json=report_path,
    )

    # Windows locks unclosed mmap handles; unlink() immediately verifies clean handle disposal.
    (ckpt_dir / "model.safetensors").unlink()

    assert report_path.is_file()
    disk_report = json.loads(report_path.read_text(encoding="utf-8"))
    for payload in (report, disk_report):
        assert payload["exact_parity"] is True
        assert payload["trunk_hbm_bytes"] > 0
        assert payload["staging_hbm_bytes"] > 0
        assert payload["total_expert_source_bytes"] > 0
        assert payload["host_store_mode"] == "mmap"

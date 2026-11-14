"""Tests for O(1) SuffixIndex, Block-TopK systolic tree packing, and DFlash XLA execution."""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import _tpuforge_ext as tpu
import jax
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.dflash_jax import load_qwen_dflash_runner
from benchmarks.qwen_jax import configure_xla_compilation_cache


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


def _build_word_tokenizer(vocab_size: int, save_path: Path) -> Tokenizer:
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = 0
    vocab["<|im_start|>"] = 1
    vocab["<|im_end|>"] = 2
    vocab["<think>"] = 3
    vocab["</think>"] = 4
    model = WordLevel(vocab=vocab, unk_token="<unk>")
    tok = Tokenizer(model)
    tok.pre_tokenizer = Whitespace()
    tok.save(str(save_path))
    return tok


def _build_qwen3_and_dflash_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create a deterministic Qwen3 (q/k norm, no attn bias) + DFlash checkpoint pair."""
    target_dir = tmp_path / "qwen3_target"
    draft_dir = tmp_path / "qwen3_dflash"
    target_dir.mkdir(parents=True, exist_ok=True)
    draft_dir.mkdir(parents=True, exist_ok=True)

    H = 128
    L_t = 4
    L_d = 2
    n_q = 8
    n_kv = 2
    hd = 64  # n_q * hd = 512 != H (exercises Qwen3 explicit head_dim projection)
    inter = 128
    vocab = 288  # ByteTokenizer requires >= 260; keep divisible by 32 for TP=2 sharding
    block_size = 8
    eos_id = 286
    mask_id = 287
    rope_theta = 10.0

    _build_word_tokenizer(vocab, target_dir / "tokenizer.json")

    (target_dir / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": H,
                "num_hidden_layers": L_t,
                "num_attention_heads": n_q,
                "num_key_value_heads": n_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "head_dim": hd,
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
                "hidden_size": H,
                "num_hidden_layers": L_d,
                "num_attention_heads": n_q,
                "num_key_value_heads": n_kv,
                "intermediate_size": inter,
                "vocab_size": vocab,
                "head_dim": hd,
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

    # Embed: one-hot on dims 0..23 plus constant energy anchor on dim 127.
    embed = np.zeros((vocab, H), dtype=np.float32)
    for t in range(24):
        embed[t, t] = 4.0
    embed[eos_id, 64] = 1.0
    embed[mask_id, 31] = 1.0
    embed[mask_id, 127] = 1.0

    # Target cyclic head: tokens 8..23 advance modulo 16; all other tokens emit 8.
    lm_head = np.zeros((vocab, H), dtype=np.float32)
    for t in range(24):
        nxt = 8 + ((t - 8 + 1) % 16) if 8 <= t <= 23 else 8
        lm_head[nxt, t] = 5.0

    t_tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": embed,
        "lm_head.weight": lm_head,
        "model.norm.weight": np.ones((H,), dtype=np.float32),
    }
    for i in range(L_t):
        t_tensors[f"model.layers.{i}.input_layernorm.weight"] = np.ones((H,), dtype=np.float32)
        t_tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones((H,), dtype=np.float32)
        t_tensors[f"model.layers.{i}.self_attn.q_norm.weight"] = np.ones((hd,), dtype=np.float32) * (1.1 if i == 0 else 1.0)
        t_tensors[f"model.layers.{i}.self_attn.k_norm.weight"] = np.ones((hd,), dtype=np.float32) * 0.9
        # Small non-zero Q/K projections exercise q_norm/k_norm without overriding cyclic residual.
        wq_t = np.zeros((n_q * hd, H), dtype=np.float32)
        wq_t[0, 127] = 0.01
        wk_t = np.zeros((n_kv * hd, H), dtype=np.float32)
        wk_t[0, 127] = 0.01
        t_tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = wq_t
        t_tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = wk_t
        t_tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = np.zeros((n_kv * hd, H), dtype=np.float32)
        t_tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = np.zeros((H, n_q * hd), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = np.zeros((inter, H), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.up_proj.weight"] = np.zeros((inter, H), dtype=np.float32)
        t_tensors[f"model.layers.{i}.mlp.down_proj.weight"] = np.zeros((H, inter), dtype=np.float32)
    _write_safetensors(target_dir / "model.safetensors", t_tensors)

    # DFlash Layer 0 uses RoPE relative offsets on freq_idx=1 to route root_tok -> step s (1..7).
    half = hd // 2  # 32
    freq_band = 1
    mask_band = 31
    inv_freq = 1.0 / (rope_theta ** (freq_band / float(half)))  # ~0.866 rad/step

    fc_w = np.zeros((H, 2 * H), dtype=np.float32)
    fc_w[32:64, :32] = 0.02

    d_tensors: dict[str, np.ndarray] = {
        "fc.weight": fc_w,
        "hidden_norm.weight": np.ones((H,), dtype=np.float32),
        "norm.weight": np.ones((H,), dtype=np.float32),
    }

    wq0 = np.zeros((n_q * hd, H), dtype=np.float32)
    wk0 = np.zeros((n_kv * hd, H), dtype=np.float32)
    wv0 = np.zeros((n_kv * hd, H), dtype=np.float32)
    wo0 = np.zeros((H, n_q * hd), dtype=np.float32)

    # KV heads 0 & 1:
    # - root_tok (any t in 8..23 -> nonzero sum) projects k onto freq_band (0.80), v carries token basis
    # - mask_tok (dim 31 == 1) projects k onto mask_band (1.0) with zero v
    # - target_hidden (dim 32) puts small value on mask_band*0 + unused freq 10 so k_ctx stays orthogonal
    for kv_h in range(n_kv):
        base = kv_h * hd
        for t in range(8, 24):
            wk0[base + freq_band, t] = 1.0
            wv0[base + (t - 8), t] = 2.0
        wk0[base + mask_band, 31] = 1.0
        wk0[base + 10, 32] = 1.0

    # Query heads 1..7 encode R(+step_s * inv_freq) on freq_band (weight 0.80) + mask_band (weight 0.60).
    for step_s in range(1, block_size):
        q_base = step_s * hd
        angle = -float(step_s) * inv_freq
        wq0[q_base + freq_band, 31] = 0.80 * math.cos(angle)
        wq0[q_base + half + freq_band, 31] = 0.80 * math.sin(angle)
        wq0[q_base + mask_band, 31] = 0.60
        # Route head step_s's value [0..15] to disjoint scratch dims [step_s*16 .. step_s*16+15]?
        # No: use SwiGLU MLP neurons! Only 16*7=112 neurons fit cleanly in inter=128!
        for j in range(16):
            # Pack (step_s, j) into o_proj dims [32 + (step_s - 1)*13? Wait 7*16=112 > 128-32=96].
            # Wait: fold into 16 shared token dims [32..47] isn't per-step; instead fold into cyclic shifted!
            # Actually at mask positions dims 8..23 are ZERO in x0! Use dims 64..79 for step_s?
            # Target lm_head adds +1 mod 16 on dims 8..23, so aim at j + step_s - 1.
            target_dim = 8 + ((j + step_s - 1) % 16)
            # Only step_s's head should strongly activate: give dim (48 + step_s) the head-activation gate,
            # OR simply sum attn_out directly into target_dim when head step_s dominates (~0.58 vs ~0.05)!
            wo0[target_dim, q_base + j] = 1.0

    # Wait: across 7 heads summing linearly in wo0, 6 off-target heads each contribute <= 0.05 (6*0.05=0.30).
    # Wait: if all tokens share dims 8..23, ALL 6 off-target heads vote for shifted variants (sum 0.30 on distinct indices,
    # Wait: only 1 off-target head votes for each OTHER token, so peak is +0.05 vs +0.58 on the TRUE offset!).
    # Scale wo0 slightly larger (~2.0) and zero MLP so residual preserves peak logit!

    for l_idx in range(L_d):
        d_tensors[f"layers.{l_idx}.input_layernorm.weight"] = np.ones((H,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.post_attention_layernorm.weight"] = np.ones((H,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.self_attn.q_norm.weight"] = np.ones((hd,), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.self_attn.k_norm.weight"] = np.ones((hd,), dtype=np.float32)
        if l_idx == 0:
            d_tensors[f"layers.{l_idx}.self_attn.q_proj.weight"] = wq0
            d_tensors[f"layers.{l_idx}.self_attn.k_proj.weight"] = wk0
            d_tensors[f"layers.{l_idx}.self_attn.v_proj.weight"] = wv0
            d_tensors[f"layers.{l_idx}.self_attn.o_proj.weight"] = wo0 * 2.0
        else:
            d_tensors[f"layers.{l_idx}.self_attn.q_proj.weight"] = np.zeros((n_q * hd, H), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.k_proj.weight"] = np.zeros((n_kv * hd, H), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.v_proj.weight"] = np.zeros((n_kv * hd, H), dtype=np.float32)
            d_tensors[f"layers.{l_idx}.self_attn.o_proj.weight"] = np.zeros((H, n_q * hd), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.gate_proj.weight"] = np.zeros((inter, H), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.up_proj.weight"] = np.zeros((inter, H), dtype=np.float32)
        d_tensors[f"layers.{l_idx}.mlp.down_proj.weight"] = np.zeros((H, inter), dtype=np.float32)

    _write_safetensors(draft_dir / "model.safetensors", d_tensors)
    return target_dir, draft_dir


def test_suffix_index_incremental_parity_and_32k_latency() -> None:
    seqs = [
        [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3],
        [42, 7, 9, 42, 7, 10, 42, 7, 9, 42, 7, 9],
        [5, 5, 5, 5, 5, 5],
    ]
    for seq in seqs:
        idx = tpu.SuffixIndex(seq[:2])
        for tok in seq[2:]:
            idx.append_token(tok)
        assert idx.size() == len(seq)

        expected_tree = tpu.SuffixTreeDrafter.build_tree(seq, 128, 16, 128)
        actual_tree = idx.build_tree(128, 16, 128)
        assert actual_tree.num_nodes() == expected_tree.num_nodes()
        for a, b in zip(actual_tree.nodes(), expected_tree.nodes()):
            assert a.token_id == b.token_id
            assert a.parent_idx == b.parent_idx
            assert a.depth == b.depth
            assert abs(a.draft_prob - b.draft_prob) < 1e-5

    # 32K long-context incremental latency check
    rng = np.random.default_rng(7)
    base_pattern = np.arange(100, 140, dtype=np.int32)
    long_ctx = np.tile(base_pattern, 820)[:32768].copy()
    noise_positions = rng.choice(32768, size=1000, replace=False)
    long_ctx[noise_positions] = rng.integers(200, 5000, size=1000, dtype=np.int32)

    index = tpu.SuffixIndex(long_ctx.tolist())
    for i in range(64):
        index.append_token(int(100 + (i % 40)))

    # Warm once, then verify single decode-step tree build + pack is sub-15ms on CPU
    _ = index.build_tree(128, 16, 128)
    t0 = time.perf_counter()
    tree = index.build_tree(128, 16, 128)
    packed = tree.pack(index.size() - 1)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 15.0, f"Expected <15ms on 32K+ context, got {elapsed_ms:.2f}ms"
    assert tree.num_nodes() > 16
    assert packed["token_ids"].shape == (128,)
    assert packed["additive_mask_f32"].shape == (128, 128)


def test_tree_builder_block_top_k_spine_and_suffix_hybrid() -> None:
    block_steps = 15
    top_k = 8
    draft_ids = np.full((block_steps, top_k), -1, dtype=np.int32)
    draft_probs = np.zeros((block_steps, top_k), dtype=np.float32)

    for s in range(block_steps):
        for k in range(top_k):
            draft_ids[s, k] = 100 + s * 10 + k
            draft_probs[s, k] = 0.90 if k == 0 else (0.10 / float(top_k - 1)) * (0.7**k)

    suffix = tpu.SuffixIndex([113, 114, 201, 202, 203, 113, 114, 201, 202, 203, 113, 114])
    tree = tpu.TreeBuilder.build_from_block_top_k(
        root_last_token=99,
        draft_token_ids=draft_ids,
        draft_probs=draft_probs,
        suffix_fallback=suffix,
        max_nodes=128,
        max_depth=20,
        tile_size=128,
    )

    nodes = tree.nodes()
    assert tree.num_nodes() <= 128
    assert nodes[0].token_id == 99

    # Walk primary spine depths 1..15, then verify suffix extension beyond depth 15
    curr = 0
    for s in range(block_steps):
        expected_tok = 100 + s * 10
        matches = [c for c in nodes[curr].children if nodes[c].token_id == expected_tok]
        assert len(matches) == 1
        curr = matches[0]
        assert nodes[curr].depth == s + 1

    # Suffix fallback continues 114 (-> depth 15: 240 here, wait let's check both ids & suffix!)
    # Check duplicate-free sibling invariant everywhere
    for node in nodes:
        child_tokens = [nodes[c].token_id for c in node.children]
        assert len(child_tokens) == len(set(child_tokens))

    # Also test exact suffix continuation beyond DFlash horizon when top-1 hits suffix trigger
    draft_ids_s = draft_ids.copy()
    draft_ids_s[13, 0] = 113
    draft_ids_s[14, 0] = 114
    tree_ext = tpu.TreeBuilder.build_from_block_top_k(
        root_last_token=99,
        draft_token_ids=draft_ids_s,
        draft_probs=draft_probs,
        suffix_fallback=suffix,
        max_nodes=128,
        max_depth=20,
        tile_size=128,
    )
    max_tree_depth = max(n.depth for n in tree_ext.nodes())
    assert max_tree_depth >= 18
    ext_tokens = {n.token_id for n in tree_ext.nodes()}
    assert {201, 202, 203}.issubset(ext_tokens)


def test_qwen3_dflash_cold_prompt_acceleration_and_exact_parity(tmp_path: Path) -> None:
    cache_dir = configure_xla_compilation_cache(tmp_path / "xla_cache")
    assert cache_dir is not None and cache_dir.is_dir()

    target_dir, draft_dir = _build_qwen3_and_dflash_dirs(tmp_path)
    runner, resolved_t, resolved_d = load_qwen_dflash_runner(
        str(target_dir),
        str(draft_dir),
        max_seq_len=512,
        dtype_name="float32",
        tile_size=128,
        tp_size=1,
        top_k=4,
        enable_xla_cache=False,
    )
    assert resolved_t == target_dir
    assert resolved_d == draft_dir

    # Verify Qwen3 optional fields loaded cleanly (no bias, with q_norm/k_norm)
    assert runner.target.weights.bq is None
    assert runner.target.weights.q_norm is not None
    assert runner.dflash_weights.q_norm is not None

    runner.warmup()
    # Warm vanilla/static on a throwaway run to verify 0 steady-state compiles
    cold_prompt = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    _ = runner.target.run_static_arena(cold_prompt, max_new_tokens=4)

    vanilla = runner.target.run_vanilla_jax(cold_prompt, max_new_tokens=21)
    static = runner.target.run_static_arena(cold_prompt, max_new_tokens=21)
    suffix_only = runner.target.run_tpuforge_tree(cold_prompt, max_new_tokens=21)
    dflash = runner.run_dflash_tree(cold_prompt, max_new_tokens=21)

    expected = [8 + ((i + 1) % 16) for i in range(21)]
    assert vanilla.token_ids == expected
    assert static.token_ids == expected
    assert suffix_only.token_ids == expected
    assert dflash.token_ids == expected

    # Cold non-repeating prompt: suffix stays near 1.0 tok/pass, DFlash hits >= 5.0 tok/pass
    assert suffix_only.mean_tokens_per_decode_pass <= 1.5
    assert dflash.mean_tokens_per_decode_pass >= 5.0
    assert dflash.decode_passes <= 5
    assert dflash.compile_count == 0


def test_qwen3_dflash_tp2_sharding_and_zero_captured_constants_subprocess(tmp_path: Path) -> None:
    target_dir, draft_dir = _build_qwen3_and_dflash_dirs(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    script = f"""
import json
from benchmarks.dflash_jax import load_qwen_dflash_runner

runner, _, _ = load_qwen_dflash_runner(
    {str(target_dir)!r},
    {str(draft_dir)!r},
    max_seq_len=512,
    dtype_name="float32",
    tile_size=128,
    tp_size=2,
    top_k=4,
    enable_xla_cache=False,
)
runner.warmup()
prompt = [0, 1, 2, 3, 4, 5, 6, 7, 8]
res = runner.run_dflash_tree(prompt, max_new_tokens=21)
assert runner.tp_size == 2
print(json.dumps({{"tokens": res.token_ids, "tok_per_pass": res.mean_tokens_per_decode_pass, "compiles": res.compile_count}}))
"""
    env = dict(os.environ)
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2").strip()
    env["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "1024"
    env["TPUFORGE_XLA_CACHE_DIR"] = "0"
    py_paths = [str(repo_root), str(repo_root / "tests")]
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
        check=False,
    )
    assert proc.returncode == 0, f"TP=2 subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "captured" not in proc.stderr.lower(), f"Constant capture warning triggered:\n{proc.stderr}"
    assert "jax.errors" not in proc.stderr
    assert len(jax.devices()) >= 1
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["tokens"] == [8 + ((i + 1) % 16) for i in range(21)]
    assert out["tok_per_pass"] >= 5.0
    assert out["compiles"] == 0

"""Tests for Milestone M-003: Live OpenXLA 128-Node Systolic Tree Speculation + 2D Causal Mask."""

from __future__ import annotations

import math
import numpy as np
import pytest

import jax
import jax.numpy as jnp

import _tpuforge_ext as tpu

VOCAB_SIZE = 32
D_MODEL = 128
NUM_HEADS = 4
HEAD_DIM = 32
MLP_DIM = 128
MAX_SEQ_LEN = 256
TILE_SIZE = 128
NUM_LAYERS = 2


def _build_induction_transformer_params(seed: int = 2026):
    """Build deterministic 2-layer Causal Transformer weights with RoPE + MLP + Induction behavior."""
    rng = np.random.default_rng(seed)
    q_mat, _ = np.linalg.qr(rng.normal(size=(32, 32)).astype(np.float32))
    embed = np.zeros((VOCAB_SIZE, D_MODEL), dtype=np.float32)
    embed[:, :32] = q_mat * 3.0
    embed[:, 32] = 1.0  # beacon bit for Layer 0 RoPE delta=+1 head

    layers = []
    # Layer 0: Head 0 (dims 0..32) detects pos_q - pos_k == 1 via RoPE pair (0, 1)
    # and writes prev_token [0:32] into residual [33:65].
    wq0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wk0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wv0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wo0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wq0[32, 0] = 8.0 * math.cos(1.0)
    wq0[32, 1] = -8.0 * math.sin(1.0)
    wk0[32, 0] = 8.0
    wv0[:32, :32] = np.eye(32, dtype=np.float32)
    wo0[:32, 33:65] = 2.0 * np.eye(32, dtype=np.float32)

    # Layer 1: Head 3 (dims 96..128, high-index pairs 2..31 where RoPE rotation is near-zero)
    # matches current token [0:30] against prev_token [33:63] and writes matched token [0:32]
    # into dedicated output subspace [65:97].
    wq1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wk1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wv1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wo1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wq1[:30, 98:128] = 15.0 * np.eye(30, dtype=np.float32)
    wk1[33:63, 98:128] = 15.0 * np.eye(30, dtype=np.float32)
    wv1[:32, 96:128] = np.eye(32, dtype=np.float32)
    wo1[96:128, 65:97] = 6.0 * np.eye(32, dtype=np.float32)

    for l_idx, (wq, wk, wv, wo) in enumerate([(wq0, wk0, wv0, wo0), (wq1, wk1, wv1, wo1)]):
        w1 = rng.normal(0.0, 0.01, size=(D_MODEL, MLP_DIM)).astype(np.float32)
        w2 = rng.normal(0.0, 0.01, size=(MLP_DIM, D_MODEL)).astype(np.float32)
        layers.append(
            {
                "wq": jnp.asarray(wq),
                "wk": jnp.asarray(wk),
                "wv": jnp.asarray(wv),
                "wo": jnp.asarray(wo),
                "w1": jnp.asarray(w1),
                "w2": jnp.asarray(w2),
            }
        )

    # LM head projects dedicated output subspace [65:97] to VOCAB_SIZE
    lm_head_np = np.zeros((D_MODEL, VOCAB_SIZE), dtype=np.float32)
    lm_head_np[65:97, :] = embed[:, :32].T * 2.0
    return {"embed": jnp.asarray(embed), "layers": layers, "lm_head": jnp.asarray(lm_head_np)}


def _apply_rope(x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position embeddings to x [N, H, D]."""
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]  # [N, half]
    cos_a = jnp.cos(angles)[:, None, :]
    sin_a = jnp.sin(angles)[:, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos_a - x1 * sin_a
    out1 = x0 * sin_a + x1 * cos_a
    out = jnp.stack([out0, out1], axis=-1).reshape(x.shape)
    return out


def _serial_layer_forward(layer_p, x, k_cache, v_cache, pos):
    """Single-token transformer block for ground-truth serial baseline."""
    q = _apply_rope((x @ layer_p["wq"]).reshape(1, NUM_HEADS, HEAD_DIM), jnp.asarray([pos]))
    k = _apply_rope((x @ layer_p["wk"]).reshape(1, NUM_HEADS, HEAD_DIM), jnp.asarray([pos]))
    v = (x @ layer_p["wv"]).reshape(1, NUM_HEADS, HEAD_DIM)

    k_cache = k_cache.at[pos].set(k[0])
    v_cache = v_cache.at[pos].set(v[0])

    valid_mask = jnp.arange(MAX_SEQ_LEN) <= pos  # [MAX_SEQ_LEN]
    scores = jnp.einsum("qhd,khd->hqk", q, k_cache) / math.sqrt(HEAD_DIM)  # [H, 1, S]
    scores = jnp.where(valid_mask[None, None, :], scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("hqk,khd->qhd", probs, v_cache).reshape(1, D_MODEL) @ layer_p["wo"]

    h = x + attn_out
    mlp_out = jax.nn.silu(h @ layer_p["w1"]) @ layer_p["w2"]
    return h + mlp_out, k_cache, v_cache


def _build_jit_models():
    """Compile serial and static 128-node systolic tree forward passes on OpenXLA."""
    serial_compile_count = [0]
    systolic_compile_count = [0]

    @jax.jit
    def serial_step(params, kv_k, kv_v, token_id, pos):
        serial_compile_count[0] += 1
        x = params["embed"][jnp.asarray([token_id])]  # [1, D]
        new_ks = []
        new_vs = []
        for l_idx in range(NUM_LAYERS):
            x, nk, nv = _serial_layer_forward(params["layers"][l_idx], x, kv_k[l_idx], kv_v[l_idx], pos)
            new_ks.append(nk)
            new_vs.append(nv)
        logits = (x @ params["lm_head"])[0]
        return logits, jnp.stack(new_ks, axis=0), jnp.stack(new_vs, axis=0)

    @jax.jit
    def systolic_tree_step(
        params,
        kv_k_prefix,
        kv_v_prefix,
        valid_prefix_len,
        tile_token_ids,
        tile_pos_offsets,
        tile_additive_mask,
    ):
        systolic_compile_count[0] += 1
        x = params["embed"][tile_token_ids]  # [128, D]
        positions = tile_pos_offsets  # [128] already offset by valid_prefix_len in SystolicTree::pack
        prefix_mask = jnp.arange(MAX_SEQ_LEN) < valid_prefix_len  # [MAX_SEQ_LEN]
        prefix_bias = jnp.where(prefix_mask, 0.0, -1e9)[None, None, :]  # [1, 1, S]
        tree_bias = tile_additive_mask[None, :, :]  # [1, 128, 128]

        step_ks = []
        step_vs = []
        for l_idx in range(NUM_LAYERS):
            lp = params["layers"][l_idx]
            q = _apply_rope((x @ lp["wq"]).reshape(TILE_SIZE, NUM_HEADS, HEAD_DIM), positions)
            k = _apply_rope((x @ lp["wk"]).reshape(TILE_SIZE, NUM_HEADS, HEAD_DIM), positions)
            v = (x @ lp["wv"]).reshape(TILE_SIZE, NUM_HEADS, HEAD_DIM)
            step_ks.append(k)
            step_vs.append(v)

            prefix_scores = jnp.einsum("qhd,khd->hqk", q, kv_k_prefix[l_idx]) / math.sqrt(HEAD_DIM)
            prefix_scores = prefix_scores + prefix_bias
            tree_scores = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(HEAD_DIM)
            tree_scores = tree_scores + tree_bias

            all_scores = jnp.concatenate([prefix_scores, tree_scores], axis=-1)  # [H, 128, S+128]
            probs = jax.nn.softmax(all_scores, axis=-1)
            probs_prefix = probs[:, :, :MAX_SEQ_LEN]
            probs_tree = probs[:, :, MAX_SEQ_LEN:]

            attn_val = (
                jnp.einsum("hqk,khd->qhd", probs_prefix, kv_v_prefix[l_idx])
                + jnp.einsum("hqk,khd->qhd", probs_tree, v)
            )
            attn_out = attn_val.reshape(TILE_SIZE, D_MODEL) @ lp["wo"]
            h = x + attn_out
            mlp_out = jax.nn.silu(h @ lp["w1"]) @ lp["w2"]
            x = h + mlp_out

        logits = x @ params["lm_head"]  # [128, VOCAB_SIZE]
        return logits, jnp.stack(step_ks, axis=0), jnp.stack(step_vs, axis=0)

    return serial_step, systolic_tree_step, serial_compile_count, systolic_compile_count


def test_openxla_tree_speculation_bit_exact_and_zero_recompile():
    assert any(d.platform == "cpu" for d in jax.devices())

    params = _build_induction_transformer_params(seed=2026)
    serial_step, systolic_tree_step, _, systolic_compile_count = _build_jit_models()

    # Use structured prompts of three distinct lengths (9, 23, 51) containing repeated idioms
    # so suffix-tree speculation achieves multi-token branch acceptance while testing zero recompilation.
    base_phrase = [4, 11, 22, 19, 8, 27, 15, 3]
    prompts = [
        (base_phrase * 2)[:9],
        (base_phrase * 3)[:23],
        (base_phrase * 7)[:51],
    ]

    # Configure per-layer StaticArena for KV row compaction verification alongside JAX KV cache.
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = MAX_SEQ_LEN
    cfg.hidden_dim = D_MODEL
    cfg.num_kv_heads = NUM_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.vocab_size = VOCAB_SIZE
    layer_arenas = [tpu.StaticArena(cfg) for _ in range(NUM_LAYERS)]
    for arena in layer_arenas:
        assert arena.is_aligned_128()

    prompt0 = prompts[0]
    gen_len = 48

    # --- 1. Ground-truth Serial Autoregressive Baseline on prompts[0] ---
    s_kv_k = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    s_kv_v = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    for pos, tok in enumerate(prompt0):
        s_logits, s_kv_k, s_kv_v = serial_step(params, s_kv_k, s_kv_v, tok, pos)

    serial_tokens = [int(jnp.argmax(s_logits))]
    cur_pos = len(prompt0)
    for _ in range(gen_len - 1):
        s_logits, s_kv_k, s_kv_v = serial_step(params, s_kv_k, s_kv_v, serial_tokens[-1], cur_pos)
        serial_tokens.append(int(jnp.argmax(s_logits)))
        cur_pos += 1

    # --- 2. TPUForge 128-Node Systolic Tree Speculation across all 3 prompt lengths (9, 23, 51) ---
    # First run full 48-token decode on prompts[0] and verify bit-exact parity + speedup,
    # then run prompts[1] (len=23) and prompts[2] (len=51) to assert jax_jit_compile_count == 1.
    all_prompt_results = []
    for p_idx, prompt in enumerate(prompts):
        t_kv_k = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
        t_kv_v = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
        for a in layer_arenas:
            a.set_valid_prefix_len(0)

        # Prefill prompt using serial_step (or tree prefill of prompt[:-1] + speculation on prompt[-1])
        valid_prefix_len = len(prompt) - 1
        for pos, tok in enumerate(prompt[:-1]):
            _, t_kv_k, t_kv_v = serial_step(params, t_kv_k, t_kv_v, tok, pos)
            # Mirror prefix KV into StaticArena to test native KV compaction
            for l_idx in range(NUM_LAYERS):
                k_row = np.asarray(t_kv_k[l_idx, pos : pos + 1]).reshape(1, NUM_HEADS * HEAD_DIM)
                v_row = np.asarray(t_kv_v[l_idx, pos : pos + 1]).reshape(1, NUM_HEADS * HEAD_DIM)
                k_tile = np.zeros((128, NUM_HEADS * HEAD_DIM), dtype=np.float32)
                v_tile = np.zeros((128, NUM_HEADS * HEAD_DIM), dtype=np.float32)
                k_tile[0] = k_row[0]
                v_tile[0] = v_row[0]
                layer_arenas[l_idx].commit_accepted_kv_rows(k_tile, v_tile, [0])

        history = list(prompt)
        generated = []
        passes = 0
        steps_needed = gen_len if p_idx == 0 else 6

        while len(generated) < steps_needed:
            tree = tpu.SuffixTreeDrafter.build_tree(history, max_nodes=128, max_depth=16)
            assert 1 <= tree.num_nodes() <= 128

            packed = tree.pack(valid_prefix_len)
            logits_128, step_ks, step_vs = systolic_tree_step(
                params,
                t_kv_k,
                t_kv_v,
                jnp.int32(valid_prefix_len),
                jnp.asarray(packed["token_ids"], dtype=jnp.int32),
                jnp.asarray(packed["position_offsets"], dtype=jnp.int32),
                jnp.asarray(packed["additive_mask_f32"], dtype=jnp.float32),
            )
            passes += 1

            logits_np = np.asarray(logits_128, dtype=np.float32)
            assert np.isfinite(logits_np).all()

            verify_res = tpu.SystolicTreeVerifier.verify_greedy(tree, logits_np)
            emitted = list(verify_res.emitted_tokens)
            accepted_indices = list(verify_res.accepted_node_indices)

            # Truncate only if emitting would overshoot target steps_needed
            remaining = steps_needed - len(generated)
            if len(emitted) > remaining:
                emitted = emitted[:remaining]
                # We commit prefix up to accepted_indices length <= remaining
                accepted_indices = accepted_indices[:remaining]

            # Compact accepted KV rows into JAX static cache and native StaticArena
            ks_np = np.asarray(step_ks, dtype=np.float32)
            vs_np = np.asarray(step_vs, dtype=np.float32)
            for slot, node_idx in enumerate(accepted_indices):
                dest_pos = valid_prefix_len + slot
                t_kv_k = t_kv_k.at[:, dest_pos].set(ks_np[:, node_idx])
                t_kv_v = t_kv_v.at[:, dest_pos].set(vs_np[:, node_idx])

            for l_idx in range(NUM_LAYERS):
                layer_arenas[l_idx].commit_accepted_kv_rows(
                    ks_np[l_idx].reshape(128, NUM_HEADS * HEAD_DIM),
                    vs_np[l_idx].reshape(128, NUM_HEADS * HEAD_DIM),
                    accepted_indices,
                )
                assert layer_arenas[l_idx].valid_prefix_len() == valid_prefix_len + len(accepted_indices)

            valid_prefix_len += len(accepted_indices)
            generated.extend(emitted)
            history.extend(emitted)

        all_prompt_results.append((generated, passes))

    tree_tokens, tree_passes = all_prompt_results[0]
    assert len(set(serial_tokens)) >= 6, f"Expected diverse induction sequence, got {serial_tokens}"
    assert tree_tokens == serial_tokens, f"Parity mismatch:\ntree:   {tree_tokens}\nserial: {serial_tokens}"
    assert tree_passes < 20, f"Expected < 20 tree forward passes for {gen_len} tokens, got {tree_passes}"
    assert systolic_compile_count[0] == 1, f"Expected 1 JAX JIT compilation, got {systolic_compile_count[0]}"


def test_partial_19_node_tree_nan_immunity():
    """Packing a partial 19-node tree into 128-row systolic tile produces finite logits everywhere."""
    params = _build_induction_transformer_params(seed=99)
    _, systolic_tree_step, _, _ = _build_jit_models()

    # Build explicit 19-node tree
    tree = tpu.SystolicTree(root_last_token=7)
    parent = 0
    for i in range(1, 19):
        parent = tree.add_child(0 if i <= 6 else (i % 6), (i * 3) % VOCAB_SIZE, 0.5)
    assert tree.num_nodes() == 19

    packed = tree.pack(valid_prefix_len=12)
    assert packed["active_nodes"] == 19
    assert packed["additive_mask_f32"].shape == (128, 128)
    # Padded diagonal rows [19..127] must have finite 0.0 on self-diagonal so softmax never yields NaN.
    for row in range(19, 128):
        assert packed["additive_mask_f32"][row, row] == 0.0

    kv_k = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    kv_v = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    logits_19, _, _ = systolic_tree_step(
        params,
        kv_k,
        kv_v,
        jnp.int32(12),
        jnp.asarray(packed["token_ids"], dtype=jnp.int32),
        jnp.asarray(packed["position_offsets"], dtype=jnp.int32),
        jnp.asarray(packed["additive_mask_f32"], dtype=jnp.float32),
    )
    logits_19_np = np.asarray(logits_19, dtype=np.float32)
    assert np.isfinite(logits_19_np).all()

    # Compare active rows [0..18] when padded tail tokens are mutated: active rows must be invariant.
    alt_tokens = np.asarray(packed["token_ids"], dtype=np.int32).copy()
    alt_tokens[19:] = 17
    logits_alt, _, _ = systolic_tree_step(
        params,
        kv_k,
        kv_v,
        jnp.int32(12),
        jnp.asarray(alt_tokens, dtype=jnp.int32),
        jnp.asarray(packed["position_offsets"], dtype=jnp.int32),
        jnp.asarray(packed["additive_mask_f32"], dtype=jnp.float32),
    )
    np.testing.assert_allclose(logits_19_np[:19], np.asarray(logits_alt, dtype=np.float32)[:19], rtol=1e-6, atol=1e-6)


def test_lossless_stochastic_branching_rejection_sampling():
    """Sequential residual verification recovers exact target marginal despite distorted draft probs."""
    temperature = 0.9
    vocab = 16
    tree = tpu.SystolicTree(root_last_token=0)
    # 3 distinct candidate tokens branching from root with heavily distorted draft probs q:
    # true target wants mostly token 3 and 7, while draft guesses mostly token 1!
    c1 = tree.add_child(0, 1, 0.85)
    c2 = tree.add_child(0, 3, 0.10)
    c3 = tree.add_child(0, 7, 0.05)
    assert c1 == 1 and c2 == 2 and c3 == 3

    rng = np.random.default_rng(123)
    logits_128 = np.zeros((128, vocab), dtype=np.float32)
    root_logits = np.array(
        [-2.0, 0.4, -1.5, 1.8, -0.5, -1.2, -0.8, 1.4, -1.0, -2.5, 0.2, -1.7, -0.9, -2.2, 0.0, -1.4],
        dtype=np.float32,
    )
    logits_128[0] = root_logits
    for idx in range(1, 4):
        logits_128[idx] = rng.normal(size=(vocab,)).astype(np.float32)

    # Exact target distribution p = softmax(root_logits / temperature)
    scaled = root_logits / temperature
    scaled = scaled - np.max(scaled)
    target_probs = np.exp(scaled) / np.sum(np.exp(scaled))

    trials = 4000
    counts = np.zeros(vocab, dtype=np.float64)
    for seed in range(1, trials + 1):
        res = tpu.SystolicTreeVerifier.verify_stochastic(tree, logits_128, temperature=temperature, seed=seed)
        first_emitted = res.emitted_tokens[0]
        counts[first_emitted] += 1.0

    empirical = counts / trials
    l1_tv = 0.5 * float(np.sum(np.abs(empirical - target_probs)))
    assert l1_tv < 0.04, f"Total variation distance {l1_tv:.4f} exceeded lossless threshold 0.04"

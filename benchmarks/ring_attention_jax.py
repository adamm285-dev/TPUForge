"""Milestone M-005: Optical Torus Ring-Attention over Inter-Chip Interconnect (ICI)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Ensure build/ and MSYS2 UCRT64 runtime DLLs are discoverable for _tpuforge_ext
_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _ROOT / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))
if sys.platform == "win32":
    _UCRT_BIN = Path(r"C:\msys64\ucrt64\bin")
    if _UCRT_BIN.is_dir():
        os.add_dll_directory(str(_UCRT_BIN))

import _tpuforge_ext as tpu

from benchmarks.qwen_jax import (
    _MASK_BLOCK_VAL,
    _gqa_attention,
    resolve_default_dtype,
    resolve_tile_size,
)


def resolve_ring_tp_mesh(
    ring_size: int | str = "auto",
    tp_size: int = 1,
) -> tuple[int, int, Mesh]:
    """Build a 2D torus Mesh with axes ('ring', 'tp') over visible JAX/PJRT devices."""
    if isinstance(ring_size, bool) or isinstance(tp_size, bool):
        raise TypeError("ring_size and tp_size cannot be bool")

    devs = jax.devices()
    num_devs = len(devs)
    resolved_tp = int(tp_size)
    if resolved_tp < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved_tp}")

    if isinstance(ring_size, str):
        norm = ring_size.strip().lower()
        if norm == "auto":
            resolved_ring = max(1, num_devs // resolved_tp)
        else:
            resolved_ring = int(norm)
    else:
        resolved_ring = int(ring_size)

    if resolved_ring < 1:
        raise ValueError(f"ring_size must be >= 1, got {resolved_ring}")
    total_chips = resolved_ring * resolved_tp
    if total_chips > num_devs:
        raise ValueError(
            f"Requested ring_size={resolved_ring} x tp_size={resolved_tp} ({total_chips} chips), "
            f"but only {num_devs} JAX device(s) are visible"
        )

    dev_grid = np.array(devs[:total_chips], dtype=object).reshape(resolved_ring, resolved_tp)
    mesh = Mesh(dev_grid, axis_names=("ring", "tp"))
    return resolved_ring, resolved_tp, mesh


def build_ici_ring_plan(
    max_seq_len: int,
    ring_size: int,
    num_layers: int = 1,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 16,
    dtype_name: str = "bf16",
) -> tpu.IciRingPlan:
    """Create a native C++20 IciRingPlan matching the JAX torus ring permutation."""
    cfg = tpu.KVCacheConfig()
    cfg.num_layers = int(num_layers)
    cfg.num_kv_heads = int(num_kv_heads)
    cfg.head_dim = int(head_dim)
    cfg.max_seq_len = int(max_seq_len)
    cfg.page_size = int(page_size)
    cfg.set_dtype(dtype_name)
    return tpu.IciRingPlanner.plan(cfg, int(ring_size))


def _online_softmax_block_update(
    m: jax.Array,
    l: jax.Array,
    acc: jax.Array,
    q_grouped: jax.Array,
    k_block: jax.Array,
    v_block: jax.Array,
    valid_mask: jax.Array,
    scale: float,
    additive_bias: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Numerically stable online log-sum-exp softmax update over one KV ring block."""
    scores = (
        jnp.einsum(
            "qkgd,nkd->kgqn",
            q_grouped,
            k_block,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    if additive_bias is not None:
        scores = scores + additive_bias[None, None, :, :]
    scores = jnp.where(valid_mask[None, None, :, :], scores, -jnp.inf)

    block_max = jnp.max(scores, axis=-1)
    m_cand = jnp.maximum(m, block_max)
    # Guard completely masked rows so exp(-inf - (-inf)) never produces NaN.
    m_safe = jnp.where(jnp.isfinite(m_cand), m_cand, 0.0)
    alpha = jnp.exp(m - m_safe)
    probs_unnorm = jnp.exp(scores - m_safe[..., None])

    l_new = alpha * l + jnp.sum(probs_unnorm, axis=-1)
    acc_new = alpha[..., None] * acc + jnp.einsum(
        "kgqn,nkd->kgqd",
        probs_unnorm,
        v_block,
        precision=jax.lax.Precision.HIGHEST,
    )
    return m_cand, l_new, acc_new


def _merge_lse_states(
    m_a: jax.Array,
    l_a: jax.Array,
    acc_a: jax.Array,
    m_b: jax.Array,
    l_b: jax.Array,
    acc_b: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Associative merge of two online log-sum-exp softmax accumulator states."""
    m_ab = jnp.maximum(m_a, m_b)
    m_safe = jnp.where(jnp.isfinite(m_ab), m_ab, 0.0)
    w_a = jnp.exp(m_a - m_safe)
    w_b = jnp.exp(m_b - m_safe)
    l_ab = w_a * l_a + w_b * l_b
    acc_ab = w_a[..., None] * acc_a + w_b[..., None] * acc_b
    return m_ab, l_ab, acc_ab


def make_ring_causal_attention_kernel(
    mesh: Mesh,
    max_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 1,
):
    """Compile a JAX shard_map + ppermute Ring-Attention causal prefill kernel."""
    ring_size = int(mesh.shape["ring"])
    tp_size = int(mesh.shape["tp"])
    if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide num_q_heads={num_q_heads} "
            f"and num_kv_heads={num_kv_heads}"
        )

    num_q_local = num_q_heads // tp_size
    num_kv_local = num_kv_heads // tp_size
    if num_q_local % num_kv_local != 0:
        raise ValueError(
            f"num_q_local={num_q_local} must be divisible by num_kv_local={num_kv_local}"
        )
    num_groups = num_q_local // num_kv_local
    scale = 1.0 / math.sqrt(float(head_dim))

    plan = build_ici_ring_plan(
        max_seq_len=max_seq_len,
        ring_size=ring_size,
        num_layers=1,
        num_kv_heads=num_kv_local,
        head_dim=head_dim,
        page_size=page_size,
    )
    perm = [(int(s), int(d)) for s, d in plan.send_perm]

    def _per_shard(
        q_chunk: jax.Array,
        k_chunk: jax.Array,
        v_chunk: jax.Array,
        valid_seq_len: jax.Array,
    ) -> jax.Array:
        chunk_len = q_chunk.shape[0]
        r = jax.lax.axis_index("ring")
        local_idx = jnp.arange(chunk_len, dtype=jnp.int32)
        q_pos = r * chunk_len + local_idx
        q_valid = q_pos < valid_seq_len

        q_grouped = q_chunk.reshape(chunk_len, num_kv_local, num_groups, head_dim).astype(
            jnp.float32
        )
        k_f32 = k_chunk.astype(jnp.float32)
        v_f32 = v_chunk.astype(jnp.float32)

        m0 = jnp.full((num_kv_local, num_groups, chunk_len), -jnp.inf, dtype=jnp.float32)
        l0 = jnp.zeros((num_kv_local, num_groups, chunk_len), dtype=jnp.float32)
        acc0 = jnp.zeros(
            (num_kv_local, num_groups, chunk_len, head_dim), dtype=jnp.float32
        )

        if ring_size == 1:
            k_pos = local_idx
            valid_mask = (
                (q_pos[:, None] >= k_pos[None, :])
                & q_valid[:, None]
                & (k_pos[None, :] < valid_seq_len)
            )
            _, l_fin, acc_fin = _online_softmax_block_update(
                m0, l0, acc0, q_grouped, k_f32, v_f32, valid_mask, scale
            )
        else:
            m0 = jax.lax.pcast(m0, ("ring", "tp"), to="varying")
            l0 = jax.lax.pcast(l0, ("ring", "tp"), to="varying")
            acc0 = jax.lax.pcast(acc0, ("ring", "tp"), to="varying")

            def hop_body(
                step: jax.Array,
                carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                k_curr, v_curr, m_curr, l_curr, acc_curr = carry
                k_next = jax.lax.ppermute(k_curr, "ring", perm)
                v_next = jax.lax.ppermute(v_curr, "ring", perm)
                kv_rank = jnp.mod(r + ring_size - step, ring_size)
                k_pos = kv_rank * chunk_len + local_idx
                valid_mask = (
                    (q_pos[:, None] >= k_pos[None, :])
                    & q_valid[:, None]
                    & (k_pos[None, :] < valid_seq_len)
                )
                m_next, l_next, acc_next = _online_softmax_block_update(
                    m_curr, l_curr, acc_curr, q_grouped, k_curr, v_curr, valid_mask, scale
                )
                return (k_next, v_next, m_next, l_next, acc_next)

            _, _, _, l_fin, acc_fin = jax.lax.fori_loop(
                0, ring_size, hop_body, (k_f32, v_f32, m0, l0, acc0)
            )

        out = acc_fin / jnp.where(l_fin > 0.0, l_fin, 1.0)[..., None]
        out = jnp.transpose(out, (2, 0, 1, 3)).reshape(chunk_len, num_q_local, head_dim)
        out = jnp.where(q_valid[:, None, None], out, 0.0)
        return out.astype(q_chunk.dtype)

    mapped = jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(
            P("ring", "tp", None),
            P("ring", "tp", None),
            P("ring", "tp", None),
            P(),
        ),
        out_specs=P("ring", "tp", None),
    )
    return jax.jit(mapped)


def make_ring_systolic_tree_attention_kernel(
    mesh: Mesh,
    max_seq_len: int,
    tile_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int = 16,
):
    """Compile an ICI-sharded stationary-KV + systolic tree verification kernel."""
    ring_size = int(mesh.shape["ring"])
    tp_size = int(mesh.shape["tp"])
    if tile_size not in (128, 256):
        raise ValueError(f"tile_size must be 128 or 256, got {tile_size}")
    if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        raise ValueError(
            f"tp_size={tp_size} must divide num_q_heads={num_q_heads} "
            f"and num_kv_heads={num_kv_heads}"
        )

    num_q_local = num_q_heads // tp_size
    num_kv_local = num_kv_heads // tp_size
    if num_q_local % num_kv_local != 0:
        raise ValueError(
            f"num_q_local={num_q_local} must be divisible by num_kv_local={num_kv_local}"
        )
    num_groups = num_q_local // num_kv_local
    scale = 1.0 / math.sqrt(float(head_dim))

    plan = build_ici_ring_plan(
        max_seq_len=max_seq_len,
        ring_size=ring_size,
        num_layers=1,
        num_kv_heads=num_kv_local,
        head_dim=head_dim,
        page_size=page_size,
    )
    perm = [(int(s), int(d)) for s, d in plan.send_perm]

    def _per_shard(
        q_tile: jax.Array,
        k_prefix: jax.Array,
        v_prefix: jax.Array,
        k_step: jax.Array,
        v_step: jax.Array,
        tile_mask: jax.Array,
        valid_prefix_len: jax.Array,
    ) -> jax.Array:
        chunk_len = k_prefix.shape[0]
        r = jax.lax.axis_index("ring")
        k_pos = r * chunk_len + jnp.arange(chunk_len, dtype=jnp.int32)
        valid_pref = jnp.broadcast_to(
            (k_pos < valid_prefix_len)[None, :], (tile_size, chunk_len)
        )

        q_grouped = q_tile.reshape(tile_size, num_kv_local, num_groups, head_dim).astype(
            jnp.float32
        )
        m0 = jnp.full((num_kv_local, num_groups, tile_size), -jnp.inf, dtype=jnp.float32)
        l0 = jnp.zeros((num_kv_local, num_groups, tile_size), dtype=jnp.float32)
        acc0 = jnp.zeros(
            (num_kv_local, num_groups, tile_size, head_dim), dtype=jnp.float32
        )

        m_loc, l_loc, acc_loc = _online_softmax_block_update(
            m0,
            l0,
            acc0,
            q_grouped,
            k_prefix.astype(jnp.float32),
            v_prefix.astype(jnp.float32),
            valid_pref,
            scale,
        )

        if ring_size == 1:
            m_pref, l_pref, acc_pref = m_loc, l_loc, acc_loc
        else:
            m_rot0 = jax.lax.ppermute(m_loc, "ring", perm)
            l_rot0 = jax.lax.ppermute(l_loc, "ring", perm)
            acc_rot0 = jax.lax.ppermute(acc_loc, "ring", perm)

            def merge_step(
                _step: jax.Array,
                carry: tuple[
                    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
                ],
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                m_tot, l_tot, acc_tot, m_cur, l_cur, acc_cur = carry
                m_next_cur = jax.lax.ppermute(m_cur, "ring", perm)
                l_next_cur = jax.lax.ppermute(l_cur, "ring", perm)
                acc_next_cur = jax.lax.ppermute(acc_cur, "ring", perm)
                m_merged, l_merged, acc_merged = _merge_lse_states(
                    m_tot, l_tot, acc_tot, m_cur, l_cur, acc_cur
                )
                return (
                    m_merged,
                    l_merged,
                    acc_merged,
                    m_next_cur,
                    l_next_cur,
                    acc_next_cur,
                )

            m_pref, l_pref, acc_pref, _, _, _ = jax.lax.fori_loop(
                0,
                ring_size - 1,
                merge_step,
                (m_loc, l_loc, acc_loc, m_rot0, l_rot0, acc_rot0),
            )

        valid_tile = tile_mask > (_MASK_BLOCK_VAL * 0.5)
        additive_bias = jnp.where(valid_tile, tile_mask.astype(jnp.float32), 0.0)
        _, l_fin, acc_fin = _online_softmax_block_update(
            m_pref,
            l_pref,
            acc_pref,
            q_grouped,
            k_step.astype(jnp.float32),
            v_step.astype(jnp.float32),
            valid_tile,
            scale,
            additive_bias=additive_bias,
        )
        out = acc_fin / jnp.where(l_fin > 0.0, l_fin, 1.0)[..., None]
        out = jnp.transpose(out, (2, 0, 1, 3)).reshape(tile_size, num_q_local, head_dim)
        return out.astype(q_tile.dtype)

    mapped = jax.shard_map(
        _per_shard,
        mesh=mesh,
        in_specs=(
            P(None, "tp", None),
            P("ring", "tp", None),
            P("ring", "tp", None),
            P(None, "tp", None),
            P(None, "tp", None),
            P(),
            P(),
        ),
        out_specs=P(None, "tp", None),
        check_vma=False,
    )
    return jax.jit(mapped)


@dataclass
class RingAttentionBenchmarkResult:
    """Structured telemetry from Milestone M-005 ICI Ring-Attention execution."""

    backend: str
    device_kind: str
    ring_size: int
    tp_size: int
    tile_size: int
    max_seq_len: int
    valid_seq_len: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    kv_bytes_per_rank: int
    kv_cluster_bytes: int
    hlo_has_collective_permute: bool
    prefill_mean_ms: float
    prefill_tok_per_s: float
    tree_step_mean_ms: float
    tree_nodes_per_s: float
    prefill_max_abs_err: float | None
    tree_max_abs_err: float | None


def run_ring_benchmark(
    max_seq_len: int = 8192,
    valid_seq_len: int = 6144,
    ring_size: int | str = "auto",
    tp_size: int = 1,
    tile_size: int | str = "auto",
    num_layers: int = 28,
    num_q_heads: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
    warmup: int = 2,
    iters: int = 5,
    check_parity: bool | None = None,
    dtype_name: str = "auto",
) -> RingAttentionBenchmarkResult:
    """Benchmark M-005 ICI Ring-Attention prefill and stationary-KV systolic tree decode."""
    if valid_seq_len <= 0 or valid_seq_len > max_seq_len:
        raise ValueError(
            f"valid_seq_len ({valid_seq_len}) must be in [1, max_seq_len={max_seq_len}]"
        )
    resolved_ring, resolved_tp, mesh = resolve_ring_tp_mesh(
        ring_size=ring_size, tp_size=tp_size
    )
    resolved_tile = resolve_tile_size(tile_size)
    dtype = resolve_default_dtype(dtype_name)
    dtype_label = "bf16" if dtype == jnp.bfloat16 else ("fp16" if dtype == jnp.float16 else "fp32")
    if check_parity is None:
        check_parity = max_seq_len <= 16384 and dtype == jnp.float32

    plan = build_ici_ring_plan(
        max_seq_len=max_seq_len,
        ring_size=resolved_ring,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads // resolved_tp,
        head_dim=head_dim,
        page_size=16,
        dtype_name=dtype_label,
    )

    prefill_fn = make_ring_causal_attention_kernel(
        mesh=mesh,
        max_seq_len=max_seq_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=16,
    )
    tree_fn = make_ring_systolic_tree_attention_kernel(
        mesh=mesh,
        max_seq_len=max_seq_len,
        tile_size=resolved_tile,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=16,
    )

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    seq_kv_sharding = NamedSharding(mesh, P("ring", "tp", None))
    tile_q_sharding = NamedSharding(mesh, P(None, "tp", None))
    rep_sharding = NamedSharding(mesh, P())

    q_pref = jax.device_put(
        (jax.random.normal(k1, (max_seq_len, num_q_heads, head_dim)) * 0.25).astype(dtype),
        seq_kv_sharding,
    )
    k_pref = jax.device_put(
        (jax.random.normal(k2, (max_seq_len, num_kv_heads, head_dim)) * 0.25).astype(dtype),
        seq_kv_sharding,
    )
    v_pref = jax.device_put(
        (jax.random.normal(k3, (max_seq_len, num_kv_heads, head_dim)) * 0.25).astype(dtype),
        seq_kv_sharding,
    )
    valid_arr = jax.device_put(jnp.asarray(valid_seq_len, dtype=jnp.int32), rep_sharding)

    q_tile = jax.device_put(
        (jax.random.normal(k4, (resolved_tile, num_q_heads, head_dim)) * 0.25).astype(dtype),
        tile_q_sharding,
    )
    k_step = jax.device_put(
        (jax.random.normal(k5, (resolved_tile, num_kv_heads, head_dim)) * 0.25).astype(dtype),
        tile_q_sharding,
    )
    v_step = jax.device_put(
        (jax.random.normal(k6, (resolved_tile, num_kv_heads, head_dim)) * 0.25).astype(dtype),
        tile_q_sharding,
    )

    draft_ctx = [100 + (i % 13) for i in range(64)]
    tree = tpu.SuffixTreeDrafter.build_tree(
        draft_ctx, max_nodes=resolved_tile, max_depth=8, tile_size=resolved_tile
    )
    packed = tree.pack(valid_prefix_len=valid_seq_len)
    tile_mask = jax.device_put(jnp.asarray(packed["additive_mask_f32"]), rep_sharding)

    prefill_compiled = prefill_fn.lower(q_pref, k_pref, v_pref, valid_arr).compile()
    tree_compiled = tree_fn.lower(
        q_tile, k_pref, v_pref, k_step, v_step, tile_mask, valid_arr
    ).compile()
    hlo_has_cp = True
    if resolved_ring > 1:
        hlo_has_cp = (
            "collective-permute" in prefill_compiled.as_text()
            and "collective-permute" in tree_compiled.as_text()
        )

    for _ in range(max(1, warmup)):
        out_p = prefill_fn(q_pref, k_pref, v_pref, valid_arr)
        out_t = tree_fn(q_tile, k_pref, v_pref, k_step, v_step, tile_mask, valid_arr)
        jax.block_until_ready((out_p, out_t))

    pref_times: list[float] = []
    tree_times: list[float] = []
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        out_p = prefill_fn(q_pref, k_pref, v_pref, valid_arr)
        jax.block_until_ready(out_p)
        pref_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        out_t = tree_fn(q_tile, k_pref, v_pref, k_step, v_step, tile_mask, valid_arr)
        jax.block_until_ready(out_t)
        tree_times.append((time.perf_counter() - t1) * 1000.0)

    prefill_max_abs_err: float | None = None
    tree_max_abs_err: float | None = None
    if check_parity:
        q_p_host = np.asarray(out_p, dtype=np.float32)[:valid_seq_len]
        q_ref = jnp.asarray(np.asarray(q_pref)[:valid_seq_len])
        k_ref = jnp.asarray(np.asarray(k_pref)[:valid_seq_len])
        v_ref = jnp.asarray(np.asarray(v_pref)[:valid_seq_len])
        idx_v = jnp.arange(valid_seq_len, dtype=jnp.int32)
        causal_ref = jnp.where(
            idx_v[:, None] >= idx_v[None, :], 0.0, _MASK_BLOCK_VAL
        ).astype(jnp.float32)
        ref_pref = _gqa_attention(
            q_ref, k_ref, v_ref, causal_ref, num_q_heads, num_kv_heads, head_dim
        ).reshape(valid_seq_len, num_q_heads, head_dim)
        prefill_max_abs_err = float(
            np.max(np.abs(q_p_host - np.asarray(ref_pref, dtype=np.float32)))
        )

        out_t_host = np.asarray(out_t, dtype=np.float32)
        pref_mask_ref = jnp.broadcast_to(
            jnp.where(
                jnp.arange(max_seq_len, dtype=jnp.int32) < valid_seq_len, 0.0, _MASK_BLOCK_VAL
            )[None, :],
            (resolved_tile, max_seq_len),
        )
        full_mask_ref = jnp.concatenate(
            [pref_mask_ref, jnp.asarray(packed["additive_mask_f32"], dtype=jnp.float32)],
            axis=-1,
        )
        k_all_ref = jnp.concatenate(
            [jnp.asarray(np.asarray(k_pref)), jnp.asarray(np.asarray(k_step))], axis=0
        )
        v_all_ref = jnp.concatenate(
            [jnp.asarray(np.asarray(v_pref)), jnp.asarray(np.asarray(v_step))], axis=0
        )
        ref_tree = _gqa_attention(
            jnp.asarray(np.asarray(q_tile)),
            k_all_ref,
            v_all_ref,
            full_mask_ref,
            num_q_heads,
            num_kv_heads,
            head_dim,
        ).reshape(resolved_tile, num_q_heads, head_dim)
        tree_max_abs_err = float(
            np.max(np.abs(out_t_host - np.asarray(ref_tree, dtype=np.float32)))
        )

    devs = jax.devices()
    dev_kind = str(getattr(devs[0], "device_kind", devs[0])) if devs else "cpu"
    pref_ms = float(sum(pref_times) / len(pref_times))
    tree_ms = float(sum(tree_times) / len(tree_times))

    return RingAttentionBenchmarkResult(
        backend=jax.default_backend(),
        device_kind=dev_kind,
        ring_size=resolved_ring,
        tp_size=resolved_tp,
        tile_size=resolved_tile,
        max_seq_len=max_seq_len,
        valid_seq_len=valid_seq_len,
        num_layers=num_layers,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_bytes_per_rank=int(plan.bytes_per_rank),
        kv_cluster_bytes=int(plan.total_cluster_bytes),
        hlo_has_collective_permute=bool(hlo_has_cp),
        prefill_mean_ms=pref_ms,
        prefill_tok_per_s=(valid_seq_len / (pref_ms / 1000.0)) if pref_ms > 0 else 0.0,
        tree_step_mean_ms=tree_ms,
        tree_nodes_per_s=(resolved_tile / (tree_ms / 1000.0)) if tree_ms > 0 else 0.0,
        prefill_max_abs_err=prefill_max_abs_err,
        tree_max_abs_err=tree_max_abs_err,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Milestone M-005 Optical Torus Ring-Attention (ICI) benchmark."
    )
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--valid-seq-len", type=int, default=6144)
    parser.add_argument("--ring-size", default="auto")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tile-size", default="auto")
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--num-q-heads", type=int, default=28)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--check-parity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compare ring outputs against full single-device _gqa_attention reference.",
    )
    parser.add_argument("--output-json", default="")
    args = parser.parse_args(argv)

    res = run_ring_benchmark(
        max_seq_len=args.max_seq_len,
        valid_seq_len=args.valid_seq_len,
        ring_size=args.ring_size,
        tp_size=args.tp_size,
        tile_size=args.tile_size,
        num_layers=args.num_layers,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        warmup=args.warmup,
        iters=args.iters,
        check_parity=args.check_parity,
        dtype_name=args.dtype,
    )

    per_rank_mb = res.kv_bytes_per_rank / (1024.0 * 1024.0)
    cluster_mb = res.kv_cluster_bytes / (1024.0 * 1024.0)
    print(
        f"=== M-005 Optical Torus Ring-Attention ({res.backend}: {res.ring_size}x ring, "
        f"{res.tp_size}x tp, tile={res.tile_size}) ==="
    )
    print(
        f"KV HBM footprint: {per_rank_mb:.1f} MiB/chip vs {cluster_mb:.1f} MiB cluster total "
        f"({res.ring_size}x reduction) | HLO collective-permute={res.hlo_has_collective_permute}"
    )
    print(
        f"Prefill ({res.valid_seq_len}/{res.max_seq_len} ctx): "
        f"{res.prefill_mean_ms:.2f} ms ({res.prefill_tok_per_s:.1f} tok/s)"
    )
    print(
        f"Systolic Tree Step ({res.tile_size} nodes): "
        f"{res.tree_step_mean_ms:.2f} ms ({res.tree_nodes_per_s:.1f} tree-nodes/s)"
    )
    if res.prefill_max_abs_err is not None or res.tree_max_abs_err is not None:
        print(
            f"Parity max abs err: prefill={res.prefill_max_abs_err:.2e}, "
            f"tree={res.tree_max_abs_err:.2e}"
        )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
        print(f"Saved JSON report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

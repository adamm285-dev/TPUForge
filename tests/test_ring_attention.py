"""Tests for Milestone M-005 Optical Torus Ring-Attention (ICI)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import _tpuforge_ext as tpu
import pytest

from benchmarks.package_kaggle_bundle import build_kaggle_bundle


def test_ici_ring_planner_topology_and_hbm_savings() -> None:
    """Verify native C++20 IciRingPlanner geometry, permutation pairs, and error bounds."""
    cfg = tpu.KVCacheConfig()
    cfg.num_layers = 32
    cfg.num_kv_heads = 8
    cfg.head_dim = 128
    cfg.max_seq_len = 131072
    cfg.page_size = 16
    cfg.set_dtype("bf16")

    plan = tpu.IciRingPlanner.plan(cfg, ring_size=8)
    assert plan.ring_size == 8
    assert plan.total_seq_len == 131072
    assert plan.chunk_seq_len == 16384
    assert plan.pages_per_rank == 1024
    assert plan.bytes_per_rank * 8 == plan.total_cluster_bytes
    expected_per_rank = 2 * 32 * 8 * 128 * 16384 * 2
    assert plan.bytes_per_rank == expected_per_rank
    assert plan.send_perm == [(i, (i + 1) % 8) for i in range(8)]
    assert plan.chunk_start_positions == [i * 16384 for i in range(8)]

    cfg_1m = tpu.KVCacheConfig()
    cfg_1m.num_layers = 28
    cfg_1m.num_kv_heads = 4
    cfg_1m.head_dim = 128
    cfg_1m.max_seq_len = 1048576
    cfg_1m.page_size = 16
    cfg_1m.set_dtype("bf16")
    plan_1m = tpu.IciRingPlanner.plan(cfg_1m, ring_size=8)
    assert plan_1m.chunk_seq_len == 131072
    assert plan_1m.bytes_per_rank * 8 == plan_1m.total_cluster_bytes

    for dst in range(8):
        visited = [tpu.IciRingPlanner.source_rank_at_step(dst, s, 8) for s in range(8)]
        assert len(set(visited)) == 8
        assert visited[0] == dst
        assert visited[1] == (dst - 1) % 8

    # With 300 valid tokens split into 4 chunks of 128 ([0..127], [128..255], [256..299], [empty]):
    hops_r1 = tpu.IciRingPlanner.causal_active_hops(
        query_rank=1, ring_size=4, chunk_seq_len=128, valid_seq_len=300
    )
    # step 0: kv_rank=1 (active), step 1: kv_rank=0 (active), step 2: kv_rank=3 (>299, skip),
    # step 3: kv_rank=2 (kv_min=256 > q_max=255, causal skip)
    assert hops_r1 == [1, 1, 0, 0]

    hops_r3 = tpu.IciRingPlanner.causal_active_hops(
        query_rank=3, ring_size=4, chunk_seq_len=128, valid_seq_len=300
    )
    assert hops_r3 == [0, 0, 0, 0]

    with pytest.raises(ValueError):
        tpu.IciRingPlanner.plan(cfg, ring_size=0)
    with pytest.raises(ValueError):
        cfg_bad = tpu.KVCacheConfig()
        cfg_bad.max_seq_len = 130
        cfg_bad.page_size = 16
        tpu.IciRingPlanner.plan(cfg_bad, ring_size=4)
    with pytest.raises(ValueError):
        cfg_unaligned = tpu.KVCacheConfig()
        cfg_unaligned.max_seq_len = 100
        cfg_unaligned.page_size = 16
        tpu.IciRingPlanner.plan(cfg_unaligned, ring_size=4)
    with pytest.raises(IndexError):
        tpu.IciRingPlanner.source_rank_at_step(4, 0, 4)
    with pytest.raises(ValueError):
        cfg.set_dtype("int4_invalid")


def test_ring_attention_four_chip_hlo_and_parity_subprocess(tmp_path: Path) -> None:
    """Verify 4-chip ICI collective-permute HLO and <1e-4 numerical parity on 1D & 2D torus."""
    out_zip = tmp_path / "bundle_check.zip"
    build_kaggle_bundle(Path(__file__).resolve().parent.parent, out_zip)
    import zipfile

    with zipfile.ZipFile(out_zip, "r") as zf:
        assert "benchmarks/ring_attention_jax.py" in zf.namelist()
        assert "tests/fixtures/pjrt_test_plugin.c" in zf.namelist()

    repo_root = Path(__file__).resolve().parent.parent
    script = r"""
import conftest
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import _tpuforge_ext as tpu
from benchmarks.qwen_jax import _MASK_BLOCK_VAL, _gqa_attention
from benchmarks.ring_attention_jax import (
    make_ring_causal_attention_kernel,
    make_ring_systolic_tree_attention_kernel,
    resolve_ring_tp_mesh,
    run_ring_benchmark,
)

assert len(jax.devices()) == 4, jax.devices()
rng = np.random.default_rng(7)
num_q, num_kv, head_dim = 8, 2, 16

for ring_size, tp_size in ((4, 1), (2, 2)):
    r_sz, t_sz, mesh = resolve_ring_tp_mesh(ring_size=ring_size, tp_size=tp_size)
    assert (r_sz, t_sz) == (ring_size, tp_size)

    # 1) Ring causal prefill parity with trailing masked chunk (valid=300 < 512)
    max_seq, valid_seq = 512, 300
    q_np = (rng.standard_normal((max_seq, num_q, head_dim)) * 0.3).astype(np.float32)
    k_np = (rng.standard_normal((max_seq, num_kv, head_dim)) * 0.3).astype(np.float32)
    v_np = (rng.standard_normal((max_seq, num_kv, head_dim)) * 0.3).astype(np.float32)

    seq_sharding = NamedSharding(mesh, P("ring", "tp", None))
    rep_sharding = NamedSharding(mesh, P())
    q_dev = jax.device_put(jnp.asarray(q_np), seq_sharding)
    k_dev = jax.device_put(jnp.asarray(k_np), seq_sharding)
    v_dev = jax.device_put(jnp.asarray(v_np), seq_sharding)
    valid_dev = jax.device_put(jnp.asarray(valid_seq, dtype=jnp.int32), rep_sharding)

    prefill_fn = make_ring_causal_attention_kernel(
        mesh=mesh,
        max_seq_len=max_seq,
        num_q_heads=num_q,
        num_kv_heads=num_kv,
        head_dim=head_dim,
        page_size=16,
    )
    prefill_hlo = prefill_fn.lower(q_dev, k_dev, v_dev, valid_dev).compile().as_text()
    assert "collective-permute" in prefill_hlo, (ring_size, tp_size)

    ring_out = np.asarray(prefill_fn(q_dev, k_dev, v_dev, valid_dev), dtype=np.float32)
    assert not np.isnan(ring_out).any(), "NaN in ring causal output"

    idx_v = jnp.arange(valid_seq, dtype=jnp.int32)
    causal_ref = jnp.where(idx_v[:, None] >= idx_v[None, :], 0.0, _MASK_BLOCK_VAL).astype(jnp.float32)
    ref_valid = np.asarray(
        _gqa_attention(
            jnp.asarray(q_np[:valid_seq]),
            jnp.asarray(k_np[:valid_seq]),
            jnp.asarray(v_np[:valid_seq]),
            causal_ref,
            num_q,
            num_kv,
            head_dim,
        ),
        dtype=np.float32,
    ).reshape(valid_seq, num_q, head_dim)
    assert np.allclose(ring_out[:valid_seq], ref_valid, atol=1e-4, rtol=1e-4)
    assert np.all(ring_out[valid_seq:] == 0.0)

    # 2) Stationary ICI-sharded KV + systolic tree verification across both 128 and 256 tile sizes
    max_pref, valid_pref = 1024, 600
    k_pref_np = (rng.standard_normal((max_pref, num_kv, head_dim)) * 0.3).astype(np.float32)
    v_pref_np = (rng.standard_normal((max_pref, num_kv, head_dim)) * 0.3).astype(np.float32)
    k_pref_dev = jax.device_put(jnp.asarray(k_pref_np), seq_sharding)
    v_pref_dev = jax.device_put(jnp.asarray(v_pref_np), seq_sharding)
    valid_pref_dev = jax.device_put(jnp.asarray(valid_pref, dtype=jnp.int32), rep_sharding)

    tile_q_sharding = NamedSharding(mesh, P(None, "tp", None))
    for tile_size in (128, 256):
        q_tile_np = (rng.standard_normal((tile_size, num_q, head_dim)) * 0.3).astype(np.float32)
        k_step_np = (rng.standard_normal((tile_size, num_kv, head_dim)) * 0.3).astype(np.float32)
        v_step_np = (rng.standard_normal((tile_size, num_kv, head_dim)) * 0.3).astype(np.float32)

        tree = tpu.SuffixTreeDrafter.build_tree(
            [1, 2, 3, 4] * 12, max_nodes=tile_size, max_depth=8, tile_size=tile_size
        )
        packed = tree.pack(valid_prefix_len=valid_pref)
        mask_np = np.asarray(packed["additive_mask_f32"], dtype=np.float32)

        tree_fn = make_ring_systolic_tree_attention_kernel(
            mesh=mesh,
            max_seq_len=max_pref,
            tile_size=tile_size,
            num_q_heads=num_q,
            num_kv_heads=num_kv,
            head_dim=head_dim,
            page_size=16,
        )
        q_tile_dev = jax.device_put(jnp.asarray(q_tile_np), tile_q_sharding)
        k_step_dev = jax.device_put(jnp.asarray(k_step_np), tile_q_sharding)
        v_step_dev = jax.device_put(jnp.asarray(v_step_np), tile_q_sharding)
        mask_dev = jax.device_put(jnp.asarray(mask_np), rep_sharding)

        tree_hlo = tree_fn.lower(
            q_tile_dev, k_pref_dev, v_pref_dev, k_step_dev, v_step_dev, mask_dev, valid_pref_dev
        ).compile().as_text()
        assert "collective-permute" in tree_hlo, (ring_size, tp_size, tile_size)

        tree_out = np.asarray(
            tree_fn(q_tile_dev, k_pref_dev, v_pref_dev, k_step_dev, v_step_dev, mask_dev, valid_pref_dev),
            dtype=np.float32,
        )
        assert not np.isnan(tree_out).any(), "NaN in ring systolic tree output"

        pref_mask_ref = jnp.broadcast_to(
            jnp.where(jnp.arange(max_pref, dtype=jnp.int32) < valid_pref, 0.0, _MASK_BLOCK_VAL)[None, :],
            (tile_size, max_pref),
        )
        full_mask_ref = jnp.concatenate([pref_mask_ref, jnp.asarray(mask_np)], axis=-1)
        ref_tree = np.asarray(
            _gqa_attention(
                jnp.asarray(q_tile_np),
                jnp.asarray(np.concatenate([k_pref_np, k_step_np], axis=0)),
                jnp.asarray(np.concatenate([v_pref_np, v_step_np], axis=0)),
                full_mask_ref,
                num_q,
                num_kv,
                head_dim,
            ),
            dtype=np.float32,
        ).reshape(tile_size, num_q, head_dim)
        assert np.allclose(tree_out, ref_tree, atol=1e-4, rtol=1e-4)

bench_res = run_ring_benchmark(
    max_seq_len=512,
    valid_seq_len=300,
    ring_size=4,
    tp_size=1,
    tile_size=128,
    num_layers=4,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=16,
    warmup=1,
    iters=2,
    check_parity=True,
    dtype_name="float32",
)
assert bench_res.hlo_has_collective_permute is True
assert bench_res.kv_bytes_per_rank * 4 == bench_res.kv_cluster_bytes
assert bench_res.prefill_max_abs_err is not None and bench_res.prefill_max_abs_err < 1e-4
assert bench_res.tree_max_abs_err is not None and bench_res.tree_max_abs_err < 1e-4

# 3) Multi-block flash-ring tiling (chunk_len=640 -> block=320, 2x2 tiles with partial tail)
multi_res = run_ring_benchmark(
    max_seq_len=2560,
    valid_seq_len=1900,
    ring_size=4,
    tp_size=1,
    tile_size=128,
    num_layers=2,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=16,
    warmup=1,
    iters=1,
    check_parity=True,
    dtype_name="float32",
)
assert multi_res.hlo_has_collective_permute is True
assert multi_res.prefill_max_abs_err is not None and multi_res.prefill_max_abs_err < 1e-4
assert multi_res.tree_max_abs_err is not None and multi_res.tree_max_abs_err < 1e-4

# 4) Native BF16 MXU accumulation parity (no NaN/Inf, matches _gqa_attention within 2e-2)
bf16_res = run_ring_benchmark(
    max_seq_len=2560,
    valid_seq_len=1900,
    ring_size=4,
    tp_size=1,
    tile_size=128,
    num_layers=2,
    num_q_heads=8,
    num_kv_heads=2,
    head_dim=16,
    warmup=1,
    iters=1,
    check_parity=True,
    dtype_name="bfloat16",
)
assert bf16_res.hlo_has_collective_permute is True
assert bf16_res.prefill_max_abs_err is not None
assert not np.isnan(bf16_res.prefill_max_abs_err) and bf16_res.prefill_max_abs_err < 2e-2
assert bf16_res.tree_max_abs_err is not None
assert not np.isnan(bf16_res.tree_max_abs_err) and bf16_res.tree_max_abs_err < 2e-2
"""
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    py_paths = [str(repo_root), str(repo_root / "tests")]
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
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"4-chip ring subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

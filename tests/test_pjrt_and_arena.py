"""Tests for Milestone M-002: Real OS PJRT Dynamic Loading, Zero-Recompile Execution, and 128-Byte StaticArena."""

from __future__ import annotations

import _tpuforge_ext as tpu
import numpy as np
import pytest
from conftest import find_pjrt_test_plugin


def test_pjrt_driver_rejects_missing_plugin_without_fake_fallback():
    driver = tpu.PjrtDriver()
    ok, err = driver.load_plugin("Z:/nonexistent_path/no_such_pjrt_plugin.dll")
    assert ok is False
    assert isinstance(err, str) and len(err) > 0
    assert driver.is_loaded() is False
    assert driver.query_devices() == []


def test_pjrt_driver_loads_c11_plugin_and_zero_recompiles():
    plugin_path = find_pjrt_test_plugin()
    driver = tpu.PjrtDriver()
    ok, err = driver.load_plugin(str(plugin_path))
    assert ok is True, f"Failed to load C11 PJRT test plugin {plugin_path}: {err}"
    assert driver.is_loaded() is True
    assert "tpu" in driver.platform_name().lower()

    devices = driver.query_devices()
    assert len(devices) == 2
    assert devices[0].kind == "Google TPU v6e"
    assert devices[0].is_tpu is True
    assert devices[0].mxu_dim_x == 256
    assert devices[0].mxu_dim_y == 256
    assert devices[0].memory_bytes == 32 * 1024 * 1024 * 1024

    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 8192
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 256
    arena = tpu.StaticArena(cfg)
    assert arena.is_aligned_128() is True

    executor = tpu.ZeroRecompileExecutor(driver)
    comp_ok, comp_err = executor.compile_once(cfg, "module @tpuforge_128 {} // tensor<128x256xf32>", "mlir")
    assert comp_ok is True, f"compile_once failed: {comp_err}"
    assert executor.is_compiled() is True
    assert executor.compile_count() == 1

    # Non-128 systolic tile row counts must be rejected without recompiling.
    with pytest.raises(ValueError):
        executor.execute_step(arena, tile_rows=19)
    assert executor.compile_count() == 1

    # Varying valid prefix lengths across [1 .. 8000] executes statically with compile_count == 1.
    prefix_lengths = [1, 7, 16, 63, 128, 255, 512, 1024, 2048, 4096, 6000, 8000]
    for step_idx, plen in enumerate(prefix_lengths, start=1):
        arena.set_valid_prefix_len(plen)
        assert arena.valid_prefix_len() == plen

        tree = tpu.SystolicTree(root_last_token=(step_idx % 250))
        tree.add_child(0, (step_idx + 1) % 250, 0.9)
        packed = tree.pack(plen)
        arena.set_tile_inputs(
            packed["token_ids"],
            packed["position_offsets"],
            packed["additive_mask_f32"],
        )

        step_ok, step_err = executor.execute_step(arena, tile_rows=128)
        assert step_ok is True, f"execute_step failed at prefix={plen}: {step_err}"
        assert executor.compile_count() == 1
        assert executor.execute_count() == step_idx

        logits = np.asarray(arena.get_logits_tile())
        assert logits.shape == (128, 256)
        assert np.isfinite(logits).all()
        expected_row0_argmax = ((step_idx % 250) + plen + 1) % 256
        assert int(np.argmax(logits[0])) == expected_row0_argmax

    driver.unload()
    assert driver.is_loaded() is False


def test_static_arena_alignment_and_kv_row_compaction():
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 256
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 2
    cfg.head_dim = 64
    cfg.vocab_size = 64

    arena = tpu.StaticArena(cfg)
    assert arena.is_aligned_128() is True
    assert arena.kv_row_elems() == 2 * 64

    arena.set_valid_prefix_len(10)
    kv_elems = arena.kv_row_elems()

    rng = np.random.default_rng(42)
    step_k = rng.normal(size=(128, kv_elems)).astype(np.float32)
    step_v = rng.normal(size=(128, kv_elems)).astype(np.float32)

    # Accepted branch nodes [0, 5, 17, 42] in the 128-node systolic tile.
    accepted_indices = [0, 5, 17, 42]
    arena.commit_accepted_kv_rows(step_k, step_v, accepted_indices)
    assert arena.valid_prefix_len() == 14

    committed_k = np.asarray(arena.get_kv_prefix_k_slice(10, 4))
    assert committed_k.shape == (4, kv_elems)
    np.testing.assert_allclose(committed_k, step_k[accepted_indices], rtol=0.0, atol=0.0)

    # Attempting to overflow max_seq_len - 128 headroom raises IndexError cleanly.
    with pytest.raises(IndexError):
        arena.set_valid_prefix_len(254)


def test_infer_mxu_dim_generations():
    for v5_like in ("Google TPU v5e", "TPU v5p", "TPU v4", "Host CPU (16 threads)"):
        assert tpu.infer_mxu_dim(v5_like) == 128
    for v6_plus in ("Google TPU v6e", "TPU Trillium", "TPU v7", "TPU Ironwood", "TPU 8t", "TPU8i"):
        assert tpu.infer_mxu_dim(v6_plus) == 256


def test_static_arena_and_pjrt_tile_256_zero_recompile():
    plugin_path = find_pjrt_test_plugin()
    driver = tpu.PjrtDriver()
    ok, err = driver.load_plugin(str(plugin_path))
    assert ok is True, f"Failed to load PJRT plugin: {err}"

    cfg = tpu.StaticArenaConfig()
    cfg.tile_size = 256
    cfg.max_seq_len = 8192
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 256
    arena = tpu.StaticArena(cfg)
    assert arena.tile_size() == 256
    assert arena.is_aligned_128() is True

    executor = tpu.ZeroRecompileExecutor(driver)
    comp_ok, comp_err = executor.compile_once(cfg, "module @tpuforge_256 {} // tensor<256x256xf32>", "mlir")
    assert comp_ok is True, f"compile_once(256) failed: {comp_err}"
    assert executor.compile_count() == 1

    # Verify a 200-node tree exceeds the legacy 128 limit and packs cleanly into 256x256.
    tree = tpu.SystolicTree(7, tile_size=256)
    assert tree.tile_size() == 256
    for i in range(1, 200):
        idx = tree.add_child(0, (i + 10) % 250, 0.9)
        assert idx == i
    assert tree.num_nodes() == 200

    for step_idx, plen in enumerate([1, 128, 256, 2048], start=1):
        arena.set_valid_prefix_len(plen)
        packed = tree.pack(plen)
        assert packed["tile_size"] == 256
        assert packed["token_ids"].shape == (256,)
        assert packed["additive_mask_f32"].shape == (256, 256)
        arena.set_tile_inputs(
            packed["token_ids"],
            packed["position_offsets"],
            packed["additive_mask_f32"],
        )
        step_ok, step_err = executor.execute_step(arena)
        assert step_ok is True, f"execute_step(256) failed at prefix={plen}: {step_err}"
        assert executor.compile_count() == 1
        assert executor.execute_count() == step_idx

        logits = np.asarray(arena.get_logits_tile())
        assert logits.shape == (256, 256)
        assert np.isfinite(logits).all()
        assert int(np.argmax(logits[0])) == (7 + plen + 1) % 256

    driver.unload()

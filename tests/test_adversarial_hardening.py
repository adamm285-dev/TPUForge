"""Adversarial regression tests covering Waves 1 through 3 findings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _tpuforge_ext as tpu
from conftest import find_pjrt_test_plugin


def test_static_arena_overflow_and_headroom_guards():
    # Overflowing max_seq_len * kv_row_elems * 4 or > INT32_MAX must throw cleanly.
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 1 << 62
    cfg.num_kv_heads = 1
    cfg.head_dim = 4
    cfg.vocab_size = 64
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        tpu.StaticArena(cfg)

    cfg_big = tpu.StaticArenaConfig()
    cfg_big.max_seq_len = (1 << 31) + 128
    cfg_big.num_kv_heads = 1
    cfg_big.head_dim = 4
    cfg_big.vocab_size = 64
    with pytest.raises((ValueError, OverflowError, RuntimeError)):
        tpu.StaticArena(cfg_big)

    # Headroom invariant: commit_accepted_kv_rows cannot fill past max_seq_len - 128.
    small_cfg = tpu.StaticArenaConfig()
    small_cfg.max_seq_len = 256
    small_cfg.hidden_dim = 128
    small_cfg.num_kv_heads = 2
    small_cfg.head_dim = 8
    small_cfg.vocab_size = 32
    arena = tpu.StaticArena(small_cfg)

    step_k = np.ones((128, arena.kv_row_elems()), dtype=np.float32)
    step_v = np.ones((128, arena.kv_row_elems()), dtype=np.float32)

    # Duplicate row indices must be rejected with ValueError before any state mutation.
    with pytest.raises(ValueError):
        arena.commit_accepted_kv_rows(step_k, step_v, [0, 0])
    assert arena.valid_prefix_len() == 0

    arena.set_valid_prefix_len(127)

    # Atomic 2-pass validation: index 999 invalid must not mutate valid_prefix_len or row 127.
    sentinel = -42.5
    step_k[0, :] = sentinel
    with pytest.raises(IndexError):
        arena.commit_accepted_kv_rows(step_k, step_v, [0, 999])
    assert arena.valid_prefix_len() == 127
    before_commit = np.asarray(arena.get_kv_prefix_k_slice(127, 1))
    assert float(before_commit[0, 0]) == 0.0

    # Committing 1 row reaches 128 (max_seq_len - 128). Committing one more must fail.
    arena.commit_accepted_kv_rows(step_k, step_v, [0])
    assert arena.valid_prefix_len() == 128
    with pytest.raises(IndexError):
        arena.commit_accepted_kv_rows(step_k, step_v, [1])

    # Wraparound integer overflow on get_kv_prefix_k_slice must fail cleanly.
    with pytest.raises(IndexError):
        arena.get_kv_prefix_k_slice((1 << 64) - 1, 1)


def test_tree_and_verifier_nan_inf_bounds_guards():
    with pytest.raises(ValueError):
        tpu.SystolicTree(-1)

    tree = tpu.SystolicTree(5)
    with pytest.raises(ValueError):
        tree.add_child(0, -1, 0.5)
    with pytest.raises(ValueError):
        tree.add_child(0, 1, float("nan"))
    with pytest.raises(ValueError):
        tree.add_child(0, 1, float("inf"))
    with pytest.raises(ValueError):
        tree.add_child(0, 1, float("-inf"))
    with pytest.raises(ValueError):
        tree.add_child(0, 1, -0.1)

    with pytest.raises(IndexError):
        tree.pack(-1)
    with pytest.raises(IndexError):
        tree.pack((1 << 31) - 1)

    # Valid child token=3, vocab=8
    tree.add_child(0, 3, 0.5)
    logits = np.zeros((128, 8), dtype=np.float32)

    # Row[0] = NaN must throw in both greedy and stochastic verifiers (never silent token 0).
    nan0 = logits.copy()
    nan0[0, 0] = float("nan")
    with pytest.raises(ValueError):
        tpu.SystolicTreeVerifier.verify_greedy(tree, nan0)
    with pytest.raises(ValueError):
        tpu.SystolicTreeVerifier.verify_stochastic(tree, nan0, temperature=0.8, seed=1)

    # +inf logit and NaN temperature must throw rather than emitting vocab_size - 1.
    inf_logits = logits.copy()
    inf_logits[0, 2] = float("inf")
    with pytest.raises(ValueError):
        tpu.SystolicTreeVerifier.verify_stochastic(tree, inf_logits, temperature=1.0, seed=1)
    with pytest.raises(ValueError):
        tpu.SystolicTreeVerifier.verify_stochastic(tree, logits, temperature=float("nan"), seed=1)

    # All -inf row must throw; single non--inf token gets 100% probability (-inf -> 0.0).
    all_neginf = np.full((128, 8), float("-inf"), dtype=np.float32)
    with pytest.raises(ValueError):
        tpu.SystolicTreeVerifier.verify_stochastic(tree, all_neginf, temperature=1.0, seed=1)

    masked = np.full((128, 8), float("-inf"), dtype=np.float32)
    masked[0, 4] = 0.0
    masked[1:, 0] = 0.0
    res_masked = tpu.SystolicTreeVerifier.verify_stochastic(tree, masked, temperature=1.0, seed=7)
    assert res_masked.emitted_tokens == [4]

    # Draft child token_id >= vocab_size must throw IndexError rather than silently skipping.
    oob_tree = tpu.SystolicTree(0)
    oob_tree.add_child(0, 99, 0.5)
    with pytest.raises(IndexError):
        tpu.SystolicTreeVerifier.verify_greedy(oob_tree, logits)
    with pytest.raises(IndexError):
        tpu.SystolicTreeVerifier.verify_stochastic(oob_tree, logits, temperature=1.0, seed=1)


def test_sole_mass_rejection_never_reemits_rejected_child():
    # Child token 1 has ~1.0 probability, non-draft token 2 has tiny exp(-27) ~ 1.87e-12.
    # Rejecting token 1 must NEVER fall back to full probs and re-emit token 1.
    vocab = 4
    tree = tpu.SystolicTree(0)
    tree.add_child(0, 1, 0.99)
    logits = np.full((128, vocab), -100.0, dtype=np.float32)
    logits[0, 1] = 27.0
    logits[0, 2] = 0.0
    logits[1, 0] = 0.0

    for seed in range(1, 200):
        res = tpu.SystolicTreeVerifier.verify_stochastic(tree, logits, temperature=1.0, seed=seed)
        if res.acceptance_count == 0:
            assert res.emitted_tokens[0] != 1


def test_pjrt_unload_reload_cache_keys_and_transactional_load():
    plugin_path = str(find_pjrt_test_plugin())
    driver = tpu.PjrtDriver()
    ok, err = driver.load_plugin(plugin_path)
    assert ok is True, err

    # Transactional load failure must preserve existing loaded plugin.
    bad_ok, bad_err = driver.load_plugin("Z:/definitely_missing/bad_plugin.dll")
    assert bad_ok is False and len(bad_err) > 0
    assert driver.is_loaded() is True
    assert len(driver.query_devices()) == 2

    executor = tpu.ZeroRecompileExecutor(driver)
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 256
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 256

    # Reject unsupported PJRT program format cleanly.
    bad_fmt_ok, bad_fmt_err = executor.compile_once(cfg, "module @a {}", "invalid_format")
    assert bad_fmt_ok is False and "format" in bad_fmt_err.lower()

    assert executor.compile_once(cfg, "module @a {} // tensor<128x256xf32>", "mlir")[0] is True
    assert executor.compile_count() == 1

    # Changing program_code triggers recompile.
    assert executor.compile_once(cfg, "module @b {} // tensor<128x256xf32>", "mlir")[0] is True
    assert executor.compile_count() == 2

    # Changing num_kv_heads triggers recompile.
    cfg2 = tpu.StaticArenaConfig()
    cfg2.max_seq_len = 256
    cfg2.hidden_dim = 128
    cfg2.num_kv_heads = 8
    cfg2.head_dim = 32
    cfg2.vocab_size = 256
    assert executor.compile_once(cfg2, "module @b {} // tensor<128x256xf32>", "stablehlo")[0] is True
    assert executor.compile_count() == 3

    # Mismatched arena config is rejected safely in execute_step.
    wrong_arena = tpu.StaticArena(cfg)
    step_ok, step_err = executor.execute_step(wrong_arena, 128)
    assert step_ok is False and "StaticShapeKey" in step_err

    # Unloading driver invalidates compiled executable cleanly; reloading compiles fresh.
    driver.unload()
    assert executor.is_compiled() is False
    step_ok2, _ = executor.execute_step(tpu.StaticArena(cfg2), 128)
    assert step_ok2 is False

    assert driver.load_plugin(plugin_path)[0] is True
    assert executor.compile_once(cfg2, "module @b {} // tensor<128x256xf32>", "stablehlo")[0] is True
    assert executor.compile_count() == 4
    good_arena = tpu.StaticArena(cfg2)
    assert executor.execute_step(good_arena, 128)[0] is True


def test_clean_environment_pyd_import_without_msys64_on_path():
    root = Path(__file__).resolve().parent.parent
    build_dir = root / "build"
    env = os.environ.copy()
    path_parts = [p for p in env.get("PATH", "").split(os.pathsep) if "msys64" not in p.lower()]
    env["PATH"] = os.pathsep.join(path_parts)

    code = (
        f"import sys; sys.path.insert(0, r'{build_dir}'); "
        "import _tpuforge_ext as t; "
        "tree = t.SystolicTree(3); "
        "assert tree.num_nodes() == 1"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, f"Standalone _tpuforge_ext import failed without MSYS64 PATH:\n{proc.stderr}"

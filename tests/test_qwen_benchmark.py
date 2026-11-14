"""Deterministic offline tests for Qwen2.5-Coder safetensors loader, EOS IDs, and 3-mode runner."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import _tpuforge_ext as tpu
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from benchmarks.package_kaggle_bundle import build_kaggle_bundle
from benchmarks.qwen_jax import (
    QwenBenchmarkRunner,
    QwenConfig,
    QwenWeights,
    load_safetensors_dir,
    resolve_tile_size,
    resolve_tp_mesh,
)


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    """Write a valid .safetensors binary file from {name: (dtype_tag, array)}."""
    header: dict[str, object] = {}
    payload_parts: list[bytes] = []
    offset = 0
    for name, (dtype_tag, arr) in tensors.items():
        raw_bytes = np.ascontiguousarray(arr).tobytes()
        header[name] = {
            "dtype": dtype_tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw_bytes)],
        }
        payload_parts.append(raw_bytes)
        offset += len(raw_bytes)
    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for part in payload_parts:
            f.write(part)


def _make_cyclic_qwen_model(vocab_size: int = 512) -> tuple[QwenConfig, dict[str, np.ndarray]]:
    """Create a compact deterministic Qwen2 checkpoint cycling tokens 10->11->12->13->10."""
    cfg = QwenConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        vocab_size=vocab_size,
        head_dim=32,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        tie_word_embeddings=False,
        eos_token_ids=(500, 501),
    )
    h = cfg.hidden_size
    kv_dim = cfg.num_key_value_heads * cfg.head_dim
    rng = np.random.default_rng(7)

    embed = rng.normal(0.0, 0.01, size=(vocab_size, h)).astype(np.float32)
    # Unit-RMS orthonormal slots in dims 0..7 for cyclic tokens 10..13
    cycle_toks = [10, 11, 12, 13]
    for idx, tok in enumerate(cycle_toks):
        embed[tok, :] = 0.0
        embed[tok, idx] = float(np.sqrt(h))

    lm_head = np.zeros((vocab_size, h), dtype=np.float32)
    for idx, tok in enumerate(cycle_toks):
        next_tok = cycle_toks[(idx + 1) % len(cycle_toks)]
        lm_head[next_tok, idx] = 10.0
    # Stop token 500 triggered by token 42
    embed[42, :] = 0.0
    embed[42, 8] = float(np.sqrt(h))
    lm_head[500, 8] = 10.0

    raw: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": embed,
        "model.norm.weight": np.ones((h,), dtype=np.float32),
        "lm_head.weight": lm_head,
    }
    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"
        raw[f"{prefix}.input_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        raw[f"{prefix}.post_attention_layernorm.weight"] = np.ones((h,), dtype=np.float32)
        raw[f"{prefix}.self_attn.q_proj.weight"] = np.zeros((h, h), dtype=np.float32)
        raw[f"{prefix}.self_attn.q_proj.bias"] = np.zeros((h,), dtype=np.float32)
        raw[f"{prefix}.self_attn.k_proj.weight"] = np.zeros((kv_dim, h), dtype=np.float32)
        raw[f"{prefix}.self_attn.k_proj.bias"] = np.zeros((kv_dim,), dtype=np.float32)
        raw[f"{prefix}.self_attn.v_proj.weight"] = np.zeros((kv_dim, h), dtype=np.float32)
        raw[f"{prefix}.self_attn.v_proj.bias"] = np.zeros((kv_dim,), dtype=np.float32)
        raw[f"{prefix}.self_attn.o_proj.weight"] = np.zeros((h, h), dtype=np.float32)
        raw[f"{prefix}.mlp.gate_proj.weight"] = np.zeros((cfg.intermediate_size, h), dtype=np.float32)
        raw[f"{prefix}.mlp.up_proj.weight"] = np.zeros((cfg.intermediate_size, h), dtype=np.float32)
        raw[f"{prefix}.mlp.down_proj.weight"] = np.zeros((h, cfg.intermediate_size), dtype=np.float32)
    return cfg, raw


def test_safetensors_bf16_single_and_sharded_loading(tmp_path: Path) -> None:
    """Verify zero-copy mmap safetensors loader reads BF16 and F32 across single & sharded checkpoints."""
    bf16_arr = np.array([[1.5, -2.25], [0.5, 4.0]], dtype=ml_dtypes.bfloat16)
    f32_arr = np.array([0.25, 0.75], dtype=np.float32)

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    _write_safetensors(
        single_dir / "model.safetensors",
        {"model.embed_tokens.weight": ("BF16", bf16_arr), "model.norm.weight": ("F32", f32_arr)},
    )
    loaded_single = load_safetensors_dir(single_dir)
    assert loaded_single["model.embed_tokens.weight"].dtype == ml_dtypes.bfloat16
    np.testing.assert_allclose(
        loaded_single["model.embed_tokens.weight"].astype(np.float32),
        bf16_arr.astype(np.float32),
    )
    np.testing.assert_allclose(loaded_single["model.norm.weight"], f32_arr)

    sharded_dir = tmp_path / "sharded"
    sharded_dir.mkdir()
    _write_safetensors(
        sharded_dir / "model-00001-of-00002.safetensors",
        {"model.embed_tokens.weight": ("BF16", bf16_arr)},
    )
    _write_safetensors(
        sharded_dir / "model-00002-of-00002.safetensors",
        {"model.norm.weight": ("F32", f32_arr)},
    )
    (sharded_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    loaded_sharded = load_safetensors_dir(sharded_dir)
    assert set(loaded_sharded.keys()) == {"model.embed_tokens.weight", "model.norm.weight"}
    np.testing.assert_allclose(loaded_sharded["model.norm.weight"], f32_arr)


def test_qwen_tied_vs_untied_embeddings() -> None:
    """Verify tie_word_embeddings reuses embed_tokens when lm_head.weight is absent."""
    cfg, raw = _make_cyclic_qwen_model()
    tied_cfg = QwenConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=cfg.vocab_size,
        head_dim=cfg.head_dim,
        tie_word_embeddings=True,
    )
    raw_no_lm = {k: v for k, v in raw.items() if k != "lm_head.weight"}
    weights_tied = QwenWeights.from_numpy_dict(raw_no_lm, tied_cfg, dtype=jnp.float32)
    assert weights_tied.lm_head.shape == (cfg.hidden_size, cfg.vocab_size)
    np.testing.assert_allclose(np.asarray(weights_tied.lm_head), np.asarray(weights_tied.embed).T)


def test_qwen_runner_zero_recompile_and_tree_parity() -> None:
    """Verify M-002 compile_count == 1 across odd prompt lengths and M-003 exact tree parity + speedup."""
    cfg, raw = _make_cyclic_qwen_model()
    weights = QwenWeights.from_numpy_dict(raw, cfg, dtype=jnp.float32)
    runner = QwenBenchmarkRunner(config=cfg, weights=weights, tokenizer=None, max_seq_len=512)

    # 1. Odd-length prompt sweep: Vanilla compiles per shape, StaticArena compiles once
    for odd_len in (17, 43, 135):
        prompt = [10, 11, 12, 13] * (odd_len // 4) + [10, 11, 12, 13][: odd_len % 4]
        runner.run_vanilla_jax(prompt, max_new_tokens=2)
        runner.run_static_arena(prompt, max_new_tokens=2)

    assert runner.vanilla_prefill_compile_count == 3
    assert runner.static_step_compile_count == 1

    # 2. Repetitive cyclic context: all 3 modes emit identical sequence, tree > 1.0 tok/pass
    rep_prompt = [10, 11, 12, 13] * 12 + [10]
    m_van = runner.run_vanilla_jax(rep_prompt, max_new_tokens=24)
    m_stat = runner.run_static_arena(rep_prompt, max_new_tokens=24)
    m_tree = runner.run_tpuforge_tree(rep_prompt, max_new_tokens=24)

    expected = [11, 12, 13, 10] * 6
    assert m_van.token_ids == expected
    assert m_stat.token_ids == expected
    assert m_tree.token_ids == expected
    assert m_tree.compile_count == 0
    assert m_tree.mean_tokens_per_decode_pass > 2.0


def test_forge_engine_configurable_eos_token_ids() -> None:
    """Verify custom eos_token_ids allow token ID 2 without stopping and halt on configured EOS ID."""
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 512
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 2
    cfg.head_dim = 64
    cfg.vocab_size = 512

    scripted = iter([2, 50, 2, 401])

    def step_fn(valid_prefix_len: int, tok_tile, pos_tile, mask):
        logits = np.full((128, cfg.vocab_size), -10.0, dtype=np.float32)
        if valid_prefix_len < 2:
            return logits
        nxt = next(scripted, 401)
        logits[0, nxt] = 10.0
        return logits

    engine = tpu.ForgeEngine(cfg, step_fn, None, "eos-test")
    req = tpu.GenerationRequest()
    req.prompt_tokens = [10, 20, 30]
    req.eos_token_ids = [400, 401]
    req.max_tokens = 16
    req.temperature = 0.0

    res = engine.generate(req)
    assert list(res.token_ids) == [2, 50, 2]
    assert res.finish_reason == "stop"

    bad_req = tpu.GenerationRequest()
    bad_req.prompt_tokens = [10, 20, 30]
    bad_req.eos_token_ids = [999]
    with pytest.raises(IndexError, match="eos_token_id out of"):
        engine.generate(bad_req)


def test_qwen_tile_256_and_streaming_loader(tmp_path: Path) -> None:
    """Verify tile_size=256 and streaming QwenWeights.from_model_dir match golden cyclic outputs."""
    cfg, raw = _make_cyclic_qwen_model()
    cfg_json = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "head_dim": cfg.head_dim,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "tie_word_embeddings": False,
        "eos_token_id": list(cfg.eos_token_ids),
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg_json), encoding="utf-8")

    keys = sorted(raw.keys())
    mid = len(keys) // 2
    s1 = {k: ("F32", raw[k].astype(np.float32)) for k in keys[:mid]}
    s2 = {k: ("F32", raw[k].astype(np.float32)) for k in keys[mid:]}
    _write_safetensors(tmp_path / "model-00001-of-00002.safetensors", s1)
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors", s2)
    weight_map = {k: ("model-00001-of-00002.safetensors" if k in s1 else "model-00002-of-00002.safetensors") for k in keys}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}), encoding="utf-8"
    )

    assert resolve_tile_size(256) == 256
    assert resolve_tile_size("128") == 128
    assert resolve_tp_mesh(1, cfg) == (1, None)

    weights = QwenWeights.from_model_dir(tmp_path, cfg, dtype=jnp.float32)
    runner = QwenBenchmarkRunner(
        config=cfg,
        weights=weights,
        tokenizer=None,
        max_seq_len=512,
        tile_size=256,
    )
    assert runner.tile_size == 256

    for odd_len in (17, 43):
        prompt = [10, 11, 12, 13] * (odd_len // 4) + [10, 11, 12, 13][: odd_len % 4]
        runner.run_static_arena(prompt, max_new_tokens=2)
    assert runner.static_step_compile_count == 1

    rep_prompt = [10, 11, 12, 13] * 12 + [10]
    m_stat = runner.run_static_arena(rep_prompt, max_new_tokens=24)
    m_tree = runner.run_tpuforge_tree(rep_prompt, max_new_tokens=24)
    expected = [11, 12, 13, 10] * 6
    assert m_stat.token_ids == expected
    assert m_tree.token_ids == expected
    assert m_tree.extra["tile_size"] == 256
    assert m_tree.mean_tokens_per_decode_pass > 2.0


def test_qwen_tp_sharding_parity_subprocess() -> None:
    """Verify multi-device TP=2 GSPMD sharding produces all-reduce HLO and exact TP=1 parity."""
    repo_root = Path(__file__).resolve().parent.parent
    script = (
        "import conftest\n"
        "import jax, jax.numpy as jnp\n"
        "from benchmarks.qwen_jax import QwenBenchmarkRunner, QwenWeights, resolve_tp_mesh\n"
        "from tests.test_qwen_benchmark import _make_cyclic_qwen_model\n"
        "assert len(jax.devices()) == 2, jax.devices()\n"
        "cfg, raw = _make_cyclic_qwen_model()\n"
        "tp, mesh = resolve_tp_mesh(2, cfg)\n"
        "assert tp == 2 and mesh is not None\n"
        "w1 = QwenWeights.from_numpy_dict(raw, cfg, dtype=jnp.float32, mesh=None)\n"
        "w2 = QwenWeights.from_numpy_dict(raw, cfg, dtype=jnp.float32, mesh=mesh)\n"
        "assert w2.wq.sharding.spec[-1] == 'tp' and w2.wo.sharding.spec[1] == 'tp'\n"
        "r1 = QwenBenchmarkRunner(cfg, w1, max_seq_len=512, tile_size=128)\n"
        "r2 = QwenBenchmarkRunner(cfg, w2, max_seq_len=512, tile_size=128, mesh=mesh)\n"
        "prompt = [10, 11, 12, 13] * 12 + [10]\n"
        "kc2, vc2 = r2._empty_kv_cache()\n"
        "hlo = r2._static_systolic_tile_step_jit.lower("
        "r2._weights_tuple, "
        "r2._put_rep(0, jnp.int32), r2._put_rep([0]*128, jnp.int32), "
        "r2._put_rep([0]*128, jnp.int32), r2._put_rep(jnp.zeros((128, 128), jnp.float32)), "
        "kc2, vc2).compile().as_text()\n"
        "assert 'all-reduce(' in hlo, 'Expected Megatron TP all-reduce in HLO'\n"
        "mv = r1.run_vanilla_jax(prompt, max_new_tokens=4)\n"
        "assert len(mv.token_ids) == 4\n"
        "m1 = r1.run_static_arena(prompt, max_new_tokens=16)\n"
        "m2 = r2.run_static_arena(prompt, max_new_tokens=16)\n"
        "m2_tree = r2.run_tpuforge_tree(prompt, max_new_tokens=16)\n"
        "assert m1.token_ids == m2.token_ids == m2_tree.token_ids == [11, 12, 13, 10] * 4\n"
        "assert m2_tree.extra['tp_size'] == 2\n"
    )
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    env["JAX_CAPTURED_CONSTANTS_WARN_BYTES"] = "1024"
    py_paths = [str(repo_root), str(repo_root / "tests")]
    venv_site = repo_root / ".venv" / "Lib" / "site-packages"
    if venv_site.is_dir():
        py_paths.append(str(venv_site))
    env["PYTHONPATH"] = os.pathsep.join(py_paths)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(repo_root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"TP=2 subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "constants were captured during lowering" not in proc.stderr, proc.stderr


def test_kaggle_bundle_packaging(tmp_path: Path) -> None:
    """Verify standalone Kaggle/Colab source bundle contains all sources and zero compiled binaries."""
    repo_root = Path(__file__).resolve().parent.parent
    out_zip = tmp_path / "tpuforge_kaggle_bundle.zip"
    built = build_kaggle_bundle(repo_root, out_zip)
    assert built.is_file()

    with zipfile.ZipFile(built, "r") as zf:
        names = set(zf.namelist())
        for required in (
            "CMakeLists.txt",
            "include/tpuforge/device.h",
            "src/main.cpp",
            "benchmarks/quantization_jax.py",
            "benchmarks/moe_streaming_jax.py",
            "benchmarks/ast_seed_jax.py",
            "benchmarks/qwen_jax.py",
            "benchmarks/block_diffusion_jax.py",
            "benchmarks/run_qwen_bench.py",
            "benchmarks/package_kaggle_bundle.py",
            "colab/TPUForge_Benchmark.ipynb",
        ):
            assert required in names, f"Missing {required} from Kaggle bundle"
        assert not any(n.endswith((".exe", ".dll", ".pyd", ".so", ".zip")) for n in names)
        nb = json.loads(zf.read("colab/TPUForge_Benchmark.ipynb").decode("utf-8"))
        assert nb["nbformat"] == 4 and len(nb["cells"]) >= 4


def test_qwen_bfloat16_native_mxu_parity() -> None:
    """Verify native BF16 MXU accumulation keeps 3-mode parity across Vanilla, StaticArena, and Tree."""
    cfg, raw = _make_cyclic_qwen_model()
    weights = QwenWeights.from_numpy_dict(raw, cfg, dtype=jnp.bfloat16)
    assert weights.wq.dtype == jnp.bfloat16

    runner = QwenBenchmarkRunner(
        config=cfg,
        weights=weights,
        tokenizer=None,
        max_seq_len=512,
        tile_size=128,
    )
    prompt = [10, 11, 12, 13] * 12 + [10]
    m_vanilla = runner.run_vanilla_jax(prompt, max_new_tokens=16)
    m_static = runner.run_static_arena(prompt, max_new_tokens=16)
    m_tree = runner.run_tpuforge_tree(prompt, max_new_tokens=16)

    expected = [11, 12, 13, 10] * 4
    assert m_vanilla.token_ids == expected
    assert m_static.token_ids == expected
    assert m_tree.token_ids == expected
    assert m_tree.mean_tokens_per_decode_pass > 2.0


def test_forge_engine_on_device_argmax_zero_logits_transfer() -> None:
    """Verify greedy prefill+decode invoke prefill_step_fn/greedy_step_fn with 0 full-logits calls."""
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 512
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 2
    cfg.head_dim = 64
    cfg.vocab_size = 512
    cfg.tile_size = 128

    prompt = [10 + (i % 10) for i in range(140)]

    def _compute_next(prev_prefix_len: int, tok_tile: np.ndarray, pos_tile: np.ndarray) -> np.ndarray:
        out = np.empty((128,), dtype=np.int32)
        for r in range(128):
            out[r] = 20 + ((int(tok_tile[r]) + int(pos_tile[r]) + int(prev_prefix_len)) % 30)
        return out

    counts_full = {"logits": 0, "kv": 0}
    counts_fast = {"logits": 0, "prefill": 0, "greedy": 0, "kv": 0}

    def full_logits_fn(valid_prefix_len: int, tok_tile, pos_tile, mask):
        counts_full["logits"] += 1
        next_toks = _compute_next(valid_prefix_len, tok_tile, pos_tile)
        logits = np.full((128, cfg.vocab_size), -10.0, dtype=np.float32)
        logits[np.arange(128), next_toks] = 10.0
        return logits

    def full_logits_tracked_fn(valid_prefix_len: int, tok_tile, pos_tile, mask):
        counts_fast["logits"] += 1
        next_toks = _compute_next(valid_prefix_len, tok_tile, pos_tile)
        logits = np.full((128, cfg.vocab_size), -10.0, dtype=np.float32)
        logits[np.arange(128), next_toks] = 10.0
        return logits

    def prefill_fn(valid_prefix_len: int, tok_tile, pos_tile, mask):
        counts_fast["prefill"] += 1

    def greedy_fn(valid_prefix_len: int, tok_tile, pos_tile, mask):
        counts_fast["greedy"] += 1
        return _compute_next(valid_prefix_len, tok_tile, pos_tile)

    def kv_counter(counter: dict[str, int]):
        def _commit(prev_len: int, accepted: list[int]) -> None:
            counter["kv"] += 1

        return _commit

    engine_ref = tpu.ForgeEngine(cfg, full_logits_fn, kv_counter(counts_full), "ref-logits")
    engine_fast = tpu.ForgeEngine(
        cfg,
        full_logits_tracked_fn,
        kv_counter(counts_fast),
        "fast-argmax",
        prefill_step_fn=prefill_fn,
        greedy_step_fn=greedy_fn,
    )

    req_greedy = tpu.GenerationRequest()
    req_greedy.prompt_tokens = prompt
    req_greedy.max_tokens = 12
    req_greedy.temperature = 0.0
    req_greedy.ignore_eos = True

    res_ref = engine_ref.generate(req_greedy)
    res_fast = engine_fast.generate(req_greedy)
    assert list(res_fast.token_ids) == list(res_ref.token_ids)
    assert len(res_fast.token_ids) == 12
    assert counts_fast["prefill"] >= 2
    assert counts_fast["greedy"] >= 1
    assert counts_fast["logits"] == 0
    assert counts_fast["kv"] == counts_full["kv"]

    # Stochastic sampling cleanly routes to target_step_fn without touching greedy_step_fn.
    greedy_before = counts_fast["greedy"]
    req_stoch = tpu.GenerationRequest()
    req_stoch.prompt_tokens = prompt
    req_stoch.max_tokens = 6
    req_stoch.temperature = 0.8
    req_stoch.seed = 42
    req_stoch.ignore_eos = True
    res_stoch = engine_fast.generate(req_stoch)
    assert len(res_stoch.token_ids) == 6
    assert counts_fast["logits"] >= 1
    assert counts_fast["greedy"] == greedy_before

    # Greedy-only engine (target_step_fn=None) works for temperature=0.0 and rejects temperature > 0.
    engine_greedy_only = tpu.ForgeEngine(
        cfg, None, kv_counter({"kv": 0}), "greedy-only", greedy_step_fn=greedy_fn
    )
    res_only = engine_greedy_only.generate(req_greedy)
    assert list(res_only.token_ids) == list(res_ref.token_ids)
    with pytest.raises(ValueError, match="target_step_fn"):
        engine_greedy_only.generate(req_stoch)

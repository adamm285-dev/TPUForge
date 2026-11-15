"""Tests for Ollama-style TPUForge Serve presets, Web HUD compliance, HTTP/SSE API, and exact parity."""

from __future__ import annotations

import http.client
import json
import shutil
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.hud_ui import render_hud_html
from benchmarks.qwen_jax import (
    _is_complete_safetensors_dir,
    discover_local_models,
    resolve_model_dir,
)
from benchmarks.tpuforge_serve import (
    ForgeServeHTTPServer,
    load_serve_session,
    main,
    resolve_preset_spec,
    start_serve_background,
)
from tests.test_block_tree_and_suffix import _build_qwen3_and_block_drafter_dirs
from tests.test_gemma4 import (
    _create_synthetic_gemma4_dflash_dir,
    _create_synthetic_gemma4_dir,
    _write_safetensors,
)


def _http_request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    parsed = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
    try:
        headers: dict[str, str] = {}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


def test_preset_catalog_and_hardware_autofit(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["--list-presets"])
    assert rc == 0
    stdout = capsys.readouterr().out
    for preset in ("gemma4:12b", "gemma4:12b-bf16", "gemma4:26b", "gemma4:31b", "qwen3:4b"):
        assert preset in stdout

    # Single-chip v5e (15.75 GiB): 26B MoE overflows HBM -> falls back cleanly to host mmap stream.
    v5e_plan = resolve_preset_spec(
        "gemma4:26b",
        chip_count=1,
        hbm_per_chip_gib=15.75,
        device_kind="TPU v5 lite",
    )
    assert v5e_plan.tile_size == 128
    assert v5e_plan.host_store == "mmap"
    assert v5e_plan.draft_model_id is None

    # Single-chip v6e Trillium (31.24 GiB): INT8 26B MoE + DFlash fits 100% in HBM on 256x256 tile.
    v6e_plan = resolve_preset_spec(
        "gemma4:26b",
        chip_count=1,
        hbm_per_chip_gib=31.24,
        device_kind="TPU v6 lite",
    )
    assert v6e_plan.tile_size == 256
    assert v6e_plan.host_store == "hbm"
    assert v6e_plan.draft_model_id == "z-lab/gemma-4-26B-A4B-it-DFlash"

    # Multi-chip v5e-8 (8 x 15.75 GiB): 31B Dense shards across TP=8 inside HBM.
    pod_plan = resolve_preset_spec(
        "gemma4:31b",
        chip_count=8,
        hbm_per_chip_gib=15.75,
        device_kind="TPU v5 lite",
    )
    assert pod_plan.tp_size == 8
    assert pod_plan.host_store == "hbm"
    assert pod_plan.tile_size == 128


def test_hud_html_modern_web_compliance() -> None:
    for tile in (128, 256):
        html = render_hud_html(
            {
                "preset": "gemma4:12b",
                "model_id": "google/gemma-4-12B-it",
                "tile_size": tile,
                "tp_size": 1,
                "weight_quant": "int8",
                "has_draft_model": True,
                "ast_seed_sequences": 3,
                "ast_seed_tokens": 24,
                "hardware_label": f"cpu x1 (tile={tile})",
                "max_seq_len": 2048,
            }
        )
        assert "\u2014" not in html
        for forbidden in ("fonts.googleapis.com", "unpkg.com", "jsdelivr", "onclick="):
            assert forbidden not in html
        for required in (
            '<meta name="color-scheme" content="light dark">',
            "@layer reset, base, theme, components, utilities;",
            "light-dark(",
            "field-sizing: content",
            "event.isComposing",
            "event.keyCode === 229",
            'closedby="any"',
            'inputmode="numeric"',
            'aria-live="polite"',
            'aria-live="assertive"',
            "requestAnimationFrame",
            "tabular-nums",
            "container-type: inline-size",
            'role="region"',
            "aria-pressed",
            "user-invalid-fallback",
            "evt.error",
        ):
            assert required in html


def test_gemma4_serve_http_sse_and_exact_parity(tmp_path: Path) -> None:
    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_serve_target",
        seed=42,
        enable_moe=False,
        num_layers=6,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=320,
    )
    draft_dir = _create_synthetic_gemma4_dflash_dir(
        tmp_path / "gemma4_serve_dflash",
        seed=99,
        hidden_size=32,
        num_target_layers=6,
        vocab_size=320,
    )
    session = load_serve_session(
        str(ckpt_dir),
        draft_model=str(draft_dir),
        max_seq_len=256,
        dtype="float32",
        tile_size=128,
        tp_size=1,
        weight_quant="int8",
        enable_xla_cache=False,
        warmup=True,
    )
    server, _thread, url = start_serve_background(session, host="127.0.0.1", port=0)
    try:
        # 1. Static GET + OPTIONS endpoints & 404 handling.
        st, hdrs, html_bytes = _http_request(url, "GET", "/")
        assert st == 200
        assert "text/html" in hdrs["content-type"]
        html_text = html_bytes.decode("utf-8")
        assert "\u2014" not in html_text
        assert "TPUForge Serve" in html_text

        st, _, health_raw = _http_request(url, "GET", "/health")
        assert st == 200
        assert json.loads(health_raw.decode("utf-8"))["status"] == "ok"

        st, _, status_raw = _http_request(url, "GET", "/api/status")
        assert st == 200
        assert json.loads(status_raw.decode("utf-8"))["has_draft_model"] is True

        st, _, models_raw = _http_request(url, "GET", "/v1/models")
        assert st == 200
        assert json.loads(models_raw.decode("utf-8"))["object"] == "list"

        st, hdrs, _ = _http_request(url, "OPTIONS", "/v1/chat/completions")
        assert st == 204
        assert hdrs["access-control-allow-origin"] == "*"

        st, _, _ = _http_request(url, "GET", "/nonexistent")
        assert st == 404

        # 2. Out-of-band AST seeding via /api/seed.
        seed_file = tmp_path / "sample_seed.py"
        seed_file.write_text(
            "def tok12(tok14: int) -> int:\n    return tok14\n",
            encoding="utf-8",
        )
        st, _, seed_raw = _http_request(
            url, "POST", "/api/seed", {"ast_seed_path": str(seed_file)}
        )
        assert st == 200
        seed_status = json.loads(seed_raw.decode("utf-8"))
        assert seed_status["ast_seed_sequences"] > 0
        assert seed_status["ast_seed_tokens"] > 0

        # 3. Exact token & text parity across block_tree, tree, and static_arena vs runners.
        prompt = [2, 12, 14, 16, 12, 14, 16, 18, 12, 14, 16]
        assert session.block_runner is not None
        ref_bt = session.block_runner.run_block_tree(prompt, max_new_tokens=10, eos_token_ids=[1])
        ref_tree = session.target_runner.run_tpuforge_tree(
            prompt, max_new_tokens=10, eos_token_ids=[1]
        )
        ref_static = session.target_runner.run_static_arena(
            prompt, max_new_tokens=10, eos_token_ids=[1]
        )

        serve_texts: dict[str, str] = {}
        serve_tokens: dict[str, list[int]] = {}
        for mode_name, ref_m in (
            ("block_tree", ref_bt),
            ("tree", ref_tree),
            ("static_arena", ref_static),
        ):
            st, _, raw = _http_request(
                url,
                "POST",
                "/v1/chat/completions",
                {
                    "model": "gemma4:12b",
                    "prompt_tokens": prompt,
                    "max_tokens": 10,
                    "mode": mode_name,
                    "stream": False,
                },
            )
            assert st == 200
            payload = json.loads(raw.decode("utf-8"))
            tokens = [int(t) for t in payload["token_ids"]]
            text = str(payload["choices"][0]["message"]["content"])
            assert tokens == ref_m.token_ids
            assert text == ref_m.text
            serve_tokens[mode_name] = tokens
            serve_texts[mode_name] = text

        assert (
            serve_tokens["block_tree"]
            == serve_tokens["tree"]
            == serve_tokens["static_arena"]
        )

        # 4. SSE streaming with live tpuforge_step telemetry + [DONE] + exact text parity.
        parsed = urllib.parse.urlparse(url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=30)
        try:
            req_bytes = json.dumps(
                {
                    "model": "gemma4:12b",
                    "prompt_tokens": prompt,
                    "max_tokens": 10,
                    "mode": "block_tree",
                    "stream": True,
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=req_bytes,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(req_bytes)),
                },
            )
            resp = conn.getresponse()
            assert resp.status == 200
            raw_sse = resp.read().decode("utf-8")
        finally:
            conn.close()

        sse_deltas: list[str] = []
        step_events: list[dict[str, Any]] = []
        final_telemetry: dict[str, Any] | None = None
        saw_done = False
        for line in raw_sse.splitlines():
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            body_str = stripped[5:].strip()
            if body_str == "[DONE]":
                saw_done = True
                continue
            evt = json.loads(body_str)
            delta = evt["choices"][0]["delta"].get("content", "")
            if delta:
                sse_deltas.append(delta)
            if "tpuforge_step" in evt:
                step_events.append(evt["tpuforge_step"])
            if "tpuforge_telemetry" in evt:
                final_telemetry = evt["tpuforge_telemetry"]

        assert saw_done
        assert len(step_events) >= 1
        for step in step_events:
            assert 0 in step["accepted_rows"]
            assert 1 <= int(step["drafted_nodes"]) <= 128
        assert final_telemetry is not None
        assert "".join(sse_deltas) == serve_texts["block_tree"]

        # 5. Zero steady-state recompilations on unaligned prompt lengths.
        st, _, reuse_raw = _http_request(
            url,
            "POST",
            "/v1/chat/completions",
            {
                "model": "gemma4:12b",
                "prompt_tokens": [2, 8, 9, 10, 11, 12, 13],
                "max_tokens": 6,
                "mode": "block_tree",
                "stream": False,
            },
        )
        assert st == 200
        assert json.loads(reuse_raw.decode("utf-8"))["tpuforge_telemetry"]["compile_count"] == 0

        # 6. Validation errors return HTTP 400.
        st, _, _ = _http_request(
            url,
            "POST",
            "/v1/chat/completions",
            {"prompt_tokens": prompt, "max_tokens": 6, "mode": "bad_mode"},
        )
        assert st == 400
        st, _, _ = _http_request(
            url,
            "POST",
            "/v1/chat/completions",
            {"prompt_tokens": prompt, "max_tokens": 9999, "mode": "tree"},
        )
        assert st == 400
    finally:
        server.shutdown()
        server.server_close()
        session.close()


def test_qwen_and_moe_mmap_serve_session_parity(tmp_path: Path) -> None:
    # 1. Gemma 4 MoE mmap streaming session parity + clean Windows mmap close.
    moe_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_moe_serve",
        seed=77,
        enable_moe=True,
        num_layers=6,
        hidden_size=32,
        intermediate_size=64,
        num_experts=12,
        top_k_experts=2,
        moe_intermediate_size=32,
    )
    moe_session = load_serve_session(
        str(moe_dir),
        max_seq_len=256,
        dtype="float32",
        tile_size=128,
        tp_size=1,
        weight_quant="int8",
        host_store="mmap",
        stage_experts=3,
        enable_xla_cache=False,
        warmup=False,
    )
    try:
        prompt = [2, 15, 22, 29, 36, 15, 22, 29, 36]
        with pytest.raises(ValueError, match="block_tree"):
            next(moe_session.stream_generate(prompt_tokens=prompt, max_tokens=6, mode="block_tree"))

        res_static = list(
            moe_session.stream_generate(prompt_tokens=prompt, max_tokens=8, mode="static_arena")
        )[-1]
        res_tree = list(
            moe_session.stream_generate(prompt_tokens=prompt, max_tokens=8, mode="tree")
        )[-1]
        assert res_static.token_ids == res_tree.token_ids
    finally:
        moe_session.close()

    shutil.rmtree(moe_dir)
    assert not moe_dir.exists()

    # 2. Cross-architecture Qwen3 + DFlash block_tree == tree == static_arena session parity.
    qwen_target_dir, qwen_draft_dir = _build_qwen3_and_block_drafter_dirs(tmp_path / "qwen_dflash")
    # Replace duplicate-ID test tokenizer with unique vocab keys so WordLevel encodes arbitrary OOD idioms cleanly.
    q_id_to_tok = [f"tok{i}" for i in range(288)]
    q_id_to_tok[0] = "<unk>"
    q_id_to_tok[1] = "<|im_start|>"
    q_id_to_tok[2] = "<|im_end|>"
    q_id_to_tok[3] = "<think>"
    q_id_to_tok[4] = "</think>"
    q_tok = Tokenizer(WordLevel(vocab={t: i for i, t in enumerate(q_id_to_tok)}, unk_token="<unk>"))
    q_tok.pre_tokenizer = Whitespace()
    q_tok.save(str(qwen_target_dir / "tokenizer.json"))

    qwen_session = load_serve_session(
        str(qwen_target_dir),
        draft_model=str(qwen_draft_dir),
        max_seq_len=256,
        dtype="float32",
        tile_size=128,
        tp_size=1,
        enable_xla_cache=False,
        warmup=True,
    )
    try:
        q_prompt = [1, 12, 14, 16, 12, 14, 16, 18]
        q_bt = list(qwen_session.stream_generate(prompt_tokens=q_prompt, max_tokens=8, mode="block_tree"))[-1]
        q_tr = list(qwen_session.stream_generate(prompt_tokens=q_prompt, max_tokens=8, mode="tree"))[-1]
        q_st = list(qwen_session.stream_generate(prompt_tokens=q_prompt, max_tokens=8, mode="static_arena"))[-1]
        assert q_bt.token_ids == q_tr.token_ids == q_st.token_ids
        assert len(q_bt.token_ids) == 8
    finally:
        qwen_session.close()


def test_model_discovery_shard_validation_and_local_only_guard(tmp_path: Path) -> None:
    import numpy as np
    import pytest

    models_root = tmp_path / "models"
    incomplete_dir = models_root / "qwen3-incomplete"
    incomplete_dir.mkdir(parents=True)
    (incomplete_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    shard_index = {
        "metadata": {"total_size": 32},
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.weight": "model-00002-of-00002.safetensors",
        },
    }
    (incomplete_dir / "model.safetensors.index.json").write_text(
        json.dumps(shard_index),
        encoding="utf-8",
    )
    _write_safetensors(
        incomplete_dir / "model-00001-of-00002.safetensors",
        {"a.weight": np.zeros((2, 2), dtype=np.float32)},
    )
    assert _is_complete_safetensors_dir(incomplete_dir) is False

    complete_dir = models_root / "gemma4-sharded"
    complete_dir.mkdir(parents=True)
    (complete_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForCausalLM"],
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )
    (complete_dir / "model.safetensors.index.json").write_text(
        json.dumps(shard_index),
        encoding="utf-8",
    )
    _write_safetensors(
        complete_dir / "model-00001-of-00002.safetensors",
        {"a.weight": np.zeros((2, 2), dtype=np.float32)},
    )
    _write_safetensors(
        complete_dir / "model-00002-of-00002.safetensors",
        {"b.weight": np.zeros((2, 2), dtype=np.float32)},
    )
    assert _is_complete_safetensors_dir(complete_dir) is True

    discovered = discover_local_models(search_roots=[models_root])
    discovered_paths = {Path(str(entry["local_dir"])).resolve() for entry in discovered}
    assert complete_dir.resolve() in discovered_paths
    assert incomplete_dir.resolve() not in discovered_paths

    with pytest.raises(FileNotFoundError):
        resolve_model_dir(
            "nonexistent/local-only-model",
            search_roots=[models_root],
            allow_download=False,
        )
    assert (
        resolve_model_dir(
            str(complete_dir),
            search_roots=[models_root],
            allow_download=False,
        ).resolve()
        == complete_dir.resolve()
    )


def test_exclusive_port_binding_cli_flags_and_strict_http_validation(
    tmp_path: Path,
    capsys,
) -> None:
    import http.client

    import pytest

    ckpt_dir, _ = _create_synthetic_gemma4_dir(
        tmp_path / "gemma4_cli_serve",
        seed=7,
        enable_moe=False,
        num_layers=6,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=320,
    )

    assert main(["--list-local"]) == 0
    _ = capsys.readouterr()

    assert main([str(ckpt_dir), "--dry-run", "--weight-quant", "int8", "--tp-size", "1"]) == 0
    _ = capsys.readouterr()

    for bad_argv in (
        [str(ckpt_dir), "--port", "70000", "--dry-run"],
        [str(ckpt_dir), "--local-only", "--allow-download", "--dry-run"],
    ):
        with pytest.raises((SystemExit, ValueError)):
            rc = main(bad_argv)
            if rc != 0:
                raise SystemExit(rc)

    with pytest.raises(TypeError):
        resolve_preset_spec("custom", tp_size=True)
    with pytest.raises(TypeError):
        resolve_preset_spec("custom", max_seq_len=True)

    session = load_serve_session(
        str(ckpt_dir),
        max_seq_len=256,
        dtype="float32",
        tile_size=128,
        tp_size=1,
        weight_quant="int8",
        enable_xla_cache=False,
        warmup=False,
    )
    server, _thread, _url = start_serve_background(session, host="127.0.0.1", port=0)
    try:
        assert server.bound_port > 0
        with pytest.raises(OSError):
            dup = ForgeServeHTTPServer(("127.0.0.1", server.bound_port), session)
            dup.server_close()

        conn = http.client.HTTPConnection("127.0.0.1", server.bound_port, timeout=10)
        try:
            conn.request("GET", "/api/local-models")
            local_resp = conn.getresponse()
            local_body = json.loads(local_resp.read().decode("utf-8"))
            assert local_resp.status == 200
            assert isinstance(local_body, dict)
            assert isinstance(local_body.get("models"), list)

            missing_repo_payload = json.dumps({"repo_path": str(tmp_path / "missing_repo_path")}).encode("utf-8")
            conn.request(
                "POST",
                "/api/seed",
                body=missing_repo_payload,
                headers={"Content-Type": "application/json"},
            )
            seed_resp = conn.getresponse()
            seed_body = json.loads(seed_resp.read().decode("utf-8"))
            assert seed_resp.status == 404
            assert "error" in seed_body

            for bad_chat in (
                {"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": True},
                {"prompt_tokens": [], "stream": True, "max_tokens": 8},
            ):
                conn.request(
                    "POST",
                    "/v1/chat/completions",
                    body=json.dumps(bad_chat).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                chat_resp = conn.getresponse()
                chat_raw = chat_resp.read().decode("utf-8")
                chat_ct = chat_resp.getheader("Content-Type", "")
                assert chat_resp.status == 400
                assert "application/json" in chat_ct
                chat_err = json.loads(chat_raw)
                assert "error" in chat_err
        finally:
            conn.close()
    finally:
        server.shutdown()
        server.server_close()
        session.close()

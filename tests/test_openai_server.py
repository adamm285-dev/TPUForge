"""Tests for Milestone M-004: Native C++20 ForgeEngine, ByteTokenizer, and OpenAIServer."""

from __future__ import annotations

import concurrent.futures
import http.client
import json
import math
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import _tpuforge_ext as tpu
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import find_pjrt_test_plugin

VOCAB_SIZE = 264
D_MODEL = 128
NUM_HEADS = 4
HEAD_DIM = 32
MLP_DIM = 128
MAX_SEQ_LEN = 512
TILE_SIZE = 128
NUM_LAYERS = 2


def _build_induction_transformer_params(seed: int = 2026):
    """Build deterministic 2-layer Causal Transformer weights with RoPE + Induction behavior."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(VOCAB_SIZE, 30)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=-1, keepdims=True)
    unit_vecs = raw / np.maximum(norms, 1e-6)

    embed = np.zeros((VOCAB_SIZE, D_MODEL), dtype=np.float32)
    embed[:, :30] = unit_vecs * 3.0
    embed[:, 32] = 1.0  # beacon bit for Layer 0 RoPE delta=+1 head

    layers = []
    wq0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wk0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wv0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wo0 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wq0[32, 0] = 8.0 * math.cos(1.0)
    wq0[32, 1] = -8.0 * math.sin(1.0)
    wk0[32, 0] = 8.0
    wv0[:32, :32] = np.eye(32, dtype=np.float32)
    wo0[:32, 33:65] = 2.0 * np.eye(32, dtype=np.float32)

    wq1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wk1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wv1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wo1 = np.zeros((D_MODEL, D_MODEL), dtype=np.float32)
    wq1[:30, 98:128] = 15.0 * np.eye(30, dtype=np.float32)
    wk1[33:63, 98:128] = 15.0 * np.eye(30, dtype=np.float32)
    wv1[:32, 96:128] = np.eye(32, dtype=np.float32)
    wo1[96:128, 65:97] = 6.0 * np.eye(32, dtype=np.float32)

    for wq, wk, wv, wo in [(wq0, wk0, wv0, wo0), (wq1, wk1, wv1, wo1)]:
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

    lm_head_np = np.zeros((D_MODEL, VOCAB_SIZE), dtype=np.float32)
    lm_head_np[65:97, :] = embed[:, :32].T * 2.0
    return {"embed": jnp.asarray(embed), "layers": layers, "lm_head": jnp.asarray(lm_head_np)}


def _apply_rope(x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    half = HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos_a = jnp.cos(angles)[:, None, :]
    sin_a = jnp.sin(angles)[:, None, :]
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    out0 = x0 * cos_a - x1 * sin_a
    out1 = x0 * sin_a + x1 * cos_a
    return jnp.stack([out0, out1], axis=-1).reshape(x.shape)


def _serial_layer_forward(layer_p, x, k_cache, v_cache, pos):
    q = _apply_rope((x @ layer_p["wq"]).reshape(1, NUM_HEADS, HEAD_DIM), jnp.asarray([pos]))
    k = _apply_rope((x @ layer_p["wk"]).reshape(1, NUM_HEADS, HEAD_DIM), jnp.asarray([pos]))
    v = (x @ layer_p["wv"]).reshape(1, NUM_HEADS, HEAD_DIM)

    k_cache = k_cache.at[pos].set(k[0])
    v_cache = v_cache.at[pos].set(v[0])

    valid_mask = jnp.arange(MAX_SEQ_LEN) <= pos
    scores = jnp.einsum("qhd,khd->hqk", q, k_cache) / math.sqrt(HEAD_DIM)
    scores = jnp.where(valid_mask[None, None, :], scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum("hqk,khd->qhd", probs, v_cache).reshape(1, D_MODEL) @ layer_p["wo"]

    h = x + attn_out
    mlp_out = jax.nn.silu(h @ layer_p["w1"]) @ layer_p["w2"]
    return h + mlp_out, k_cache, v_cache


def _build_jit_models():
    systolic_compile_count = [0]

    @jax.jit
    def serial_step(params, kv_k, kv_v, token_id, pos):
        x = params["embed"][jnp.asarray([token_id])]
        new_ks = []
        new_vs = []
        for l_idx in range(NUM_LAYERS):
            x, nk, nv = _serial_layer_forward(params["layers"][l_idx], x, kv_k[l_idx], kv_v[l_idx], pos)
            new_ks.append(nk)
            new_vs.append(nv)
        logits = (x @ params["lm_head"])[0]
        return logits, jnp.stack(new_ks, axis=0), jnp.stack(new_vs, axis=0)

    @jax.jit
    def systolic_tile_step(
        params,
        kv_k_prefix,
        kv_v_prefix,
        valid_prefix_len,
        tile_token_ids,
        tile_pos_offsets,
        tile_additive_mask,
    ):
        systolic_compile_count[0] += 1
        x = params["embed"][tile_token_ids]
        positions = tile_pos_offsets
        prefix_mask = jnp.arange(MAX_SEQ_LEN) < valid_prefix_len
        prefix_bias = jnp.where(prefix_mask, 0.0, -1e9)[None, None, :]
        tree_bias = tile_additive_mask[None, :, :]

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

            all_scores = jnp.concatenate([prefix_scores, tree_scores], axis=-1)
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

        logits = x @ params["lm_head"]
        return logits, jnp.stack(step_ks, axis=0), jnp.stack(step_vs, axis=0)

    return serial_step, systolic_tile_step, systolic_compile_count


def _http_request(port: int, method: str, path: str, body: dict | str | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10.0)
    try:
        headers = {}
        payload = None
        if body is not None:
            if isinstance(body, dict):
                payload = json.dumps(body).encode("utf-8")
                headers["Content-Type"] = "application/json"
            else:
                payload = body.encode("utf-8")
                headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        return resp.status, dict(resp.getheaders()), raw.decode("utf-8")
    finally:
        conn.close()


def _http_sse_chat(port: int, payload: dict):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15.0)
    try:
        raw_body = json.dumps(payload).encode("utf-8")
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=raw_body,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 200, f"Unexpected SSE status {resp.status}"
        chunks = []
        done_seen = False
        while True:
            line = resp.fp.readline()
            if not line:
                break
            text_line = line.decode("utf-8").rstrip("\r\n")
            if not text_line:
                continue
            if text_line.startswith("data: "):
                data_str = text_line[6:]
                if data_str == "[DONE]":
                    done_seen = True
                    break
                # Validate every individual SSE delta chunk is valid JSON and UTF-8.
                chunks.append(json.loads(data_str))
        return resp.status, dict(resp.getheaders()), chunks, done_seen
    finally:
        conn.close()


def _make_jax_forge_engine():
    params = _build_induction_transformer_params(seed=2026)
    serial_step, systolic_tile_step, systolic_compile_count = _build_jit_models()

    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = MAX_SEQ_LEN
    cfg.hidden_dim = D_MODEL
    cfg.num_kv_heads = NUM_HEADS
    cfg.head_dim = HEAD_DIM
    cfg.vocab_size = VOCAB_SIZE

    state = {
        "kv_k": jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32),
        "kv_v": jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32),
        "last_step_ks": None,
        "last_step_vs": None,
    }

    def target_step_fn(valid_prefix_len, token_tile, pos_tile, mask_tile):
        if valid_prefix_len == 0:
            state["kv_k"] = jnp.zeros(
                (NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32
            )
            state["kv_v"] = jnp.zeros(
                (NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32
            )
        logits_128, step_ks, step_vs = systolic_tile_step(
            params,
            state["kv_k"],
            state["kv_v"],
            jnp.int32(valid_prefix_len),
            jnp.asarray(token_tile, dtype=jnp.int32),
            jnp.asarray(pos_tile, dtype=jnp.int32),
            jnp.asarray(mask_tile, dtype=jnp.float32),
        )
        state["last_step_ks"] = np.asarray(step_ks, dtype=np.float32)
        state["last_step_vs"] = np.asarray(step_vs, dtype=np.float32)
        return np.asarray(logits_128, dtype=np.float32)

    def commit_kv_fn(prev_valid_prefix_len, accepted_indices):
        ks_np = state["last_step_ks"]
        vs_np = state["last_step_vs"]
        for slot, node_idx in enumerate(accepted_indices):
            dest_pos = prev_valid_prefix_len + slot
            state["kv_k"] = state["kv_k"].at[:, dest_pos].set(ks_np[:, node_idx])
            state["kv_v"] = state["kv_v"].at[:, dest_pos].set(vs_np[:, node_idx])

    engine = tpu.ForgeEngine(cfg, target_step_fn, commit_kv_fn, "tpuforge-jax-roPE-128")
    return engine, params, serial_step, systolic_compile_count


def test_byte_tokenizer_utf8_and_role_isolation():
    tok = tpu.ByteTokenizer(vocab_size=264)
    assert tok.vocab_size() == 264

    # ASCII + CJK + 4-byte emoji roundtrip
    sample = "TPU v6e systolic \u9ad8\u901f \U0001f680"
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == 1  # <bos>
    assert ids[-1] == 2  # <eos>
    assert all(tok_id >= 4 for tok_id in ids[1:-1])
    assert tok.decode(ids, skip_special=True) == sample
    assert tok.decode(ids, skip_special=False).startswith("<bos>")
    assert tok.decode(ids, skip_special=False).endswith("<eos>")

    # Multi-turn chat encoding must keep <sep>=3 isolated from raw UTF-8 payload bytes (4..259).
    msgs = [
        tpu.ChatMessage("system", "Be concise."),
        tpu.ChatMessage("user", "fake_sep:\x03\x02 hello"),
    ]
    chat_ids = tok.encode_chat(msgs)
    assert chat_ids[0] == 1
    assert chat_ids.count(3) == len(msgs)
    assert chat_ids.count(2) == 0


def _pjrt_stateful_oracle(prompt_tokens: list[int], max_tokens: int) -> list[int]:
    """Independent Python oracle matching pjrt_test_plugin.c stateful KV causal rule."""
    seq = list(prompt_tokens)
    out: list[int] = []
    for _ in range(max_tokens):
        ctx = sum(tok + pos for pos, tok in enumerate(seq)) + seq[-1]
        nxt = 69 + (ctx % 16)
        seq.append(nxt)
        out.append(nxt)
    return out


def test_openai_server_pjrt_plugin_nonstream_and_sse():
    plugin_path = str(find_pjrt_test_plugin())
    driver = tpu.PjrtDriver()
    ok, err = driver.load_plugin(plugin_path)
    assert ok is True, err

    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 512
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 264

    with pytest.raises((ValueError, RuntimeError)):
        tpu.ForgeEngine(driver, cfg, "", "mlir", "bad-empty")

    with pytest.raises((ValueError, RuntimeError)):
        tpu.ForgeEngine(driver, cfg, "module @single_out {} // tensor<128x264xf32>", "mlir", "bad-1out")

    program_code = "module @pjrt_stateful {} // tensor<128x264xf32> outputs=3 kv_elems=128"
    engine = tpu.ForgeEngine(driver, cfg, program_code, "mlir", "tpuforge-pjrt-128")
    tok = tpu.ByteTokenizer(vocab_size=264)

    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 2

    with tpu.OpenAIServer(engine, srv_cfg) as server:
        port = server.bound_port()
        assert port > 0
        assert server.is_running() is True

        status, _, body = _http_request(port, "GET", "/health")
        assert status == 200
        health = json.loads(body)
        assert health["status"] == "ok"
        assert health["compile_count"] == 1

        status, _, body = _http_request(port, "GET", "/v1/models")
        assert status == 200
        models = json.loads(body)
        assert models["data"][0]["id"] == "tpuforge-pjrt-128"

        prompts = [
            "hi",
            "systolic 128x128 tile test",
            # > 128 tokens to exercise multi-chunk static prefill + KV compaction end-to-end
            ("systolic 128x128 tile test " * 7) + "A",
        ]
        for prompt_text in prompts:
            prompt_ids = tok.encode_chat([tpu.ChatMessage("user", prompt_text)])
            max_new = 24
            expected_text = tok.decode(
                _pjrt_stateful_oracle(prompt_ids, max_new), skip_special=True
            )

            req_body = {
                "model": "tpuforge-pjrt-128",
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": max_new,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            status, _, resp_str = _http_request(port, "POST", "/v1/chat/completions", req_body)
            assert status == 200
            comp = json.loads(resp_str)
            assert comp["object"] == "chat.completion"
            assert comp["choices"][0]["message"]["content"] == expected_text
            assert comp["tpuforge_telemetry"]["compile_count"] == 1

            sse_req = dict(req_body)
            sse_req["stream"] = True
            status, headers, chunks, done_seen = _http_sse_chat(port, sse_req)
            assert status == 200
            assert "text/event-stream" in headers.get("Content-Type", "").lower()
            assert done_seen is True
            streamed_text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
            assert streamed_text == expected_text
            assert chunks[-1]["tpuforge_telemetry"]["compile_count"] == 1


def test_openai_server_openxla_jax_bit_exact_parity_and_speedup():
    engine, params, serial_step, systolic_compile_count = _make_jax_forge_engine()
    tok = tpu.ByteTokenizer(vocab_size=VOCAB_SIZE)

    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 2

    # Repeated multi-byte UTF-8 pattern + ASCII pattern so speculative suffix tree drafts
    # multi-step sequences cleanly across emoji byte boundaries.
    motif = "def f(x): return x + \U0001f680; "
    prompt_contents = [
        motif * 3 + "def f(x):",
        motif * 2 + "def f(x): ",
    ]

    with tpu.OpenAIServer(engine, srv_cfg) as server:
        port = server.bound_port()
        for idx, content in enumerate(prompt_contents):
            msg = tpu.ChatMessage("user", content)
            prompt_ids = tok.encode_chat([msg])
            max_new = 36

            # Ground-truth serial autoregressive decoding on OpenXLA
            s_kv_k = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
            s_kv_v = jnp.zeros((NUM_LAYERS, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
            cur_logits = None
            for pos, tid in enumerate(prompt_ids):
                cur_logits, s_kv_k, s_kv_v = serial_step(params, s_kv_k, s_kv_v, tid, pos)
            cur_pos = len(prompt_ids)
            serial_tokens = []
            for _ in range(max_new):
                nxt = int(np.argmax(np.asarray(cur_logits)))
                serial_tokens.append(nxt)
                cur_logits, s_kv_k, s_kv_v = serial_step(params, s_kv_k, s_kv_v, nxt, cur_pos)
                cur_pos += 1
            serial_text = tok.decode(serial_tokens, skip_special=True)

            req_body = {
                "model": "tpuforge-jax-roPE-128",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_new,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": False,
            }
            status, _, resp_str = _http_request(port, "POST", "/v1/chat/completions", req_body)
            assert status == 200
            comp = json.loads(resp_str)
            assert comp["choices"][0]["message"]["content"] == serial_text

            # Check SSE streaming chunks never split multi-byte UTF-8 and concatenate identically
            sse_req = dict(req_body)
            sse_req["stream"] = True
            _, _, chunks, done_seen = _http_sse_chat(port, sse_req)
            assert done_seen is True
            streamed_text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
            assert streamed_text == serial_text

            telemetry = comp["tpuforge_telemetry"]
            if idx == 0:
                assert telemetry["mean_tokens_per_decode_pass"] > 1.0

    # Zero-recompilation XLA invariant: one compilation covers both prefill and decode across prompts
    assert systolic_compile_count[0] == 1


def test_openai_server_concurrent_multithreaded_requests():
    engine, _, _, systolic_compile_count = _make_jax_forge_engine()
    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 4

    motif = "abc\U0001f680xyz " * 3
    payload_nonstream = {
        "messages": [{"role": "user", "content": motif + "abc"}],
        "max_tokens": 24,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    payload_stream = dict(payload_nonstream)
    payload_stream["stream"] = True

    with tpu.OpenAIServer(engine, srv_cfg) as server:
        port = server.bound_port()
        # Baseline single result
        status, _, base_raw = _http_request(port, "POST", "/v1/chat/completions", payload_nonstream)
        assert status == 200
        expected_text = json.loads(base_raw)["choices"][0]["message"]["content"]

        def run_worker(worker_id: int):
            if worker_id % 2 == 0:
                st, _, body = _http_request(port, "POST", "/v1/chat/completions", payload_nonstream)
                assert st == 200
                return json.loads(body)["choices"][0]["message"]["content"]
            st, _, chunks, done = _http_sse_chat(port, payload_stream)
            assert st == 200 and done is True
            return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            outputs = list(pool.map(run_worker, range(6)))

        assert all(out == expected_text for out in outputs)
        assert systolic_compile_count[0] == 1


def test_openai_server_adversarial_http_and_json_errors():
    plugin_path = str(find_pjrt_test_plugin())
    driver = tpu.PjrtDriver()
    assert driver.load_plugin(plugin_path)[0] is True

    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 256
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 264

    program_code = "module @pjrt_adv {} // tensor<128x264xf32> outputs=3 kv_elems=128"
    engine = tpu.ForgeEngine(driver, cfg, program_code, "mlir", "tpuforge-adversarial-128")
    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 2

    server = tpu.OpenAIServer(engine, srv_cfg)
    ok, err = server.start()
    assert ok is True, err
    port = server.bound_port()

    def _raw_http_send(raw_bytes: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as s:
            s.sendall(raw_bytes)
            chunks = []
            while True:
                part = s.recv(4096)
                if not part:
                    break
                chunks.append(part)
            return b"".join(chunks)

    try:
        # 404 Unknown endpoint
        st, _, body = _http_request(port, "GET", "/v1/nonexistent")
        assert st == 404 and "error" in json.loads(body)

        # 405 Method not allowed
        st, headers, _ = _http_request(port, "GET", "/v1/chat/completions")
        assert st == 405 and headers.get("Allow") == "POST"
        st, headers, _ = _http_request(port, "POST", "/health", "{}")
        assert st == 405 and headers.get("Allow") == "GET"

        # 501 Transfer-Encoding: chunked via raw socket so http.client won't rewrite headers
        raw_resp = _raw_http_send(
            b"POST /v1/chat/completions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Connection: close\r\n\r\n"
            b"0\r\n\r\n"
        )
        assert raw_resp.startswith(b"HTTP/1.1 501 ")

        # 400 Whitespace before header colon and obs-fold continuation (RFC 7230)
        for bad_hdr in (
            b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Bad : 1\r\nConnection: close\r\n\r\n",
            b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Fold: a\r\n b\r\nConnection: close\r\n\r\n",
        ):
            assert _raw_http_send(bad_hdr).startswith(b"HTTP/1.1 400 ")

        # 400 Malformed JSON and duplicate keys
        st, _, _ = _http_request(port, "POST", "/v1/chat/completions", "{bad json")
        assert st == 400
        dup_key = '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"max_tokens":8}'
        st, _, _ = _http_request(port, "POST", "/v1/chat/completions", dup_key)
        assert st == 400

        # 400 Excessive JSON nesting (> 32)
        deep = ("{\"a\":" * 33) + "1" + ("}" * 33)
        st, _, _ = _http_request(port, "POST", "/v1/chat/completions", deep)
        assert st == 400

        # M1: Valid literal U+FFFD passes; invalid overlong / out-of-range raw UTF-8 fails with 400
        st, _, _ = _http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "ok \ufffd ok"}], "max_tokens": 4},
        )
        assert st == 200

        for bad_utf8 in (b"\xe0\x80", b"\xf4\x90a"):
            payload = (
                b'{"messages":[{"role":"user","content":"'
                + bad_utf8
                + b'"}],"max_tokens":4}'
            )
            req_bytes = (
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + payload
            )
            assert _raw_http_send(req_bytes).startswith(b"HTTP/1.1 400 ")

        # 400 Unpaired UTF-16 surrogate escape
        bad_surrogate = '{"messages":[{"role":"user","content":"\\uD800"}]}'
        st, _, _ = _http_request(port, "POST", "/v1/chat/completions", bad_surrogate)
        assert st == 400

        # 400 Wrong field types
        st, _, _ = _http_request(
            port, "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "stream": "true"},
        )
        assert st == 400
        st, _, _ = _http_request(
            port, "POST", "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1},
        )
        assert st == 400

        # 400 Empty messages
        st, _, _ = _http_request(port, "POST", "/v1/chat/completions", {"messages": []})
        assert st == 400

        # 400 Oversized prompt (> max_seq_len - 128 = 128) BEFORE streaming headers!
        for stream_flag in (False, True):
            st, _, err_body = _http_request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "A" * 200}],
                    "stream": stream_flag,
                },
            )
            assert st == 400
            assert "capacity" in json.loads(err_body)["error"]["message"].lower()
    finally:
        server.stop()
        assert server.is_running() is False
        assert server.bound_port() == 0


def test_openai_server_wave5_slowloris_rng_gil_and_cli_regressions():
    import queue

    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 256
    cfg.hidden_dim = 128
    cfg.num_kv_heads = 4
    cfg.head_dim = 32
    cfg.vocab_size = 264

    callback_entered = threading.Event()
    release_sse_tail = threading.Event()
    sse_mode = [False]
    sse_step_counter = [0]
    sse_unblocked_by_client = [False]

    def custom_step_fn(valid_prefix_len, token_tile, pos_tile, mask_tile):
        active_tokens = {int(x) for x in token_tile}
        # M4 trigger: exclamation mark byte (4 + 33 = 37) deterministically fires on prefill tile
        if 37 in active_tokens:
            raise RuntimeError("boom from python callback")
        # H1 trigger: '~' (4 + 126 = 130) sleeps briefly while other threads query server state
        if 130 in active_tokens:
            callback_entered.set()
            time.sleep(0.08)
        # W6-M3 trigger: '^' (4 + 94 = 98) enables pause on 2nd decode step until client reads 1st SSE delta
        if valid_prefix_len == 0:
            sse_mode[0] = (98 in active_tokens)
            sse_step_counter[0] = 0
        elif sse_mode[0]:
            sse_step_counter[0] += 1
            if sse_step_counter[0] == 2:
                sse_unblocked_by_client[0] = release_sse_tail.wait(timeout=2.5)
        logits = np.full((128, 264), -20.0, dtype=np.float32)
        logits[:, 69:85] = 0.0
        return logits

    engine = tpu.ForgeEngine(cfg, custom_step_fn, None, "tpuforge-custom-wave5")
    assert engine.compile_count() == 0

    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 2
    srv_cfg.socket_timeout_ms = 250

    with tpu.OpenAIServer(engine, srv_cfg) as server:
        port = server.bound_port()

        # L9: Custom callback engine reports compile_count == 0 on /health
        st, _, health_raw = _http_request(port, "GET", "/health")
        assert st == 200 and json.loads(health_raw)["compile_count"] == 0

        # M4: Python RuntimeError inside target_step_fn maps to clean HTTP 500 JSON, and server recovers
        st, _, err_raw = _http_request(
            port,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "trigger!"}], "max_tokens": 4},
        )
        assert st == 500
        assert "boom from python callback" in json.loads(err_raw)["error"]["message"]

        # W6-M3: Verify SSE streams first delta BEFORE decode step 2 is unblocked
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        try:
            sse_payload = json.dumps(
                {
                    "messages": [{"role": "user", "content": "^"}],
                    "max_tokens": 6,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "stream": True,
                }
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=sse_payload,
                headers={"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            assert resp.status == 200
            seen_first_delta = False
            while True:
                raw_line = resp.fp.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    chunk_obj = json.loads(data_str)
                    if chunk_obj["choices"][0]["delta"].get("content"):
                        seen_first_delta = True
                        release_sse_tail.set()
            assert seen_first_delta is True
            assert sse_step_counter[0] >= 2
            assert sse_unblocked_by_client[0] is True
        finally:
            release_sse_tail.set()
            conn.close()

        # M2: Seedless stochastic generation varies; explicit seeds (including 0) stay deterministic
        seedless_outputs = set()
        for _ in range(6):
            st, _, body = _http_request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "temperature": 1.0,
                    "ignore_eos": True,
                },
            )
            assert st == 200
            seedless_outputs.add(json.loads(body)["choices"][0]["message"]["content"])
        assert len(seedless_outputs) >= 2

        seeded_pair = []
        for _ in range(2):
            st, _, body = _http_request(
                port,
                "POST",
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "temperature": 1.0,
                    "seed": 0,
                    "ignore_eos": True,
                },
            )
            assert st == 200
            seeded_pair.append(json.loads(body)["choices"][0]["message"]["content"])
        assert seeded_pair[0] == seeded_pair[1]

        # H3: Slowloris header drip exceeding 250ms budget triggers HTTP 408 promptly
        t_start = time.monotonic()
        with socket.create_connection(("127.0.0.1", port), timeout=2.5) as slow_sock:
            slow_sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n")
            time.sleep(0.14)
            try:
                slow_sock.sendall(b"X-Slow: 1\r\n")
            except OSError:
                pass
            time.sleep(0.22)
            resp_chunks = []
            while True:
                part = slow_sock.recv(4096)
                if not part:
                    break
                resp_chunks.append(part)
            slow_resp = b"".join(resp_chunks)
        assert time.monotonic() - t_start < 1.8
        assert slow_resp.startswith(b"HTTP/1.1 408 ")

        # W6-H1: Immediate client disconnect mid-SSE never wedges engine or server workers
        with socket.create_connection(("127.0.0.1", port), timeout=2.5) as drop_sock:
            sse_body = json.dumps(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 256,
                    "ignore_eos": True,
                    "stream": True,
                }
            ).encode("utf-8")
            req_bytes = (
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"Content-Type: application/json\r\n"
                + f"Content-Length: {len(sse_body)}\r\n".encode("ascii")
                + b"Connection: close\r\n\r\n"
                + sse_body
            )
            drop_sock.sendall(req_bytes)
            _ = drop_sock.recv(128)
        st, _, post_drop_health = _http_request(port, "GET", "/health")
        assert st == 200 and json.loads(post_drop_health)["status"] == "ok"

        # H1: Concurrent is_running()/bound_port()/stop() alongside active Python callback never deadlocks
        def trigger_slow_cb():
            return _http_request(
                port,
                "POST",
                "/v1/chat/completions",
                {"messages": [{"role": "user", "content": "~"}], "max_tokens": 2},
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            req_fut = pool.submit(trigger_slow_cb)
            assert callback_entered.wait(timeout=3.0)
            poll_fut = pool.submit(lambda: (server.is_running(), server.bound_port()))
            running_flag, port_val = poll_fut.result(timeout=3.0)
            assert isinstance(running_flag, bool) and isinstance(port_val, int)
            server.stop()
            try:
                st_cb, _, _ = req_fut.result(timeout=3.0)
                assert st_cb in (200, 500, 503)
            except (http.client.HTTPException, OSError) as exc:
                assert exc is not None
        assert server.is_running() is False

        # W6-M1: Concurrent start()/stop() calls serialize cleanly without hanging
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for _ in range(4):
                f_start = pool.submit(server.start)
                f_stop = pool.submit(server.stop)
                f_start.result(timeout=3.0)
                f_stop.result(timeout=3.0)
        server.stop()
        assert server.is_running() is False

    # M5 CLI: --plugin without --program fails; --serve with --program succeeds end-to-end
    root = Path(__file__).resolve().parent.parent
    exe_path = root / "build" / ("tpuforge_bench.exe" if sys.platform == "win32" else "tpuforge_bench")
    plugin_path = find_pjrt_test_plugin()
    stateful_mlir = root / "tests" / "fixtures" / "tpuforge_stateful_128x32000.mlir"

    missing_prog = subprocess.run(
        [str(exe_path), "--plugin", str(plugin_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_prog.returncode != 0

    proc = subprocess.Popen(
        [
            str(exe_path),
            "--serve",
            "--plugin",
            str(plugin_path),
            "--program",
            str(stateful_mlir),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
        ],
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    line_q: queue.Queue[str | None] = queue.Queue()

    def _drain_stdout():
        assert proc.stdout is not None
        for ln in proc.stdout:
            line_q.put(ln)
        line_q.put(None)

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()
    try:
        cli_port = 0
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            try:
                line = line_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                break
            marker = "Listening on http://127.0.0.1:"
            if marker in line:
                cli_port = int(line.split(marker, 1)[1].strip())
                break
        assert cli_port > 0, "CLI --serve did not bind listening port"
        st, _, h_body = _http_request(cli_port, "GET", "/health")
        assert st == 200 and json.loads(h_body)["compile_count"] == 1
        st, _, c_body = _http_request(
            cli_port,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "cli serve check"}], "max_tokens": 6},
        )
        assert st == 200 and json.loads(c_body)["tpuforge_telemetry"]["compile_count"] == 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
        reader.join(timeout=2.0)

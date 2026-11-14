"""Tests for out-of-band AST and idiom SuffixIndex seeding for cold prompts."""

from __future__ import annotations

import http.client
import json
from pathlib import Path
from typing import Any

import _tpuforge_ext as tpu
import jax.numpy as jnp
import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from benchmarks.ast_seed_jax import (
    DEFAULT_CODE_IDIOMS,
    encode_with_tokenizer,
    extract_cpp_header_snippets,
    extract_python_ast_snippets,
    extract_repo_ast_snippets,
)
from benchmarks.qwen_jax import QwenBenchmarkRunner, QwenWeights
from tests.test_qwen_benchmark import _make_cyclic_qwen_model


def _http_post_json(port: int, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10.0)
    try:
        conn.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)
    finally:
        conn.close()


def _build_word_level_tokenizer(vocab_size: int) -> Tokenizer:
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = 0
    vocab["<|im_start|>"] = 1
    vocab["<|im_end|>"] = 2
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return tok


def test_suffix_index_out_of_band_seeding_zero_boundary_bleed_and_live_recency() -> None:
    """Out-of-band seeds accelerate cold prompts with zero cross-snippet bleed and live recency priority."""
    idx = tpu.SuffixIndex([100])
    idx.seed_sequences([[100, 101, 102, 103], [200, 201, 202]])

    assert idx.size() == 1
    assert idx.seeded_tokens() == 7

    # Adjacent seed sequences and prompt tokens must never merge into synthetic cross-boundary n-grams.
    assert idx.query([101, 102, 103], top_k=4) == []
    assert idx.query([201, 202], top_k=4) == []

    tree = idx.build_tree(max_nodes=32, max_depth=8, tile_size=128)
    nodes = tree.nodes()
    assert nodes[0].token_id == 100
    assert max(n.depth for n in nodes) >= 3

    chain: list[int] = []
    curr = 0
    while nodes[curr].children:
        curr = nodes[curr].children[0]
        chain.append(nodes[curr].token_id)
    assert chain[:3] == [101, 102, 103]

    # Live context transitions (pos >= 1) strictly outrank 1x out-of-band seeds (pos == 0).
    idx2 = tpu.SuffixIndex([50])
    idx2.seed_sequence([100, 101, 102])
    idx2.append_tokens(np.array([100, 150, 99, 100], dtype=np.int32))
    candidates = idx2.query([], top_k=2)
    assert len(candidates) == 2
    assert candidates[0].token_id == 150


def test_forge_engine_and_openai_server_cold_codegen_speedup_with_zero_kv_cost() -> None:
    """AST/idiom seeds cut cold-codegen decode passes >= 3x with bit-exact parity and zero prompt KV inflation."""
    cfg = tpu.StaticArenaConfig()
    cfg.max_seq_len = 512
    cfg.hidden_dim = 64
    cfg.num_kv_heads = 2
    cfg.head_dim = 16
    cfg.vocab_size = 320
    cfg.tile_size = 128

    tok = tpu.ByteTokenizer(cfg.vocab_size)
    target_code = "def compute_systolic_shard(x: np.ndarray) -> int:"
    prompt_msgs = [tpu.ChatMessage("user", "hi")]
    prompt_ids = list(tok.encode_chat(prompt_msgs))
    target_ids = list(tok.encode(target_code, False, False))
    expected_full = prompt_ids + target_ids + [2]

    committed: list[int] = []
    last_tile: list[np.ndarray | None] = [None]

    def target_step_fn(valid_prefix_len: int, token_tile, pos_tile, mask_tile) -> np.ndarray:
        del pos_tile
        del committed[valid_prefix_len:]
        tok_arr = np.asarray(token_tile, dtype=np.int32)
        mask_arr = np.asarray(mask_tile, dtype=np.float32)
        last_tile[0] = tok_arr.copy()

        logits = np.zeros((cfg.tile_size, cfg.vocab_size), dtype=np.float32)
        for r in range(cfg.tile_size):
            ancestors = [int(tok_arr[c]) for c in range(r + 1) if mask_arr[r, c] > -1.0]
            full_prefix = committed + ancestors
            if len(full_prefix) < len(expected_full) and full_prefix == expected_full[: len(full_prefix)]:
                logits[r, expected_full[len(full_prefix)]] = 20.0
            else:
                logits[r, 2] = 20.0
        return logits

    def commit_kv_fn(prev_valid_prefix_len: int, accepted_indices) -> None:
        del committed[prev_valid_prefix_len:]
        tile = last_tile[0]
        assert tile is not None
        for idx in accepted_indices:
            committed.append(int(tile[int(idx)]))

    engine = tpu.ForgeEngine(cfg, target_step_fn, commit_kv_fn, "tpuforge-ast-oracle")
    req = tpu.GenerationRequest()
    req.messages = prompt_msgs
    req.max_tokens = 64
    req.temperature = 0.0

    unseeded = engine.generate(req)
    assert unseeded.text == target_code
    assert unseeded.prompt_tokens_count == len(prompt_ids)

    engine.seed_ast_text([target_code])
    assert engine.ast_seed_tokens() == len(target_ids)
    seeded = engine.generate(req)

    assert seeded.text == target_code
    assert seeded.token_ids == unseeded.token_ids
    assert seeded.prompt_tokens_count == unseeded.prompt_tokens_count == len(prompt_ids)
    assert seeded.decode_forward_passes < unseeded.decode_forward_passes
    assert seeded.mean_tokens_per_decode_pass >= 3.0 * unseeded.mean_tokens_per_decode_pass

    engine.clear_ast_seeds()
    assert engine.ast_seed_tokens() == 0
    with pytest.raises((IndexError, ValueError), match="out of"):
        engine.seed_ast_sequences([[999]])

    srv_cfg = tpu.OpenAIServerConfig()
    srv_cfg.host = "127.0.0.1"
    srv_cfg.port = 0
    srv_cfg.worker_threads = 2
    base_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    with tpu.OpenAIServer(engine, srv_cfg) as server:
        port = server.bound_port()
        st_plain, plain_json = _http_post_json(port, "/v1/chat/completions", base_payload)
        assert st_plain == 200
        assert plain_json["choices"][0]["message"]["content"] == target_code

        seeded_payload = dict(base_payload)
        seeded_payload["seed_snippets"] = [target_code]
        st_seeded, seeded_json = _http_post_json(port, "/v1/chat/completions", seeded_payload)
        assert st_seeded == 200
        assert seeded_json["choices"][0]["message"]["content"] == target_code
        assert seeded_json["usage"]["prompt_tokens"] == plain_json["usage"]["prompt_tokens"]
        assert (
            seeded_json["tpuforge_telemetry"]["mean_tokens_per_decode_pass"]
            >= 3.0 * plain_json["tpuforge_telemetry"]["mean_tokens_per_decode_pass"]
        )

        # Malformed or oversized HTTP "seed_snippets" must return HTTP 400.
        for bad_val in ("not-an-array", [123], ["x"] * 257):
            bad_payload = dict(base_payload)
            bad_payload["seed_snippets"] = bad_val
            st_bad, _ = _http_post_json(port, "/v1/chat/completions", bad_payload)
            assert st_bad == 400


def test_python_and_cpp_ast_extractor_and_qwen_cold_prompt_tree_parity(tmp_path: Path) -> None:
    """AST/header seed extraction speeds up short cold Qwen prompts with zero recompiles and exact parity."""
    py_file = tmp_path / "sample.py"
    py_src = (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class ShardSpec:\n"
        "    tile_size: int = 128\n\n"
        "def compute_cycle(x: int) -> int:\n"
        '    cycle_seed = " tok10 tok11 tok12 tok13 tok10 tok11 tok12 tok13 "\n'
        "    return x\n"
    )
    py_file.write_text(py_src, encoding="utf-8")

    h_file = tmp_path / "sample.h"
    h_src = (
        "#include <vector>\n"
        "struct TileConfig {\n"
        "    int size;\n"
        "};\n"
        "int32_t run_tile(int32_t tok);\n"
    )
    h_file.write_text(h_src, encoding="utf-8")

    py_only = extract_python_ast_snippets(py_src)
    assert "def compute_cycle(x: int) -> int:" in py_only
    cpp_only = extract_cpp_header_snippets(h_src)
    assert "#include <vector>" in cpp_only
    assert "struct TileConfig" in cpp_only

    snippets = extract_repo_ast_snippets([tmp_path], include_default_idioms=True)
    assert len(snippets) == len(set(snippets))
    assert any("def compute_cycle(x: int) -> int:" in s for s in snippets)
    assert any("tok10 tok11 tok12 tok13" in s for s in snippets)
    assert "#include <vector>" in snippets
    assert "struct TileConfig" in snippets
    for idiom in DEFAULT_CODE_IDIOMS:
        assert idiom.strip() in snippets

    byte_seqs = encode_with_tokenizer(snippets, tpu.ByteTokenizer(320))
    assert len(byte_seqs) == len(snippets)

    word_tok = _build_word_level_tokenizer(512)
    seed_seqs = encode_with_tokenizer(snippets, word_tok)
    assert any(seq for seq in seed_seqs if 10 in seq and 11 in seq and 12 in seq and 13 in seq)

    cfg, raw = _make_cyclic_qwen_model(vocab_size=512)
    weights = QwenWeights.from_numpy_dict(raw, cfg, dtype=jnp.float32)
    runner = QwenBenchmarkRunner(config=cfg, weights=weights, tokenizer=None, max_seq_len=512)

    # Warm JIT graphs once so subsequent cold-prompt measurements report zero recompilations.
    runner.run_vanilla_jax([10, 11, 12, 13], max_new_tokens=2)
    runner.run_tpuforge_tree([10, 11, 12, 13], max_new_tokens=2)

    cold_prompt = [5, 6, 7, 10]
    m_van = runner.run_vanilla_jax(cold_prompt, max_new_tokens=16)
    m_unseeded = runner.run_tpuforge_tree(cold_prompt, max_new_tokens=16)
    m_seeded = runner.run_tpuforge_tree(
        cold_prompt,
        max_new_tokens=16,
        seed_token_sequences=seed_seqs,
    )

    expected = [11, 12, 13, 10] * 4
    assert m_van.token_ids == expected
    assert m_unseeded.token_ids == expected
    assert m_seeded.token_ids == expected
    assert m_seeded.compile_count == 0
    assert m_seeded.decode_passes < m_unseeded.decode_passes
    assert m_seeded.mean_tokens_per_decode_pass > m_unseeded.mean_tokens_per_decode_pass
    assert m_seeded.extra["ast_seed_tokens"] > 0


def test_block_top_k_hybrid_fallback_with_one_token_cold_prompt_and_ast_seeds() -> None:
    """TreeBuilder extends block-diffusion spines using out-of-band AST seeds even on 1-token prompts."""
    draft_ids = np.array(
        [
            [101, 201],
            [102, 202],
            [103, 203],
            [104, 204],
        ],
        dtype=np.int32,
    )
    draft_probs = np.array([[0.8, 0.2]] * 4, dtype=np.float32)

    idx = tpu.SuffixIndex([100])
    idx.seed_sequence([100, 101, 102, 103, 104, 105, 106, 107])
    assert idx.size() == 1
    assert idx.seeded_tokens() == 8

    tree_no_fb = tpu.TreeBuilder.build_from_block_top_k(100, draft_ids, draft_probs, None, 64, 12, 128)
    tree_fb = tpu.TreeBuilder.build_from_block_top_k(100, draft_ids, draft_probs, idx, 64, 12, 128)

    assert max(n.depth for n in tree_no_fb.nodes()) == 4
    assert max(n.depth for n in tree_fb.nodes()) >= 7
    fb_tokens = {n.token_id for n in tree_fb.nodes()}
    assert {105, 106, 107}.issubset(fb_tokens)

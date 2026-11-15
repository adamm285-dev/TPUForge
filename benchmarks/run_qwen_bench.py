"""3-Workload OpenXLA benchmark comparing Vanilla JAX, StaticArena (M-002), and TPUForge (M-002 + M-003)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from benchmarks.ast_seed_jax import encode_with_tokenizer, extract_repo_ast_snippets
from benchmarks.block_diffusion_jax import (
    BlockDiffusionConfig,
    BlockDiffusionWeights,
    QwenBlockTreeRunner,
    load_qwen_block_tree_runner,
)
from benchmarks.gemma4_jax import (
    Gemma4BlockTreeRunner,
    Gemma4MoeStreamRunner,
    Gemma4Runner,
    encode_gemma4_chat_messages,
    is_gemma4_model_dir,
    load_gemma4_block_tree_runner,
    load_gemma4_moe_stream_runner,
    load_gemma4_runner,
)
from benchmarks.qwen_jax import (
    QwenBenchmarkRunner,
    RunMetrics,
    encode_chat_messages,
    load_qwen_runner,
    resolve_model_dir,
)


def _encode_bench_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = None,
    is_gemma4: bool = False,
) -> list[int]:
    if is_gemma4:
        return encode_gemma4_chat_messages(
            tokenizer, messages, enable_thinking=bool(enable_thinking)
        )
    return encode_chat_messages(tokenizer, messages, enable_thinking=enable_thinking)


def _make_sweep_prompts(
    tokenizer: Any,
    enable_thinking: bool | None = None,
    is_gemma4: bool = False,
) -> list[tuple[str, list[int]]]:
    """Build 4 unaligned prompt lengths to expose dynamic-shape XLA recompilations."""
    raw_texts = [
        "Return only a one-line Python lambda that squares x.",
        (
            "Write a clean Python helper named clamp_aligned(value: int, low: int, high: int) -> int "
            "that clamps an integer inclusively and raises ValueError if low > high. Include a concise docstring."
        ),
        (
            "You are optimizing a fixed-shape ring buffer for LLM KV cache pages. "
            "Given page_size=16, num_kv_heads=4, and head_dim=128, write a Python class PageTable "
            "that maps logical sequence blocks to physical frame indices, tracks free frames in a deque, "
            "and raises IndexError when out of frames. Keep methods typed and concise."
        ),
        (
            "Below is an unaligned configuration dictionary for an inference runner:\n"
            "cfg = {'tile_rows': 128, 'max_seq_len': 2048, 'vocab_size': 151936, 'num_layers': 24, "
            "'num_heads': 14, 'num_kv_heads': 2, 'head_dim': 64, 'rope_theta': 1000000.0, "
            "'rms_norm_eps': 1e-6, 'speculative_max_depth': 16, 'speculative_max_nodes': 128}\n"
            "Write a Python dataclass EngineSpec with a validate() method that checks tile_rows == 128, "
            "num_heads % num_kv_heads == 0, max_seq_len >= 256, and positive eps and rope_theta."
        ),
    ]
    prompts: list[tuple[str, list[int]]] = []
    seen_lens: set[int] = set()
    for idx, text in enumerate(raw_texts):
        toks = _encode_bench_messages(
            tokenizer,
            [
                {"role": "system", "content": "You are a concise expert systems programmer."},
                {"role": "user", "content": text},
            ],
            enable_thinking=enable_thinking,
            is_gemma4=is_gemma4,
        )
        while len(toks) in seen_lens:
            toks.append(toks[-1])
        seen_lens.add(len(toks))
        prompts.append((f"sweep_{idx + 1}_len_{len(toks)}", toks))
    return prompts


def _make_refactor_prompt(
    tokenizer: Any,
    enable_thinking: bool | None = None,
    is_gemma4: bool = False,
) -> list[int]:
    """Code refactor prompt with high context overlap for SuffixTreeDrafter evaluation."""
    source_code = (
        "class StaticArena:\n"
        "    def __init__(self, max_seq_len, num_kv_heads, head_dim, vocab_size):\n"
        "        self.max_seq_len = max_seq_len\n"
        "        self.num_kv_heads = num_kv_heads\n"
        "        self.head_dim = head_dim\n"
        "        self.vocab_size = vocab_size\n"
        "        self.valid_prefix_len = 0\n\n"
        "    def kv_row_elems(self):\n"
        "        return self.num_kv_heads * self.head_dim\n\n"
        "    def set_valid_prefix_len(self, length):\n"
        "        if length < 0 or length > self.max_seq_len:\n"
        "            raise ValueError('valid_prefix_len out of range')\n"
        "        self.valid_prefix_len = length\n\n"
        "    def remaining_capacity(self):\n"
        "        return self.max_seq_len - self.valid_prefix_len\n"
    )
    user_msg = (
        "Refactor the Python class below by adding full Python type annotations "
        "(int, None) to every method signature and parameter while preserving the exact "
        "class name, method names, attribute names, and error messages. Output only code:\n\n"
        f"```python\n{source_code}```"
    )
    return _encode_bench_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are a concise Python refactoring assistant."},
            {"role": "user", "content": user_msg},
        ],
        enable_thinking=enable_thinking,
        is_gemma4=is_gemma4,
    )


def _make_cold_codegen_prompt(
    tokenizer: Any,
    enable_thinking: bool | None = None,
    is_gemma4: bool = False,
) -> list[int]:
    """Fresh code generation prompt with minimal prompt n-gram overlap."""
    return _encode_bench_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are a concise Python programmer."},
            {
                "role": "user",
                "content": (
                    "Write a Python function binary_search(arr: list[int], target: int) -> int "
                    "that returns the index of target or -1 if missing, with a short docstring."
                ),
            },
        ],
        enable_thinking=enable_thinking,
        is_gemma4=is_gemma4,
    )


def _make_cold_chat_prompt(
    tokenizer: Any,
    enable_thinking: bool | None = None,
    is_gemma4: bool = False,
) -> list[int]:
    """Natural-language cold chat prompt for block-diffusion systolic tree evaluation."""
    return _encode_bench_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are a concise computer architecture instructor."},
            {
                "role": "user",
                "content": (
                    "Explain in 4 clear bullet points why fixed-shape matrix tiles prevent "
                    "XLA graph recompilation stalls on systolic array accelerators."
                ),
            },
        ],
        enable_thinking=enable_thinking,
        is_gemma4=is_gemma4,
    )


def _metrics_dict(m: RunMetrics) -> dict[str, Any]:
    d = asdict(m)
    d["text_preview"] = m.text[:240]
    return d


def _count_prefix_matches(a: list[int], b: list[int]) -> tuple[int, int]:
    min_len = min(len(a), len(b))
    matches = sum(1 for x, y in zip(a[:min_len], b[:min_len]) if x == y)
    return matches, min_len


def _print_mode_row(m: RunMetrics) -> None:
    print(
        f"  {m.mode:<16} | gen={m.generated_len:3d} | fwd_passes={m.forward_passes:3d} | "
        f"dec_passes={m.decode_passes:3d} | tok/pass={m.mean_tokens_per_decode_pass:4.2f} | "
        f"dec_s={m.decode_s:6.2f}s | tok/s={m.decode_tok_per_s:6.2f} | compiles={m.compile_count}"
    )


def run_benchmark_suite(
    model_id: str,
    max_seq_len: int = 1024,
    max_new_tokens: int = 64,
    dtype_name: str = "auto",
    output_json: Path | None = None,
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    preloaded: tuple[QwenBenchmarkRunner | Gemma4Runner, Path] | None = None,
    draft_model_id: str | None = None,
    top_k: int = 8,
    weight_quant: str = "none",
    ast_seed_path: str | Path | None = None,
    host_store: str = "hbm",
    stage_experts: int = 16,
) -> dict[str, Any]:
    normalized_draft = draft_model_id.strip() if draft_model_id else ""
    norm_host_store = host_store.strip().lower()
    if norm_host_store not in ("hbm", "mmap", "dram"):
        raise ValueError(
            f"Unsupported host_store={host_store!r}; expected 'hbm', 'mmap', or 'dram'"
        )
    block_runner: QwenBlockTreeRunner | Gemma4BlockTreeRunner | None = None
    draft_dir: Path | None = None
    block_warmup_s = 0.0
    owned_stream_runner: Gemma4MoeStreamRunner | None = None

    t_load0 = time.perf_counter()
    if preloaded is not None:
        prev_runner, model_dir = preloaded
        is_gemma4 = isinstance(prev_runner, (Gemma4Runner, Gemma4MoeStreamRunner))
        if isinstance(prev_runner, Gemma4Runner):
            base_runner: Any = Gemma4Runner(
                config=prev_runner.config,
                weights=prev_runner.weights,
                tokenizer=prev_runner.tokenizer,
                max_seq_len=max_seq_len,
                tile_size=tile_size,
                mesh=prev_runner.mesh,
            )
            if normalized_draft:
                draft_dir = resolve_model_dir(normalized_draft)
                draft_cfg = BlockDiffusionConfig.from_dir(draft_dir, target_config=base_runner.config)
                draft_weights = BlockDiffusionWeights.from_model_dir(
                    draft_dir,
                    draft_cfg,
                    dtype=base_runner.dtype,
                    mesh=base_runner.mesh,
                    weight_quant=base_runner.weight_quant,
                )
                block_runner = Gemma4BlockTreeRunner(
                    target_runner=base_runner,
                    draft_config=draft_cfg,
                    draft_weights=draft_weights,
                    top_k=top_k,
                )
        else:
            assert isinstance(prev_runner, QwenBenchmarkRunner)
            base_runner = QwenBenchmarkRunner(
                config=prev_runner.config,
                weights=prev_runner.weights,
                tokenizer=prev_runner.tokenizer,
                max_seq_len=max_seq_len,
                tile_size=tile_size,
                mesh=prev_runner.mesh,
            )
            if normalized_draft:
                draft_dir = resolve_model_dir(normalized_draft)
                draft_cfg = BlockDiffusionConfig.from_dir(draft_dir, target_config=base_runner.config)
                draft_weights = BlockDiffusionWeights.from_model_dir(
                    draft_dir, draft_cfg, dtype=base_runner.dtype, mesh=base_runner.mesh
                )
                block_runner = QwenBlockTreeRunner(
                    target_runner=base_runner,
                    draft_config=draft_cfg,
                    draft_weights=draft_weights,
                    top_k=top_k,
                )
    else:
        resolved_dir = resolve_model_dir(model_id)
        is_gemma4 = is_gemma4_model_dir(resolved_dir)
        if is_gemma4:
            if norm_host_store != "hbm" and normalized_draft:
                raise ValueError(
                    "--draft-model is only supported with --host-store hbm; "
                    "omit --draft-model when streaming MoE experts"
                )
            if norm_host_store != "hbm":
                base_runner, model_dir = load_gemma4_moe_stream_runner(
                    model_id_or_path=str(resolved_dir),
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    tile_size=tile_size,
                    tp_size=tp_size,
                    weight_quant=weight_quant,
                    host_store_mode=norm_host_store,
                    stage_experts=stage_experts,
                )
                owned_stream_runner = base_runner
            elif normalized_draft:
                block_runner, model_dir, draft_dir = load_gemma4_block_tree_runner(
                    model_id_or_path=str(resolved_dir),
                    draft_model_id_or_path=normalized_draft,
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    tile_size=tile_size,
                    tp_size=tp_size,
                    top_k=top_k,
                    weight_quant=weight_quant,
                )
                base_runner = block_runner.target
            else:
                base_runner, model_dir = load_gemma4_runner(
                    model_id_or_path=str(resolved_dir),
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    tile_size=tile_size,
                    tp_size=tp_size,
                    weight_quant=weight_quant,
                )
        else:
            if norm_host_store != "hbm":
                raise ValueError(
                    "--host-store != 'hbm' in run_qwen_bench targets Gemma 4 MoE; "
                    "for Qwen3-MoE/OLMoE use python -m benchmarks.moe_streaming_jax"
                )
            if normalized_draft:
                block_runner, model_dir, draft_dir = load_qwen_block_tree_runner(
                    model_id_or_path=str(resolved_dir),
                    draft_model_id_or_path=normalized_draft,
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    tile_size=tile_size,
                    tp_size=tp_size,
                    top_k=top_k,
                    weight_quant=weight_quant,
                )
                base_runner = block_runner.target
            else:
                base_runner, model_dir = load_qwen_runner(
                    model_id_or_path=str(resolved_dir),
                    max_seq_len=max_seq_len,
                    dtype_name=dtype_name,
                    tile_size=tile_size,
                    tp_size=tp_size,
                    weight_quant=weight_quant,
                )
    load_s = time.perf_counter() - t_load0

    try:
        if block_runner is not None:
            block_warmup_s = block_runner.warmup()

        backend = jax.default_backend()
        devices = [str(d) for d in jax.devices()]
        cfg = base_runner.config
        tok_to_id = getattr(base_runner.tokenizer, "token_to_id", None)
        has_thinking_marker = tok_to_id is not None and (
            tok_to_id("</think>") is not None or tok_to_id("<|think|>") is not None
        )
        enable_thinking: bool | None = False if has_thinking_marker else None

        print("=" * 78)
        print(f"TPUForge Real-Weight Benchmark | Model: {model_id}")
        if normalized_draft and block_runner is not None:
            print(
                f"Draft Model: {normalized_draft} | block_size={block_runner.draft_config.block_size} | "
                f"top_k={top_k} | warmup={block_warmup_s:.2f}s"
            )
        print(
            f"Backend: {backend} ({len(devices)}x {devices[0]}) | TP={base_runner.tp_size} | "
            f"Tile={base_runner.tile_size}x{base_runner.tile_size} | Dtype: {base_runner.dtype} | "
            f"Quant: {base_runner.weight_quant} ({base_runner.model_hbm_bytes / (1024 * 1024):.1f} MiB) | "
            f"Layers: {cfg.num_hidden_layers} | Vocab: {cfg.vocab_size} | Load: {load_s:.2f}s"
        )
        if backend == "cpu":
            print(
                "Note: Running on OpenXLA CPU (no physical systolic MXU array). "
                "Wall-clock tok/s reflects CPU simulation; compile_count and "
                "tokens/forward_pass measure the exact M-002 and M-003 algorithmic levers."
            )
        print("=" * 78)

        sweep_runner = base_runner
        sweep_prompts = _make_sweep_prompts(
            base_runner.tokenizer, enable_thinking=enable_thinking, is_gemma4=is_gemma4
        )
        w1_vanilla: list[dict[str, Any]] = []
        w1_static: list[dict[str, Any]] = []
        for name, toks in sweep_prompts:
            mv = sweep_runner.run_vanilla_jax(toks, max_new_tokens=4)
            ms = sweep_runner.run_static_arena(toks, max_new_tokens=4)
            w1_vanilla.append({"name": name, **_metrics_dict(mv)})
            w1_static.append({"name": name, **_metrics_dict(ms)})

        w1_v_compiles = sum(r["compile_count"] for r in w1_vanilla)
        w1_s_compiles = sum(r["compile_count"] for r in w1_static)
        w1_v_total_s = sum(r["total_s"] for r in w1_vanilla)
        w1_s_total_s = sum(r["total_s"] for r in w1_static)

        print(
            f"\n[Workload 1: Unaligned Prompt-Shape Sweep (4 distinct lengths: "
            f"{[len(t) for _, t in sweep_prompts]})]"
        )
        print(
            f"  vanilla_jax  -> compiles: {w1_v_compiles} | total sweep time: {w1_v_total_s:.2f}s"
        )
        print(
            f"  static_arena -> compiles: {w1_s_compiles} | total sweep time: {w1_s_total_s:.2f}s"
        )

        refactor_prompt = _make_refactor_prompt(
            base_runner.tokenizer, enable_thinking=enable_thinking, is_gemma4=is_gemma4
        )
        w2_vanilla = sweep_runner.run_vanilla_jax(refactor_prompt, max_new_tokens=max_new_tokens)
        w2_static = sweep_runner.run_static_arena(refactor_prompt, max_new_tokens=max_new_tokens)
        w2_tree = sweep_runner.run_tpuforge_tree(refactor_prompt, max_new_tokens=max_new_tokens)
        w2_block = (
            block_runner.run_block_tree(refactor_prompt, max_new_tokens=max_new_tokens)
            if block_runner is not None
            else None
        )

        exact_static_tree_parity = w2_static.token_ids == w2_tree.token_ids
        vanilla_match_tokens, min_len = _count_prefix_matches(w2_vanilla.token_ids, w2_tree.token_ids)

        print(f"\n[Workload 2: Code Refactor (prompt_len={len(refactor_prompt)}, max_new={max_new_tokens})]")
        w2_runs = (w2_vanilla, w2_static, w2_tree) + ((w2_block,) if w2_block is not None else ())
        for m in w2_runs:
            _print_mode_row(m)
        print(
            f"  Parity: static_arena == tpuforge_tree: {exact_static_tree_parity} | "
            f"vanilla == tpuforge_tree: {vanilla_match_tokens}/{min_len} tokens"
        )
        if w2_block is not None:
            w2_bt_static_parity = w2_static.token_ids == w2_block.token_ids
            bt_v_match, bt_v_len = _count_prefix_matches(w2_vanilla.token_ids, w2_block.token_ids)
            print(
                f"  Block-Tree Parity: static_arena == tpuforge_block_tree: {w2_bt_static_parity} | "
                f"vanilla == tpuforge_block_tree: {bt_v_match}/{bt_v_len} tokens"
            )
        print(f"  Sample output ({w2_runs[-1].mode}):\n{w2_runs[-1].text[:320]}")

        ast_seed_seqs: list[list[int]] = []
        if ast_seed_path:
            ast_snippets = extract_repo_ast_snippets([ast_seed_path])
            ast_seed_seqs = encode_with_tokenizer(ast_snippets, base_runner.tokenizer)

        cold_prompt = _make_cold_codegen_prompt(
            base_runner.tokenizer, enable_thinking=enable_thinking, is_gemma4=is_gemma4
        )
        w3_vanilla = sweep_runner.run_vanilla_jax(cold_prompt, max_new_tokens=max_new_tokens)
        w3_tree = sweep_runner.run_tpuforge_tree(
            cold_prompt,
            max_new_tokens=max_new_tokens,
            seed_token_sequences=ast_seed_seqs or None,
        )
        w3_block = (
            block_runner.run_block_tree(
                cold_prompt,
                max_new_tokens=max_new_tokens,
                seed_token_sequences=ast_seed_seqs or None,
            )
            if block_runner is not None
            else None
        )
        w3_match, w3_min = _count_prefix_matches(w3_vanilla.token_ids, w3_tree.token_ids)

        print(f"\n[Workload 3: Fresh Code Generation Floor (prompt_len={len(cold_prompt)}, max_new={max_new_tokens})]")
        w3_runs = (w3_vanilla, w3_tree) + ((w3_block,) if w3_block is not None else ())
        for m in w3_runs:
            _print_mode_row(m)
        print(f"  Parity: vanilla == tpuforge_tree: {w3_match}/{w3_min} tokens")
        if w3_block is not None:
            w3_bt_tree_parity = w3_tree.token_ids == w3_block.token_ids
            w3_bt_match, w3_bt_len = _count_prefix_matches(w3_vanilla.token_ids, w3_block.token_ids)
            print(
                f"  Block-Tree Parity: tpuforge_tree == tpuforge_block_tree: {w3_bt_tree_parity} | "
                f"vanilla == tpuforge_block_tree: {w3_bt_match}/{w3_bt_len} tokens"
            )

        w2_report: dict[str, Any] = {
            "static_tree_exact_parity": exact_static_tree_parity,
            "vanilla_tree_matching_tokens": vanilla_match_tokens,
            "vanilla_tree_compared_tokens": min_len,
            "vanilla_jax": _metrics_dict(w2_vanilla),
            "static_arena": _metrics_dict(w2_static),
            "tpuforge_tree": _metrics_dict(w2_tree),
        }
        if w2_block is not None:
            bt_v_match, bt_v_len = _count_prefix_matches(w2_vanilla.token_ids, w2_block.token_ids)
            w2_report["static_block_tree_exact_parity"] = w2_static.token_ids == w2_block.token_ids
            w2_report["vanilla_block_tree_matching_tokens"] = bt_v_match
            w2_report["vanilla_block_tree_compared_tokens"] = bt_v_len
            w2_report["tpuforge_block_tree"] = _metrics_dict(w2_block)

        w3_report: dict[str, Any] = {
            "vanilla_tree_matching_tokens": w3_match,
            "vanilla_tree_compared_tokens": w3_min,
            "vanilla_jax": _metrics_dict(w3_vanilla),
            "tpuforge_tree": _metrics_dict(w3_tree),
        }
        if w3_block is not None:
            w3_bt_match, w3_bt_len = _count_prefix_matches(w3_vanilla.token_ids, w3_block.token_ids)
            w3_report["tree_block_tree_exact_parity"] = w3_tree.token_ids == w3_block.token_ids
            w3_report["vanilla_block_tree_matching_tokens"] = w3_bt_match
            w3_report["vanilla_block_tree_compared_tokens"] = w3_bt_len
            w3_report["tpuforge_block_tree"] = _metrics_dict(w3_block)

        report: dict[str, Any] = {
            "model_id": model_id,
            "arch": "gemma4" if is_gemma4 else "qwen",
            "model_dir": str(model_dir),
            "backend": backend,
            "device": devices[0],
            "num_devices": len(devices),
            "tp_size": base_runner.tp_size,
            "tile_size": base_runner.tile_size,
            "dtype": str(base_runner.dtype),
            "weight_quant": base_runner.weight_quant,
            "ast_seed_tokens": sum(len(seq) for seq in ast_seed_seqs),
            "model_hbm_bytes": base_runner.model_hbm_bytes,
            "max_seq_len": max_seq_len,
            "max_new_tokens": max_new_tokens,
            "load_seconds": round(load_s, 3),
            "workload_1_prompt_sweep": {
                "prompt_lengths": [len(t) for _, t in sweep_prompts],
                "vanilla_compiles": w1_v_compiles,
                "static_arena_compiles": w1_s_compiles,
                "vanilla_total_s": round(w1_v_total_s, 4),
                "static_arena_total_s": round(w1_s_total_s, 4),
                "vanilla_runs": w1_vanilla,
                "static_arena_runs": w1_static,
            },
            "workload_2_code_refactor": w2_report,
            "workload_3_cold_codegen": w3_report,
        }
        if norm_host_store != "hbm":
            report["host_store"] = norm_host_store
            report["stage_experts"] = stage_experts

        if block_runner is not None:
            chat_prompt = _make_cold_chat_prompt(
                base_runner.tokenizer, enable_thinking=enable_thinking, is_gemma4=is_gemma4
            )
            w4_vanilla = sweep_runner.run_vanilla_jax(chat_prompt, max_new_tokens=max_new_tokens)
            w4_tree = sweep_runner.run_tpuforge_tree(chat_prompt, max_new_tokens=max_new_tokens)
            w4_block = block_runner.run_block_tree(chat_prompt, max_new_tokens=max_new_tokens)
            w4_match_tree, w4_len_tree = _count_prefix_matches(w4_vanilla.token_ids, w4_tree.token_ids)
            w4_match_bt, w4_len_bt = _count_prefix_matches(w4_vanilla.token_ids, w4_block.token_ids)
            print(f"\n[Workload 4: Cold Chat Explanation (prompt_len={len(chat_prompt)}, max_new={max_new_tokens})]")
            for m in (w4_vanilla, w4_tree, w4_block):
                _print_mode_row(m)
            print(
                f"  Parity: tpuforge_tree == tpuforge_block_tree: {w4_tree.token_ids == w4_block.token_ids} | "
                f"vanilla == tpuforge_block_tree: {w4_match_bt}/{w4_len_bt} tokens"
            )
            report["draft_model_id"] = normalized_draft
            report["draft_model_dir"] = str(draft_dir)
            report["draft_top_k"] = top_k
            report["draft_warmup_seconds"] = round(block_warmup_s, 3)
            report["workload_4_cold_chat"] = {
                "vanilla_tree_matching_tokens": w4_match_tree,
                "vanilla_tree_compared_tokens": w4_len_tree,
                "vanilla_block_tree_matching_tokens": w4_match_bt,
                "vanilla_block_tree_compared_tokens": w4_len_bt,
                "tree_block_tree_exact_parity": w4_tree.token_ids == w4_block.token_ids,
                "vanilla_jax": _metrics_dict(w4_vanilla),
                "tpuforge_tree": _metrics_dict(w4_tree),
                "tpuforge_block_tree": _metrics_dict(w4_block),
            }

        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"\nSaved JSON report to: {output_json}")

        return report
    finally:
        if owned_stream_runner is not None:
            owned_stream_runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Vanilla JAX vs TPUForge (plus optional parallel block-diffusion systolic tree) "
            "benchmark on Qwen weights."
        )
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Hugging Face repo ID or local model directory.",
    )
    parser.add_argument(
        "--draft-model",
        default="",
        help=(
            "Optional Hugging Face repo ID or local directory for a parallel block-diffusion "
            "systolic tree drafter (compatible with Hugging Face block-diffusion / Z Lab DFlash checkpoints)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K branching width per block-diffusion step packed into the systolic tree.",
    )
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument(
        "--tile-size",
        default="auto",
        choices=["auto", "128", "256"],
        help="Systolic tile width ('auto' infers 128 on v2-v5e/CPU, 256 on v6e/v7/TPU8).",
    )
    parser.add_argument(
        "--tp-size",
        default="auto",
        help="Megatron tensor-parallelism degree ('auto', or integer e.g. 1, 2, 4, 8).",
    )
    parser.add_argument(
        "--weight-quant",
        default="none",
        choices=["none", "int8", "fp8"],
        help="Weight-only systolic quantization mode ('none', 'int8', or 'fp8').",
    )
    parser.add_argument(
        "--ast-seed-path",
        default=None,
        help="Optional repository file/directory to seed C++ SuffixIndex out-of-band on cold prompts.",
    )
    parser.add_argument(
        "--host-store",
        default="hbm",
        choices=["hbm", "mmap", "dram"],
        help="Weight placement mode ('hbm' full on-device, or 'mmap'/'dram' MoE expert streaming).",
    )
    parser.add_argument(
        "--stage-experts",
        type=int,
        default=16,
        help="Fixed [E_stage] HBM expert staging capacity when --host-store is mmap or dram.",
    )
    parser.add_argument(
        "--output-json",
        default="benchmarks/results/latest_qwen_bench.json",
        help="Path to write JSON telemetry output.",
    )
    args = parser.parse_args()

    out_path = Path(args.output_json) if args.output_json else None
    run_benchmark_suite(
        model_id=args.model,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        dtype_name=args.dtype,
        output_json=out_path,
        tile_size=args.tile_size,
        tp_size=args.tp_size,
        draft_model_id=args.draft_model or None,
        top_k=args.top_k,
        weight_quant=args.weight_quant,
        ast_seed_path=args.ast_seed_path,
        host_store=args.host_store,
        stage_experts=args.stage_experts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

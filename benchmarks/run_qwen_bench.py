"""3-Workload OpenXLA benchmark comparing Vanilla JAX, StaticArena (M-002), and TPUForge (M-002 + M-003)."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax

from benchmarks.dflash_jax import (
    DFlashConfig,
    DFlashWeights,
    QwenDFlashRunner,
    load_qwen_dflash_runner,
)
from benchmarks.qwen_jax import (
    QwenBenchmarkRunner,
    RunMetrics,
    encode_chat_messages,
    load_qwen_runner,
    resolve_model_dir,
)


def _make_sweep_prompts(
    tokenizer: Any, enable_thinking: bool | None = None
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
        toks = encode_chat_messages(
            tokenizer,
            [
                {"role": "system", "content": "You are a concise expert systems programmer."},
                {"role": "user", "content": text},
            ],
            enable_thinking=enable_thinking,
        )
        while len(toks) in seen_lens:
            toks.append(toks[-1])
        seen_lens.add(len(toks))
        prompts.append((f"sweep_{idx + 1}_len_{len(toks)}", toks))
    return prompts


def _make_refactor_prompt(
    tokenizer: Any, enable_thinking: bool | None = None
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
    return encode_chat_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are a concise Python refactoring assistant."},
            {"role": "user", "content": user_msg},
        ],
        enable_thinking=enable_thinking,
    )


def _make_cold_codegen_prompt(
    tokenizer: Any, enable_thinking: bool | None = None
) -> list[int]:
    """Fresh code generation prompt with minimal prompt n-gram overlap."""
    return encode_chat_messages(
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
    )


def _make_cold_chat_prompt(
    tokenizer: Any, enable_thinking: bool | None = None
) -> list[int]:
    """Natural-language cold chat prompt for DFlash block-diffusion evaluation."""
    return encode_chat_messages(
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
    preloaded: tuple[QwenBenchmarkRunner, Path] | None = None,
    draft_model_id: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    normalized_draft = draft_model_id.strip() if draft_model_id else ""
    dflash_runner: QwenDFlashRunner | None = None
    draft_dir: Path | None = None
    dflash_warmup_s = 0.0

    t_load0 = time.perf_counter()
    if preloaded is not None:
        prev_runner, model_dir = preloaded
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
            dflash_cfg = DFlashConfig.from_dir(draft_dir, target_config=base_runner.config)
            dflash_weights = DFlashWeights.from_model_dir(
                draft_dir, dflash_cfg, dtype=base_runner.dtype, mesh=base_runner.mesh
            )
            dflash_runner = QwenDFlashRunner(
                target_runner=base_runner,
                dflash_config=dflash_cfg,
                dflash_weights=dflash_weights,
                top_k=top_k,
            )
    elif normalized_draft:
        dflash_runner, model_dir, draft_dir = load_qwen_dflash_runner(
            model_id_or_path=model_id,
            draft_model_id_or_path=normalized_draft,
            max_seq_len=max_seq_len,
            dtype_name=dtype_name,
            tile_size=tile_size,
            tp_size=tp_size,
            top_k=top_k,
        )
        base_runner = dflash_runner.target
    else:
        base_runner, model_dir = load_qwen_runner(
            model_id_or_path=model_id,
            max_seq_len=max_seq_len,
            dtype_name=dtype_name,
            tile_size=tile_size,
            tp_size=tp_size,
        )
    load_s = time.perf_counter() - t_load0

    if dflash_runner is not None:
        dflash_warmup_s = dflash_runner.warmup()

    backend = jax.default_backend()
    devices = [str(d) for d in jax.devices()]
    cfg = base_runner.config
    tok_to_id = getattr(base_runner.tokenizer, "token_to_id", None)
    enable_thinking: bool | None = (
        False if (tok_to_id is not None and tok_to_id("</think>") is not None) else None
    )

    print("=" * 78)
    print(f"TPUForge Real-Weight Benchmark | Model: {model_id}")
    if normalized_draft:
        print(
            f"Draft Model: {normalized_draft} | block_size={dflash_runner.dflash_config.block_size} | "
            f"top_k={top_k} | warmup={dflash_warmup_s:.2f}s"
        )
    print(
        f"Backend: {backend} ({len(devices)}x {devices[0]}) | TP={base_runner.tp_size} | "
        f"Tile={base_runner.tile_size}x{base_runner.tile_size} | Dtype: {base_runner.dtype} | "
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
    sweep_prompts = _make_sweep_prompts(base_runner.tokenizer, enable_thinking=enable_thinking)
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

    refactor_prompt = _make_refactor_prompt(base_runner.tokenizer, enable_thinking=enable_thinking)
    w2_vanilla = sweep_runner.run_vanilla_jax(refactor_prompt, max_new_tokens=max_new_tokens)
    w2_static = sweep_runner.run_static_arena(refactor_prompt, max_new_tokens=max_new_tokens)
    w2_tree = sweep_runner.run_tpuforge_tree(refactor_prompt, max_new_tokens=max_new_tokens)
    w2_dflash = (
        dflash_runner.run_dflash_tree(refactor_prompt, max_new_tokens=max_new_tokens)
        if dflash_runner is not None
        else None
    )

    exact_static_tree_parity = w2_static.token_ids == w2_tree.token_ids
    vanilla_match_tokens, min_len = _count_prefix_matches(w2_vanilla.token_ids, w2_tree.token_ids)

    print(f"\n[Workload 2: Code Refactor (prompt_len={len(refactor_prompt)}, max_new={max_new_tokens})]")
    w2_runs = (w2_vanilla, w2_static, w2_tree) + ((w2_dflash,) if w2_dflash is not None else ())
    for m in w2_runs:
        _print_mode_row(m)
    print(
        f"  Parity: static_arena == tpuforge_tree: {exact_static_tree_parity} | "
        f"vanilla == tpuforge_tree: {vanilla_match_tokens}/{min_len} tokens"
    )
    if w2_dflash is not None:
        w2_df_static_parity = w2_static.token_ids == w2_dflash.token_ids
        df_v_match, df_v_len = _count_prefix_matches(w2_vanilla.token_ids, w2_dflash.token_ids)
        print(
            f"  DFlash Parity: static_arena == tpuforge_dflash: {w2_df_static_parity} | "
            f"vanilla == tpuforge_dflash: {df_v_match}/{df_v_len} tokens"
        )
    print(f"  Sample output ({w2_runs[-1].mode}):\n{w2_runs[-1].text[:320]}")

    cold_prompt = _make_cold_codegen_prompt(base_runner.tokenizer, enable_thinking=enable_thinking)
    w3_vanilla = sweep_runner.run_vanilla_jax(cold_prompt, max_new_tokens=max_new_tokens)
    w3_tree = sweep_runner.run_tpuforge_tree(cold_prompt, max_new_tokens=max_new_tokens)
    w3_dflash = (
        dflash_runner.run_dflash_tree(cold_prompt, max_new_tokens=max_new_tokens)
        if dflash_runner is not None
        else None
    )
    w3_match, w3_min = _count_prefix_matches(w3_vanilla.token_ids, w3_tree.token_ids)

    print(f"\n[Workload 3: Fresh Code Generation Floor (prompt_len={len(cold_prompt)}, max_new={max_new_tokens})]")
    w3_runs = (w3_vanilla, w3_tree) + ((w3_dflash,) if w3_dflash is not None else ())
    for m in w3_runs:
        _print_mode_row(m)
    print(f"  Parity: vanilla == tpuforge_tree: {w3_match}/{w3_min} tokens")
    if w3_dflash is not None:
        w3_df_tree_parity = w3_tree.token_ids == w3_dflash.token_ids
        w3_df_match, w3_df_len = _count_prefix_matches(w3_vanilla.token_ids, w3_dflash.token_ids)
        print(
            f"  DFlash Parity: tpuforge_tree == tpuforge_dflash: {w3_df_tree_parity} | "
            f"vanilla == tpuforge_dflash: {w3_df_match}/{w3_df_len} tokens"
        )

    w2_report: dict[str, Any] = {
        "static_tree_exact_parity": exact_static_tree_parity,
        "vanilla_tree_matching_tokens": vanilla_match_tokens,
        "vanilla_tree_compared_tokens": min_len,
        "vanilla_jax": _metrics_dict(w2_vanilla),
        "static_arena": _metrics_dict(w2_static),
        "tpuforge_tree": _metrics_dict(w2_tree),
    }
    if w2_dflash is not None:
        df_v_match, df_v_len = _count_prefix_matches(w2_vanilla.token_ids, w2_dflash.token_ids)
        w2_report["static_dflash_exact_parity"] = w2_static.token_ids == w2_dflash.token_ids
        w2_report["vanilla_dflash_matching_tokens"] = df_v_match
        w2_report["vanilla_dflash_compared_tokens"] = df_v_len
        w2_report["tpuforge_dflash"] = _metrics_dict(w2_dflash)

    w3_report: dict[str, Any] = {
        "vanilla_tree_matching_tokens": w3_match,
        "vanilla_tree_compared_tokens": w3_min,
        "vanilla_jax": _metrics_dict(w3_vanilla),
        "tpuforge_tree": _metrics_dict(w3_tree),
    }
    if w3_dflash is not None:
        w3_df_match, w3_df_len = _count_prefix_matches(w3_vanilla.token_ids, w3_dflash.token_ids)
        w3_report["tree_dflash_exact_parity"] = w3_tree.token_ids == w3_dflash.token_ids
        w3_report["vanilla_dflash_matching_tokens"] = w3_df_match
        w3_report["vanilla_dflash_compared_tokens"] = w3_df_len
        w3_report["tpuforge_dflash"] = _metrics_dict(w3_dflash)

    report: dict[str, Any] = {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "backend": backend,
        "device": devices[0],
        "num_devices": len(devices),
        "tp_size": base_runner.tp_size,
        "tile_size": base_runner.tile_size,
        "dtype": str(base_runner.dtype),
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

    if dflash_runner is not None:
        chat_prompt = _make_cold_chat_prompt(base_runner.tokenizer, enable_thinking=enable_thinking)
        w4_vanilla = sweep_runner.run_vanilla_jax(chat_prompt, max_new_tokens=max_new_tokens)
        w4_tree = sweep_runner.run_tpuforge_tree(chat_prompt, max_new_tokens=max_new_tokens)
        w4_dflash = dflash_runner.run_dflash_tree(chat_prompt, max_new_tokens=max_new_tokens)
        w4_match_tree, w4_len_tree = _count_prefix_matches(w4_vanilla.token_ids, w4_tree.token_ids)
        w4_match_df, w4_len_df = _count_prefix_matches(w4_vanilla.token_ids, w4_dflash.token_ids)
        print(f"\n[Workload 4: Cold Chat Explanation (prompt_len={len(chat_prompt)}, max_new={max_new_tokens})]")
        for m in (w4_vanilla, w4_tree, w4_dflash):
            _print_mode_row(m)
        print(
            f"  Parity: tpuforge_tree == tpuforge_dflash: {w4_tree.token_ids == w4_dflash.token_ids} | "
            f"vanilla == tpuforge_dflash: {w4_match_df}/{w4_len_df} tokens"
        )
        report["draft_model_id"] = normalized_draft
        report["draft_model_dir"] = str(draft_dir)
        report["dflash_top_k"] = top_k
        report["dflash_warmup_seconds"] = round(dflash_warmup_s, 3)
        report["workload_4_cold_chat"] = {
            "vanilla_tree_matching_tokens": w4_match_tree,
            "vanilla_tree_compared_tokens": w4_len_tree,
            "vanilla_dflash_matching_tokens": w4_match_df,
            "vanilla_dflash_compared_tokens": w4_len_df,
            "tree_dflash_exact_parity": w4_tree.token_ids == w4_dflash.token_ids,
            "vanilla_jax": _metrics_dict(w4_vanilla),
            "tpuforge_tree": _metrics_dict(w4_tree),
            "tpuforge_dflash": _metrics_dict(w4_dflash),
        }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {output_json}")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Vanilla JAX vs TPUForge (plus optional DFlash) benchmark on Qwen weights."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Hugging Face repo ID or local model directory.",
    )
    parser.add_argument(
        "--draft-model",
        default="",
        help="Optional Hugging Face repo ID or local directory for a DFlash block drafter.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K branching width per DFlash step packed into systolic tree.",
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ollama-style preset server, OpenAI HTTP API, and Live Systolic HUD for TPUForge."""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import _tpuforge_ext as tpu
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ast_seed_jax import (
    DEFAULT_CODE_IDIOMS,
    encode_with_tokenizer,
    extract_repo_ast_snippets,
)
from benchmarks.block_diffusion_jax import (
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
from benchmarks.hud_ui import render_hud_html
from benchmarks.qwen_jax import (
    _MASK_BLOCK_VAL,
    QwenBenchmarkRunner,
    default_local_model_roots,
    discover_local_models,
    encode_chat_messages,
    load_qwen_runner,
    resolve_model_dir,
)

VALID_SERVE_MODES: tuple[str, ...] = ("block_tree", "tree", "static_arena")
VALID_WEIGHT_QUANTS: tuple[str, ...] = ("none", "int8", "fp8")
VALID_DTYPES: tuple[str, ...] = ("auto", "float32", "bfloat16", "float16")
VALID_HOST_STORES: tuple[str, ...] = ("auto", "hbm", "mmap", "dram")
_CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError)


def _parse_strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _parse_strict_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return int(value)


def _parse_strict_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite")
    if exclusive_minimum is not None and val <= exclusive_minimum:
        raise ValueError(f"{name} must be > {exclusive_minimum}, got {val}")
    if minimum is not None and val < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {val}")
    if maximum is not None and val > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {val}")
    return val


@dataclass(frozen=True)
class PresetEntry:
    """Declarative Ollama-style preset definition mapped to TPUForge weights & defaults."""

    name: str
    model_id: str
    draft_model_id: str | None
    weight_quant: str
    dtype: str
    host_store: str
    min_hbm_gib: float
    default_max_seq_len: int
    description: str


PRESET_CATALOG: dict[str, PresetEntry] = {
    "gemma4:12b": PresetEntry(
        name="gemma4:12b",
        model_id="google/gemma-4-12B-it",
        draft_model_id="z-lab/gemma4-12B-it-DFlash",
        weight_quant="int8",
        dtype="auto",
        host_store="hbm",
        min_hbm_gib=14.5,
        default_max_seq_len=2048,
        description="Gemma 4 12B Dense + DFlash (INT8, fits single 16 GiB v5e-1 & 32 GiB v6e-1)",
    ),
    "gemma4:12b-bf16": PresetEntry(
        name="gemma4:12b-bf16",
        model_id="google/gemma-4-12B-it",
        draft_model_id="z-lab/gemma4-12B-it-DFlash",
        weight_quant="none",
        dtype="bfloat16",
        host_store="hbm",
        min_hbm_gib=23.5,
        default_max_seq_len=2048,
        description="Gemma 4 12B Dense + DFlash (BF16, fits 32 GiB v6e-1 or multi-chip pods)",
    ),
    "gemma4:26b": PresetEntry(
        name="gemma4:26b",
        model_id="google/gemma-4-26B-A4B-it",
        draft_model_id="z-lab/gemma-4-26B-A4B-it-DFlash",
        weight_quant="int8",
        dtype="auto",
        host_store="auto",
        min_hbm_gib=25.5,
        default_max_seq_len=2048,
        description="Gemma 4 26B-A4B MoE + DFlash (INT8 HBM on v6e-1/pods, host-mmap stream on v5e-1)",
    ),
    "gemma4:31b": PresetEntry(
        name="gemma4:31b",
        model_id="google/gemma-4-31B-it",
        draft_model_id="z-lab/gemma-4-31B-it-DFlash",
        weight_quant="none",
        dtype="bfloat16",
        host_store="hbm",
        min_hbm_gib=62.0,
        default_max_seq_len=2048,
        description="Gemma 4 31B Dense + DFlash (BF16 tied-embed, multi-chip TP pod)",
    ),
    "qwen3:4b": PresetEntry(
        name="qwen3:4b",
        model_id="Qwen/Qwen3-4B",
        draft_model_id="z-lab/Qwen3-4B-DFlash-b16",
        weight_quant="none",
        dtype="auto",
        host_store="hbm",
        min_hbm_gib=9.0,
        default_max_seq_len=2048,
        description="Qwen3-4B + DFlash (BF16 single-chip sanity preset)",
    ),
}


@dataclass(frozen=True)
class HardwareProfile:
    """Detected (or simulated) JAX/TPU runtime characteristics."""

    backend: str
    device_kind: str
    chip_count: int
    hbm_per_chip_gib: float | None
    tile_size: int
    label: str


@dataclass(frozen=True)
class ResolvedPresetSpec:
    """Fully resolved concrete checkpoint + hardware placement plan for a preset or path."""

    preset_name: str
    model_id: str
    draft_model_id: str | None
    weight_quant: str
    dtype: str
    host_store: str
    tile_size: int
    tp_size: int
    max_seq_len: int
    top_k: int
    stage_experts: int
    hardware: HardwareProfile


@dataclass(frozen=True)
class StepPulse:
    """Per-forward-pass systolic tile telemetry streamed to SSE clients and the Web HUD."""

    step_index: int
    mode: str
    drafted_nodes: int
    accepted_rows: list[int]
    accepted_tokens: list[int]
    text_delta: str
    step_ms: float
    prefix_len: int


@dataclass(frozen=True)
class GenerationSummary:
    """Final completion result + aggregate OpenXLA / speculative telemetry."""

    token_ids: list[int]
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prefill_ms: float
    decode_ms: float
    total_ms: float
    decode_tok_per_s: float
    target_forward_passes: int
    decode_forward_passes: int
    draft_tokens_accepted: int
    mean_tokens_per_decode_pass: float
    compile_count: int
    total_compiles: int
    mode: str
    tile_size: int


def detect_tpu_hardware(
    *,
    chip_count: int | None = None,
    hbm_per_chip_gib: float | None = None,
    device_kind: str | None = None,
) -> HardwareProfile:
    """Inspect active JAX runtime devices (with optional explicit overrides for planning/tests)."""
    devs = jax.devices()
    backend = jax.default_backend()
    if chip_count is not None:
        eff_chips = _parse_strict_int(chip_count, "chip_count", minimum=1)
    else:
        eff_chips = max(1, len(devs))

    if device_kind is not None:
        if not isinstance(device_kind, str) or not device_kind.strip():
            raise ValueError("device_kind must be a non-empty string")
        eff_kind = device_kind.strip()
    else:
        eff_kind = str(getattr(devs[0], "device_kind", backend))

    if hbm_per_chip_gib is not None:
        eff_hbm: float | None = _parse_strict_float(
            hbm_per_chip_gib, "hbm_per_chip_gib", exclusive_minimum=0.0
        )
    else:
        eff_hbm = None
        try:
            mem_stats = devs[0].memory_stats()
            if isinstance(mem_stats, dict) and mem_stats.get("bytes_limit"):
                eff_hbm = float(mem_stats["bytes_limit"]) / float(1024**3)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            eff_hbm = None

    tile_size = int(tpu.infer_mxu_dim(eff_kind))
    hbm_str = f"{eff_hbm:.2f} GiB/chip" if eff_hbm is not None else "host RAM"
    label = f"{eff_kind} x{eff_chips} ({hbm_str}, tile={tile_size})"
    return HardwareProfile(
        backend=backend,
        device_kind=eff_kind,
        chip_count=eff_chips,
        hbm_per_chip_gib=eff_hbm,
        tile_size=tile_size,
        label=label,
    )


def resolve_preset_spec(
    preset_or_model: str,
    *,
    draft_model: str | None = None,
    weight_quant: str | None = None,
    dtype: str | None = None,
    host_store: str | None = None,
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    max_seq_len: int | None = None,
    top_k: int = 8,
    stage_experts: int = 16,
    chip_count: int | None = None,
    hbm_per_chip_gib: float | None = None,
    device_kind: str | None = None,
) -> ResolvedPresetSpec:
    """Resolve an Ollama-style preset name or HF/local checkpoint path against TPU hardware."""
    if not isinstance(preset_or_model, str):
        raise TypeError("preset_or_model must be a string")
    raw_target = preset_or_model.strip()
    if not raw_target:
        raise ValueError("preset_or_model cannot be empty")
    if draft_model is not None and not isinstance(draft_model, str):
        raise TypeError("draft_model must be a string or None")

    hw = detect_tpu_hardware(
        chip_count=chip_count,
        hbm_per_chip_gib=hbm_per_chip_gib,
        device_kind=device_kind,
    )

    if isinstance(tile_size, str):
        cleaned_tile = tile_size.strip().lower()
        if cleaned_tile == "auto":
            eff_tile = hw.tile_size
        elif cleaned_tile.isdigit():
            eff_tile = int(cleaned_tile)
        else:
            raise ValueError(f"Unsupported tile_size={tile_size!r}; expected 'auto', 128, or 256")
    else:
        eff_tile = _parse_strict_int(tile_size, "tile_size")
    if eff_tile not in (128, 256):
        raise ValueError(f"tile_size must be 128 or 256, got {eff_tile}")

    entry = PRESET_CATALOG.get(raw_target.lower())
    if entry is not None:
        preset_name = entry.name
        model_id = entry.model_id
        eff_draft = entry.draft_model_id if draft_model is None else (draft_model.strip() or None)
        eff_quant = entry.weight_quant if weight_quant is None else weight_quant
        eff_dtype = entry.dtype if dtype is None else dtype
        eff_store = entry.host_store if host_store is None else host_store
        default_max_seq = entry.default_max_seq_len
        min_hbm = entry.min_hbm_gib
    else:
        preset_name = Path(raw_target).name or raw_target
        model_id = raw_target
        eff_draft = draft_model.strip() if draft_model else None
        eff_quant = "none" if weight_quant is None else weight_quant
        eff_dtype = "auto" if dtype is None else dtype
        eff_store = "hbm" if host_store is None else host_store
        default_max_seq = 1024
        min_hbm = 0.0

    if not isinstance(eff_quant, str) or eff_quant.strip().lower() not in VALID_WEIGHT_QUANTS:
        raise ValueError(f"weight_quant must be one of {VALID_WEIGHT_QUANTS}, got {eff_quant!r}")
    eff_quant = eff_quant.strip().lower()

    if not isinstance(eff_dtype, str) or eff_dtype.strip().lower() not in VALID_DTYPES:
        raise ValueError(f"dtype must be one of {VALID_DTYPES}, got {eff_dtype!r}")
    eff_dtype = eff_dtype.strip().lower()

    if not isinstance(eff_store, str) or eff_store.strip().lower() not in VALID_HOST_STORES:
        raise ValueError(f"host_store must be one of {VALID_HOST_STORES}, got {eff_store!r}")
    eff_store = eff_store.strip().lower()

    eff_max_seq = (
        default_max_seq
        if max_seq_len is None
        else _parse_strict_int(max_seq_len, "max_seq_len", minimum=eff_tile + 2)
    )
    eff_top_k = _parse_strict_int(top_k, "top_k", minimum=1)
    eff_stage_experts = _parse_strict_int(stage_experts, "stage_experts", minimum=1)

    if isinstance(tp_size, str):
        cleaned_tp = tp_size.strip().lower()
        if cleaned_tp == "auto":
            if (
                entry is not None
                and entry.name == "gemma4:12b"
                and hw.hbm_per_chip_gib is not None
                and hw.hbm_per_chip_gib >= 14.5
            ):
                eff_tp = 1
            elif hw.device_kind.lower().startswith("tpu") or hw.chip_count > 1:
                eff_tp = hw.chip_count
            else:
                eff_tp = 1
        elif cleaned_tp.isdigit():
            eff_tp = _parse_strict_int(int(cleaned_tp), "tp_size", minimum=1)
        else:
            raise ValueError(f"tp_size must be 'auto' or a positive integer, got {tp_size!r}")
    else:
        eff_tp = _parse_strict_int(tp_size, "tp_size", minimum=1)

    # Automatic single-chip HBM overflow fallback (e.g. gemma4:26b on 15.75 GiB v5e-1 -> host mmap).
    if eff_store == "auto":
        total_visible_hbm = (
            hw.hbm_per_chip_gib * eff_tp if hw.hbm_per_chip_gib is not None else None
        )
        if total_visible_hbm is not None and min_hbm > 0.0 and total_visible_hbm < min_hbm:
            eff_store = "mmap"
            eff_draft = None
        else:
            eff_store = "hbm"

    return ResolvedPresetSpec(
        preset_name=preset_name,
        model_id=model_id,
        draft_model_id=eff_draft,
        weight_quant=eff_quant,
        dtype=eff_dtype,
        host_store=eff_store,
        tile_size=eff_tile,
        tp_size=eff_tp,
        max_seq_len=eff_max_seq,
        top_k=eff_top_k,
        stage_experts=eff_stage_experts,
        hardware=hw,
    )


def format_preset_catalog_table(hw: HardwareProfile | None = None) -> str:
    """Format an aligned CLI table of built-in TPUForge presets and hardware-fit verdicts."""
    profile = hw if hw is not None else detect_tpu_hardware()
    lines = [
        f"Hardware: {profile.label}",
        "",
        f"{'Preset':<18} {'Quant':<6} {'Store':<6} {'Tile':<5} {'Draft Model':<32} Description",
        "-" * 112,
    ]
    for name, entry in PRESET_CATALOG.items():
        plan = resolve_preset_spec(
            name,
            chip_count=profile.chip_count,
            hbm_per_chip_gib=profile.hbm_per_chip_gib,
            device_kind=profile.device_kind,
        )
        draft_label = plan.draft_model_id or "(none: host mmap stream)"
        lines.append(
            f"{name:<18} {plan.weight_quant:<6} {plan.host_store:<6} {plan.tile_size:<5} "
            f"{draft_label:<32} {entry.description}"
        )
    return "\n".join(lines)


def format_local_models_table(search_roots: Sequence[str | Path] | None = None) -> str:
    """Format discovered complete local safetensors checkpoints and search roots for CLI display."""
    roots = (
        [Path(r).expanduser() for r in search_roots]
        if search_roots is not None
        else default_local_model_roots()
    )
    models = discover_local_models(roots)
    lines = [
        "Search Roots: " + (", ".join(str(r) for r in roots) if roots else "(none detected)"),
        "",
    ]
    if not models:
        lines.append("No complete local safetensors checkpoints found.")
        return "\n".join(lines)
    lines.append(f"{'Repo / Name':<38} {'Arch':<10} {'Source':<10} Path")
    lines.append("-" * 100)
    for item in models:
        lines.append(
            f"{item['repo_id']:<38} {item['arch']:<10} {item['source']:<10} {item['local_dir']}"
        )
    return "\n".join(lines)


def _decode_safe_prefix(tokenizer: Any, generated: list[int], emitted_len: int) -> tuple[str, int]:
    """Decode generated token IDs cleanly without emitting trailing replacement bytes early."""
    if not generated:
        return "", emitted_len
    full_text = str(tokenizer.decode(generated, skip_special_tokens=True))
    safe_end = len(full_text)
    if full_text.endswith("\ufffd"):
        idx = full_text.rfind("\ufffd")
        while idx > 0 and full_text[idx - 1] == "\ufffd":
            idx -= 1
        safe_end = max(emitted_len, idx)
    if safe_end <= emitted_len:
        return "", emitted_len
    return full_text[emitted_len:safe_end], safe_end


def _prefill_mask(chunk: int, tile_size: int) -> np.ndarray:
    """Match ForgeEngine::generate static T-row lower-triangular causal + idle diagonal mask."""
    mask = np.full((tile_size, tile_size), _MASK_BLOCK_VAL, dtype=np.float32)
    for r in range(chunk):
        mask[r, : r + 1] = 0.0
    for r in range(chunk, tile_size):
        mask[r, r] = 0.0
    return mask


class ForgeServeSession:
    """Stateful OpenXLA + _tpuforge_ext serving engine backing the HTTP server & HUD."""

    def __init__(
        self,
        spec: ResolvedPresetSpec,
        *,
        target_runner: Gemma4Runner | Gemma4MoeStreamRunner | QwenBenchmarkRunner,
        block_runner: Gemma4BlockTreeRunner | QwenBlockTreeRunner | None,
        is_gemma4: bool,
        model_dir: Path,
        draft_dir: Path | None = None,
        ast_seed_path: str | Path | None = None,
        search_roots: Sequence[str | Path] | None = None,
        warmup: bool = True,
    ) -> None:
        self.spec = spec
        self.target_runner = target_runner
        self.block_runner = block_runner
        self.is_gemma4 = is_gemma4
        self.model_dir = model_dir
        self.draft_dir = draft_dir
        self.tile_size = int(target_runner.tile_size)
        self.max_seq_len = int(target_runner.max_seq_len)
        self.tokenizer = target_runner.tokenizer
        self.config = target_runner.config
        self.eos_token_ids: list[int] = [int(x) for x in self.config.eos_token_ids]
        self._lock = threading.Lock()
        self.base_seed_sequences: list[list[int]] = []
        self.ast_seed_source: str = ""
        self._warmed_modes: set[str] = set()
        self._search_roots: tuple[Path, ...] | None = (
            tuple(Path(r) for r in search_roots) if search_roots is not None else None
        )
        self._local_models_cache: list[dict[str, Any]] = discover_local_models(self._search_roots)

        if ast_seed_path is not None:
            self.update_ast_seed(ast_seed_path=ast_seed_path, append=False)
        else:
            self.update_ast_seed(seed_texts=list(DEFAULT_CODE_IDIOMS), append=False)

        if warmup:
            self.warmup()

    def refresh_local_models(self) -> list[dict[str, Any]]:
        """Scan configured local model roots outside the inference lock and update cached inventory."""
        discovered = discover_local_models(self._search_roots)
        with self._lock:
            self._local_models_cache = discovered
            return list(self._local_models_cache)

    @property
    def default_mode(self) -> str:
        return "block_tree" if self.block_runner is not None else "tree"

    def _empty_target_kv(self) -> tuple[Any, Any]:
        if isinstance(self.target_runner, QwenBenchmarkRunner):
            return self.target_runner._empty_kv_cache()
        return self.target_runner._empty_kv_caches()

    def total_compile_count(self) -> int:
        if self.block_runner is not None:
            return int(self.block_runner.total_compile_count())
        return int(self.target_runner.static_step_compile_count)

    def warmup(self) -> dict[str, Any]:
        """Warm OpenXLA graphs upfront so interactive HUD requests run with zero recompiles."""
        with self._lock:
            block_s = 0.0
            if self.block_runner is not None and "block_tree" not in self._warmed_modes:
                block_s = float(self.block_runner.warmup())
                self._warmed_modes.add("block_tree")
            return {
                "warmup_block_seconds": round(block_s, 4),
                "total_compiles": self.total_compile_count(),
            }

    def close(self) -> None:
        close_fn = getattr(self.target_runner, "close", None)
        if callable(close_fn):
            close_fn()

    def status_dict(self) -> dict[str, Any]:
        seed_snapshot = list(self.base_seed_sequences)
        local_models_snapshot = list(self._local_models_cache)
        return {
            "preset": self.spec.preset_name,
            "model_id": self.spec.model_id,
            "model_dir": str(self.model_dir),
            "draft_model_id": self.spec.draft_model_id,
            "draft_model_dir": str(self.draft_dir) if self.draft_dir else None,
            "has_draft_model": self.block_runner is not None,
            "default_mode": self.default_mode,
            "arch": "gemma4" if self.is_gemma4 else "qwen",
            "host_store": self.spec.host_store,
            "weight_quant": self.target_runner.weight_quant,
            "tile_size": self.tile_size,
            "tp_size": int(self.target_runner.tp_size),
            "max_seq_len": self.max_seq_len,
            "hardware_label": self.spec.hardware.label,
            "total_compiles": self.total_compile_count(),
            "ast_seed_sequences": len(seed_snapshot),
            "ast_seed_tokens": sum(len(seq) for seq in seed_snapshot),
            "ast_seed_source": self.ast_seed_source,
            "local_models_count": len(local_models_snapshot),
            "available_presets": list(PRESET_CATALOG.keys()),
        }

    def update_ast_seed(
        self,
        *,
        ast_seed_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        seed_texts: Sequence[str] | None = None,
        snippets: Sequence[str] | None = None,
        append: bool = False,
        reset: bool | None = None,
    ) -> dict[str, Any]:
        eff_append = _parse_strict_bool(append, "append")
        if reset is not None:
            eff_append = not _parse_strict_bool(reset, "reset")

        collected: list[str] = []
        new_source: str | None = None
        eff_path = ast_seed_path if ast_seed_path is not None else repo_path
        if eff_path is not None:
            if not isinstance(eff_path, (str, Path)):
                raise TypeError("ast_seed_path / repo_path must be a string or Path")
            raw_path = str(eff_path).strip()
            if raw_path:
                target_path = Path(raw_path).expanduser()
                if not target_path.exists():
                    raise FileNotFoundError(f"AST seed path does not exist: {target_path}")
                collected.extend(extract_repo_ast_snippets(target_path))
                new_source = str(target_path)

        eff_texts = seed_texts if seed_texts is not None else snippets
        if eff_texts is not None:
            if isinstance(eff_texts, (str, bytes)) or not isinstance(eff_texts, Sequence):
                raise TypeError("seed_texts / snippets must be a list of strings")
            for idx, txt in enumerate(eff_texts):
                if not isinstance(txt, str):
                    raise TypeError(f"seed_texts[{idx}] must be a string")
                stripped = txt.strip()
                if stripped:
                    collected.append(stripped)

        if not collected and not eff_append:
            collected = list(DEFAULT_CODE_IDIOMS)

        new_seqs = encode_with_tokenizer(collected, self.tokenizer)
        with self._lock:
            if new_source is not None:
                self.ast_seed_source = new_source
            elif not eff_append:
                self.ast_seed_source = ""
            if eff_append:
                self.base_seed_sequences.extend(new_seqs)
            else:
                self.base_seed_sequences = new_seqs
        return self.status_dict()

    def encode_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        enable_thinking: bool = False,
    ) -> list[int]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a non-empty sequence of message objects")
        if not messages:
            raise ValueError("messages cannot be empty")
        norm: list[dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise TypeError(f"messages[{idx}] must be an object")
            role = msg.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"messages[{idx}].role must be a non-empty string")
            raw_content = msg.get("content", "")
            if isinstance(raw_content, str):
                text_content = raw_content
            elif isinstance(raw_content, Sequence) and not isinstance(raw_content, (bytes, bytearray)):
                parts: list[str] = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text")
                        if not isinstance(txt, str):
                            raise TypeError(f"messages[{idx}] text part requires string 'text'")
                        parts.append(txt)
                    else:
                        raise TypeError(
                            f"messages[{idx}].content list items must be {{\"type\": \"text\", \"text\": ...}}"
                        )
                text_content = "".join(parts)
            else:
                raise TypeError(f"messages[{idx}].content must be a string or text-part list")
            norm.append({"role": role.strip(), "content": text_content})

        if self.is_gemma4:
            token_ids = encode_gemma4_chat_messages(
                self.tokenizer, norm, enable_thinking=enable_thinking
            )
        else:
            token_ids = encode_chat_messages(self.tokenizer, norm, enable_thinking=enable_thinking)
        if not token_ids:
            raise ValueError("Encoded messages produced zero prompt tokens")
        return token_ids

    def _run_prefill_static(
        self,
        prompt_tokens: list[int],
    ) -> tuple[Any, Any, int, float, dict[str, Any]]:
        """Prefill target KV caches for static_arena / tree modes across all target runner types."""
        T = self.tile_size
        t0 = time.perf_counter()
        if isinstance(self.target_runner, Gemma4MoeStreamRunner):
            stats = self.target_runner._new_stats()
            k_caches, v_caches = self.target_runner._empty_kv_caches()
            jax.block_until_ready((k_caches[0], v_caches[0]))
            prefill_len = len(prompt_tokens) - 1
            forward_passes = 0
            all_rows_mask = np.ones((T,), dtype=bool)
            for offset in range(0, prefill_len, T):
                chunk = min(T, prefill_len - offset)
                tok_np = np.zeros((T,), dtype=np.int32)
                pos_np = np.zeros((T,), dtype=np.int32)
                tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
                pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
                mask_np = _prefill_mask(chunk, T)
                _, step_ks, step_vs = self.target_runner._run_step(
                    offset,
                    tok_np,
                    pos_np,
                    mask_np,
                    True,
                    all_rows_mask,
                    k_caches,
                    v_caches,
                    stats,
                    mode_tag="st",
                    count_as_decode=False,
                )
                src_rows = np.zeros((T,), dtype=np.int32)
                src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
                kt, vt = self.target_runner._commit_all_kv_jit(
                    tuple(k_caches),
                    tuple(v_caches),
                    step_ks,
                    step_vs,
                    self.target_runner._put_rep(offset, dtype=jnp.int32),
                    self.target_runner._put_rep(src_rows),
                    self.target_runner._put_rep(chunk, dtype=jnp.int32),
                )
                k_caches, v_caches = list(kt), list(vt)
                forward_passes += 1
            jax.block_until_ready((k_caches[0], v_caches[0]))
            return k_caches, v_caches, forward_passes, (time.perf_counter() - t0) * 1000.0, stats

        k_caches, v_caches = self._empty_target_kv()
        jax.block_until_ready((k_caches, v_caches))
        prefill_len = len(prompt_tokens) - 1
        forward_passes = 0
        for offset in range(0, prefill_len, T):
            chunk = min(T, prefill_len - offset)
            tok_np = np.zeros((T,), dtype=np.int32)
            pos_np = np.zeros((T,), dtype=np.int32)
            tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
            pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
            mask_np = _prefill_mask(chunk, T)

            _, step_k, step_v = self.target_runner._static_systolic_tile_step_jit(
                self.target_runner._weights_tuple,
                self.target_runner._put_rep(offset, dtype=jnp.int32),
                self.target_runner._put_rep(tok_np),
                self.target_runner._put_rep(pos_np),
                self.target_runner._put_rep(mask_np),
                k_caches,
                v_caches,
            )
            src_rows = np.zeros((T,), dtype=np.int32)
            src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
            k_caches, v_caches = self.target_runner._commit_kv_jit(
                k_caches,
                v_caches,
                step_k,
                step_v,
                self.target_runner._put_rep(offset, dtype=jnp.int32),
                self.target_runner._put_rep(src_rows),
                self.target_runner._put_rep(chunk, dtype=jnp.int32),
            )
            forward_passes += 1
        jax.block_until_ready((k_caches, v_caches))
        return k_caches, v_caches, forward_passes, (time.perf_counter() - t0) * 1000.0, {}

    def stream_generate(
        self,
        *,
        prompt_tokens: list[int],
        max_tokens: int = 128,
        mode: str | None = None,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> Iterator[StepPulse | GenerationSummary]:
        """Execute real OpenXLA + _tpuforge_ext decoding, yielding StepPulse events & Summary."""
        chosen_mode = (mode or self.default_mode).strip().lower()
        if chosen_mode not in VALID_SERVE_MODES:
            raise ValueError(f"Unsupported mode={chosen_mode!r}; expected one of {VALID_SERVE_MODES}")
        if chosen_mode == "block_tree" and self.block_runner is None:
            raise ValueError(
                "mode='block_tree' requires a DFlash draft model; use 'tree' or 'static_arena'"
            )

        self.target_runner._validate_bounds(prompt_tokens, max_tokens)
        T = self.tile_size
        max_prefix = self.max_seq_len - T
        vocab_size = int(self.config.vocab_size)
        eos_set = set(self.eos_token_ids)

        with self._lock:
            compiles_before = self.total_compile_count()
            generated: list[int] = []
            emitted_len = 0
            hit_eos = False
            draft_tokens_accepted = 0
            decode_passes = 0
            target_passes = 0

            if chosen_mode == "static_arena":
                k_caches, v_caches, forward_passes, prefill_ms, stats = self._run_prefill_static(
                    prompt_tokens
                )
                target_passes = forward_passes
                decode_mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
                np.fill_diagonal(decode_mask_np, 0.0)
                decode_mask_dev = self.target_runner._put_rep(decode_mask_np)
                src_row_0 = self.target_runner._put_rep(np.zeros((T,), dtype=np.int32))
                one_jnp = self.target_runner._put_rep(1, dtype=jnp.int32)
                valid_row0 = np.zeros((T,), dtype=bool)
                valid_row0[0] = True
                cur_prefix = len(prompt_tokens) - 1
                cur_tok = int(prompt_tokens[-1])

                t_dec0 = time.perf_counter()
                while len(generated) < max_tokens and cur_prefix < max_prefix:
                    step_t0 = time.perf_counter()
                    tok_np = np.zeros((T,), dtype=np.int32)
                    pos_np = np.zeros((T,), dtype=np.int32)
                    tok_np[0] = cur_tok
                    pos_np[0] = cur_prefix

                    if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                        argmax_dev, step_ks, step_vs = self.target_runner._run_step(
                            cur_prefix,
                            tok_np,
                            pos_np,
                            decode_mask_np,
                            True,
                            valid_row0,
                            k_caches,
                            v_caches,
                            stats,
                            mode_tag="st",
                            count_as_decode=True,
                        )
                    else:
                        argmax_dev, step_k, step_v = self.target_runner._static_systolic_tile_step_jit(
                            self.target_runner._weights_tuple,
                            self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                            self.target_runner._put_rep(tok_np),
                            self.target_runner._put_rep(pos_np),
                            decode_mask_dev,
                            k_caches,
                            v_caches,
                        )
                    target_passes += 1
                    decode_passes += 1
                    next_tok = int(argmax_dev[0])
                    if next_tok in eos_set:
                        hit_eos = True
                        yield StepPulse(
                            step_index=decode_passes,
                            mode=chosen_mode,
                            drafted_nodes=1,
                            accepted_rows=[],
                            accepted_tokens=[],
                            text_delta="",
                            step_ms=(time.perf_counter() - step_t0) * 1000.0,
                            prefix_len=cur_prefix,
                        )
                        break

                    if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                        kt, vt = self.target_runner._commit_all_kv_jit(
                            tuple(k_caches),
                            tuple(v_caches),
                            step_ks,
                            step_vs,
                            self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                            src_row_0,
                            one_jnp,
                        )
                        k_caches, v_caches = list(kt), list(vt)
                    else:
                        k_caches, v_caches = self.target_runner._commit_kv_jit(
                            k_caches,
                            v_caches,
                            step_k,
                            step_v,
                            self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                            src_row_0,
                            one_jnp,
                        )
                    generated.append(next_tok)
                    cur_prefix += 1
                    cur_tok = next_tok
                    delta, emitted_len = _decode_safe_prefix(self.tokenizer, generated, emitted_len)
                    yield StepPulse(
                        step_index=decode_passes,
                        mode=chosen_mode,
                        drafted_nodes=1,
                        accepted_rows=[0],
                        accepted_tokens=[next_tok],
                        text_delta=delta,
                        step_ms=(time.perf_counter() - step_t0) * 1000.0,
                        prefix_len=cur_prefix,
                    )

                if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                    jax.block_until_ready((k_caches[0], v_caches[0]))
                else:
                    jax.block_until_ready((k_caches, v_caches))
                decode_ms = (time.perf_counter() - t_dec0) * 1000.0

            elif chosen_mode == "tree":
                k_caches, v_caches, forward_passes, prefill_ms, stats = self._run_prefill_static(
                    prompt_tokens
                )
                target_passes = forward_passes
                suffix_index = tpu.SuffixIndex()
                if self.base_seed_sequences:
                    suffix_index.seed_sequences(self.base_seed_sequences)
                if seed_token_sequences:
                    extra_seeds = [[int(t) for t in s] for s in seed_token_sequences if len(s) >= 2]
                    if extra_seeds:
                        suffix_index.seed_sequences(extra_seeds)
                suffix_index.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))
                cur_prefix = len(prompt_tokens) - 1

                t_dec0 = time.perf_counter()
                while len(generated) < max_tokens and cur_prefix < max_prefix:
                    step_t0 = time.perf_counter()
                    remaining_prefix = max_prefix - cur_prefix
                    budget = min(max_tokens - len(generated), remaining_prefix)
                    max_depth = max(1, min(16, remaining_prefix))

                    tree = suffix_index.build_tree(T, max_depth, T)
                    packed = tree.pack(cur_prefix)
                    tok_np = np.asarray(packed["token_ids"], dtype=np.int32)
                    pos_np = np.asarray(packed["position_offsets"], dtype=np.int32)
                    mask_np = np.asarray(packed["additive_mask_f32"], dtype=np.float32).reshape(T, T)

                    if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                        argmax_dev, step_ks, step_vs = self.target_runner._run_step(
                            cur_prefix,
                            tok_np,
                            pos_np,
                            mask_np,
                            True,
                            (pos_np >= cur_prefix),
                            k_caches,
                            v_caches,
                            stats,
                            mode_tag="st",
                            count_as_decode=True,
                        )
                    else:
                        argmax_dev, step_k, step_v = self.target_runner._static_systolic_tile_step_jit(
                            self.target_runner._weights_tuple,
                            self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                            self.target_runner._put_rep(tok_np),
                            self.target_runner._put_rep(pos_np),
                            self.target_runner._put_rep(mask_np),
                            k_caches,
                            v_caches,
                        )
                    target_passes += 1
                    decode_passes += 1

                    greedy_np = np.asarray(jax.device_get(argmax_dev), dtype=np.int32)
                    verify_res = tpu.SystolicTreeVerifier.verify_greedy_tokens(
                        tree, greedy_np, vocab_size
                    )
                    accepted_slice: list[int] = []
                    for tok in verify_res.emitted_tokens[:budget]:
                        tok_int = int(tok)
                        if tok_int in eos_set:
                            hit_eos = True
                            break
                        generated.append(tok_int)
                        suffix_index.append_token(tok_int)
                        accepted_slice.append(tok_int)

                    commit_count = len(accepted_slice)
                    accepted_rows = [
                        int(r) for r in verify_res.accepted_node_indices[:commit_count]
                    ]
                    if commit_count > 0:
                        if commit_count > 1:
                            draft_tokens_accepted += commit_count - 1
                        src_rows = np.zeros((T,), dtype=np.int32)
                        src_rows[: len(accepted_rows)] = np.asarray(accepted_rows, dtype=np.int32)
                        if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                            kt, vt = self.target_runner._commit_all_kv_jit(
                                tuple(k_caches),
                                tuple(v_caches),
                                step_ks,
                                step_vs,
                                self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                                self.target_runner._put_rep(src_rows),
                                self.target_runner._put_rep(commit_count, dtype=jnp.int32),
                            )
                            k_caches, v_caches = list(kt), list(vt)
                        else:
                            k_caches, v_caches = self.target_runner._commit_kv_jit(
                                k_caches,
                                v_caches,
                                step_k,
                                step_v,
                                self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                                self.target_runner._put_rep(src_rows),
                                self.target_runner._put_rep(commit_count, dtype=jnp.int32),
                            )
                        cur_prefix += commit_count

                    delta, emitted_len = _decode_safe_prefix(self.tokenizer, generated, emitted_len)
                    yield StepPulse(
                        step_index=decode_passes,
                        mode=chosen_mode,
                        drafted_nodes=int(tree.num_nodes()),
                        accepted_rows=accepted_rows,
                        accepted_tokens=accepted_slice,
                        text_delta=delta,
                        step_ms=(time.perf_counter() - step_t0) * 1000.0,
                        prefix_len=cur_prefix,
                    )
                    if hit_eos or commit_count == 0:
                        break

                if isinstance(self.target_runner, Gemma4MoeStreamRunner):
                    jax.block_until_ready((k_caches[0], v_caches[0]))
                else:
                    jax.block_until_ready((k_caches, v_caches))
                decode_ms = (time.perf_counter() - t_dec0) * 1000.0

            else:
                # mode == "block_tree" (DFlash block-diffusion + SuffixIndex hybrid verification)
                assert self.block_runner is not None
                br = self.block_runner
                target_wt = self.target_runner._weights_tuple
                draft_wt = br._draft_weights_tuple
                embed = target_wt.embed
                if isinstance(br, Gemma4BlockTreeRunner):
                    lm_arr = embed if target_wt.lm_head is None else target_wt.lm_head
                    lm_head_scale = target_wt.lm_head_scale
                else:
                    assert isinstance(br, QwenBlockTreeRunner)
                    lm_arr = target_wt.lm_head
                    lm_head_scale = target_wt.lm_head_scale
                k_caches, v_caches = self._empty_target_kv()
                dk_cache, dv_cache = br._empty_draft_kv_cache()
                jax.block_until_ready((k_caches, v_caches, dk_cache, dv_cache))

                t_pre0 = time.perf_counter()
                prefill_len = len(prompt_tokens) - 1
                for offset in range(0, prefill_len, T):
                    chunk = min(T, prefill_len - offset)
                    tok_np = np.zeros((T,), dtype=np.int32)
                    pos_np = np.zeros((T,), dtype=np.int32)
                    tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
                    pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
                    mask_np = _prefill_mask(chunk, T)

                    _, step_k, step_v, step_ctx = br._target_tile_with_ctx_jit(
                        target_wt,
                        br._capture_slots,
                        self.target_runner._put_rep(offset, dtype=jnp.int32),
                        self.target_runner._put_rep(tok_np),
                        self.target_runner._put_rep(pos_np),
                        self.target_runner._put_rep(mask_np),
                        k_caches,
                        v_caches,
                    )
                    src_rows = np.zeros((T,), dtype=np.int32)
                    src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
                    k_caches, v_caches, dk_cache, dv_cache = br._commit_both_kv_jit(
                        draft_wt,
                        k_caches,
                        v_caches,
                        step_k,
                        step_v,
                        dk_cache,
                        dv_cache,
                        step_ctx,
                        self.target_runner._put_rep(offset, dtype=jnp.int32),
                        self.target_runner._put_rep(src_rows),
                        self.target_runner._put_rep(chunk, dtype=jnp.int32),
                    )
                    target_passes += 1
                jax.block_until_ready((k_caches, v_caches, dk_cache, dv_cache))
                prefill_ms = (time.perf_counter() - t_pre0) * 1000.0

                suffix_index = tpu.SuffixIndex()
                if self.base_seed_sequences:
                    suffix_index.seed_sequences(self.base_seed_sequences)
                if seed_token_sequences:
                    extra_seeds = [[int(t) for t in s] for s in seed_token_sequences if len(s) >= 2]
                    if extra_seeds:
                        suffix_index.seed_sequences(extra_seeds)
                suffix_index.append_tokens(np.asarray(prompt_tokens, dtype=np.int32))

                history: list[int] = [int(t) for t in prompt_tokens]
                cur_prefix = prefill_len

                t_dec0 = time.perf_counter()
                while len(generated) < max_tokens and cur_prefix < max_prefix:
                    step_t0 = time.perf_counter()
                    remaining_prefix = max_prefix - cur_prefix
                    budget = min(max_tokens - len(generated), remaining_prefix)
                    max_depth = max(1, min(16, remaining_prefix))
                    root_tok = int(history[-1])

                    top_ids_dev, top_probs_dev = br._block_draft_top_k_jit(
                        embed,
                        lm_arr,
                        lm_head_scale,
                        draft_wt,
                        self.target_runner._put_rep(root_tok, dtype=jnp.int32),
                        self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                        dk_cache,
                        dv_cache,
                    )
                    top_ids_np, top_probs_np = jax.device_get((top_ids_dev, top_probs_dev))
                    steps = min(int(top_ids_np.shape[0]), max_depth)
                    ids_slice = np.ascontiguousarray(top_ids_np[:steps], dtype=np.int32)
                    probs_slice = np.ascontiguousarray(top_probs_np[:steps], dtype=np.float32)

                    tree = tpu.TreeBuilder.build_from_block_top_k(
                        root_tok,
                        ids_slice,
                        probs_slice,
                        suffix_fallback=suffix_index,
                        max_nodes=T,
                        max_depth=max_depth,
                        tile_size=T,
                    )
                    packed = tree.pack(cur_prefix)
                    tok_np = np.asarray(packed["token_ids"], dtype=np.int32)
                    pos_np = np.asarray(packed["position_offsets"], dtype=np.int32)
                    mask_np = np.asarray(packed["additive_mask_f32"], dtype=np.float32).reshape(T, T)

                    argmax_dev, step_k, step_v, step_ctx = br._target_tile_with_ctx_jit(
                        target_wt,
                        br._capture_slots,
                        self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                        self.target_runner._put_rep(tok_np),
                        self.target_runner._put_rep(pos_np),
                        self.target_runner._put_rep(mask_np),
                        k_caches,
                        v_caches,
                    )
                    target_passes += 1
                    decode_passes += 1

                    greedy_np = np.asarray(jax.device_get(argmax_dev), dtype=np.int32)
                    verify_res = tpu.SystolicTreeVerifier.verify_greedy_tokens(
                        tree, greedy_np, vocab_size
                    )
                    accepted_slice = []
                    for tok in verify_res.emitted_tokens[:budget]:
                        tok_int = int(tok)
                        if tok_int in eos_set:
                            hit_eos = True
                            break
                        generated.append(tok_int)
                        history.append(tok_int)
                        suffix_index.append_token(tok_int)
                        accepted_slice.append(tok_int)

                    commit_count = len(accepted_slice)
                    accepted_rows = [
                        int(r) for r in verify_res.accepted_node_indices[:commit_count]
                    ]
                    if commit_count > 0:
                        if commit_count > 1:
                            draft_tokens_accepted += commit_count - 1
                        src_rows = np.zeros((T,), dtype=np.int32)
                        src_rows[: len(accepted_rows)] = np.asarray(accepted_rows, dtype=np.int32)
                        k_caches, v_caches, dk_cache, dv_cache = br._commit_both_kv_jit(
                            draft_wt,
                            k_caches,
                            v_caches,
                            step_k,
                            step_v,
                            dk_cache,
                            dv_cache,
                            step_ctx,
                            self.target_runner._put_rep(cur_prefix, dtype=jnp.int32),
                            self.target_runner._put_rep(src_rows),
                            self.target_runner._put_rep(commit_count, dtype=jnp.int32),
                        )
                        cur_prefix += commit_count

                    delta, emitted_len = _decode_safe_prefix(self.tokenizer, generated, emitted_len)
                    yield StepPulse(
                        step_index=decode_passes,
                        mode=chosen_mode,
                        drafted_nodes=int(tree.num_nodes()),
                        accepted_rows=accepted_rows,
                        accepted_tokens=accepted_slice,
                        text_delta=delta,
                        step_ms=(time.perf_counter() - step_t0) * 1000.0,
                        prefix_len=cur_prefix,
                    )
                    if hit_eos or commit_count == 0:
                        break

                jax.block_until_ready((k_caches, v_caches, dk_cache, dv_cache))
                decode_ms = (time.perf_counter() - t_dec0) * 1000.0

            full_text = self.target_runner._decode_text(generated) if generated else ""
            if len(full_text) > emitted_len:
                yield StepPulse(
                    step_index=decode_passes,
                    mode=chosen_mode,
                    drafted_nodes=0,
                    accepted_rows=[],
                    accepted_tokens=[],
                    text_delta=full_text[emitted_len:],
                    step_ms=0.0,
                    prefix_len=len(prompt_tokens) - 1 + len(generated),
                )

            compiles_after = self.total_compile_count()
            compiles_delta = compiles_after - compiles_before
            decode_s = decode_ms / 1000.0
            gen_len = len(generated)
            yield GenerationSummary(
                token_ids=generated,
                text=full_text,
                finish_reason="stop" if hit_eos else "length",
                prompt_tokens=len(prompt_tokens),
                completion_tokens=gen_len,
                total_tokens=len(prompt_tokens) + gen_len,
                prefill_ms=prefill_ms,
                decode_ms=decode_ms,
                total_ms=prefill_ms + decode_ms,
                decode_tok_per_s=(gen_len / decode_s) if decode_s > 0.0 else 0.0,
                target_forward_passes=target_passes,
                decode_forward_passes=decode_passes,
                draft_tokens_accepted=draft_tokens_accepted,
                mean_tokens_per_decode_pass=(
                    float(gen_len) / float(decode_passes) if decode_passes > 0 else 0.0
                ),
                compile_count=compiles_delta,
                total_compiles=compiles_after,
                mode=chosen_mode,
                tile_size=T,
            )


def load_serve_session(
    preset_or_model: str = "gemma4:12b",
    *,
    draft_model: str | None = None,
    max_seq_len: int | None = None,
    dtype: str | None = None,
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    weight_quant: str | None = None,
    host_store: str | None = None,
    stage_experts: int = 16,
    top_k: int = 8,
    ast_seed_path: str | Path | None = None,
    enable_xla_cache: bool = True,
    allow_download: bool = True,
    search_roots: Sequence[str | Path] | None = None,
    warmup: bool = True,
) -> ForgeServeSession:
    """Resolve preset or local/HF model directory and load target + optional DFlash runner."""
    spec = resolve_preset_spec(
        preset_or_model,
        draft_model=draft_model,
        weight_quant=weight_quant,
        dtype=dtype,
        host_store=host_store,
        tile_size=tile_size,
        tp_size=tp_size,
        max_seq_len=max_seq_len,
        top_k=top_k,
        stage_experts=stage_experts,
    )
    model_dir = resolve_model_dir(
        spec.model_id, search_roots=search_roots, allow_download=allow_download
    )
    gemma4 = is_gemma4_model_dir(model_dir)
    tile_arg: int | str = spec.tile_size if tile_size != "auto" else "auto"
    tp_arg: int | str = spec.tp_size if tp_size != "auto" else "auto"

    if gemma4:
        if spec.host_store in ("mmap", "dram"):
            target_runner, model_dir = load_gemma4_moe_stream_runner(
                str(model_dir),
                max_seq_len=spec.max_seq_len,
                dtype_name=spec.dtype,
                tile_size=tile_arg,
                tp_size=tp_arg,
                weight_quant=spec.weight_quant,
                host_store_mode=spec.host_store,
                stage_experts=spec.stage_experts,
                enable_xla_cache=enable_xla_cache,
            )
            return ForgeServeSession(
                spec,
                target_runner=target_runner,
                block_runner=None,
                is_gemma4=True,
                model_dir=model_dir,
                draft_dir=None,
                ast_seed_path=ast_seed_path,
                search_roots=search_roots,
                warmup=warmup,
            )

        if spec.draft_model_id:
            draft_resolved = resolve_model_dir(
                spec.draft_model_id, search_roots=search_roots, allow_download=allow_download
            )
            block_runner, model_dir, draft_dir = load_gemma4_block_tree_runner(
                str(model_dir),
                str(draft_resolved),
                max_seq_len=spec.max_seq_len,
                dtype_name=spec.dtype,
                tile_size=tile_arg,
                tp_size=tp_arg,
                top_k=spec.top_k,
                enable_xla_cache=enable_xla_cache,
                weight_quant=spec.weight_quant,
            )
            return ForgeServeSession(
                spec,
                target_runner=block_runner.target,
                block_runner=block_runner,
                is_gemma4=True,
                model_dir=model_dir,
                draft_dir=draft_dir,
                ast_seed_path=ast_seed_path,
                search_roots=search_roots,
                warmup=warmup,
            )

        target_runner, model_dir = load_gemma4_runner(
            str(model_dir),
            max_seq_len=spec.max_seq_len,
            dtype_name=spec.dtype,
            tile_size=tile_arg,
            tp_size=tp_arg,
            enable_xla_cache=enable_xla_cache,
            weight_quant=spec.weight_quant,
        )
        return ForgeServeSession(
            spec,
            target_runner=target_runner,
            block_runner=None,
            is_gemma4=True,
            model_dir=model_dir,
            draft_dir=None,
            ast_seed_path=ast_seed_path,
            search_roots=search_roots,
            warmup=warmup,
        )

    if spec.draft_model_id:
        draft_resolved = resolve_model_dir(
            spec.draft_model_id, search_roots=search_roots, allow_download=allow_download
        )
        qwen_block, model_dir, draft_dir = load_qwen_block_tree_runner(
            str(model_dir),
            str(draft_resolved),
            max_seq_len=spec.max_seq_len,
            dtype_name=spec.dtype,
            tile_size=tile_arg,
            tp_size=tp_arg,
            top_k=spec.top_k,
            enable_xla_cache=enable_xla_cache,
            weight_quant=spec.weight_quant,
            draft_weight_quant=spec.weight_quant,
        )
        return ForgeServeSession(
            spec,
            target_runner=qwen_block.target,
            block_runner=qwen_block,
            is_gemma4=False,
            model_dir=model_dir,
            draft_dir=draft_dir,
            ast_seed_path=ast_seed_path,
            search_roots=search_roots,
            warmup=warmup,
        )

    qwen_target, model_dir = load_qwen_runner(
        str(model_dir),
        max_seq_len=spec.max_seq_len,
        dtype_name=spec.dtype,
        tile_size=tile_arg,
        tp_size=tp_arg,
        enable_xla_cache=enable_xla_cache,
        weight_quant=spec.weight_quant,
    )
    return ForgeServeSession(
        spec,
        target_runner=qwen_target,
        block_runner=None,
        is_gemma4=False,
        model_dir=model_dir,
        draft_dir=None,
        ast_seed_path=ast_seed_path,
        search_roots=search_roots,
        warmup=warmup,
    )


class ForgeServeHTTPServer(ThreadingHTTPServer):
    """Multi-threaded HTTP server holding a single ForgeServeSession with exclusive port binding on Windows."""

    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

    def __init__(
        self,
        server_address: tuple[str, int],
        session: ForgeServeSession,
    ) -> None:
        self.session = session
        super().__init__(server_address, _ForgeServeHandler)

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])


def _validate_seed_token_sequences(
    raw_seqs: Any, vocab_size: int
) -> list[list[int]] | None:
    if raw_seqs is None:
        return None
    if isinstance(raw_seqs, (str, bytes)) or not isinstance(raw_seqs, Sequence):
        raise TypeError("seed_token_sequences must be a list of token-ID lists")
    out: list[list[int]] = []
    for idx, seq in enumerate(raw_seqs):
        if isinstance(seq, (str, bytes)) or not isinstance(seq, Sequence):
            raise TypeError(f"seed_token_sequences[{idx}] must be a list of integers")
        out.append(
            [
                _parse_strict_int(tok, f"seed_token_sequences[{idx}]", minimum=0, maximum=vocab_size - 1)
                for tok in seq
            ]
        )
    return out


class _ForgeServeHandler(BaseHTTPRequestHandler):
    """HTTP/1.1 handler for Live Systolic HUD, status/seed APIs, and OpenAI /v1 endpoints."""

    server: ForgeServeHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", raw)

    def _send_error_json(self, status: int, message: str, err_type: str = "invalid_request_error") -> None:
        self._send_json(
            status,
            {"error": {"message": message, "type": err_type, "code": status}},
        )

    def _read_json_body(self) -> dict[str, Any]:
        length_str = self.headers.get("Content-Length")
        if not length_str:
            raise ValueError("Missing Content-Length header")
        try:
            length = int(length_str)
        except ValueError as exc:
            raise ValueError(f"Invalid Content-Length header: {length_str!r}") from exc
        if length < 0 or length > 8 * 1024 * 1024:
            raise ValueError("Request body size out of bounds")
        body = self.rfile.read(length)
        parsed = json.loads(body.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise TypeError("JSON body must be an object")
        return parsed

    def _prepare_chat_request(
        self, session: ForgeServeSession, body: dict[str, Any]
    ) -> tuple[str, list[int], int, bool, str | None, list[list[int]] | None]:
        raw_model = body.get("model")
        if raw_model is None:
            req_model = session.spec.preset_name
        elif isinstance(raw_model, str) and raw_model.strip():
            req_model = raw_model.strip()
        else:
            raise ValueError("model must be a non-empty string")

        max_tokens = _parse_strict_int(
            body.get("max_tokens", 128), "max_tokens", minimum=1, maximum=session.max_seq_len
        )
        stream = _parse_strict_bool(body.get("stream", False), "stream")
        enable_thinking = _parse_strict_bool(body.get("enable_thinking", False), "enable_thinking")
        if body.get("temperature") is not None:
            _parse_strict_float(body["temperature"], "temperature", minimum=0.0, maximum=2.0)

        mode = body.get("mode")
        if mode is not None and (not isinstance(mode, str) or not mode.strip()):
            raise ValueError("mode must be a non-empty string")

        vocab_size = int(session.config.vocab_size)
        if "prompt_tokens" in body:
            raw_ptoks = body["prompt_tokens"]
            if isinstance(raw_ptoks, (str, bytes)) or not isinstance(raw_ptoks, Sequence):
                raise TypeError("prompt_tokens must be a list of integers")
            if not raw_ptoks:
                raise ValueError("prompt_tokens cannot be empty")
            prompt_tokens = [
                _parse_strict_int(t, "prompt_tokens[]", minimum=0, maximum=vocab_size - 1)
                for t in raw_ptoks
            ]
        elif "messages" in body:
            prompt_tokens = session.encode_messages(
                body["messages"], enable_thinking=enable_thinking
            )
        elif "prompt" in body:
            raw_prompt = body["prompt"]
            if not isinstance(raw_prompt, str) or not raw_prompt.strip():
                raise ValueError("prompt must be a non-empty string")
            prompt_tokens = session.encode_messages(
                [{"role": "user", "content": raw_prompt}],
                enable_thinking=enable_thinking,
            )
        else:
            raise ValueError("Request must include 'messages', 'prompt', or 'prompt_tokens'")

        seed_seqs = _validate_seed_token_sequences(body.get("seed_token_sequences"), vocab_size)
        raw_seed_texts = (
            body.get("seed_texts") if "seed_texts" in body else body.get("seed_snippets")
        )
        if seed_seqs is None and raw_seed_texts is not None:
            if isinstance(raw_seed_texts, (str, bytes)) or not isinstance(raw_seed_texts, Sequence):
                raise TypeError("seed_texts must be a list of strings")
            if len(raw_seed_texts) > 256:
                raise ValueError("seed_texts cannot exceed 256 entries")
            for idx, txt in enumerate(raw_seed_texts):
                if not isinstance(txt, str):
                    raise TypeError(f"seed_texts[{idx}] must be a string")
            seed_seqs = encode_with_tokenizer(raw_seed_texts, session.tokenizer)

        return (
            req_model,
            prompt_tokens,
            max_tokens,
            stream,
            mode.strip() if isinstance(mode, str) else None,
            seed_seqs,
        )

    def do_OPTIONS(self) -> None:
        try:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self._send_cors_headers()
            self.end_headers()
        except _CLIENT_DISCONNECT_ERRORS:
            self.close_connection = True

    def do_GET(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        session = self.server.session

        if path == "/" or path == "/index.html":
            html_bytes = render_hud_html(session.status_dict()).encode("utf-8")
            self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html_bytes)
            return

        if path == "/health":
            payload = {"status": "ok", **session.status_dict()}
            self._send_json(HTTPStatus.OK, payload)
            return

        if path == "/api/status":
            self._send_json(HTTPStatus.OK, session.status_dict())
            return

        if path == "/api/local-models":
            models = session.refresh_local_models()
            roots = (
                list(session._search_roots)
                if session._search_roots is not None
                else default_local_model_roots()
            )
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "models": models,
                    "count": len(models),
                    "search_roots": [str(r) for r in roots],
                },
            )
            return

        if path == "/v1/models":
            st = session.status_dict()
            created = int(time.time())
            items = [
                {
                    "id": st["preset"],
                    "object": "model",
                    "created": created,
                    "owned_by": "tpuforge",
                    "tpuforge_model_id": st["model_id"],
                    "tpuforge_tile_size": st["tile_size"],
                }
            ]
            for name in PRESET_CATALOG:
                if name != st["preset"]:
                    items.append(
                        {
                            "id": name,
                            "object": "model",
                            "created": created,
                            "owned_by": "tpuforge-preset",
                        }
                    )
            self._send_json(HTTPStatus.OK, {"object": "list", "data": items})
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}", "not_found_error")

    def do_POST(self) -> None:
        path = urllib.parse.urlparse(self.path).path
        session = self.server.session

        if path == "/api/seed":
            try:
                body = self._read_json_body()
                updated = session.update_ast_seed(
                    ast_seed_path=body.get("ast_seed_path"),
                    repo_path=body.get("repo_path"),
                    seed_texts=body.get("seed_texts"),
                    snippets=body.get("snippets"),
                    append=body.get("append", False),
                    reset=body.get("reset"),
                )
            except FileNotFoundError as exc:
                self._send_error_json(HTTPStatus.NOT_FOUND, str(exc), "not_found_error")
                return
            except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_json(HTTPStatus.OK, updated)
            return

        if path == "/v1/chat/completions":
            try:
                body = self._read_json_body()
                (
                    req_model,
                    prompt_tokens,
                    max_tokens,
                    stream,
                    mode,
                    seed_seqs,
                ) = self._prepare_chat_request(session, body)

                gen_iter = session.stream_generate(
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    mode=mode,
                    seed_token_sequences=seed_seqs,
                )
                first_or_summary = next(gen_iter)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except (RuntimeError, LookupError, ArithmeticError, OSError) as exc:
                self._send_error_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR, str(exc), "internal_server_error"
                )
                return

            completion_id = f"chatcmpl-tpuforge-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if not stream:
                summary: GenerationSummary | None = (
                    first_or_summary if isinstance(first_or_summary, GenerationSummary) else None
                )
                for item in gen_iter:
                    if isinstance(item, GenerationSummary):
                        summary = item
                assert summary is not None
                resp = {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": req_model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": summary.text},
                            "finish_reason": summary.finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": summary.prompt_tokens,
                        "completion_tokens": summary.completion_tokens,
                        "total_tokens": summary.total_tokens,
                    },
                    "token_ids": summary.token_ids,
                    "tpuforge_telemetry": asdict(summary),
                }
                self._send_json(HTTPStatus.OK, resp)
                return

            try:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self._send_cors_headers()
                self.end_headers()
            except _CLIENT_DISCONNECT_ERRORS:
                self.close_connection = True
                return

            def _write_sse(payload: Any) -> None:
                line = "data: " + (
                    payload if isinstance(payload, str) else json.dumps(payload)
                ) + "\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            def _emit_sse_item(evt: StepPulse | GenerationSummary) -> None:
                if isinstance(evt, StepPulse):
                    chunk_payload: dict[str, Any] = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": ({"content": evt.text_delta} if evt.text_delta else {}),
                                "finish_reason": None,
                            }
                        ],
                    }
                    if evt.drafted_nodes > 0 or evt.accepted_rows:
                        chunk_payload["tpuforge_step"] = asdict(evt)
                    _write_sse(chunk_payload)
                elif isinstance(evt, GenerationSummary):
                    _write_sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": req_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": evt.finish_reason,
                                }
                            ],
                            "usage": {
                                "prompt_tokens": evt.prompt_tokens,
                                "completion_tokens": evt.completion_tokens,
                                "total_tokens": evt.total_tokens,
                            },
                            "tpuforge_telemetry": asdict(evt),
                        }
                    )

            try:
                _write_sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": req_model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                )
                _emit_sse_item(first_or_summary)
                for evt in gen_iter:
                    _emit_sse_item(evt)
                _write_sse("[DONE]")
            except _CLIENT_DISCONNECT_ERRORS:
                self.close_connection = True
                return
            except (RuntimeError, LookupError, ArithmeticError, ValueError) as exc:
                try:
                    _write_sse({"error": {"message": str(exc), "type": "server_error"}})
                    _write_sse("[DONE]")
                except _CLIENT_DISCONNECT_ERRORS:
                    pass
                self.close_connection = True
                return
            self.close_connection = True
            return

        self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}", "not_found_error")


def start_serve_background(
    session: ForgeServeSession,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> tuple[ForgeServeHTTPServer, threading.Thread, str]:
    """Start ForgeServeHTTPServer on a background daemon thread and return (server, thread, url)."""
    server = ForgeServeHTTPServer((host, int(port)), session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{server.bound_port}"
    return server, thread, url


def launch_colab_hud(
    preset_or_session: str | ForgeServeSession = "gemma4:12b",
    *,
    port: int = 8000,
    height: int = 760,
    host: str = "127.0.0.1",
    **load_kwargs: Any,
) -> tuple[ForgeServeHTTPServer, str]:
    """Launch TPUForge Serve in background and embed the Live Systolic HUD inside Google Colab."""
    session = (
        preset_or_session
        if isinstance(preset_or_session, ForgeServeSession)
        else load_serve_session(preset_or_model=preset_or_session, **load_kwargs)
    )
    server, _thread, url = start_serve_background(session, host=host, port=port)
    try:
        from google.colab import output  # type: ignore[import-not-found]

        output.serve_kernel_port_as_iframe(server.bound_port, height=height)
    except ImportError:
        pass
    print(f"TPUForge Serve Live Systolic HUD ready at: {url}")
    print(f"OpenAI Base URL: {url}/v1")
    return server, url


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `python -m benchmarks.tpuforge_serve`."""
    parser = argparse.ArgumentParser(
        description="TPUForge Serve: Ollama-style preset launcher, OpenAI HTTP API, and Live Systolic HUD."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="gemma4:12b",
        help="Preset name (gemma4:12b, gemma4:26b, gemma4:31b, qwen3:4b) or HF repo / local path.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all presets with automatic TPU hardware fit detection and exit.",
    )
    parser.add_argument(
        "--list-local",
        action="store_true",
        help="Scan local cache roots for complete safetensors checkpoints and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve preset and hardware placement plan as JSON and exit without loading weights.",
    )
    dl_group = parser.add_mutually_exclusive_group()
    dl_group.add_argument(
        "--local-only",
        action="store_true",
        help="Require target and draft checkpoints to already exist locally; never download from HF Hub.",
    )
    dl_group.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow falling back to Hugging Face Hub download when a model is absent locally.",
    )
    parser.add_argument("--draft-model", default=None, help="Override DFlash draft model repo/path.")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Static KV arena sequence length.")
    parser.add_argument("--dtype", default=None, choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--tile-size", default="auto", choices=["auto", "128", "256"])
    parser.add_argument("--tp-size", default="auto")
    parser.add_argument("--weight-quant", default=None, choices=["none", "int8", "fp8"])
    parser.add_argument("--host-store", default=None, choices=["auto", "hbm", "mmap", "dram"])
    parser.add_argument("--stage-experts", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ast-seed-path", default=None, help="Initial repository/file AST seed path.")
    parser.add_argument("--no-xla-cache", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    if args.stage_experts < 1:
        parser.error("--stage-experts must be >= 1")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    tp_clean = str(args.tp_size).strip().lower()
    if tp_clean != "auto":
        try:
            tp_val = int(tp_clean)
        except ValueError:
            parser.error("--tp-size must be 'auto' or a positive integer")
        if tp_val < 1:
            parser.error("--tp-size must be >= 1")

    if args.list_presets:
        print(format_preset_catalog_table())
        return 0
    if args.list_local:
        print(format_local_models_table())
        return 0

    try:
        spec = resolve_preset_spec(
            args.target,
            draft_model=args.draft_model,
            weight_quant=args.weight_quant,
            dtype=args.dtype,
            host_store=args.host_store,
            tile_size=args.tile_size,
            tp_size=args.tp_size,
            max_seq_len=args.max_seq_len,
            top_k=args.top_k,
            stage_experts=args.stage_experts,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))

    allow_download = not args.local_only
    if args.dry_run:
        payload = asdict(spec)
        payload["allow_download"] = allow_download
        print(json.dumps(payload, indent=2))
        return 0

    session = load_serve_session(
        args.target,
        draft_model=args.draft_model,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        tile_size=args.tile_size,
        tp_size=args.tp_size,
        weight_quant=args.weight_quant,
        host_store=args.host_store,
        stage_experts=args.stage_experts,
        top_k=args.top_k,
        ast_seed_path=args.ast_seed_path,
        enable_xla_cache=not args.no_xla_cache,
        allow_download=allow_download,
        warmup=not args.no_warmup,
    )
    server = ForgeServeHTTPServer((args.host, args.port), session)
    url = f"http://{args.host}:{server.bound_port}"
    print(f"TPUForge Serve running on {url}")
    print(f"  Preset      : {session.spec.preset_name} ({session.spec.model_id})")
    print(f"  Hardware    : {session.spec.hardware.label}")
    print(
        f"  Config      : tile={session.tile_size}, tp={session.target_runner.tp_size}, "
        f"quant={session.target_runner.weight_quant}, store={session.spec.host_store}"
    )
    print(f"  OpenAI Base : {url}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down TPUForge Serve...")
    finally:
        server.server_close()
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

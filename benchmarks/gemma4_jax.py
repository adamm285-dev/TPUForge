"""Pure-JAX + zero-copy safetensors Google Gemma 4 runner (12B dense, 26B-A4B MoE, 31B dense)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import _tpuforge_ext as tpu
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from benchmarks.block_diffusion_jax import (
    BlockDiffusionConfig,
    BlockDiffusionWeights,
    DraftWeightsTuple,
    _place_np_sharded,
    commit_draft_context_kv,
    run_dflash_block_step,
)
from benchmarks.moe_streaming_jax import (
    VALID_HOST_STORE_MODES,
    ExpertHostStore,
    normalize_host_store_mode,
)
from benchmarks.quantization_jax import (
    batched_expert_systolic_linear,
    normalize_weight_quant,
    place_np_sharded_preserved,
    pytree_nbytes,
    quant_np_dtype,
    quantize_and_place_2d,
    quantize_np_per_channel,
    scale_spec_for_weight,
    split_np_shard,
    systolic_linear,
)
from benchmarks.qwen_jax import (
    _MASK_BLOCK_VAL,
    RunMetrics,
    _iter_safetensors_index,
    _jnp_to_numpy_dtype,
    _MappedSafetensorsReader,
    _read_indexed_tensor,
    _split_tensor_shard,
    configure_xla_compilation_cache,
    resolve_default_dtype,
    resolve_model_dir,
    resolve_tile_size,
)

_GEMMA4_MODEL_TYPES = ("gemma4", "gemma4_text", "gemma4_unified_text")


def _build_rope_inv_freq(dim: int, theta: float, partial_factor: float) -> np.ndarray:
    """Compute half-dim RoPE inverse frequencies (supporting proportional partial rotation)."""
    half_dim = dim // 2
    rope_angles = max(0, min(half_dim, int(float(partial_factor) * float(dim) // 2.0)))
    if rope_angles == 0:
        return np.zeros((half_dim,), dtype=np.float32)
    exponents = (2.0 * np.arange(rope_angles, dtype=np.float32)) / float(dim)
    inv_rot = (1.0 / (float(theta) ** exponents)).astype(np.float32)
    if rope_angles == half_dim:
        return inv_rot
    return np.concatenate([inv_rot, np.zeros((half_dim - rope_angles,), dtype=np.float32)], axis=0)


def _quantize_expert_bank(arr: np.ndarray, quant_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel quantization across [E, in_dim, out_dim]."""
    q_dtype = quant_np_dtype(quant_mode)
    finfo = np.finfo(q_dtype) if quant_mode == "fp8" else None
    max_q = float(finfo.max) if finfo is not None else 127.0
    w_f32 = arr.astype(np.float32, copy=False)
    max_abs = np.max(np.abs(w_f32), axis=-2)
    scale = np.where(max_abs > 0.0, max_abs / max_q, 1.0).astype(np.float32)
    scaled = w_f32 / scale[..., None, :]
    if quant_mode == "fp8":
        q = np.clip(scaled, -max_q, max_q).astype(q_dtype)
    else:
        q = np.clip(np.rint(scaled), -127.0, 127.0).astype(q_dtype)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)


def _split_tensor_shard_3d_or_nd(arr: np.ndarray, spec: P, rank: int, tp_size: int) -> np.ndarray:
    """Split a NumPy array across arbitrary PartitionSpec axes for a single TP rank."""
    return split_np_shard(arr, spec, rank, tp_size)


def _gemma4_rms_norm(x: jax.Array, weight: jax.Array | None, eps: float) -> jax.Array:
    """Gemma 4 RMSNorm: float32 mean-square, direct * weight (no +1), optional scale-free."""
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(var + eps)
    if weight is not None:
        normed = normed * weight.astype(jnp.float32)
    return normed.astype(x.dtype)


def _apply_gemma4_rope(
    x: jax.Array,
    positions: jax.Array,
    inv_freq: jax.Array,
) -> jax.Array:
    """Apply half-split rotary embeddings with [head_dim // 2] inverse frequencies."""
    orig_dtype = x.dtype
    head_dim = x.shape[-1]
    half = head_dim // 2
    angles = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    cos = jnp.cos(angles)[:, None, :]
    sin = jnp.sin(angles)[:, None, :]
    x_f32 = x.astype(jnp.float32)
    x1 = x_f32[..., :half]
    x2 = x_f32[..., half:]
    rot1 = x1 * cos - x2 * sin
    rot2 = x2 * cos + x1 * sin
    return jnp.concatenate([rot1, rot2], axis=-1).astype(orig_dtype)


def is_gemma4_model_dir(model_dir: Path) -> bool:
    """Return True if model_dir/config.json describes a Google Gemma 4 checkpoint."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    top_type = str(raw.get("model_type", "")).lower()
    if top_type in _GEMMA4_MODEL_TYPES:
        return True
    tc = raw.get("text_config")
    if isinstance(tc, dict) and str(tc.get("model_type", "")).lower() in _GEMMA4_MODEL_TYPES:
        return True
    archs = raw.get("architectures") or []
    return any(str(a).startswith("Gemma4") for a in archs)


def _collect_eos_ids(raw_val: Any, vocab_size: int) -> list[int]:
    if raw_val is None:
        return []
    vals = raw_val if isinstance(raw_val, list) else [raw_val]
    out: list[int] = []
    for item in vals:
        tid = int(item)
        if 0 <= tid < vocab_size:
            out.append(tid)
    return out


@dataclass(frozen=True)
class Gemma4Config:
    """Architecture hyperparameters parsed from Hugging Face Gemma 4 config.json."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    rms_norm_eps: float
    vocab_size: int
    sliding_window: int
    layer_types: tuple[str, ...]
    sliding_rope_theta: float = 10000.0
    sliding_partial_rotary_factor: float = 1.0
    full_rope_theta: float = 1_000_000.0
    full_partial_rotary_factor: float = 0.25
    attention_k_eq_v: bool = False
    tie_word_embeddings: bool = True
    final_logit_softcapping: float | None = 30.0
    enable_moe_block: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    bos_token_id: int = 2
    eos_token_ids: tuple[int, ...] = (1, 106, 50)
    hidden_size_per_layer_input: int = 0
    num_kv_shared_layers: int = 0

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Duck-typed compatibility with ExpertHostStore."""
        return bool(
            self.enable_moe_block
            and self.num_experts > 0
            and 0 <= layer_idx < self.num_hidden_layers
        )

    def periodic_block_params(self) -> tuple[int, int] | None:
        """Return (sliding_per_block, num_blocks) if layer_types repeats (S*k, F)*B."""
        if not self.layer_types or self.num_hidden_layers < 2:
            return None
        try:
            first_full = self.layer_types.index("full_attention")
        except ValueError:
            return None
        if first_full < 1:
            return None
        block_len = first_full + 1
        if self.num_hidden_layers % block_len != 0:
            return None
        expected_block = ("sliding_attention",) * first_full + ("full_attention",)
        num_blocks = self.num_hidden_layers // block_len
        if self.layer_types != expected_block * num_blocks:
            return None
        return first_full, num_blocks

    def sliding_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, lt in enumerate(self.layer_types) if lt == "sliding_attention")

    def full_layer_indices(self) -> tuple[int, ...]:
        return tuple(i for i, lt in enumerate(self.layer_types) if lt == "full_attention")

    @classmethod
    def from_dir(cls, model_dir: Path) -> Gemma4Config:
        cfg_path = model_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        root = json.loads(cfg_path.read_text(encoding="utf-8"))
        tc = root.get("text_config") if isinstance(root.get("text_config"), dict) else root

        ple_dim = int(tc.get("hidden_size_per_layer_input") or 0)
        kv_shared = int(tc.get("num_kv_shared_layers") or 0)
        if ple_dim != 0 or kv_shared != 0:
            raise ValueError(
                "Gemma 4 Per-Layer Embeddings / KV-sharing (E2B/E4B) not supported in TPUForge yet: "
                f"hidden_size_per_layer_input={ple_dim}, num_kv_shared_layers={kv_shared}"
            )

        hidden_size = int(tc["hidden_size"])
        intermediate_size = int(tc["intermediate_size"])
        num_layers = int(tc["num_hidden_layers"])
        num_heads = int(tc["num_attention_heads"])
        num_kv = int(tc["num_key_value_heads"])
        num_global_kv = int(tc.get("num_global_key_value_heads") or num_kv)
        head_dim = int(tc.get("head_dim") or (hidden_size // num_heads))
        global_head_dim = int(tc.get("global_head_dim") or head_dim)
        rms_norm_eps = float(tc.get("rms_norm_eps", 1e-6))
        vocab_size = int(tc.get("vocab_size", 262144))
        sliding_window = int(tc.get("sliding_window") or 1024)
        attention_k_eq_v = bool(tc.get("attention_k_eq_v", False))
        tie_word = bool(tc.get("tie_word_embeddings", root.get("tie_word_embeddings", True)))

        raw_cap = tc.get("final_logit_softcapping", 30.0)
        softcap = float(raw_cap) if raw_cap is not None and float(raw_cap) > 0.0 else None

        enable_moe = bool(tc.get("enable_moe_block", False))
        num_experts = int(tc.get("num_experts") or 0)
        top_k = int(tc.get("top_k_experts") or tc.get("num_experts_per_tok") or 0)
        moe_inter = int(tc.get("moe_intermediate_size") or 0)
        if enable_moe and (num_experts <= 0 or top_k <= 0 or moe_inter <= 0):
            raise ValueError(
                f"Invalid Gemma 4 MoE parameters: num_experts={num_experts}, "
                f"top_k_experts={top_k}, moe_intermediate_size={moe_inter}"
            )

        raw_types = tc.get("layer_types")
        if raw_types is None:
            layer_types = tuple(
                "full_attention" if (i % 6 == 5) else "sliding_attention" for i in range(num_layers)
            )
        else:
            layer_types = tuple(str(x) for x in raw_types)
        if len(layer_types) != num_layers:
            raise ValueError(
                f"Gemma 4 layer_types length ({len(layer_types)}) != num_hidden_layers ({num_layers})"
            )
        for lt in layer_types:
            if lt not in ("sliding_attention", "full_attention"):
                raise ValueError(f"Unsupported Gemma 4 layer_type: {lt!r}")

        rope_params = tc.get("rope_parameters") or {}
        if not isinstance(rope_params, dict):
            rope_params = {}
        slide_rp = rope_params.get("sliding_attention")
        full_rp = rope_params.get("full_attention")
        if not isinstance(slide_rp, dict):
            slide_rp = rope_params
        if not isinstance(full_rp, dict):
            full_rp = rope_params

        sliding_theta = float(slide_rp.get("rope_theta", tc.get("rope_theta", 10000.0)))
        sliding_partial = float(slide_rp.get("partial_rotary_factor", 1.0))
        full_theta = float(full_rp.get("rope_theta", tc.get("rope_theta", 1_000_000.0)))
        full_partial = float(full_rp.get("partial_rotary_factor", 0.25))

        eos_list: list[int] = []
        for candidate in (
            tc.get("eos_token_id"),
            root.get("eos_token_id"),
        ):
            for tid in _collect_eos_ids(candidate, vocab_size):
                if tid not in eos_list:
                    eos_list.append(tid)
        gen_cfg_path = model_dir / "generation_config.json"
        if gen_cfg_path.is_file():
            gen_cfg = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            for tid in _collect_eos_ids(gen_cfg.get("eos_token_id"), vocab_size):
                if tid not in eos_list:
                    eos_list.append(tid)
        if not eos_list:
            eos_list = [1]

        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv,
            num_global_key_value_heads=num_global_kv,
            head_dim=head_dim,
            global_head_dim=global_head_dim,
            rms_norm_eps=rms_norm_eps,
            vocab_size=vocab_size,
            sliding_window=sliding_window,
            layer_types=layer_types,
            sliding_rope_theta=sliding_theta,
            sliding_partial_rotary_factor=sliding_partial,
            full_rope_theta=full_theta,
            full_partial_rotary_factor=full_partial,
            attention_k_eq_v=attention_k_eq_v,
            tie_word_embeddings=tie_word,
            final_logit_softcapping=softcap,
            enable_moe_block=enable_moe,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
            moe_intermediate_size=moe_inter,
            bos_token_id=int(tc.get("bos_token_id", root.get("bos_token_id", 2))),
            eos_token_ids=tuple(eos_list),
            hidden_size_per_layer_input=ple_dim,
            num_kv_shared_layers=kv_shared,
        )


def resolve_gemma4_tp_mesh(
    tp_size: int | str = "auto",
    config: Gemma4Config | None = None,
) -> tuple[int, Mesh | None]:
    """Resolve Megatron TP degree for Gemma 4 (replicates global KV heads when < tp_size)."""
    devs = jax.devices()
    num_devs = len(devs)

    if isinstance(tp_size, bool):
        raise ValueError("tp_size cannot be bool")  # noqa: TRY004

    def _supports_tp(cand: int) -> bool:
        if config is None:
            return True
        if config.num_attention_heads % cand != 0 or config.intermediate_size % cand != 0:
            return False
        return not (config.enable_moe_block and config.moe_intermediate_size % cand != 0)

    if isinstance(tp_size, str):
        norm = tp_size.strip().lower()
        if norm == "auto":
            if jax.default_backend() != "tpu" or num_devs <= 1:
                return 1, None
            candidates = [c for c in (16, 8, 4, 2, 1) if c <= num_devs]
            if num_devs not in candidates:
                candidates.insert(0, num_devs)
            resolved_tp = 1
            for cand in sorted(candidates, reverse=True):
                if _supports_tp(cand):
                    resolved_tp = cand
                    break
        else:
            resolved_tp = int(norm)
    else:
        resolved_tp = int(tp_size)

    if resolved_tp < 1 or resolved_tp > num_devs:
        raise ValueError(
            f"Requested tp_size={resolved_tp}, but only {num_devs} JAX device(s) are visible"
        )
    if resolved_tp > 1 and not _supports_tp(resolved_tp):
        raise ValueError(
            f"tp_size={resolved_tp} must divide num_attention_heads={config.num_attention_heads}, "  # type: ignore[union-attr]
            f"intermediate_size={config.intermediate_size}"  # type: ignore[union-attr]
            + (
                f", and moe_intermediate_size={config.moe_intermediate_size}"  # type: ignore[union-attr]
                if config and config.enable_moe_block
                else ""
            )
        )
    if resolved_tp == 1:
        return 1, None
    mesh = Mesh(np.array(devs[:resolved_tp]), axis_names=("tp",))
    return resolved_tp, mesh


def encode_gemma4_chat_messages(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Encode chat messages using Gemma 4 canonical `<|turn>...<turn|>` turn markers."""
    sys_parts: list[str] = []
    conv: list[tuple[str, str]] = []
    for msg in messages:
        role = str(msg["role"]).strip().lower()
        content = str(msg["content"])
        if role == "system" and not conv:
            sys_parts.append(content)
        else:
            mapped = "model" if role == "assistant" else role
            conv.append((mapped, content))

    chunks: list[str] = []
    if sys_parts or enable_thinking:
        sys_prefix = "<|think|>\n" if enable_thinking else ""
        sys_body = "\n".join(sys_parts)
        chunks.append(f"<|turn>system\n{sys_prefix}{sys_body}<turn|>\n")
    for role, content in conv:
        chunks.append(f"<|turn>{role}\n{content}<turn|>\n")
    chunks.append("<|turn>model\n")
    if not enable_thinking:
        chunks.append("<|channel>thought\n<channel|>")
    else:
        chunks.append("<|channel>thought\n")

    body_text = "".join(chunks)
    bos_id = tokenizer.token_to_id("<bos>") if hasattr(tokenizer, "token_to_id") else None
    full_text = f"<bos>{body_text}"
    ids = [int(t) for t in tokenizer.encode(full_text, add_special_tokens=False).ids]
    if isinstance(bos_id, int) and (not ids or ids[0] != bos_id):
        body_ids = [int(t) for t in tokenizer.encode(body_text, add_special_tokens=False).ids]
        return [int(bos_id), *body_ids]
    return ids


class _Gemma4GroupScanTuple(NamedTuple):
    """Constant-free grouped parameters for sliding [B, k, ...] or full [B, ...] scan."""

    in_norm: jax.Array
    post_attn_norm: jax.Array
    pre_ffn_norm: jax.Array
    post_ffn_norm: jax.Array
    layer_scalar: jax.Array
    wq: jax.Array
    q_norm: jax.Array
    wk: jax.Array
    k_norm: jax.Array
    wv: jax.Array | None
    wo: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    wq_scale: jax.Array | None
    wk_scale: jax.Array | None
    wv_scale: jax.Array | None
    wo_scale: jax.Array | None
    w_gate_scale: jax.Array | None
    w_up_scale: jax.Array | None
    w_down_scale: jax.Array | None
    post_ffn_norm_1: jax.Array | None = None
    pre_ffn_norm_2: jax.Array | None = None
    post_ffn_norm_2: jax.Array | None = None
    router_proj: jax.Array | None = None
    router_scale: jax.Array | None = None
    per_expert_scale: jax.Array | None = None
    exp_gate: jax.Array | None = None
    exp_up: jax.Array | None = None
    exp_down: jax.Array | None = None
    exp_gate_scale: jax.Array | None = None
    exp_up_scale: jax.Array | None = None
    exp_down_scale: jax.Array | None = None


class _Gemma4PeriodicTuple(NamedTuple):
    """Constant-free weights pytree for periodic Gemma 4 target execution."""

    embed: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array | None
    lm_head_scale: jax.Array | None
    sliding_inv_freq: jax.Array
    full_inv_freq: jax.Array
    sliding_group: _Gemma4GroupScanTuple
    full_group: _Gemma4GroupScanTuple


class _Gemma4LayerListTuple(NamedTuple):
    """Fallback constant-free weights pytree for non-periodic synthetic Gemma 4 models."""

    embed: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array | None
    lm_head_scale: jax.Array | None
    sliding_inv_freq: jax.Array
    full_inv_freq: jax.Array
    layers: tuple[_Gemma4GroupScanTuple, ...]


def _detect_gemma4_prefix(tensor_loc: dict[str, Any]) -> str:
    for candidate in ("model.language_model.", "model.", ""):
        if f"{candidate}embed_tokens.weight" in tensor_loc:
            return candidate
    raise KeyError("Could not find embed_tokens.weight with Gemma 4 prefix in checkpoint")


def _load_group_weights(
    reader: _MappedSafetensorsReader,
    tensor_loc: dict[str, tuple[Path, str, tuple[int, ...], int, int]],
    prefix: str,
    indices_nd: np.ndarray,
    *,
    has_v_proj: bool,
    enable_moe: bool,
    include_experts_in_hbm: bool,
    kv_shardable: bool,
    dtype: jnp.dtype,
    np_dtype: Any,
    norm_quant: str,
    mesh: Mesh | None,
    num_experts: int,
    moe_inter: int,
    hidden_size: int,
) -> _Gemma4GroupScanTuple:
    """Stream stacked layer groups ([B, k, ...] or [B, ...]) directly onto device/mesh."""
    tp_size = mesh.size if mesh is not None else 1
    lead_shape = indices_nd.shape
    lead_axes = (None,) * len(lead_shape)
    flat_indices = [int(i) for i in indices_nd.reshape(-1)]

    def to_jnp(arr: np.ndarray, spec: P) -> jax.Array:
        return _place_np_sharded(arr, spec, mesh, dtype, np_dtype)

    def kv_spec(col_shard: bool) -> P:
        if tp_size <= 1 or not kv_shardable:
            return P(*lead_axes)
        return P(*lead_axes, None, "tp") if col_shard else P(*lead_axes, "tp", None)

    def stack_vec(suffix: str, *, default_ones_shape: tuple[int, ...] | None = None) -> jax.Array:
        spec = P(*lead_axes)
        key0 = f"{prefix}layers.{flat_indices[0]}.{suffix}"
        if key0 not in tensor_loc:
            if default_ones_shape is None:
                raise KeyError(f"Missing tensor {key0!r} in safetensors checkpoint")
            ones = np.ones(lead_shape + default_ones_shape, dtype=np_dtype)
            return to_jnp(ones, spec)
        arrs = []
        for idx in flat_indices:
            key = f"{prefix}layers.{idx}.{suffix}"
            val = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
            arrs.append(val.reshape(val.shape) if val.ndim > 0 else val.reshape((1,)))
        stacked = np.stack(arrs, axis=0).reshape(lead_shape + arrs[0].shape)
        return to_jnp(stacked, spec)

    def stack_linear(
        suffix: str,
        tail_spec: tuple[str | None, str | None],
        *,
        force_unquantized: bool = False,
    ) -> tuple[jax.Array, jax.Array | None]:
        key0 = f"{prefix}layers.{flat_indices[0]}.{suffix}"
        if key0 not in tensor_loc:
            raise KeyError(f"Missing linear tensor {key0!r} in safetensors checkpoint")
        mat_shape = (tensor_loc[key0][2][1], tensor_loc[key0][2][0])
        full_shape = lead_shape + mat_shape
        eff_quant = "none" if force_unquantized else norm_quant
        full_spec = P(*lead_axes, *tail_spec)

        if eff_quant == "none":
            if mesh is None:
                buf = np.empty((len(flat_indices),) + mat_shape, dtype=np_dtype)
                for pos, idx in enumerate(flat_indices):
                    arr = _read_indexed_tensor(
                        tensor_loc[f"{prefix}layers.{idx}.{suffix}"], np_dtype, reader=reader
                    )
                    buf[pos] = np.ascontiguousarray(arr.T)
                reshaped = buf.reshape(full_shape)
                out = jax.device_put(jnp.asarray(reshaped, dtype=dtype))
                del buf, reshaped
                return out, None

            sharding = NamedSharding(mesh, full_spec)
            all_devs = list(mesh.devices.flat)
            local_devs = set(mesh.local_devices)
            sample = _split_tensor_shard(np.empty(full_shape, dtype=np_dtype), full_spec, 0, tp_size)
            shard_shape = sample.shape
            del sample
            flat_shard = (len(flat_indices),) + shard_shape[len(lead_shape) :]
            mat_spec = P(*tail_spec)
            dev_arrays = []
            for dev in all_devs:
                if dev not in local_devs:
                    continue
                rank = all_devs.index(dev)
                rank_buf = np.empty(flat_shard, dtype=np_dtype)
                for pos, idx in enumerate(flat_indices):
                    arr = _read_indexed_tensor(
                        tensor_loc[f"{prefix}layers.{idx}.{suffix}"], np_dtype, reader=reader
                    )
                    rank_buf[pos] = _split_tensor_shard(
                        np.ascontiguousarray(arr.T), mat_spec, rank, tp_size
                    )
                dev_arrays.append(
                    jax.device_put(jnp.asarray(rank_buf.reshape(shard_shape), dtype=dtype), dev)
                )
                del rank_buf
            return jax.make_array_from_single_device_arrays(full_shape, sharding, dev_arrays), None

        q_dtype = quant_np_dtype(eff_quant)
        scale_shape = lead_shape + (mat_shape[1],)
        w_sharding = NamedSharding(mesh, full_spec) if mesh is not None else None
        s_spec = scale_spec_for_weight(full_spec)
        s_sharding = NamedSharding(mesh, s_spec) if mesh is not None else None

        if mesh is None:
            q_buf = np.empty((len(flat_indices),) + mat_shape, dtype=q_dtype)
            s_buf = np.empty((len(flat_indices), mat_shape[1]), dtype=np.float32)
            for pos, idx in enumerate(flat_indices):
                arr = _read_indexed_tensor(
                    tensor_loc[f"{prefix}layers.{idx}.{suffix}"], None, reader=reader
                )
                qi, si = quantize_np_per_channel(np.ascontiguousarray(arr.T), eff_quant)
                q_buf[pos] = qi
                s_buf[pos] = si
            return (
                place_np_sharded_preserved(q_buf.reshape(full_shape), full_spec, None),
                place_np_sharded_preserved(s_buf.reshape(scale_shape), s_spec, None),
            )

        all_devs = list(mesh.devices.flat)
        local_devs = set(mesh.local_devices)
        w_sample = _split_tensor_shard_3d_or_nd(
            np.empty(full_shape, dtype=q_dtype), full_spec, 0, tp_size
        )
        s_sample = _split_tensor_shard_3d_or_nd(
            np.empty(scale_shape, dtype=np.float32), s_spec, 0, tp_size
        )
        w_shard_shape = w_sample.shape
        s_shard_shape = s_sample.shape
        del w_sample, s_sample

        mat_spec = P(*tail_spec)
        mat_s_spec = scale_spec_for_weight(mat_spec)
        w_arrays, s_arrays = [], []
        for dev in all_devs:
            if dev not in local_devs:
                continue
            rank = all_devs.index(dev)
            w_buf = np.empty((len(flat_indices),) + w_shard_shape[len(lead_shape) :], dtype=q_dtype)
            s_buf = np.empty(
                (len(flat_indices),) + s_shard_shape[len(lead_shape) :], dtype=np.float32
            )
            for pos, idx in enumerate(flat_indices):
                arr = _read_indexed_tensor(
                    tensor_loc[f"{prefix}layers.{idx}.{suffix}"], None, reader=reader
                )
                qi, si = quantize_np_per_channel(np.ascontiguousarray(arr.T), eff_quant)
                w_buf[pos] = _split_tensor_shard(qi, mat_spec, rank, tp_size)
                s_buf[pos] = _split_tensor_shard(si, mat_s_spec, rank, tp_size)
            w_arrays.append(jax.device_put(jnp.asarray(w_buf.reshape(w_shard_shape)), dev))
            s_arrays.append(jax.device_put(jnp.asarray(s_buf.reshape(s_shard_shape)), dev))
            del w_buf, s_buf

        return (
            jax.make_array_from_single_device_arrays(full_shape, w_sharding, w_arrays),
            jax.make_array_from_single_device_arrays(scale_shape, s_sharding, s_arrays),
        )

    def stack_experts() -> tuple[
        jax.Array, jax.Array, jax.Array, jax.Array | None, jax.Array | None, jax.Array | None
    ]:
        e_gu_shape = (num_experts, hidden_size, moe_inter)
        e_d_shape = (num_experts, moe_inter, hidden_size)
        gu_full_shape = lead_shape + e_gu_shape
        d_full_shape = lead_shape + e_d_shape
        gu_spec = P(*lead_axes, None, None, "tp") if tp_size > 1 else P(*lead_axes)
        d_spec = P(*lead_axes, None, "tp", None) if tp_size > 1 else P(*lead_axes)

        def read_layer_experts(layer_idx: int, read_dt: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            base = f"{prefix}layers.{layer_idx}.experts"
            if f"{base}.gate_up_proj" in tensor_loc:
                gu = _read_indexed_tensor(tensor_loc[f"{base}.gate_up_proj"], read_dt, reader=reader)
                dp = _read_indexed_tensor(tensor_loc[f"{base}.down_proj"], read_dt, reader=reader)
                raw_g = np.transpose(gu[:, :moe_inter, :], (0, 2, 1))
                raw_u = np.transpose(gu[:, moe_inter:, :], (0, 2, 1))
                raw_d = np.transpose(dp, (0, 2, 1))
                return (
                    np.ascontiguousarray(raw_g),
                    np.ascontiguousarray(raw_u),
                    np.ascontiguousarray(raw_d),
                )
            gs, us, ds = [], [], []
            for e in range(num_experts):
                eb = f"{base}.{e}"
                gs.append(
                    _read_indexed_tensor(
                        tensor_loc[f"{eb}.gate_proj.weight"], read_dt, reader=reader
                    ).T
                )
                us.append(
                    _read_indexed_tensor(
                        tensor_loc[f"{eb}.up_proj.weight"], read_dt, reader=reader
                    ).T
                )
                ds.append(
                    _read_indexed_tensor(
                        tensor_loc[f"{eb}.down_proj.weight"], read_dt, reader=reader
                    ).T
                )
            return (
                np.ascontiguousarray(np.stack(gs, axis=0)),
                np.ascontiguousarray(np.stack(us, axis=0)),
                np.ascontiguousarray(np.stack(ds, axis=0)),
            )

        if norm_quant == "none":
            if mesh is None:
                bg = np.empty((len(flat_indices),) + e_gu_shape, dtype=np_dtype)
                bu = np.empty((len(flat_indices),) + e_gu_shape, dtype=np_dtype)
                bd = np.empty((len(flat_indices),) + e_d_shape, dtype=np_dtype)
                for pos, idx in enumerate(flat_indices):
                    bg[pos], bu[pos], bd[pos] = read_layer_experts(idx, np_dtype)
                return (
                    jax.device_put(jnp.asarray(bg.reshape(gu_full_shape), dtype=dtype)),
                    jax.device_put(jnp.asarray(bu.reshape(gu_full_shape), dtype=dtype)),
                    jax.device_put(jnp.asarray(bd.reshape(d_full_shape), dtype=dtype)),
                    None,
                    None,
                    None,
                )

            all_devs = list(mesh.devices.flat)
            local_devs = set(mesh.local_devices)
            gu_sample = _split_tensor_shard_3d_or_nd(
                np.empty(gu_full_shape, dtype=np_dtype), gu_spec, 0, tp_size
            )
            d_sample = _split_tensor_shard_3d_or_nd(
                np.empty(d_full_shape, dtype=np_dtype), d_spec, 0, tp_size
            )
            gu_shard = gu_sample.shape
            d_shard = d_sample.shape
            del gu_sample, d_sample
            lay_gu_spec = P(None, None, "tp")
            lay_d_spec = P(None, "tp", None)
            g_arrs, u_arrs, d_arrs = [], [], []
            for dev in all_devs:
                if dev not in local_devs:
                    continue
                rank = all_devs.index(dev)
                bg = np.empty((len(flat_indices),) + gu_shard[len(lead_shape) :], dtype=np_dtype)
                bu = np.empty((len(flat_indices),) + gu_shard[len(lead_shape) :], dtype=np_dtype)
                bd = np.empty((len(flat_indices),) + d_shard[len(lead_shape) :], dtype=np_dtype)
                for pos, idx in enumerate(flat_indices):
                    g_i, u_i, d_i = read_layer_experts(idx, np_dtype)
                    bg[pos] = _split_tensor_shard_3d_or_nd(g_i, lay_gu_spec, rank, tp_size)
                    bu[pos] = _split_tensor_shard_3d_or_nd(u_i, lay_gu_spec, rank, tp_size)
                    bd[pos] = _split_tensor_shard_3d_or_nd(d_i, lay_d_spec, rank, tp_size)
                g_arrs.append(jax.device_put(jnp.asarray(bg.reshape(gu_shard), dtype=dtype), dev))
                u_arrs.append(jax.device_put(jnp.asarray(bu.reshape(gu_shard), dtype=dtype), dev))
                d_arrs.append(jax.device_put(jnp.asarray(bd.reshape(d_shard), dtype=dtype), dev))
            return (
                jax.make_array_from_single_device_arrays(
                    gu_full_shape, NamedSharding(mesh, gu_spec), g_arrs
                ),
                jax.make_array_from_single_device_arrays(
                    gu_full_shape, NamedSharding(mesh, gu_spec), u_arrs
                ),
                jax.make_array_from_single_device_arrays(
                    d_full_shape, NamedSharding(mesh, d_spec), d_arrs
                ),
                None,
                None,
                None,
            )

        q_dtype = quant_np_dtype(norm_quant)
        gu_s_shape = lead_shape + (num_experts, moe_inter)
        d_s_shape = lead_shape + (num_experts, hidden_size)
        gu_s_spec = scale_spec_for_weight(gu_spec)
        d_s_spec = scale_spec_for_weight(d_spec)
        if mesh is None:
            qg = np.empty((len(flat_indices),) + e_gu_shape, dtype=q_dtype)
            qu = np.empty((len(flat_indices),) + e_gu_shape, dtype=q_dtype)
            qd = np.empty((len(flat_indices),) + e_d_shape, dtype=q_dtype)
            sg = np.empty((len(flat_indices), num_experts, moe_inter), dtype=np.float32)
            su = np.empty((len(flat_indices), num_experts, moe_inter), dtype=np.float32)
            sd = np.empty((len(flat_indices), num_experts, hidden_size), dtype=np.float32)
            for pos, idx in enumerate(flat_indices):
                rg, ru, rd = read_layer_experts(idx, None)
                qg[pos], sg[pos] = _quantize_expert_bank(rg, norm_quant)
                qu[pos], su[pos] = _quantize_expert_bank(ru, norm_quant)
                qd[pos], sd[pos] = _quantize_expert_bank(rd, norm_quant)
            return (
                place_np_sharded_preserved(qg.reshape(gu_full_shape), gu_spec, None),
                place_np_sharded_preserved(qu.reshape(gu_full_shape), gu_spec, None),
                place_np_sharded_preserved(qd.reshape(d_full_shape), d_spec, None),
                place_np_sharded_preserved(sg.reshape(gu_s_shape), gu_s_spec, None),
                place_np_sharded_preserved(su.reshape(gu_s_shape), gu_s_spec, None),
                place_np_sharded_preserved(sd.reshape(d_s_shape), d_s_spec, None),
            )

        all_devs = list(mesh.devices.flat)
        local_devs = set(mesh.local_devices)
        lay_gu_spec = P(None, None, "tp")
        lay_d_spec = P(None, "tp", None)
        lay_gu_s_spec = scale_spec_for_weight(lay_gu_spec)
        lay_d_s_spec = scale_spec_for_weight(lay_d_spec)
        g_s = _split_tensor_shard_3d_or_nd(np.empty(gu_full_shape, dtype=q_dtype), gu_spec, 0, tp_size).shape
        d_s = _split_tensor_shard_3d_or_nd(np.empty(d_full_shape, dtype=q_dtype), d_spec, 0, tp_size).shape
        gs_s = _split_tensor_shard_3d_or_nd(np.empty(gu_s_shape, dtype=np.float32), gu_s_spec, 0, tp_size).shape
        ds_s = _split_tensor_shard_3d_or_nd(np.empty(d_s_shape, dtype=np.float32), d_s_spec, 0, tp_size).shape
        ga, ua, da, gsa, usa, dsa = [], [], [], [], [], []
        for dev in all_devs:
            if dev not in local_devs:
                continue
            rank = all_devs.index(dev)
            bg = np.empty((len(flat_indices),) + g_s[len(lead_shape) :], dtype=q_dtype)
            bu = np.empty((len(flat_indices),) + g_s[len(lead_shape) :], dtype=q_dtype)
            bd = np.empty((len(flat_indices),) + d_s[len(lead_shape) :], dtype=q_dtype)
            bsg = np.empty((len(flat_indices),) + gs_s[len(lead_shape) :], dtype=np.float32)
            bsu = np.empty((len(flat_indices),) + gs_s[len(lead_shape) :], dtype=np.float32)
            bsd = np.empty((len(flat_indices),) + ds_s[len(lead_shape) :], dtype=np.float32)
            for pos, idx in enumerate(flat_indices):
                rg, ru, rd = read_layer_experts(idx, None)
                qgi, sgi = _quantize_expert_bank(rg, norm_quant)
                qui, sui = _quantize_expert_bank(ru, norm_quant)
                qdi, sdi = _quantize_expert_bank(rd, norm_quant)
                bg[pos] = _split_tensor_shard_3d_or_nd(qgi, lay_gu_spec, rank, tp_size)
                bu[pos] = _split_tensor_shard_3d_or_nd(qui, lay_gu_spec, rank, tp_size)
                bd[pos] = _split_tensor_shard_3d_or_nd(qdi, lay_d_spec, rank, tp_size)
                bsg[pos] = _split_tensor_shard_3d_or_nd(sgi, lay_gu_s_spec, rank, tp_size)
                bsu[pos] = _split_tensor_shard_3d_or_nd(sui, lay_gu_s_spec, rank, tp_size)
                bsd[pos] = _split_tensor_shard_3d_or_nd(sdi, lay_d_s_spec, rank, tp_size)
            ga.append(jax.device_put(jnp.asarray(bg.reshape(g_s)), dev))
            ua.append(jax.device_put(jnp.asarray(bu.reshape(g_s)), dev))
            da.append(jax.device_put(jnp.asarray(bd.reshape(d_s)), dev))
            gsa.append(jax.device_put(jnp.asarray(bsg.reshape(gs_s)), dev))
            usa.append(jax.device_put(jnp.asarray(bsu.reshape(gs_s)), dev))
            dsa.append(jax.device_put(jnp.asarray(bsd.reshape(ds_s)), dev))
        return (
            jax.make_array_from_single_device_arrays(gu_full_shape, NamedSharding(mesh, gu_spec), ga),
            jax.make_array_from_single_device_arrays(gu_full_shape, NamedSharding(mesh, gu_spec), ua),
            jax.make_array_from_single_device_arrays(d_full_shape, NamedSharding(mesh, d_spec), da),
            jax.make_array_from_single_device_arrays(gu_s_shape, NamedSharding(mesh, gu_s_spec), gsa),
            jax.make_array_from_single_device_arrays(gu_s_shape, NamedSharding(mesh, gu_s_spec), usa),
            jax.make_array_from_single_device_arrays(d_s_shape, NamedSharding(mesh, d_s_spec), dsa),
        )

    wq, wq_scale = stack_linear("self_attn.q_proj.weight", (None, "tp"))
    kv_col_tail = (None, "tp") if (tp_size > 1 and kv_shardable) else (None, None)
    wk, wk_scale = stack_linear("self_attn.k_proj.weight", kv_col_tail)
    wv, wv_scale = (
        stack_linear("self_attn.v_proj.weight", kv_col_tail) if has_v_proj else (None, None)
    )
    wo, wo_scale = stack_linear("self_attn.o_proj.weight", ("tp", None))
    w_gate, w_gate_scale = stack_linear("mlp.gate_proj.weight", (None, "tp"))
    w_up, w_up_scale = stack_linear("mlp.up_proj.weight", (None, "tp"))
    w_down, w_down_scale = stack_linear("mlp.down_proj.weight", ("tp", None))

    in_norm = stack_vec("input_layernorm.weight")
    post_attn_norm = stack_vec("post_attention_layernorm.weight")
    pre_ffn_norm = stack_vec("pre_feedforward_layernorm.weight")
    post_ffn_norm = stack_vec("post_feedforward_layernorm.weight")
    layer_scalar = stack_vec("layer_scalar", default_ones_shape=(1,))
    q_norm = stack_vec("self_attn.q_norm.weight")
    k_norm = stack_vec("self_attn.k_norm.weight")

    if enable_moe:
        post_ffn_norm_1 = stack_vec("post_feedforward_layernorm_1.weight")
        pre_ffn_norm_2 = stack_vec("pre_feedforward_layernorm_2.weight")
        post_ffn_norm_2 = stack_vec("post_feedforward_layernorm_2.weight")
        router_proj, _ = stack_linear("router.proj.weight", (None, None), force_unquantized=True)
        router_scale = stack_vec("router.scale")
        per_expert_scale = stack_vec("router.per_expert_scale")
        if include_experts_in_hbm:
            exp_g, exp_u, exp_d, exp_gs, exp_us, exp_ds = stack_experts()
        else:
            exp_g = exp_u = exp_d = exp_gs = exp_us = exp_ds = None
    else:
        post_ffn_norm_1 = pre_ffn_norm_2 = post_ffn_norm_2 = None
        router_proj = router_scale = per_expert_scale = None
        exp_g = exp_u = exp_d = exp_gs = exp_us = exp_ds = None

    return _Gemma4GroupScanTuple(
        in_norm=in_norm,
        post_attn_norm=post_attn_norm,
        pre_ffn_norm=pre_ffn_norm,
        post_ffn_norm=post_ffn_norm,
        layer_scalar=layer_scalar,
        wq=wq,
        q_norm=q_norm,
        wk=wk,
        k_norm=k_norm,
        wv=wv,
        wo=wo,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        wq_scale=wq_scale,
        wk_scale=wk_scale,
        wv_scale=wv_scale,
        wo_scale=wo_scale,
        w_gate_scale=w_gate_scale,
        w_up_scale=w_up_scale,
        w_down_scale=w_down_scale,
        post_ffn_norm_1=post_ffn_norm_1,
        pre_ffn_norm_2=pre_ffn_norm_2,
        post_ffn_norm_2=post_ffn_norm_2,
        router_proj=router_proj,
        router_scale=router_scale,
        per_expert_scale=per_expert_scale,
        exp_gate=exp_g,
        exp_up=exp_u,
        exp_down=exp_d,
        exp_gate_scale=exp_gs,
        exp_up_scale=exp_us,
        exp_down_scale=exp_ds,
    )


@dataclass
class Gemma4Weights:
    """Device-placed Gemma 4 weights (periodic grouped or fallback per-layer, HBM or trunk-only)."""

    config: Gemma4Config
    dtype: jnp.dtype
    mesh: Mesh | None
    weight_quant: str
    layer_prefix: str
    is_periodic: bool
    weights_tuple: _Gemma4PeriodicTuple | _Gemma4LayerListTuple
    include_experts_in_hbm: bool = True

    @property
    def embed(self) -> jax.Array:
        return self.weights_tuple.embed

    @property
    def lm_head(self) -> jax.Array | None:
        return self.weights_tuple.lm_head

    @property
    def lm_head_scale(self) -> jax.Array | None:
        return self.weights_tuple.lm_head_scale

    @property
    def total_nbytes(self) -> int:
        return pytree_nbytes(self.weights_tuple)

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        config: Gemma4Config,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        weight_quant: str = "none",
        include_experts_in_hbm: bool = True,
    ) -> Gemma4Weights:
        norm_quant = normalize_weight_quant(weight_quant)
        tensor_loc = _iter_safetensors_index(model_dir)
        prefix = _detect_gemma4_prefix(tensor_loc)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        tp_size = mesh.size if mesh is not None else 1

        sliding_kv_shardable = tp_size > 1 and (config.num_key_value_heads % tp_size == 0)
        full_kv_shardable = tp_size > 1 and (config.num_global_key_value_heads % tp_size == 0)

        with _MappedSafetensorsReader() as reader:
            embed_np = _read_indexed_tensor(
                tensor_loc[f"{prefix}embed_tokens.weight"], np_dtype, reader=reader
            )
            embed = _place_np_sharded(embed_np, P(), mesh, dtype, np_dtype)

            final_norm_np = _read_indexed_tensor(
                tensor_loc[f"{prefix}norm.weight"], np_dtype, reader=reader
            )
            final_norm = _place_np_sharded(final_norm_np, P(), mesh, dtype, np_dtype)

            lm_spec = (
                P(None, "tp") if (tp_size > 1 and config.vocab_size % tp_size == 0) else P()
            )
            lm_head: jax.Array | None
            lm_head_scale: jax.Array | None
            if "lm_head.weight" in tensor_loc or f"{prefix}lm_head.weight" in tensor_loc:
                lm_key = "lm_head.weight" if "lm_head.weight" in tensor_loc else f"{prefix}lm_head.weight"
                raw_lm = _read_indexed_tensor(
                    tensor_loc[lm_key], None if norm_quant != "none" else np_dtype, reader=reader
                )
                lm_head, lm_head_scale = quantize_and_place_2d(
                    np.ascontiguousarray(raw_lm.T),
                    lm_spec,
                    mesh,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )
                del raw_lm
            elif config.tie_word_embeddings and norm_quant == "none":
                # Alias raw BF16 embed [vocab, hidden] directly to avoid duplicating 2.62 GiB on 31B.
                lm_head = None
                lm_head_scale = None
            elif config.tie_word_embeddings:
                raw_embed_f32 = _read_indexed_tensor(
                    tensor_loc[f"{prefix}embed_tokens.weight"], None, reader=reader
                )
                lm_head, lm_head_scale = quantize_and_place_2d(
                    np.ascontiguousarray(raw_embed_f32.T),
                    lm_spec,
                    mesh,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )
                del raw_embed_f32
            else:
                raise KeyError("lm_head.weight missing and tie_word_embeddings is False")
            del embed_np

            sliding_inv = jax.device_put(
                jnp.asarray(
                    _build_rope_inv_freq(
                        config.head_dim, config.sliding_rope_theta, config.sliding_partial_rotary_factor
                    ),
                    dtype=jnp.float32,
                )
            )
            full_inv = jax.device_put(
                jnp.asarray(
                    _build_rope_inv_freq(
                        config.global_head_dim, config.full_rope_theta, config.full_partial_rotary_factor
                    ),
                    dtype=jnp.float32,
                )
            )

            periodic = config.periodic_block_params()
            if periodic is not None:
                k_slide, n_blocks = periodic
                block_len = k_slide + 1
                slide_grid = np.array(
                    [[b * block_len + s for s in range(k_slide)] for b in range(n_blocks)],
                    dtype=np.int32,
                )
                full_grid = np.array(
                    [b * block_len + k_slide for b in range(n_blocks)], dtype=np.int32
                )
                sliding_group = _load_group_weights(
                    reader,
                    tensor_loc,
                    prefix,
                    slide_grid,
                    has_v_proj=True,
                    enable_moe=config.enable_moe_block,
                    include_experts_in_hbm=include_experts_in_hbm,
                    kv_shardable=sliding_kv_shardable,
                    dtype=dtype,
                    np_dtype=np_dtype,
                    norm_quant=norm_quant,
                    mesh=mesh,
                    num_experts=config.num_experts,
                    moe_inter=config.moe_intermediate_size,
                    hidden_size=config.hidden_size,
                )
                full_has_v = not config.attention_k_eq_v
                if full_has_v is False:
                    # Honor safetensors if a test explicitly included full-layer v_proj.weight.
                    full_has_v = f"{prefix}layers.{int(full_grid[0])}.self_attn.v_proj.weight" in tensor_loc
                full_group = _load_group_weights(
                    reader,
                    tensor_loc,
                    prefix,
                    full_grid,
                    has_v_proj=full_has_v,
                    enable_moe=config.enable_moe_block,
                    include_experts_in_hbm=include_experts_in_hbm,
                    kv_shardable=full_kv_shardable,
                    dtype=dtype,
                    np_dtype=np_dtype,
                    norm_quant=norm_quant,
                    mesh=mesh,
                    num_experts=config.num_experts,
                    moe_inter=config.moe_intermediate_size,
                    hidden_size=config.hidden_size,
                )
                wt: _Gemma4PeriodicTuple | _Gemma4LayerListTuple = _Gemma4PeriodicTuple(
                    embed=embed,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    lm_head_scale=lm_head_scale,
                    sliding_inv_freq=sliding_inv,
                    full_inv_freq=full_inv,
                    sliding_group=sliding_group,
                    full_group=full_group,
                )
                is_periodic = True
            else:
                groups = []
                for idx, lt in enumerate(config.layer_types):
                    is_full = lt == "full_attention"
                    has_v = (
                        not config.attention_k_eq_v
                        or not is_full
                        or f"{prefix}layers.{idx}.self_attn.v_proj.weight" in tensor_loc
                    )
                    grp = _load_group_weights(
                        reader,
                        tensor_loc,
                        prefix,
                        np.array([idx], dtype=np.int32),
                        has_v_proj=has_v,
                        enable_moe=config.enable_moe_block,
                        include_experts_in_hbm=include_experts_in_hbm,
                        kv_shardable=full_kv_shardable if is_full else sliding_kv_shardable,
                        dtype=dtype,
                        np_dtype=np_dtype,
                        norm_quant=norm_quant,
                        mesh=mesh,
                        num_experts=config.num_experts,
                        moe_inter=config.moe_intermediate_size,
                        hidden_size=config.hidden_size,
                    )
                    # Unwrap length-1 leading layer axis to per-layer shapes.
                    groups.append(
                        _Gemma4GroupScanTuple(
                            *[
                                (arr[0] if isinstance(arr, jax.Array) else None)
                                for arr in grp
                            ]
                        )
                    )
                wt = _Gemma4LayerListTuple(
                    embed=embed,
                    final_norm=final_norm,
                    lm_head=lm_head,
                    lm_head_scale=lm_head_scale,
                    sliding_inv_freq=sliding_inv,
                    full_inv_freq=full_inv,
                    layers=tuple(groups),
                )
                is_periodic = False

        return cls(
            config=config,
            dtype=dtype,
            mesh=mesh,
            weight_quant=norm_quant,
            layer_prefix=prefix,
            is_periodic=is_periodic,
            weights_tuple=wt,
            include_experts_in_hbm=include_experts_in_hbm,
        )


def _compute_gemma4_router(
    x_mid: jax.Array,
    layer: _Gemma4GroupScanTuple,
    cfg: Gemma4Config,
    rep_sharding: NamedSharding | None,
) -> tuple[jax.Array, jax.Array]:
    """Compute top-k expert indices [seq, K] and normalized weights * per_expert_scale."""
    assert layer.router_proj is not None and layer.router_scale is not None
    assert layer.per_expert_scale is not None
    r_norm = _gemma4_rms_norm(x_mid, None, cfg.rms_norm_eps)
    r_scaled = r_norm.astype(jnp.float32) * layer.router_scale.astype(jnp.float32) * (
        float(cfg.hidden_size) ** -0.5
    )
    router_logits = jax.lax.dot_general(
        r_scaled,
        layer.router_proj.astype(jnp.float32),
        (((1,), (0,)), ((), ())),
        precision=jax.lax.Precision.HIGHEST,
    )
    probs = jax.nn.softmax(router_logits, axis=-1)
    top_vals, top_idx = jax.lax.top_k(probs, k=cfg.num_experts_per_tok)
    denom = jnp.sum(top_vals, axis=-1, keepdims=True)
    top_norm = jnp.where(denom > 0.0, top_vals / denom, 0.0)
    top_w = top_norm * layer.per_expert_scale[top_idx].astype(jnp.float32)
    if rep_sharding is not None:
        top_idx = jax.lax.with_sharding_constraint(top_idx, rep_sharding)
        top_w = jax.lax.with_sharding_constraint(top_w, rep_sharding)
    return top_idx.astype(jnp.int32), top_w.astype(jnp.float32)


def _eval_hbm_experts(
    x_exp: jax.Array,
    top_idx: jax.Array,
    top_w: jax.Array,
    layer: _Gemma4GroupScanTuple,
    cfg: Gemma4Config,
    rep_sharding: NamedSharding | None,
    exp_inter_sharding: NamedSharding | None,
) -> jax.Array:
    """Evaluate Gemma 4 GeGLU experts (active gather on seq=1, all-expert batched dot on seq>1)."""
    assert layer.exp_gate is not None and layer.exp_up is not None and layer.exp_down is not None
    seq_q = x_exp.shape[0]
    if seq_q == 1:
        e_idx = top_idx[0]
        wg = layer.exp_gate[e_idx]
        wu = layer.exp_up[e_idx]
        wd = layer.exp_down[e_idx]
        sg = layer.exp_gate_scale[e_idx] if layer.exp_gate_scale is not None else None
        su = layer.exp_up_scale[e_idx] if layer.exp_up_scale is not None else None
        sd = layer.exp_down_scale[e_idx] if layer.exp_down_scale is not None else None
        x_in = jnp.broadcast_to(x_exp[None, :, :], (cfg.num_experts_per_tok, 1, cfg.hidden_size))
        gate = batched_expert_systolic_linear(x_in, wg, sg, x_exp.dtype)
        up = batched_expert_systolic_linear(x_in, wu, su, x_exp.dtype)
        if exp_inter_sharding is not None:
            gate = jax.lax.with_sharding_constraint(gate, exp_inter_sharding)
            up = jax.lax.with_sharding_constraint(up, exp_inter_sharding)
        act = (
            jax.nn.gelu(gate.astype(jnp.float32), approximate=True) * up.astype(jnp.float32)
        ).astype(x_exp.dtype)
        expert_outs = batched_expert_systolic_linear(act, wd, sd, x_exp.dtype)[:, 0, :]
        if rep_sharding is not None:
            expert_outs = jax.lax.with_sharding_constraint(expert_outs, rep_sharding)
        out = jnp.sum(expert_outs.astype(jnp.float32) * top_w[0, :, None], axis=0, keepdims=True)
        return out.astype(x_exp.dtype)

    x_b = jnp.broadcast_to(x_exp[None, :, :], (cfg.num_experts, seq_q, cfg.hidden_size))
    gate = batched_expert_systolic_linear(x_b, layer.exp_gate, layer.exp_gate_scale, x_exp.dtype)
    up = batched_expert_systolic_linear(x_b, layer.exp_up, layer.exp_up_scale, x_exp.dtype)
    if exp_inter_sharding is not None:
        gate = jax.lax.with_sharding_constraint(gate, exp_inter_sharding)
        up = jax.lax.with_sharding_constraint(up, exp_inter_sharding)
    act = (
        jax.nn.gelu(gate.astype(jnp.float32), approximate=True) * up.astype(jnp.float32)
    ).astype(x_exp.dtype)
    all_out = batched_expert_systolic_linear(act, layer.exp_down, layer.exp_down_scale, x_exp.dtype)
    if rep_sharding is not None:
        all_out = jax.lax.with_sharding_constraint(all_out, rep_sharding)
    seq_arange = jnp.arange(seq_q, dtype=jnp.int32)[:, None]
    picked = all_out[top_idx, seq_arange, :].astype(jnp.float32)
    out = jnp.sum(picked * top_w[:, :, None], axis=1)
    return out.astype(x_exp.dtype)


def _gemma4_attn_and_dense_mlp(
    x: jax.Array,
    layer: _Gemma4GroupScanTuple,
    k_cache: jax.Array,
    v_cache: jax.Array,
    positions: jax.Array,
    cache_mask: jax.Array | None,
    tile_mask: jax.Array,
    inv_freq: jax.Array,
    cfg: Gemma4Config,
    num_kv_heads: int,
    head_dim: int,
    rep_sharding: NamedSharding | None,
    head_sharding: NamedSharding | None,
    flat_head_sharding: NamedSharding | None,
    kv_sharding: NamedSharding | None,
    heads_first_sharding: NamedSharding | None,
    kv_cache_sharding: NamedSharding | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run Gemma 4 self-attention (scaling=1.0, k_eq_v, scale-free v_norm) + residual."""
    seq_q = x.shape[0]
    h = _gemma4_rms_norm(x, layer.in_norm, cfg.rms_norm_eps)
    q_raw = systolic_linear(h, layer.wq, w_scale=layer.wq_scale).reshape(
        seq_q, cfg.num_attention_heads, head_dim
    )
    k_raw = systolic_linear(h, layer.wk, w_scale=layer.wk_scale).reshape(
        seq_q, num_kv_heads, head_dim
    )
    if layer.wv is not None:
        v_raw = systolic_linear(h, layer.wv, w_scale=layer.wv_scale).reshape(
            seq_q, num_kv_heads, head_dim
        )
    else:
        v_raw = k_raw

    if head_sharding is not None:
        q_raw = jax.lax.with_sharding_constraint(q_raw, head_sharding)
    if kv_sharding is not None:
        k_raw = jax.lax.with_sharding_constraint(k_raw, kv_sharding)
        v_raw = jax.lax.with_sharding_constraint(v_raw, kv_sharding)

    q = _apply_gemma4_rope(_gemma4_rms_norm(q_raw, layer.q_norm, cfg.rms_norm_eps), positions, inv_freq)
    k_new = _apply_gemma4_rope(
        _gemma4_rms_norm(k_raw, layer.k_norm, cfg.rms_norm_eps), positions, inv_freq
    )
    v_new = _gemma4_rms_norm(v_raw, None, cfg.rms_norm_eps)

    if head_sharding is not None:
        q = jax.lax.with_sharding_constraint(q, head_sharding)
    if kv_sharding is not None:
        k_new = jax.lax.with_sharding_constraint(k_new, kv_sharding)
        v_new = jax.lax.with_sharding_constraint(v_new, kv_sharding)

    k_new_t = jnp.transpose(k_new, (1, 0, 2))
    v_new_t = jnp.transpose(v_new, (1, 0, 2))
    if kv_cache_sharding is not None:
        k_new_t = jax.lax.with_sharding_constraint(k_new_t, kv_cache_sharding)
        v_new_t = jax.lax.with_sharding_constraint(v_new_t, kv_cache_sharding)

    if cache_mask is not None:
        k_all = jnp.concatenate([k_cache, k_new_t], axis=1)
        v_all = jnp.concatenate([v_cache, v_new_t], axis=1)
        full_mask = jnp.concatenate([cache_mask, tile_mask], axis=-1)
    else:
        k_all = k_new_t
        v_all = v_new_t
        full_mask = tile_mask

    repeat = cfg.num_attention_heads // num_kv_heads
    k_exp = jnp.repeat(k_all, repeat, axis=0)
    v_exp = jnp.repeat(v_all, repeat, axis=0)
    q_hf = jnp.transpose(q, (1, 0, 2))
    if heads_first_sharding is not None:
        q_hf = jax.lax.with_sharding_constraint(q_hf, heads_first_sharding)
        k_exp = jax.lax.with_sharding_constraint(k_exp, heads_first_sharding)
        v_exp = jax.lax.with_sharding_constraint(v_exp, heads_first_sharding)

    # Gemma 4 uses scaling = 1.0 (no 1/sqrt(head_dim)).
    attn_logits = jax.lax.dot_general(
        q_hf.astype(jnp.float32),
        k_exp.astype(jnp.float32),
        (((2,), (2,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
    )
    attn_weights = jax.nn.softmax(attn_logits + full_mask[None, :, :], axis=-1)
    ctx_hf = jax.lax.dot_general(
        attn_weights.astype(v_exp.dtype),
        v_exp,
        (((2,), (1,)), ((0,), (0,))),
        preferred_element_type=(
            jnp.float32 if x.dtype in (jnp.bfloat16, jnp.float16) else None
        ),
    ).astype(x.dtype)
    ctx = jnp.transpose(ctx_hf, (1, 0, 2)).reshape(seq_q, cfg.num_attention_heads * head_dim)
    if flat_head_sharding is not None:
        ctx = jax.lax.with_sharding_constraint(ctx, flat_head_sharding)
    attn_proj = systolic_linear(ctx, layer.wo, w_scale=layer.wo_scale)
    if rep_sharding is not None:
        attn_proj = jax.lax.with_sharding_constraint(attn_proj, rep_sharding)
    x_mid = x + _gemma4_rms_norm(attn_proj, layer.post_attn_norm, cfg.rms_norm_eps)
    if rep_sharding is not None:
        x_mid = jax.lax.with_sharding_constraint(x_mid, rep_sharding)
    return x_mid, k_new_t, v_new_t, _eval_dense_geglu(
        _gemma4_rms_norm(x_mid, layer.pre_ffn_norm, cfg.rms_norm_eps), layer, x.dtype, rep_sharding
    )


def _eval_dense_geglu(
    h_in: jax.Array,
    layer: _Gemma4GroupScanTuple,
    out_dtype: Any,
    rep_sharding: NamedSharding | None,
) -> jax.Array:
    gate = systolic_linear(h_in, layer.w_gate, w_scale=layer.w_gate_scale)
    up = systolic_linear(h_in, layer.w_up, w_scale=layer.w_up_scale)
    act = (
        jax.nn.gelu(gate.astype(jnp.float32), approximate=True) * up.astype(jnp.float32)
    ).astype(out_dtype)
    out = systolic_linear(act, layer.w_down, w_scale=layer.w_down_scale)
    if rep_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, rep_sharding)
    return out


def _gemma4_layer_forward(
    x: jax.Array,
    layer: _Gemma4GroupScanTuple,
    k_cache: jax.Array,
    v_cache: jax.Array,
    positions: jax.Array,
    cache_mask: jax.Array | None,
    tile_mask: jax.Array,
    inv_freq: jax.Array,
    cfg: Gemma4Config,
    num_kv_heads: int,
    head_dim: int,
    rep_sharding: NamedSharding | None,
    head_sharding: NamedSharding | None,
    flat_head_sharding: NamedSharding | None,
    kv_sharding: NamedSharding | None,
    heads_first_sharding: NamedSharding | None,
    kv_cache_sharding: NamedSharding | None,
    exp_inter_sharding: NamedSharding | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x_mid, k_new_t, v_new_t, mlp_out = _gemma4_attn_and_dense_mlp(
        x,
        layer,
        k_cache,
        v_cache,
        positions,
        cache_mask,
        tile_mask,
        inv_freq,
        cfg,
        num_kv_heads,
        head_dim,
        rep_sharding,
        head_sharding,
        flat_head_sharding,
        kv_sharding,
        heads_first_sharding,
        kv_cache_sharding,
    )
    if cfg.enable_moe_block:
        assert (
            layer.post_ffn_norm_1 is not None
            and layer.pre_ffn_norm_2 is not None
            and layer.post_ffn_norm_2 is not None
        )
        h1 = _gemma4_rms_norm(mlp_out, layer.post_ffn_norm_1, cfg.rms_norm_eps)
        top_idx, top_w = _compute_gemma4_router(x_mid, layer, cfg, rep_sharding)
        x_exp = _gemma4_rms_norm(x_mid, layer.pre_ffn_norm_2, cfg.rms_norm_eps)
        moe_out = _eval_hbm_experts(
            x_exp, top_idx, top_w, layer, cfg, rep_sharding, exp_inter_sharding
        )
        h2 = _gemma4_rms_norm(moe_out, layer.post_ffn_norm_2, cfg.rms_norm_eps)
        ffn_in = h1 + h2
    else:
        ffn_in = mlp_out

    ffn_out = _gemma4_rms_norm(ffn_in, layer.post_ffn_norm, cfg.rms_norm_eps)
    out = (x_mid + ffn_out) * layer.layer_scalar.astype(x.dtype)
    if rep_sharding is not None:
        out = jax.lax.with_sharding_constraint(out, rep_sharding)
    return out, k_new_t, v_new_t


def _build_cache_masks(
    positions: jax.Array,
    valid_prefix_len: jax.Array,
    max_seq_len: int,
    sliding_window: int,
) -> tuple[jax.Array, jax.Array]:
    """Return (full_cache_mask, sliding_cache_mask) shaped [seq_q, max_seq_len]."""
    cache_pos = jnp.arange(max_seq_len, dtype=jnp.int32)[None, :]
    q_pos = positions[:, None]
    valid_prefix = cache_pos < valid_prefix_len
    valid_full = valid_prefix & (cache_pos <= q_pos)
    diff = q_pos - cache_pos
    valid_slide = valid_full & (diff < jnp.int32(sliding_window))
    full_mask = jnp.where(valid_full, jnp.float32(0.0), jnp.float32(_MASK_BLOCK_VAL))
    slide_mask = jnp.where(valid_slide, jnp.float32(0.0), jnp.float32(_MASK_BLOCK_VAL))
    return full_mask, slide_mask


def _apply_sliding_to_tile_mask(
    positions: jax.Array,
    tile_mask: jax.Array,
    sliding_window: int,
) -> jax.Array:
    diff = positions[:, None] - positions[None, :]
    in_win = (diff >= 0) & (diff < jnp.int32(sliding_window))
    return jnp.where(in_win, tile_mask, jnp.float32(_MASK_BLOCK_VAL))


def _compute_gemma4_logits(
    x_final: jax.Array,
    final_norm: jax.Array,
    embed: jax.Array,
    lm_head: jax.Array | None,
    lm_head_scale: jax.Array | None,
    cfg: Gemma4Config,
) -> jax.Array:
    """Apply final RMSNorm, linear projection (tied or quantized), and softcapping in float32."""
    normed = _gemma4_rms_norm(x_final, final_norm, cfg.rms_norm_eps)
    if lm_head is None:
        if normed.dtype in (jnp.bfloat16, jnp.float16):
            logits = jax.lax.dot_general(
                normed,
                embed,
                (((normed.ndim - 1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            logits = jax.lax.dot_general(
                normed,
                embed,
                (((normed.ndim - 1,), (1,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
            ).astype(jnp.float32)
    else:
        logits = systolic_linear(normed, lm_head, w_scale=lm_head_scale, out_dtype=jnp.float32)
    cap = cfg.final_logit_softcapping
    if cap is not None and cap > 0.0:
        cap_f32 = jnp.float32(cap)
        logits = jnp.tanh(logits / cap_f32) * cap_f32
    return logits


def _commit_kv_pair(
    k_cache: jax.Array,
    v_cache: jax.Array,
    step_k: jax.Array,
    step_v: jax.Array,
    prefix_len: jax.Array,
    src_rows: jax.Array,
    n_valid: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Write [..., H, T, D] step slices into [..., H, max_seq, D] caches."""
    tile_size = step_k.shape[-2]
    max_seq = k_cache.shape[-2]
    gathered_k = jnp.take(step_k, src_rows, axis=-2)
    gathered_v = jnp.take(step_v, src_rows, axis=-2)

    valid_row = jnp.arange(tile_size, dtype=jnp.int32) < n_valid
    dest_cols = prefix_len + jnp.arange(tile_size, dtype=jnp.int32)
    dest_cols = jnp.where(valid_row, dest_cols, jnp.int32(max_seq))

    k_move = jnp.moveaxis(gathered_k, -2, 0)
    v_move = jnp.moveaxis(gathered_v, -2, 0)
    scatter_dims = jax.lax.ScatterDimensionNumbers(
        update_window_dims=tuple(range(1, k_move.ndim)),
        inserted_window_dims=(k_cache.ndim - 2,),
        scatter_dims_to_operand_dims=(k_cache.ndim - 2,),
    )
    k_next = jax.lax.scatter(
        k_cache,
        dest_cols[:, None],
        k_move,
        scatter_dims,
        indices_are_sorted=False,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
    )
    v_next = jax.lax.scatter(
        v_cache,
        dest_cols[:, None],
        v_move,
        scatter_dims,
        indices_are_sorted=False,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.FILL_OR_DROP,
    )
    return k_next, v_next


class _Gemma4ShardingHelper:
    """Shared TP Sharding descriptors + static cache helpers for Gemma4Runner & MoE stream."""

    def __init__(self, cfg: Gemma4Config, mesh: Mesh | None) -> None:
        self.cfg = cfg
        self.mesh = mesh
        self.tp_size = mesh.size if mesh is not None else 1
        sliding_kv_shardable = self.tp_size > 1 and (cfg.num_key_value_heads % self.tp_size == 0)
        full_kv_shardable = self.tp_size > 1 and (cfg.num_global_key_value_heads % self.tp_size == 0)

        def ns(spec: P) -> NamedSharding | None:
            return NamedSharding(mesh, spec) if mesh is not None else None

        self.rep = ns(P())
        self.head = ns(P(None, "tp", None))
        self.flat_head = ns(P(None, "tp"))
        self.heads_first = ns(P("tp", None, None))
        self.exp_inter = ns(P(None, None, "tp"))
        self.sliding_kv = ns(P(None, "tp", None) if sliding_kv_shardable else P())
        self.full_kv = ns(P(None, "tp", None) if full_kv_shardable else P())
        self.sliding_kv_cache = ns(P("tp", None, None) if sliding_kv_shardable else P())
        self.full_kv_cache = ns(P("tp", None, None) if full_kv_shardable else P())
        self.periodic_sliding_cache = ns(
            P(None, None, "tp", None, None) if sliding_kv_shardable else P()
        )
        self.periodic_full_cache = ns(P(None, "tp", None, None) if full_kv_shardable else P())

    def put_rep(self, arr: Any, dtype: Any | None = None) -> jax.Array:
        jarr = jnp.asarray(arr) if dtype is None else jnp.asarray(arr, dtype=dtype)
        return jax.device_put(jarr, self.rep) if self.rep is not None else jax.device_put(jarr)


class Gemma4Runner:
    """Pure-JAX + C++ ForgeEngine Gemma 4 runner (periodic nested scan + fallback)."""

    def __init__(
        self,
        config: Gemma4Config,
        weights: Gemma4Weights,
        tokenizer: Any,
        max_seq_len: int = 1024,
        tile_size: int | str = "auto",
        mesh: Mesh | None = None,
    ) -> None:
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.tile_size = resolve_tile_size(tile_size)
        self.mesh = mesh
        self.tp_size = mesh.size if mesh is not None else 1
        self.dtype = weights.dtype
        self.weight_quant = weights.weight_quant
        self.model_hbm_bytes = weights.total_nbytes
        self._weights_tuple = weights.weights_tuple
        self._sh = _Gemma4ShardingHelper(config, mesh)
        self._embed_scale = float(math.sqrt(config.hidden_size))
        self._periodic = config.periodic_block_params()

        self.vanilla_prefill_compile_count = 0
        self.vanilla_decode_compile_count = 0
        self.static_step_compile_count = 0

        self._compile_kernels()

    def _put_rep(self, arr: Any, dtype: Any | None = None) -> jax.Array:
        return self._sh.put_rep(arr, dtype=dtype)

    def _decode_text(self, token_ids: list[int]) -> str:
        if not token_ids or self.tokenizer is None:
            return ""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _validate_bounds(self, prompt_tokens: list[int], max_new_tokens: int) -> None:
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if len(prompt_tokens) + max_new_tokens > self.max_seq_len - self.tile_size:
            raise ValueError(
                f"prompt_len ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_seq_len - tile_size ({self.max_seq_len - self.tile_size})"
            )

    def _empty_kv_caches(self) -> tuple[Any, Any]:
        cfg = self.config
        if self._periodic is not None:
            k_slide, n_blocks = self._periodic
            s_shape = (n_blocks, k_slide, cfg.num_key_value_heads, self.max_seq_len, cfg.head_dim)
            f_shape = (n_blocks, cfg.num_global_key_value_heads, self.max_seq_len, cfg.global_head_dim)
            sk = jnp.zeros(s_shape, dtype=self.dtype)
            sv = jnp.zeros(s_shape, dtype=self.dtype)
            fk = jnp.zeros(f_shape, dtype=self.dtype)
            fv = jnp.zeros(f_shape, dtype=self.dtype)
            if self._sh.periodic_sliding_cache is not None:
                sk = jax.device_put(sk, self._sh.periodic_sliding_cache)
                sv = jax.device_put(sv, self._sh.periodic_sliding_cache)
            if self._sh.periodic_full_cache is not None:
                fk = jax.device_put(fk, self._sh.periodic_full_cache)
                fv = jax.device_put(fv, self._sh.periodic_full_cache)
            return (sk, fk), (sv, fv)

        ks, vs = [], []
        for lt in cfg.layer_types:
            is_full = lt == "full_attention"
            kv_h = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
            hd = cfg.global_head_dim if is_full else cfg.head_dim
            sh = self._sh.full_kv_cache if is_full else self._sh.sliding_kv_cache
            k_l = jnp.zeros((kv_h, self.max_seq_len, hd), dtype=self.dtype)
            v_l = jnp.zeros((kv_h, self.max_seq_len, hd), dtype=self.dtype)
            if sh is not None:
                k_l = jax.device_put(k_l, sh)
                v_l = jax.device_put(v_l, sh)
            ks.append(k_l)
            vs.append(v_l)
        return tuple(ks), tuple(vs)

    def _forward_trunk(
        self,
        wt: _Gemma4PeriodicTuple | _Gemma4LayerListTuple,
        token_ids: jax.Array,
        positions: jax.Array,
        valid_prefix_len: jax.Array | None,
        tile_mask: jax.Array,
        k_caches: Any,
        v_caches: Any,
        *,
        capture_slots: jax.Array | None = None,
        num_captured: int = 0,
    ) -> tuple[jax.Array, Any, Any, jax.Array | None]:
        cfg = self.config
        sh = self._sh
        x = wt.embed[token_ids] * jnp.asarray(self._embed_scale, dtype=wt.embed.dtype)
        if sh.rep is not None:
            x = jax.lax.with_sharding_constraint(x, sh.rep)

        if valid_prefix_len is not None:
            full_cache_mask, slide_cache_mask = _build_cache_masks(
                positions, valid_prefix_len, self.max_seq_len, cfg.sliding_window
            )
        else:
            full_cache_mask = None
            slide_cache_mask = None
        slide_tile_mask = _apply_sliding_to_tile_mask(positions, tile_mask, cfg.sliding_window)

        seq_q = x.shape[0]
        cap_buf = (
            jnp.zeros((num_captured + 1, seq_q, cfg.hidden_size), dtype=x.dtype)
            if capture_slots is not None
            else None
        )

        if self._periodic is not None:
            assert isinstance(wt, _Gemma4PeriodicTuple)
            k_slide, n_blocks = self._periodic
            sk, fk = k_caches
            sv, fv = v_caches
            block_len = k_slide + 1
            block_ids = jnp.arange(n_blocks, dtype=jnp.int32)

            def outer_body(
                carry: tuple[jax.Array, jax.Array | None],
                scan_in: tuple[
                    _Gemma4GroupScanTuple,
                    _Gemma4GroupScanTuple,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                    jax.Array,
                ],
            ) -> tuple[
                tuple[jax.Array, jax.Array | None],
                tuple[jax.Array, jax.Array, jax.Array, jax.Array],
            ]:
                x_b, c_buf = carry
                sg_b, fg_b, sk_b, sv_b, fk_b, fv_b, b_idx = scan_in
                if sh.rep is not None:
                    x_b = jax.lax.with_sharding_constraint(x_b, sh.rep)
                base_lid = b_idx * block_len
                s_indices = jnp.arange(k_slide, dtype=jnp.int32)

                def inner_slide(
                    inner_carry: tuple[jax.Array, jax.Array | None],
                    s_in: tuple[_Gemma4GroupScanTuple, jax.Array, jax.Array, jax.Array],
                ) -> tuple[tuple[jax.Array, jax.Array | None], tuple[jax.Array, jax.Array]]:
                    x_s, ic_buf = inner_carry
                    sl, sk_s, sv_s, s_off = s_in
                    x_next, ks_new, vs_new = _gemma4_layer_forward(
                        x_s,
                        sl,
                        sk_s,
                        sv_s,
                        positions,
                        slide_cache_mask,
                        slide_tile_mask,
                        wt.sliding_inv_freq,
                        cfg,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        sh.rep,
                        sh.head,
                        sh.flat_head,
                        sh.sliding_kv,
                        sh.heads_first,
                        sh.sliding_kv_cache,
                        sh.exp_inter,
                    )
                    if ic_buf is not None and capture_slots is not None:
                        ic_buf = ic_buf.at[capture_slots[base_lid + s_off]].set(x_next, mode="drop")
                    return (x_next, ic_buf), (ks_new, vs_new)

                (x_after_slide, c_buf), (ks_step_b, vs_step_b) = jax.lax.scan(
                    inner_slide, (x_b, c_buf), (sg_b, sk_b, sv_b, s_indices)
                )
                x_after_full, kf_new, vf_new = _gemma4_layer_forward(
                    x_after_slide,
                    fg_b,
                    fk_b,
                    fv_b,
                    positions,
                    full_cache_mask,
                    tile_mask,
                    wt.full_inv_freq,
                    cfg,
                    cfg.num_global_key_value_heads,
                    cfg.global_head_dim,
                    sh.rep,
                    sh.head,
                    sh.flat_head,
                    sh.full_kv,
                    sh.heads_first,
                    sh.full_kv_cache,
                    sh.exp_inter,
                )
                if c_buf is not None and capture_slots is not None:
                    c_buf = c_buf.at[capture_slots[base_lid + k_slide]].set(
                        x_after_full, mode="drop"
                    )
                return (x_after_full, c_buf), (ks_step_b, vs_step_b, kf_new, vf_new)

            (x_out, cap_out), (sk_steps, sv_steps, fk_steps, fv_steps) = jax.lax.scan(
                outer_body,
                (x, cap_buf),
                (wt.sliding_group, wt.full_group, sk, sv, fk, fv, block_ids),
            )
            step_k: Any = (sk_steps, fk_steps)
            step_v: Any = (sv_steps, fv_steps)
            cap_ctx = (
                jnp.transpose(cap_out[:num_captured], (1, 0, 2)).reshape(
                    seq_q, num_captured * cfg.hidden_size
                )
                if cap_out is not None
                else None
            )
            return x_out, step_k, step_v, cap_ctx

        assert isinstance(wt, _Gemma4LayerListTuple)
        step_ks: list[jax.Array] = []
        step_vs: list[jax.Array] = []
        cur = x
        for lid, (layer, lt, kc, vc) in enumerate(zip(wt.layers, cfg.layer_types, k_caches, v_caches)):
            is_full = lt == "full_attention"
            cur, kn, vn = _gemma4_layer_forward(
                cur,
                layer,
                kc,
                vc,
                positions,
                full_cache_mask if is_full else slide_cache_mask,
                tile_mask if is_full else slide_tile_mask,
                wt.full_inv_freq if is_full else wt.sliding_inv_freq,
                cfg,
                cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads,
                cfg.global_head_dim if is_full else cfg.head_dim,
                sh.rep,
                sh.head,
                sh.flat_head,
                sh.full_kv if is_full else sh.sliding_kv,
                sh.heads_first,
                sh.full_kv_cache if is_full else sh.sliding_kv_cache,
                sh.exp_inter,
            )
            step_ks.append(kn)
            step_vs.append(vn)
            if cap_buf is not None and capture_slots is not None:
                cap_buf = cap_buf.at[capture_slots[lid]].set(cur, mode="drop")
        cap_ctx = (
            jnp.transpose(cap_buf[:num_captured], (1, 0, 2)).reshape(
                seq_q, num_captured * cfg.hidden_size
            )
            if cap_buf is not None
            else None
        )
        return cur, tuple(step_ks), tuple(step_vs), cap_ctx

    def _commit_caches(
        self,
        k_caches: Any,
        v_caches: Any,
        step_k: Any,
        step_v: Any,
        prefix_len: jax.Array,
        src_rows: jax.Array,
        n_valid: jax.Array,
    ) -> tuple[Any, Any]:
        if self._periodic is not None:
            sk_next, sv_next = _commit_kv_pair(
                k_caches[0], v_caches[0], step_k[0], step_v[0], prefix_len, src_rows, n_valid
            )
            fk_next, fv_next = _commit_kv_pair(
                k_caches[1], v_caches[1], step_k[1], step_v[1], prefix_len, src_rows, n_valid
            )
            if self._sh.periodic_sliding_cache is not None:
                sk_next = jax.lax.with_sharding_constraint(sk_next, self._sh.periodic_sliding_cache)
                sv_next = jax.lax.with_sharding_constraint(sv_next, self._sh.periodic_sliding_cache)
            if self._sh.periodic_full_cache is not None:
                fk_next = jax.lax.with_sharding_constraint(fk_next, self._sh.periodic_full_cache)
                fv_next = jax.lax.with_sharding_constraint(fv_next, self._sh.periodic_full_cache)
            return (sk_next, fk_next), (sv_next, fv_next)

        k_out, v_out = [], []
        for kc, vc, sk_l, sv_l, lt in zip(
            k_caches, v_caches, step_k, step_v, self.config.layer_types
        ):
            kn, vn = _commit_kv_pair(kc, vc, sk_l, sv_l, prefix_len, src_rows, n_valid)
            sh = self._sh.full_kv_cache if lt == "full_attention" else self._sh.sliding_kv_cache
            if sh is not None:
                kn = jax.lax.with_sharding_constraint(kn, sh)
                vn = jax.lax.with_sharding_constraint(vn, sh)
            k_out.append(kn)
            v_out.append(vn)
        return tuple(k_out), tuple(v_out)

    def _compile_kernels(self) -> None:
        def vanilla_prefill(
            wt: Any,
            prompt_tokens: jax.Array,
            k_caches: Any,
            v_caches: Any,
        ) -> tuple[jax.Array, Any, Any]:
            self.vanilla_prefill_compile_count += 1
            seq_len = prompt_tokens.shape[0]
            positions = jnp.arange(seq_len, dtype=jnp.int32)
            rows = positions[:, None]
            cols = positions[None, :]
            causal = jnp.where(cols <= rows, jnp.float32(0.0), jnp.float32(_MASK_BLOCK_VAL))
            x_out, step_k, step_v, _ = self._forward_trunk(
                wt, prompt_tokens, positions, None, causal, k_caches, v_caches
            )
            all_rows = jnp.arange(seq_len, dtype=jnp.int32)
            k_next, v_next = self._commit_caches(
                k_caches, v_caches, step_k, step_v, jnp.int32(0), all_rows, jnp.int32(seq_len)
            )
            logits = _compute_gemma4_logits(
                x_out[-1], wt.final_norm, wt.embed, wt.lm_head, wt.lm_head_scale, self.config
            )
            return logits, k_next, v_next

        def vanilla_decode_step(
            wt: Any,
            token_id: jax.Array,
            pos_idx: jax.Array,
            k_caches: Any,
            v_caches: Any,
        ) -> tuple[jax.Array, Any, Any]:
            self.vanilla_decode_compile_count += 1
            tok_arr = token_id.reshape((1,))
            pos_arr = pos_idx.reshape((1,))
            self_mask = jnp.zeros((1, 1), dtype=jnp.float32)
            x_out, step_k, step_v, _ = self._forward_trunk(
                wt, tok_arr, pos_arr, pos_idx, self_mask, k_caches, v_caches
            )
            k_next, v_next = self._commit_caches(
                k_caches, v_caches, step_k, step_v, pos_idx, jnp.zeros((1,), dtype=jnp.int32), jnp.int32(1)
            )
            logits = _compute_gemma4_logits(
                x_out[0], wt.final_norm, wt.embed, wt.lm_head, wt.lm_head_scale, self.config
            )
            return logits, k_next, v_next

        def static_systolic_tile_step(
            wt: Any,
            valid_prefix_len: jax.Array,
            tile_tokens: jax.Array,
            tile_positions: jax.Array,
            tile_mask: jax.Array,
            k_caches: Any,
            v_caches: Any,
        ) -> tuple[jax.Array, Any, Any]:
            self.static_step_compile_count += 1
            x_out, step_k, step_v, _ = self._forward_trunk(
                wt,
                tile_tokens,
                tile_positions,
                valid_prefix_len,
                tile_mask,
                k_caches,
                v_caches,
            )
            logits = _compute_gemma4_logits(
                x_out, wt.final_norm, wt.embed, wt.lm_head, wt.lm_head_scale, self.config
            )
            argmax_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            return argmax_tokens, step_k, step_v

        def commit_kv_step(
            k_caches: Any,
            v_caches: Any,
            step_k: Any,
            step_v: Any,
            prev_prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[Any, Any]:
            return self._commit_caches(
                k_caches, v_caches, step_k, step_v, prev_prefix_len, src_rows, n_valid
            )

        self._vanilla_prefill_jit = jax.jit(vanilla_prefill)
        self._vanilla_decode_step_jit = jax.jit(vanilla_decode_step)
        self._static_systolic_tile_step_jit = jax.jit(static_systolic_tile_step)
        self._commit_kv_jit = jax.jit(commit_kv_step)

    def _run_prefill_tile_chunk(
        self,
        prompt_tokens: list[int],
        k_caches: Any,
        v_caches: Any,
    ) -> tuple[Any, Any, int, jax.Array | None]:
        T = self.tile_size
        prefill_len = len(prompt_tokens) - 1
        forward_passes = 0
        last_argmax = None
        for offset in range(0, prefill_len, T):
            chunk = min(T, prefill_len - offset)
            tok_np = np.zeros((T,), dtype=np.int32)
            pos_np = np.zeros((T,), dtype=np.int32)
            mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
            tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
            pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
            for r in range(chunk):
                mask_np[r, : r + 1] = 0.0
            for r in range(chunk, T):
                mask_np[r, r] = 0.0
            argmax_dev, step_k, step_v = self._static_systolic_tile_step_jit(
                self._weights_tuple,
                self._put_rep(offset, dtype=jnp.int32),
                self._put_rep(tok_np),
                self._put_rep(pos_np),
                self._put_rep(mask_np),
                k_caches,
                v_caches,
            )
            src_rows = np.zeros((T,), dtype=np.int32)
            src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
            k_caches, v_caches = self._commit_kv_jit(
                k_caches,
                v_caches,
                step_k,
                step_v,
                self._put_rep(offset, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(chunk, dtype=jnp.int32),
            )
            last_argmax = argmax_dev
            forward_passes += 1
        return k_caches, v_caches, forward_passes, last_argmax

    def compute_prefill_last_logits(self, prompt_tokens: Sequence[int]) -> np.ndarray:
        """Compute exact float32 next-token logits after prompt_tokens (for tests)."""
        k_caches, v_caches = self._empty_kv_caches()
        prompt_arr = self._put_rep(list(prompt_tokens), dtype=jnp.int32)
        logits, _, _ = self._vanilla_prefill_jit(self._weights_tuple, prompt_arr, k_caches, v_caches)
        return np.asarray(jax.device_get(logits), dtype=np.float32)

    def run_vanilla_jax(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        """Mode 1: Dynamic-shape Vanilla JAX prefill + batch=1 decode baseline."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.vanilla_prefill_compile_count + self.vanilla_decode_compile_count

        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches, v_caches))

        t0 = time.perf_counter()
        prompt_arr = self._put_rep(prompt_tokens, dtype=jnp.int32)
        logits, k_caches, v_caches = self._vanilla_prefill_jit(
            self._weights_tuple, prompt_arr, k_caches, v_caches
        )
        next_tok = int(jnp.argmax(logits))
        jax.block_until_ready((logits, k_caches, v_caches))
        t1 = time.perf_counter()

        generated: list[int] = []
        decode_passes = 0
        t_dec0 = time.perf_counter()
        if next_tok not in eos_set:
            generated.append(next_tok)
            cur_pos = len(prompt_tokens)
            while len(generated) < max_new_tokens:
                logits, k_caches, v_caches = self._vanilla_decode_step_jit(
                    self._weights_tuple,
                    self._put_rep(next_tok, dtype=jnp.int32),
                    self._put_rep(cur_pos, dtype=jnp.int32),
                    k_caches,
                    v_caches,
                )
                decode_passes += 1
                next_tok = int(jnp.argmax(logits))
                jax.block_until_ready(logits)
                if next_tok in eos_set:
                    break
                generated.append(next_tok)
                cur_pos += 1
        t_dec1 = time.perf_counter()

        compiles_delta = (
            self.vanilla_prefill_compile_count + self.vanilla_decode_compile_count
        ) - compiles_before
        prefill_s = t1 - t0
        decode_s = t_dec1 - t_dec0
        gen_len = len(generated)
        return RunMetrics(
            mode="vanilla_jax",
            token_ids=generated,
            text=self._decode_text(generated),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=prefill_s + decode_s,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=1 + decode_passes,
            decode_passes=decode_passes,
            draft_tokens_accepted=0,
            mean_tokens_per_decode_pass=1.0,
            extra={
                "prefill_compiles": self.vanilla_prefill_compile_count,
                "decode_compiles": self.vanilla_decode_compile_count,
                "tile_size": self.tile_size,
                "tp_size": self.tp_size,
                "weight_quant": self.weight_quant,
                "arch": "gemma4",
            },
        )

    def run_static_arena(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        """Mode 2 (M-002): Fixed-tile OpenXLA step, zero recompilations, serial decode."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.static_step_compile_count

        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches, v_caches))

        t0 = time.perf_counter()
        k_caches, v_caches, forward_passes, _ = self._run_prefill_tile_chunk(
            prompt_tokens, k_caches, v_caches
        )
        jax.block_until_ready((k_caches, v_caches))
        t1 = time.perf_counter()

        decode_mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
        np.fill_diagonal(decode_mask_np, 0.0)
        decode_mask_dev = self._put_rep(decode_mask_np)
        src_row_0_jnp = self._put_rep(np.zeros((T,), dtype=np.int32))
        one_jnp = self._put_rep(1, dtype=jnp.int32)

        generated: list[int] = []
        decode_passes = 0
        cur_prefix = len(prompt_tokens) - 1
        cur_tok = prompt_tokens[-1]

        t_dec0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            tok_np = np.zeros((T,), dtype=np.int32)
            pos_np = np.zeros((T,), dtype=np.int32)
            tok_np[0] = cur_tok
            pos_np[0] = cur_prefix

            argmax_dev, step_k, step_v = self._static_systolic_tile_step_jit(
                self._weights_tuple,
                self._put_rep(cur_prefix, dtype=jnp.int32),
                self._put_rep(tok_np),
                self._put_rep(pos_np),
                decode_mask_dev,
                k_caches,
                v_caches,
            )
            forward_passes += 1
            decode_passes += 1
            next_tok = int(argmax_dev[0])
            if next_tok in eos_set:
                break
            k_caches, v_caches = self._commit_kv_jit(
                k_caches,
                v_caches,
                step_k,
                step_v,
                self._put_rep(cur_prefix, dtype=jnp.int32),
                src_row_0_jnp,
                one_jnp,
            )
            generated.append(next_tok)
            cur_prefix += 1
            cur_tok = next_tok
        jax.block_until_ready((k_caches, v_caches))
        t_dec1 = time.perf_counter()

        compiles_delta = self.static_step_compile_count - compiles_before
        prefill_s = t1 - t0
        decode_s = t_dec1 - t_dec0
        gen_len = len(generated)
        return RunMetrics(
            mode="static_arena",
            token_ids=generated,
            text=self._decode_text(generated),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=prefill_s + decode_s,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=forward_passes,
            decode_passes=decode_passes,
            draft_tokens_accepted=0,
            mean_tokens_per_decode_pass=(gen_len / decode_passes) if decode_passes > 0 else 0.0,
            extra={
                "total_static_compiles": self.static_step_compile_count,
                "tile_size": self.tile_size,
                "tp_size": self.tp_size,
                "weight_quant": self.weight_quant,
                "arch": "gemma4",
            },
        )

    def run_tpuforge_tree(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        """Mode 3 (M-002 + M-003): Native C++ ForgeEngine systolic tree."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.config.eos_token_ids)
        )
        compiles_before = self.static_step_compile_count

        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches, v_caches))
        state: dict[str, Any] = {
            "k_caches": k_caches,
            "v_caches": v_caches,
            "step_k": None,
            "step_v": None,
            "t_prefill_done": None,
            "step_idx": 0,
        }
        prefill_tiles = (max(0, len(prompt_tokens) - 1) + T - 1) // T

        def _dispatch_static_tile(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> jax.Array:
            argmax_tokens, step_k, step_v = self._static_systolic_tile_step_jit(
                self._weights_tuple,
                self._put_rep(valid_prefix_len, dtype=jnp.int32),
                self._put_rep(tok_tile),
                self._put_rep(pos_tile),
                self._put_rep(mask_tile),
                state["k_caches"],
                state["v_caches"],
            )
            state["step_k"] = step_k
            state["step_v"] = step_v
            state["step_idx"] += 1
            return argmax_tokens

        def prefill_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> None:
            _dispatch_static_tile(valid_prefix_len, tok_tile, pos_tile, mask_tile)

        def greedy_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> np.ndarray:
            argmax_tokens = _dispatch_static_tile(
                valid_prefix_len, tok_tile, pos_tile, mask_tile
            )
            return np.asarray(jax.device_get(argmax_tokens), dtype=np.int32)

        def commit_kv_fn(prev_prefix_len: int, accepted_row_indices: list[int]) -> None:
            n_valid = len(accepted_row_indices)
            src_rows = np.zeros((T,), dtype=np.int32)
            if n_valid > 0:
                src_rows[:n_valid] = np.asarray(accepted_row_indices, dtype=np.int32)
            k_next, v_next = self._commit_kv_jit(
                state["k_caches"],
                state["v_caches"],
                state["step_k"],
                state["step_v"],
                self._put_rep(prev_prefix_len, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(n_valid, dtype=jnp.int32),
            )
            state["k_caches"] = k_next
            state["v_caches"] = v_next
            if state["step_idx"] == prefill_tiles and state["t_prefill_done"] is None:
                jax.block_until_ready((k_next, v_next))
                state["t_prefill_done"] = time.perf_counter()

        arena_cfg = tpu.StaticArenaConfig()
        arena_cfg.max_seq_len = self.max_seq_len
        arena_cfg.hidden_dim = self.config.hidden_size
        arena_cfg.num_kv_heads = self.config.num_key_value_heads
        arena_cfg.head_dim = self.config.head_dim
        arena_cfg.vocab_size = self.config.vocab_size
        arena_cfg.tile_size = T

        engine = tpu.ForgeEngine(
            arena_cfg,
            None,
            commit_kv_fn,
            f"gemma4-tpuforge-{T}",
            prefill_step_fn=prefill_step_fn,
            greedy_step_fn=greedy_step_fn,
        )
        seed_seqs = (
            [[int(t) for t in seq] for seq in seed_token_sequences if len(seq) >= 2]
            if seed_token_sequences is not None
            else []
        )
        req = tpu.GenerationRequest()
        req.prompt_tokens = list(prompt_tokens)
        req.eos_token_ids = eos_list
        if seed_seqs:
            req.seed_token_sequences = seed_seqs
        req.max_tokens = int(max_new_tokens)
        req.temperature = 0.0

        t_all0 = time.perf_counter()
        res = engine.generate(req)
        jax.block_until_ready((state["k_caches"], state["v_caches"]))
        t_all1 = time.perf_counter()

        compiles_delta = self.static_step_compile_count - compiles_before
        token_ids = [int(t) for t in res.token_ids]
        t_split = state["t_prefill_done"] if state["t_prefill_done"] is not None else t_all0
        prefill_s = t_split - t_all0
        decode_s = t_all1 - t_split
        gen_len = len(token_ids)
        return RunMetrics(
            mode="tpuforge_tree",
            token_ids=token_ids,
            text=self._decode_text(token_ids),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=t_all1 - t_all0,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=int(res.target_forward_passes),
            decode_passes=int(res.decode_forward_passes),
            draft_tokens_accepted=int(res.draft_tokens_accepted),
            mean_tokens_per_decode_pass=float(res.mean_tokens_per_decode_pass),
            extra={
                "total_static_compiles": self.static_step_compile_count,
                "tile_size": self.tile_size,
                "tp_size": self.tp_size,
                "weight_quant": self.weight_quant,
                "ast_seed_sequences": len(seed_seqs),
                "ast_seed_tokens": sum(len(seq) for seq in seed_seqs),
                "arch": "gemma4",
            },
        )


class Gemma4MoeStreamRunner:
    """M-008 + Gemma 4 sparse MoE host-mmap/DRAM streaming runner amortized across 128-node trees."""

    def __init__(
        self,
        config: Gemma4Config,
        trunk_weights: Gemma4Weights,
        expert_store: ExpertHostStore,
        tokenizer: Any,
        max_seq_len: int = 1024,
        tile_size: int | str = "auto",
        mesh: Mesh | None = None,
        stage_experts: int = 16,
    ) -> None:
        if not config.enable_moe_block:
            raise ValueError("Gemma4MoeStreamRunner requires an MoE Gemma 4 checkpoint")
        if int(stage_experts) <= 0:
            raise ValueError(f"stage_experts must be positive, got {stage_experts}")

        self.config = config
        self.weights = trunk_weights
        self.expert_store = expert_store
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.tile_size = resolve_tile_size(tile_size)
        self.mesh = mesh
        self.tp_size = mesh.size if mesh is not None else 1
        self.dtype = trunk_weights.dtype
        self.weight_quant = trunk_weights.weight_quant
        self.host_store_mode = expert_store.host_store_mode
        self.stage_experts = int(stage_experts)
        self._sh = _Gemma4ShardingHelper(config, mesh)
        self._embed_scale = float(math.sqrt(config.hidden_size))
        self._periodic = config.periodic_block_params()

        self.trunk_hbm_bytes = trunk_weights.total_nbytes
        self.staging_hbm_bytes = expert_store.staging_buffer_bytes(self.stage_experts)
        self.expert_source_bytes = expert_store.total_source_bytes()
        self.model_hbm_bytes = self.trunk_hbm_bytes

        self.vanilla_prefill_compile_count = 0
        self.vanilla_decode_compile_count = 0
        self.static_step_compile_count = 0

        self._compile_kernels()

    def close(self) -> None:
        self.expert_store.close()

    def _put_rep(self, arr: Any, dtype: Any | None = None) -> jax.Array:
        return self._sh.put_rep(arr, dtype=dtype)

    def _decode_text(self, token_ids: list[int]) -> str:
        if not token_ids or self.tokenizer is None:
            return ""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _validate_bounds(self, prompt_tokens: list[int], max_new_tokens: int) -> None:
        if not prompt_tokens:
            raise ValueError("prompt_tokens cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if len(prompt_tokens) + max_new_tokens > self.max_seq_len - self.tile_size:
            raise ValueError(
                f"prompt_len ({len(prompt_tokens)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_seq_len - tile_size ({self.max_seq_len - self.tile_size})"
            )

    def _layer_view(self, lid: int) -> _Gemma4GroupScanTuple:
        wt = self.weights.weights_tuple
        if self._periodic is not None:
            assert isinstance(wt, _Gemma4PeriodicTuple)
            k_slide, _ = self._periodic
            block_len = k_slide + 1
            b = lid // block_len
            rem = lid % block_len
            grp = wt.full_group if rem == k_slide else wt.sliding_group
            idx_pair = (b,) if rem == k_slide else (b, rem)
            return _Gemma4GroupScanTuple(
                *[
                    (arr[idx_pair] if isinstance(arr, jax.Array) else None)
                    for arr in grp
                ]
            )
        assert isinstance(wt, _Gemma4LayerListTuple)
        return wt.layers[lid]

    def _empty_kv_caches(self) -> tuple[list[jax.Array], list[jax.Array]]:
        cfg = self.config
        ks, vs = [], []
        for lt in cfg.layer_types:
            is_full = lt == "full_attention"
            kv_h = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
            hd = cfg.global_head_dim if is_full else cfg.head_dim
            sh = self._sh.full_kv_cache if is_full else self._sh.sliding_kv_cache
            k_l = jnp.zeros((kv_h, self.max_seq_len, hd), dtype=self.dtype)
            v_l = jnp.zeros((kv_h, self.max_seq_len, hd), dtype=self.dtype)
            if sh is not None:
                k_l = jax.device_put(k_l, sh)
                v_l = jax.device_put(v_l, sh)
            ks.append(k_l)
            vs.append(v_l)
        return ks, vs

    def _compile_kernels(self) -> None:
        cfg = self.config
        sh = self._sh
        max_seq = self.max_seq_len

        @jax.jit
        def embed_step(embed: jax.Array, tokens: jax.Array) -> jax.Array:
            x = embed[tokens] * jnp.asarray(self._embed_scale, dtype=embed.dtype)
            return jax.lax.with_sharding_constraint(x, sh.rep) if sh.rep is not None else x

        @jax.jit
        def final_logits_step(
            x_final: jax.Array,
            final_norm: jax.Array,
            embed: jax.Array,
            lm_head: jax.Array | None,
            lm_head_scale: jax.Array | None,
        ) -> jax.Array:
            logits = _compute_gemma4_logits(x_final, final_norm, embed, lm_head, lm_head_scale, cfg)
            return jnp.argmax(logits, axis=-1).astype(jnp.int32)

        def _make_pre_route(is_full: bool, count_tag: str):
            num_kv = cfg.num_global_key_value_heads if is_full else cfg.num_key_value_heads
            hd = cfg.global_head_dim if is_full else cfg.head_dim
            kv_sh = sh.full_kv if is_full else sh.sliding_kv
            cache_sh = sh.full_kv_cache if is_full else sh.sliding_kv_cache

            @jax.jit
            def pre_route(
                x: jax.Array,
                layer: _Gemma4GroupScanTuple,
                k_cache: jax.Array,
                v_cache: jax.Array,
                inv_freq: jax.Array,
                positions: jax.Array,
                valid_prefix_len: jax.Array,
                tile_mask: jax.Array,
                has_cache: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                if count_tag == "vp":
                    self.vanilla_prefill_compile_count += 1
                elif count_tag == "vd":
                    self.vanilla_decode_compile_count += 1
                elif count_tag == "st":
                    self.static_step_compile_count += 1

                full_cm, slide_cm = _build_cache_masks(
                    positions, valid_prefix_len, max_seq, cfg.sliding_window
                )
                cm = full_cm if is_full else slide_cm
                # Masked out via has_cache=False on vanilla prefill while keeping identical signature/shape.
                eff_cm = jnp.where(has_cache, cm, jnp.float32(_MASK_BLOCK_VAL))
                eff_tm = tile_mask if is_full else _apply_sliding_to_tile_mask(
                    positions, tile_mask, cfg.sliding_window
                )
                x_mid, k_new, v_new, mlp_out = _gemma4_attn_and_dense_mlp(
                    x,
                    layer,
                    k_cache,
                    v_cache,
                    positions,
                    eff_cm,
                    eff_tm,
                    inv_freq,
                    cfg,
                    num_kv,
                    hd,
                    sh.rep,
                    sh.head,
                    sh.flat_head,
                    kv_sh,
                    sh.heads_first,
                    cache_sh,
                )
                h1 = _gemma4_rms_norm(mlp_out, layer.post_ffn_norm_1, cfg.rms_norm_eps)
                top_idx, top_w = _compute_gemma4_router(x_mid, layer, cfg, sh.rep)
                x_exp = _gemma4_rms_norm(x_mid, layer.pre_ffn_norm_2, cfg.rms_norm_eps)
                return x_mid, h1, x_exp, top_idx, top_w, k_new, v_new

            return pre_route

        def _make_stage_accum():
            @jax.jit
            def stage_accum(
                acc: jax.Array,
                x_exp: jax.Array,
                wg: jax.Array,
                wu: jax.Array,
                wd: jax.Array,
                sg: jax.Array | None,
                su: jax.Array | None,
                sd: jax.Array | None,
                slot_idx: jax.Array,
                slot_weights: jax.Array,
            ) -> jax.Array:
                e_stage = wg.shape[0]
                seq_q = x_exp.shape[0]
                x_b = jnp.broadcast_to(x_exp[None, :, :], (e_stage, seq_q, cfg.hidden_size))
                gate = batched_expert_systolic_linear(x_b, wg, sg, x_exp.dtype)
                up = batched_expert_systolic_linear(x_b, wu, su, x_exp.dtype)
                if sh.exp_inter is not None:
                    gate = jax.lax.with_sharding_constraint(gate, sh.exp_inter)
                    up = jax.lax.with_sharding_constraint(up, sh.exp_inter)
                act = (
                    jax.nn.gelu(gate.astype(jnp.float32), approximate=True) * up.astype(jnp.float32)
                ).astype(x_exp.dtype)
                exp_outs = batched_expert_systolic_linear(act, wd, sd, x_exp.dtype)
                if sh.rep is not None:
                    exp_outs = jax.lax.with_sharding_constraint(exp_outs, sh.rep)
                zero_pad = jnp.zeros((1, seq_q, cfg.hidden_size), dtype=exp_outs.dtype)
                padded = jnp.concatenate([exp_outs, zero_pad], axis=0)
                safe_slot = jnp.where(slot_idx >= 0, slot_idx, jnp.int32(e_stage))
                row_idx = jnp.arange(seq_q, dtype=jnp.int32)[:, None]
                picked = padded[safe_slot, row_idx, :].astype(jnp.float32)
                contrib = jnp.sum(picked * slot_weights[:, :, None], axis=1)
                return acc + contrib

            return stage_accum

        @jax.jit
        def post_ffn_step(
            x_mid: jax.Array,
            h1: jax.Array,
            moe_acc_f32: jax.Array,
            layer: _Gemma4GroupScanTuple,
        ) -> jax.Array:
            h2 = _gemma4_rms_norm(moe_acc_f32.astype(x_mid.dtype), layer.post_ffn_norm_2, cfg.rms_norm_eps)
            ffn_out = _gemma4_rms_norm(h1 + h2, layer.post_ffn_norm, cfg.rms_norm_eps)
            out = (x_mid + ffn_out) * layer.layer_scalar.astype(x_mid.dtype)
            return jax.lax.with_sharding_constraint(out, sh.rep) if sh.rep is not None else out

        @jax.jit
        def commit_all_kv_step(
            k_caches: tuple[jax.Array, ...],
            v_caches: tuple[jax.Array, ...],
            step_ks: tuple[jax.Array, ...],
            step_vs: tuple[jax.Array, ...],
            prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
            k_out, v_out = [], []
            for kc, vc, sk_l, sv_l, lt in zip(k_caches, v_caches, step_ks, step_vs, cfg.layer_types):
                kn, vn = _commit_kv_pair(kc, vc, sk_l, sv_l, prefix_len, src_rows, n_valid)
                csh = sh.full_kv_cache if lt == "full_attention" else sh.sliding_kv_cache
                if csh is not None:
                    kn = jax.lax.with_sharding_constraint(kn, csh)
                    vn = jax.lax.with_sharding_constraint(vn, csh)
                k_out.append(kn)
                v_out.append(vn)
            return tuple(k_out), tuple(v_out)

        self._embed_jit = embed_step
        self._final_argmax_jit = final_logits_step
        self._pre_route_static_slide = _make_pre_route(False, "st")
        self._pre_route_static_full = _make_pre_route(True, "st")
        self._pre_route_vp_slide = _make_pre_route(False, "vp")
        self._pre_route_vp_full = _make_pre_route(True, "vp")
        self._pre_route_vd_slide = _make_pre_route(False, "vd")
        self._pre_route_vd_full = _make_pre_route(True, "vd")
        self._stage_accum_jit = _make_stage_accum()
        self._post_ffn_jit = post_ffn_step
        self._commit_all_kv_jit = commit_all_kv_step

    def _new_stats(self) -> dict[str, int]:
        return {
            "h2d_bytes": 0,
            "decode_h2d_bytes": 0,
            "expert_loads": 0,
            "decode_expert_loads": 0,
            "staged_chunks": 0,
        }

    def _run_step(
        self,
        valid_prefix_len: int,
        tok_arr: np.ndarray,
        pos_arr: np.ndarray,
        mask_arr: np.ndarray,
        has_cache: bool,
        valid_row_mask: np.ndarray,
        k_caches: list[jax.Array],
        v_caches: list[jax.Array],
        stats: dict[str, int],
        *,
        mode_tag: str,
        count_as_decode: bool,
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        wt = self.weights.weights_tuple
        x = self._embed_jit(wt.embed, self._put_rep(tok_arr, dtype=jnp.int32))
        pos_dev = self._put_rep(pos_arr, dtype=jnp.int32)
        pref_dev = self._put_rep(valid_prefix_len, dtype=jnp.int32)
        mask_dev = self._put_rep(mask_arr, dtype=jnp.float32)
        hc_dev = self._put_rep(has_cache, dtype=jnp.bool_)

        step_ks: list[jax.Array] = []
        step_vs: list[jax.Array] = []
        single_expert_bytes = self.expert_store.single_expert_bytes()

        for lid, lt in enumerate(self.config.layer_types):
            is_full = lt == "full_attention"
            layer_v = self._layer_view(lid)
            inv_freq = wt.full_inv_freq if is_full else wt.sliding_inv_freq
            if mode_tag == "vp":
                fn = self._pre_route_vp_full if is_full else self._pre_route_vp_slide
            elif mode_tag == "vd":
                fn = self._pre_route_vd_full if is_full else self._pre_route_vd_slide
            else:
                fn = self._pre_route_static_full if is_full else self._pre_route_static_slide

            x_mid, h1, x_exp, top_idx_d, top_w_d, k_new, v_new = fn(
                x, layer_v, k_caches[lid], v_caches[lid], inv_freq, pos_dev, pref_dev, mask_dev, hc_dev
            )
            step_ks.append(k_new)
            step_vs.append(v_new)

            top_idx_np, top_w_np = jax.device_get((top_idx_d, top_w_d))
            moe_acc = jnp.zeros((x.shape[0], self.config.hidden_size), dtype=jnp.float32)
            if self._sh.rep is not None:
                moe_acc = jax.device_put(moe_acc, self._sh.rep)

            for (
                chunk_len,
                wg_buf,
                wu_buf,
                wd_buf,
                sg_buf,
                su_buf,
                sd_buf,
                slot_idx_np,
                slot_w_np,
            ) in self.expert_store.iter_staged_chunks(
                lid, top_idx_np, top_w_np, valid_row_mask, self.stage_experts
            ):
                stats["staged_chunks"] += 1
                stats["expert_loads"] += chunk_len
                chunk_bytes = chunk_len * single_expert_bytes
                stats["h2d_bytes"] += chunk_bytes
                if count_as_decode:
                    stats["decode_expert_loads"] += chunk_len
                    stats["decode_h2d_bytes"] += chunk_bytes

                exp_spec = P(None, None, "tp") if self.tp_size > 1 else P()
                exp_down_spec = P(None, "tp", None) if self.tp_size > 1 else P()

                def _put_e(arr_np: np.ndarray, sp: P) -> jax.Array:
                    if self.mesh is None:
                        return jax.device_put(jnp.asarray(arr_np))
                    return place_np_sharded_preserved(arr_np, sp, self.mesh)

                wg_d = _put_e(wg_buf, exp_spec)
                wu_d = _put_e(wu_buf, exp_spec)
                wd_d = _put_e(wd_buf, exp_down_spec)
                sg_d = (
                    _put_e(sg_buf, scale_spec_for_weight(exp_spec))
                    if sg_buf is not None
                    else None
                )
                su_d = (
                    _put_e(su_buf, scale_spec_for_weight(exp_spec))
                    if su_buf is not None
                    else None
                )
                sd_d = (
                    _put_e(sd_buf, scale_spec_for_weight(exp_down_spec))
                    if sd_buf is not None
                    else None
                )

                moe_acc = self._stage_accum_jit(
                    moe_acc,
                    x_exp,
                    wg_d,
                    wu_d,
                    wd_d,
                    sg_d,
                    su_d,
                    sd_d,
                    self._put_rep(slot_idx_np, dtype=jnp.int32),
                    self._put_rep(slot_w_np, dtype=jnp.float32),
                )

            x = self._post_ffn_jit(x_mid, h1, moe_acc, layer_v)

        argmax_tokens = self._final_argmax_jit(
            x, wt.final_norm, wt.embed, wt.lm_head, wt.lm_head_scale
        )
        return argmax_tokens, tuple(step_ks), tuple(step_vs)

    def _extra_metrics(self, stats: dict[str, int], gen_len: int) -> dict[str, Any]:
        return {
            "host_store_mode": self.host_store_mode,
            "stage_experts": self.stage_experts,
            "trunk_hbm_bytes": self.trunk_hbm_bytes,
            "staging_hbm_bytes": self.staging_hbm_bytes,
            "expert_source_bytes": self.expert_source_bytes,
            "h2d_bytes_total": stats["h2d_bytes"],
            "decode_h2d_bytes": stats["decode_h2d_bytes"],
            "expert_loads_total": stats["expert_loads"],
            "decode_expert_loads": stats["decode_expert_loads"],
            "decode_bytes_per_token": (stats["decode_h2d_bytes"] / gen_len) if gen_len > 0 else 0.0,
            "staged_chunks_total": stats["staged_chunks"],
            "tile_size": self.tile_size,
            "tp_size": self.tp_size,
            "weight_quant": self.weight_quant,
            "arch": "gemma4",
        }

    def run_vanilla_jax(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        self._validate_bounds(prompt_tokens, max_new_tokens)
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.vanilla_prefill_compile_count + self.vanilla_decode_compile_count
        stats = self._new_stats()

        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches[0], v_caches[0]))

        L = len(prompt_tokens)
        pos_np = np.arange(L, dtype=np.int32)
        rows = pos_np[:, None]
        cols = pos_np[None, :]
        causal_np = np.where(cols <= rows, np.float32(0.0), np.float32(_MASK_BLOCK_VAL))
        valid_prefill = np.ones((L,), dtype=bool)
        valid_decode = np.ones((1,), dtype=bool)

        t0 = time.perf_counter()
        argmax_pre, step_ks, step_vs = self._run_step(
            0,
            np.asarray(prompt_tokens, dtype=np.int32),
            pos_np,
            causal_np,
            False,
            valid_prefill,
            k_caches,
            v_caches,
            stats,
            mode_tag="vp",
            count_as_decode=False,
        )
        all_rows = self._put_rep(pos_np, dtype=jnp.int32)
        kt, vt = self._commit_all_kv_jit(
            tuple(k_caches),
            tuple(v_caches),
            step_ks,
            step_vs,
            self._put_rep(0, dtype=jnp.int32),
            all_rows,
            self._put_rep(L, dtype=jnp.int32),
        )
        k_caches, v_caches = list(kt), list(vt)
        next_tok = int(argmax_pre[-1])
        jax.block_until_ready((k_caches[0], v_caches[0]))
        t1 = time.perf_counter()

        generated: list[int] = []
        decode_passes = 0
        self_mask = np.zeros((1, 1), dtype=np.float32)
        src_0 = self._put_rep(np.zeros((1,), dtype=jnp.int32))
        one_jnp = self._put_rep(1, dtype=jnp.int32)
        t_dec0 = time.perf_counter()
        if next_tok not in eos_set:
            generated.append(next_tok)
            cur_pos = L
            while len(generated) < max_new_tokens:
                argmax_dec, step_ks, step_vs = self._run_step(
                    cur_pos,
                    np.asarray([next_tok], dtype=np.int32),
                    np.asarray([cur_pos], dtype=np.int32),
                    self_mask,
                    True,
                    valid_decode,
                    k_caches,
                    v_caches,
                    stats,
                    mode_tag="vd",
                    count_as_decode=True,
                )
                kt, vt = self._commit_all_kv_jit(
                    tuple(k_caches),
                    tuple(v_caches),
                    step_ks,
                    step_vs,
                    self._put_rep(cur_pos, dtype=jnp.int32),
                    src_0,
                    one_jnp,
                )
                k_caches, v_caches = list(kt), list(vt)
                decode_passes += 1
                next_tok = int(argmax_dec[0])
                if next_tok in eos_set:
                    break
                generated.append(next_tok)
                cur_pos += 1
        t_dec1 = time.perf_counter()

        compiles_delta = (
            self.vanilla_prefill_compile_count + self.vanilla_decode_compile_count
        ) - compiles_before
        prefill_s = t1 - t0
        decode_s = t_dec1 - t_dec0
        gen_len = len(generated)
        return RunMetrics(
            mode="vanilla_moe",
            token_ids=generated,
            text=self._decode_text(generated),
            prompt_len=L,
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=prefill_s + decode_s,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=1 + decode_passes,
            decode_passes=decode_passes,
            draft_tokens_accepted=0,
            mean_tokens_per_decode_pass=1.0,
            extra=self._extra_metrics(stats, gen_len),
        )

    def run_static_arena(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.static_step_compile_count
        stats = self._new_stats()

        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches[0], v_caches[0]))

        t0 = time.perf_counter()
        prefill_len = len(prompt_tokens) - 1
        forward_passes = 0
        for offset in range(0, prefill_len, T):
            chunk = min(T, prefill_len - offset)
            tok_np = np.zeros((T,), dtype=np.int32)
            pos_np = np.zeros((T,), dtype=np.int32)
            mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
            valid_mask = np.zeros((T,), dtype=bool)
            tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
            pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
            valid_mask[:chunk] = True
            for r in range(chunk):
                mask_np[r, : r + 1] = 0.0
            for r in range(chunk, T):
                mask_np[r, r] = 0.0

            _, step_ks, step_vs = self._run_step(
                offset,
                tok_np,
                pos_np,
                mask_np,
                True,
                valid_mask,
                k_caches,
                v_caches,
                stats,
                mode_tag="st",
                count_as_decode=False,
            )
            src_rows = np.zeros((T,), dtype=np.int32)
            src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
            kt, vt = self._commit_all_kv_jit(
                tuple(k_caches),
                tuple(v_caches),
                step_ks,
                step_vs,
                self._put_rep(offset, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(chunk, dtype=jnp.int32),
            )
            k_caches, v_caches = list(kt), list(vt)
            forward_passes += 1
        jax.block_until_ready((k_caches[0], v_caches[0]))
        t1 = time.perf_counter()

        decode_mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
        np.fill_diagonal(decode_mask_np, 0.0)
        valid_row0 = np.zeros((T,), dtype=bool)
        valid_row0[0] = True
        src_0 = self._put_rep(np.zeros((T,), dtype=np.int32))
        one_jnp = self._put_rep(1, dtype=jnp.int32)

        generated: list[int] = []
        decode_passes = 0
        cur_prefix = prefill_len
        cur_tok = prompt_tokens[-1]

        t_dec0 = time.perf_counter()
        while len(generated) < max_new_tokens:
            tok_np = np.zeros((T,), dtype=np.int32)
            pos_np = np.zeros((T,), dtype=np.int32)
            tok_np[0] = cur_tok
            pos_np[0] = cur_prefix

            argmax_dev, step_ks, step_vs = self._run_step(
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
            forward_passes += 1
            decode_passes += 1
            next_tok = int(argmax_dev[0])
            if next_tok in eos_set:
                break
            kt, vt = self._commit_all_kv_jit(
                tuple(k_caches),
                tuple(v_caches),
                step_ks,
                step_vs,
                self._put_rep(cur_prefix, dtype=jnp.int32),
                src_0,
                one_jnp,
            )
            k_caches, v_caches = list(kt), list(vt)
            generated.append(next_tok)
            cur_prefix += 1
            cur_tok = next_tok
        jax.block_until_ready((k_caches[0], v_caches[0]))
        t_dec1 = time.perf_counter()

        compiles_delta = self.static_step_compile_count - compiles_before
        prefill_s = t1 - t0
        decode_s = t_dec1 - t_dec0
        gen_len = len(generated)
        return RunMetrics(
            mode="static_arena_moe",
            token_ids=generated,
            text=self._decode_text(generated),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=prefill_s + decode_s,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=forward_passes,
            decode_passes=decode_passes,
            draft_tokens_accepted=0,
            mean_tokens_per_decode_pass=1.0,
            extra=self._extra_metrics(stats, gen_len),
        )

    def run_tpuforge_tree(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.config.eos_token_ids)
        )
        compiles_before = self.static_step_compile_count
        stats = self._new_stats()
        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches[0], v_caches[0]))

        state: dict[str, Any] = {
            "k_caches": k_caches,
            "v_caches": v_caches,
            "step_ks": None,
            "step_vs": None,
            "t_prefill_done": None,
            "step_idx": 0,
        }
        prefill_tiles = (max(0, len(prompt_tokens) - 1) + T - 1) // T
        all_rows_mask = np.ones((T,), dtype=bool)

        def _dispatch(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
            count_as_decode: bool,
        ) -> jax.Array:
            valid_mask = (
                (pos_tile >= valid_prefix_len)
                if count_as_decode
                else np.ones((T,), dtype=bool)
            )
            if not count_as_decode:
                valid_mask = all_rows_mask
            argmax_tokens, step_ks, step_vs = self._run_step(
                valid_prefix_len,
                tok_tile,
                pos_tile,
                mask_tile,
                True,
                valid_mask,
                state["k_caches"],
                state["v_caches"],
                stats,
                mode_tag="st",
                count_as_decode=count_as_decode,
            )
            state["step_ks"] = step_ks
            state["step_vs"] = step_vs
            state["step_idx"] += 1
            return argmax_tokens

        def prefill_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> None:
            _dispatch(valid_prefix_len, tok_tile, pos_tile, mask_tile, count_as_decode=False)

        def greedy_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> np.ndarray:
            argmax_tokens = _dispatch(
                valid_prefix_len, tok_tile, pos_tile, mask_tile, count_as_decode=True
            )
            return np.asarray(jax.device_get(argmax_tokens), dtype=np.int32)

        def commit_kv_fn(prev_prefix_len: int, accepted_row_indices: list[int]) -> None:
            n_valid = len(accepted_row_indices)
            src_rows = np.zeros((T,), dtype=np.int32)
            if n_valid > 0:
                src_rows[:n_valid] = np.asarray(accepted_row_indices, dtype=np.int32)
            kt, vt = self._commit_all_kv_jit(
                tuple(state["k_caches"]),
                tuple(state["v_caches"]),
                state["step_ks"],
                state["step_vs"],
                self._put_rep(prev_prefix_len, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(n_valid, dtype=jnp.int32),
            )
            state["k_caches"] = list(kt)
            state["v_caches"] = list(vt)
            if state["step_idx"] == prefill_tiles and state["t_prefill_done"] is None:
                jax.block_until_ready((kt[0], vt[0]))
                state["t_prefill_done"] = time.perf_counter()

        arena_cfg = tpu.StaticArenaConfig()
        arena_cfg.max_seq_len = self.max_seq_len
        arena_cfg.hidden_dim = self.config.hidden_size
        arena_cfg.num_kv_heads = self.config.num_key_value_heads
        arena_cfg.head_dim = self.config.head_dim
        arena_cfg.vocab_size = self.config.vocab_size
        arena_cfg.tile_size = T

        engine = tpu.ForgeEngine(
            arena_cfg,
            None,
            commit_kv_fn,
            f"gemma4-tpuforge-moe-{T}",
            prefill_step_fn=prefill_step_fn,
            greedy_step_fn=greedy_step_fn,
        )
        seed_seqs = (
            [[int(t) for t in seq] for seq in seed_token_sequences if len(seq) >= 2]
            if seed_token_sequences is not None
            else []
        )
        req = tpu.GenerationRequest()
        req.prompt_tokens = list(prompt_tokens)
        req.eos_token_ids = eos_list
        if seed_seqs:
            req.seed_token_sequences = seed_seqs
        req.max_tokens = int(max_new_tokens)
        req.temperature = 0.0

        t_all0 = time.perf_counter()
        res = engine.generate(req)
        jax.block_until_ready((state["k_caches"][0], state["v_caches"][0]))
        t_all1 = time.perf_counter()

        compiles_delta = self.static_step_compile_count - compiles_before
        token_ids = [int(t) for t in res.token_ids]
        t_split = state["t_prefill_done"] if state["t_prefill_done"] is not None else t_all0
        prefill_s = t_split - t_all0
        decode_s = t_all1 - t_split
        gen_len = len(token_ids)
        extra = self._extra_metrics(stats, gen_len)
        extra["ast_seed_tokens"] = sum(len(seq) for seq in seed_seqs)
        return RunMetrics(
            mode="tpuforge_tree_moe",
            token_ids=token_ids,
            text=self._decode_text(token_ids),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=t_all1 - t_all0,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=int(res.target_forward_passes),
            decode_passes=int(res.decode_forward_passes),
            draft_tokens_accepted=int(res.draft_tokens_accepted),
            mean_tokens_per_decode_pass=float(res.mean_tokens_per_decode_pass),
            extra=extra,
        )

    run_vanilla_moe = run_vanilla_jax
    run_static_arena_moe = run_static_arena
    run_tpuforge_tree_moe = run_tpuforge_tree


class Gemma4BlockTreeRunner:
    """M-006 + Gemma 4: parallel DFlash block-diffusion + SuffixIndex hybrid systolic tree."""

    def __init__(
        self,
        target_runner: Gemma4Runner,
        draft_config: BlockDiffusionConfig,
        draft_weights: BlockDiffusionWeights,
        top_k: int = 8,
    ) -> None:
        if top_k <= 0 or top_k > draft_config.vocab_size:
            raise ValueError(f"top_k={top_k} must be in [1, {draft_config.vocab_size}]")
        if draft_config.hidden_size != target_runner.config.hidden_size:
            raise ValueError(
                f"BlockDiffusion hidden_size ({draft_config.hidden_size}) != target ({target_runner.config.hidden_size})"
            )
        if draft_config.num_target_layers != target_runner.config.num_hidden_layers:
            raise ValueError(
                f"BlockDiffusion num_target_layers ({draft_config.num_target_layers}) != "
                f"target num_hidden_layers ({target_runner.config.num_hidden_layers})"
            )

        self.target = target_runner
        self.draft_config = draft_config
        self.draft_weights = draft_weights
        self.top_k = int(top_k)

        self.max_seq_len = target_runner.max_seq_len
        self.tile_size = target_runner.tile_size
        self.mesh = target_runner.mesh
        self.tp_size = target_runner.tp_size
        self.dtype = target_runner.dtype
        self.weight_quant = draft_weights.weight_quant

        self.target_with_ctx_compile_count = 0
        self.commit_both_kv_compile_count = 0
        self.draft_step_compile_count = 0

        layer_map = {lid: idx for idx, lid in enumerate(draft_config.target_layer_ids)}
        self._num_captured = len(draft_config.target_layer_ids)
        capture_slots_np = np.array(
            [layer_map.get(i, self._num_captured) for i in range(target_runner.config.num_hidden_layers)],
            dtype=np.int32,
        )
        self._capture_slots = target_runner._put_rep(capture_slots_np, dtype=jnp.int32)
        self._draft_weights_tuple: DraftWeightsTuple = draft_weights.to_draft_weights_tuple()

        def ns(spec: P) -> NamedSharding | None:
            return NamedSharding(self.mesh, spec) if self.mesh is not None else None

        self._rep_sharding = ns(P())
        self._head_sharding = ns(P(None, "tp", None))
        draft_kv_shardable = self.tp_size > 1 and (
            draft_config.num_key_value_heads % self.tp_size == 0
        )
        self._kv_head_sharding = ns(P(None, "tp", None) if draft_kv_shardable else P())
        self._kv_cache_sharding = ns(P(None, None, "tp", None) if draft_kv_shardable else P())
        self._transpose_lm_head = target_runner.weights.lm_head is None

        self._compile_block_tree_kernels()

    def total_compile_count(self) -> int:
        return (
            self.target_with_ctx_compile_count
            + self.commit_both_kv_compile_count
            + self.draft_step_compile_count
        )

    def _empty_draft_kv_cache(self) -> tuple[jax.Array, jax.Array]:
        dcfg = self.draft_config
        shape = (dcfg.num_hidden_layers, self.max_seq_len, dcfg.num_key_value_heads, dcfg.head_dim)
        dk = jnp.zeros(shape, dtype=self.dtype)
        dv = jnp.zeros(shape, dtype=self.dtype)
        if self._kv_cache_sharding is not None:
            dk = jax.device_put(dk, self._kv_cache_sharding)
            dv = jax.device_put(dv, self._kv_cache_sharding)
        return dk, dv

    def _compile_block_tree_kernels(self) -> None:
        tcfg = self.target.config
        dcfg = self.draft_config
        max_seq = self.max_seq_len
        top_k = self.top_k
        num_captured = self._num_captured
        transpose_lm = self._transpose_lm_head

        def target_tile_with_ctx(
            target_wt: Any,
            capture_slots: jax.Array,
            valid_prefix_len: jax.Array,
            tile_tokens: jax.Array,
            tile_positions: jax.Array,
            tile_mask: jax.Array,
            k_caches: Any,
            v_caches: Any,
        ) -> tuple[jax.Array, Any, Any, jax.Array]:
            self.target_with_ctx_compile_count += 1
            x_out, step_k, step_v, cap_ctx = self.target._forward_trunk(
                target_wt,
                tile_tokens,
                tile_positions,
                valid_prefix_len,
                tile_mask,
                k_caches,
                v_caches,
                capture_slots=capture_slots,
                num_captured=num_captured,
            )
            assert cap_ctx is not None
            logits = _compute_gemma4_logits(
                x_out, target_wt.final_norm, target_wt.embed, target_wt.lm_head, target_wt.lm_head_scale, tcfg
            )
            argmax_tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            return argmax_tokens, step_k, step_v, cap_ctx

        def commit_both_kv(
            draft_wt: DraftWeightsTuple,
            k_caches: Any,
            v_caches: Any,
            step_k: Any,
            step_v: Any,
            draft_k_cache: jax.Array,
            draft_v_cache: jax.Array,
            step_ctx: jax.Array,
            prev_prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[Any, Any, jax.Array, jax.Array]:
            self.commit_both_kv_compile_count += 1
            k_next, v_next = self.target._commit_caches(
                k_caches, v_caches, step_k, step_v, prev_prefix_len, src_rows, n_valid
            )
            gathered_ctx = jnp.take(step_ctx, src_rows, axis=0)
            tile_len = src_rows.shape[0]
            valid_mask = jnp.arange(tile_len, dtype=jnp.int32) < n_valid
            dest_positions = prev_prefix_len + jnp.arange(tile_len, dtype=jnp.int32)
            dest_slots = jnp.where(valid_mask, dest_positions, jnp.int32(max_seq))
            dk_next, dv_next = commit_draft_context_kv(
                draft_wt,
                dcfg,
                draft_k_cache,
                draft_v_cache,
                gathered_ctx,
                dest_positions,
                dest_slots,
                self._rep_sharding,
                self._kv_head_sharding,
                self._kv_cache_sharding,
            )
            return k_next, v_next, dk_next, dv_next

        def block_draft_top_k_step(
            raw_embed: jax.Array,
            lm_head: jax.Array,
            lm_head_scale: jax.Array | None,
            draft_wt: DraftWeightsTuple,
            root_token_id: jax.Array,
            cur_prefix: jax.Array,
            draft_k_cache: jax.Array,
            draft_v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self.draft_step_compile_count += 1
            return run_dflash_block_step(
                raw_embed,
                lm_head,
                lm_head_scale,
                draft_wt,
                dcfg,
                root_token_id,
                cur_prefix,
                draft_k_cache,
                draft_v_cache,
                max_seq,
                top_k,
                self._rep_sharding,
                self._head_sharding,
                self._kv_head_sharding,
                transpose_lm_head=transpose_lm,
            )

        self._target_tile_with_ctx_jit = jax.jit(target_tile_with_ctx)
        self._commit_both_kv_jit = jax.jit(commit_both_kv)
        self._block_draft_top_k_jit = jax.jit(block_draft_top_k_step)

    def warmup(self) -> float:
        """Pre-compile the 3 fixed-shape Gemma 4 block-tree kernels outside timed inference."""
        T = self.tile_size
        t0 = time.perf_counter()
        k_caches, v_caches = self.target._empty_kv_caches()
        dk_cache, dv_cache = self._empty_draft_kv_cache()
        tok_tile = self.target._put_rep(np.zeros((T,), dtype=np.int32))
        pos_tile = self.target._put_rep(np.arange(T, dtype=np.int32))
        mask_tile = self.target._put_rep(np.zeros((T, T), dtype=np.float32))
        src_rows = self.target._put_rep(np.arange(T, dtype=np.int32))
        target_wt = self.target._weights_tuple
        draft_wt = self._draft_weights_tuple
        lm_arr = target_wt.embed if target_wt.lm_head is None else target_wt.lm_head

        arg, sk, sv, ctx = self._target_tile_with_ctx_jit(
            target_wt,
            self._capture_slots,
            self.target._put_rep(0, dtype=jnp.int32),
            tok_tile,
            pos_tile,
            mask_tile,
            k_caches,
            v_caches,
        )
        kn, vn, dkn, dvn = self._commit_both_kv_jit(
            draft_wt,
            k_caches,
            v_caches,
            sk,
            sv,
            dk_cache,
            dv_cache,
            ctx,
            self.target._put_rep(0, dtype=jnp.int32),
            src_rows,
            self.target._put_rep(1, dtype=jnp.int32),
        )
        t_ids, t_probs = self._block_draft_top_k_jit(
            target_wt.embed,
            lm_arr,
            target_wt.lm_head_scale,
            draft_wt,
            self.target._put_rep(0, dtype=jnp.int32),
            self.target._put_rep(1, dtype=jnp.int32),
            dkn,
            dvn,
        )
        jax.block_until_ready((arg, kn, vn, t_ids, t_probs))
        return time.perf_counter() - t0

    def run_block_tree(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        use_suffix_fallback: bool = True,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        """Run native C++ ForgeEngine with DFlash + SuffixIndex on Gemma 4 target."""
        self.target._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.target.config.eos_token_ids)
        )
        compiles_before = self.total_compile_count()

        k_caches, v_caches = self.target._empty_kv_caches()
        dk_cache, dv_cache = self._empty_draft_kv_cache()
        jax.block_until_ready((k_caches, v_caches, dk_cache, dv_cache))

        state: dict[str, Any] = {
            "k_caches": k_caches,
            "v_caches": v_caches,
            "dk_cache": dk_cache,
            "dv_cache": dv_cache,
            "step_k": None,
            "step_v": None,
            "step_ctx": None,
            "t_prefill_done": None,
            "step_idx": 0,
        }
        prefill_tiles = (max(0, len(prompt_tokens) - 1) + T - 1) // T
        target_wt = self.target._weights_tuple
        draft_wt = self._draft_weights_tuple
        embed = target_wt.embed
        lm_arr = embed if target_wt.lm_head is None else target_wt.lm_head
        lm_head_scale = target_wt.lm_head_scale

        def _dispatch_target_tile(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> jax.Array:
            argmax_tokens, step_k, step_v, step_ctx = self._target_tile_with_ctx_jit(
                target_wt,
                self._capture_slots,
                self.target._put_rep(valid_prefix_len, dtype=jnp.int32),
                self.target._put_rep(tok_tile),
                self.target._put_rep(pos_tile),
                self.target._put_rep(mask_tile),
                state["k_caches"],
                state["v_caches"],
            )
            state["step_k"] = step_k
            state["step_v"] = step_v
            state["step_ctx"] = step_ctx
            state["step_idx"] += 1
            return argmax_tokens

        def prefill_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> None:
            _dispatch_target_tile(valid_prefix_len, tok_tile, pos_tile, mask_tile)

        def greedy_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> np.ndarray:
            argmax_tokens = _dispatch_target_tile(
                valid_prefix_len, tok_tile, pos_tile, mask_tile
            )
            return np.asarray(jax.device_get(argmax_tokens), dtype=np.int32)

        def commit_kv_fn(prev_prefix_len: int, accepted_row_indices: list[int]) -> None:
            n_valid = len(accepted_row_indices)
            src_rows = np.zeros((T,), dtype=np.int32)
            if n_valid > 0:
                src_rows[:n_valid] = np.asarray(accepted_row_indices, dtype=np.int32)
            k_n, v_n, dk_n, dv_n = self._commit_both_kv_jit(
                draft_wt,
                state["k_caches"],
                state["v_caches"],
                state["step_k"],
                state["step_v"],
                state["dk_cache"],
                state["dv_cache"],
                state["step_ctx"],
                self.target._put_rep(prev_prefix_len, dtype=jnp.int32),
                self.target._put_rep(src_rows),
                self.target._put_rep(n_valid, dtype=jnp.int32),
            )
            state["k_caches"] = k_n
            state["v_caches"] = v_n
            state["dk_cache"] = dk_n
            state["dv_cache"] = dv_n
            if state["step_idx"] == prefill_tiles and state["t_prefill_done"] is None:
                jax.block_until_ready((k_n, v_n, dk_n, dv_n))
                state["t_prefill_done"] = time.perf_counter()

        def draft_tree_fn(
            valid_prefix_len: int,
            history_arr: np.ndarray,
            suffix_index: tpu.SuffixIndex,
            max_nodes: int,
            max_depth: int,
            tile_size: int,
        ) -> tpu.SystolicTree:
            root_tok = int(history_arr[-1])
            top_ids_dev, top_probs_dev = self._block_draft_top_k_jit(
                embed,
                lm_arr,
                lm_head_scale,
                draft_wt,
                self.target._put_rep(root_tok, dtype=jnp.int32),
                self.target._put_rep(valid_prefix_len, dtype=jnp.int32),
                state["dk_cache"],
                state["dv_cache"],
            )
            top_ids_np, top_probs_np = jax.device_get((top_ids_dev, top_probs_dev))
            steps = min(int(top_ids_np.shape[0]), int(max_depth))
            ids_slice = np.ascontiguousarray(top_ids_np[:steps], dtype=np.int32)
            probs_slice = np.ascontiguousarray(top_probs_np[:steps], dtype=np.float32)
            return tpu.TreeBuilder.build_from_block_top_k(
                root_tok,
                ids_slice,
                probs_slice,
                suffix_fallback=suffix_index if use_suffix_fallback else None,
                max_nodes=int(max_nodes),
                max_depth=int(max_depth),
                tile_size=int(tile_size),
            )

        arena_cfg = tpu.StaticArenaConfig()
        arena_cfg.max_seq_len = self.max_seq_len
        arena_cfg.hidden_dim = self.target.config.hidden_size
        arena_cfg.num_kv_heads = self.target.config.num_key_value_heads
        arena_cfg.head_dim = self.target.config.head_dim
        arena_cfg.vocab_size = self.target.config.vocab_size
        arena_cfg.tile_size = T

        engine = tpu.ForgeEngine(
            arena_cfg,
            None,
            commit_kv_fn,
            f"gemma4-tpuforge-block-tree-{T}",
            prefill_step_fn=prefill_step_fn,
            greedy_step_fn=greedy_step_fn,
            draft_tree_fn=draft_tree_fn,
        )
        seed_seqs = (
            [[int(t) for t in seq] for seq in seed_token_sequences if len(seq) >= 2]
            if seed_token_sequences is not None
            else []
        )
        req = tpu.GenerationRequest()
        req.prompt_tokens = list(prompt_tokens)
        req.eos_token_ids = eos_list
        if seed_seqs:
            req.seed_token_sequences = seed_seqs
        req.max_tokens = int(max_new_tokens)
        req.temperature = 0.0

        t_all0 = time.perf_counter()
        res = engine.generate(req)
        jax.block_until_ready((state["k_caches"], state["v_caches"], state["dk_cache"], state["dv_cache"]))
        t_all1 = time.perf_counter()

        compiles_delta = self.total_compile_count() - compiles_before
        token_ids = [int(t) for t in res.token_ids]
        t_split = state["t_prefill_done"] if state["t_prefill_done"] is not None else t_all0
        prefill_s = t_split - t_all0
        decode_s = t_all1 - t_split
        gen_len = len(token_ids)
        return RunMetrics(
            mode="tpuforge_block_tree",
            token_ids=token_ids,
            text=self.target._decode_text(token_ids),
            prompt_len=len(prompt_tokens),
            generated_len=gen_len,
            compile_count=compiles_delta,
            prefill_s=prefill_s,
            decode_s=decode_s,
            total_s=t_all1 - t_all0,
            decode_tok_per_s=(gen_len / decode_s) if decode_s > 0 else 0.0,
            forward_passes=int(res.target_forward_passes),
            decode_passes=int(res.decode_forward_passes),
            draft_tokens_accepted=int(res.draft_tokens_accepted),
            mean_tokens_per_decode_pass=float(res.mean_tokens_per_decode_pass),
            extra={
                "block_size": self.draft_config.block_size,
                "top_k": self.top_k,
                "draft_layers": self.draft_config.num_hidden_layers,
                "tile_size": self.tile_size,
                "tp_size": self.tp_size,
                "weight_quant": self.target.weight_quant,
                "draft_weight_quant": self.weight_quant,
                "ast_seed_tokens": sum(len(seq) for seq in seed_seqs),
                "arch": "gemma4",
            },
        )


def _augment_gemma4_eos(config: Gemma4Config, tokenizer: Any) -> Gemma4Config:
    eos_list = list(config.eos_token_ids)
    tok_to_id = getattr(tokenizer, "token_to_id", None)
    if tok_to_id is not None:
        for tok_str in ("<turn|>", "<end_of_turn>", "<eos>"):
            tid = tok_to_id(tok_str)
            if isinstance(tid, int) and 0 <= tid < config.vocab_size and tid not in eos_list:
                eos_list.append(tid)
    if tuple(eos_list) == config.eos_token_ids:
        return config
    from dataclasses import replace

    return replace(config, eos_token_ids=tuple(eos_list))


def load_gemma4_runner(
    model_id_or_path: str,
    max_seq_len: int = 1024,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    enable_xla_cache: bool = True,
    weight_quant: str = "none",
) -> tuple[Gemma4Runner, Path]:
    """Load config, tokenizer, and Gemma 4 weights onto TPU HBM."""
    from tokenizers import Tokenizer

    if enable_xla_cache:
        configure_xla_compilation_cache()

    model_dir = resolve_model_dir(model_id_or_path)
    config = Gemma4Config.from_dir(model_dir)
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    config = _augment_gemma4_eos(config, tokenizer)

    dtype = resolve_default_dtype(dtype_name)
    _, mesh = resolve_gemma4_tp_mesh(tp_size, config=config)
    weights = Gemma4Weights.from_model_dir(
        model_dir, config, dtype=dtype, mesh=mesh, weight_quant=weight_quant
    )
    runner = Gemma4Runner(
        config=config,
        weights=weights,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tile_size=tile_size,
        mesh=mesh,
    )
    return runner, model_dir


def load_gemma4_block_tree_runner(
    model_id_or_path: str,
    draft_model_id_or_path: str,
    max_seq_len: int = 1024,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    top_k: int = 8,
    enable_xla_cache: bool = True,
    weight_quant: str = "none",
    draft_weight_quant: str | None = None,
) -> tuple[Gemma4BlockTreeRunner, Path, Path]:
    """Load target Gemma 4 runner + parallel DFlash drafter into Gemma4BlockTreeRunner."""
    target_runner, model_dir = load_gemma4_runner(
        model_id_or_path=model_id_or_path,
        max_seq_len=max_seq_len,
        dtype_name=dtype_name,
        tile_size=tile_size,
        tp_size=tp_size,
        enable_xla_cache=enable_xla_cache,
        weight_quant=weight_quant,
    )
    draft_dir = resolve_model_dir(draft_model_id_or_path)
    draft_cfg = BlockDiffusionConfig.from_dir(draft_dir, target_config=target_runner.config)
    eff_draft_quant = (
        target_runner.weight_quant if draft_weight_quant is None else draft_weight_quant
    )
    draft_weights = BlockDiffusionWeights.from_model_dir(
        draft_dir,
        draft_cfg,
        dtype=target_runner.dtype,
        mesh=target_runner.mesh,
        weight_quant=eff_draft_quant,
    )
    runner = Gemma4BlockTreeRunner(
        target_runner=target_runner,
        draft_config=draft_cfg,
        draft_weights=draft_weights,
        top_k=top_k,
    )
    return runner, model_dir, draft_dir


def load_gemma4_moe_stream_runner(
    model_id_or_path: str,
    max_seq_len: int = 1024,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    weight_quant: str = "none",
    host_store_mode: str = "mmap",
    stage_experts: int = 16,
    lru_cache_experts: int = 64,
    enable_xla_cache: bool = True,
) -> tuple[Gemma4MoeStreamRunner, Path]:
    """Load Gemma 4 MoE trunk on HBM while keeping routed experts in host mmap/DRAM."""
    from tokenizers import Tokenizer

    if enable_xla_cache:
        configure_xla_compilation_cache()

    norm_store = normalize_host_store_mode(host_store_mode)
    if norm_store not in VALID_HOST_STORE_MODES:
        raise ValueError(f"Unsupported host_store_mode={norm_store!r}")

    model_dir = resolve_model_dir(model_id_or_path)
    config = Gemma4Config.from_dir(model_dir)
    if not config.enable_moe_block:
        raise ValueError(
            f"{model_id_or_path!r} is dense (enable_moe_block=False); "
            "use --host-store hbm for dense checkpoints"
        )
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))
    config = _augment_gemma4_eos(config, tokenizer)

    dtype = resolve_default_dtype(dtype_name)
    _, mesh = resolve_gemma4_tp_mesh(tp_size, config=config)
    trunk_weights = Gemma4Weights.from_model_dir(
        model_dir,
        config,
        dtype=dtype,
        mesh=mesh,
        weight_quant=weight_quant,
        include_experts_in_hbm=False,
    )
    expert_store = ExpertHostStore(
        model_dir=model_dir,
        config=config,  # type: ignore[arg-type]
        dtype=dtype,
        weight_quant=weight_quant,
        host_store_mode=norm_store,
        lru_cache_experts=lru_cache_experts,
        layer_prefix=trunk_weights.layer_prefix,
        experts_subpath="experts",
    )
    runner = Gemma4MoeStreamRunner(
        config=config,
        trunk_weights=trunk_weights,
        expert_store=expert_store,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tile_size=tile_size,
        mesh=mesh,
        stage_experts=stage_experts,
    )
    return runner, model_dir

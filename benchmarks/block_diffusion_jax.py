"""Pure-JAX block-diffusion systolic tree drafter for TPUForge."""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from benchmarks.quantization_jax import (
    DraftWeightsTuple,
    LayerScanWeights,
    TargetWeightsTuple,
    normalize_weight_quant,
    place_np_sharded_preserved,
    quant_np_dtype,
    quantize_and_place_2d,
    quantize_np_per_channel,
    scale_spec_for_weight,
)
from benchmarks.qwen_jax import (
    _MASK_BLOCK_VAL,
    QwenBenchmarkRunner,
    QwenConfig,
    QwenWeights,
    RunMetrics,
    _apply_qwen_rope,
    _gqa_attention,
    _iter_safetensors_index,
    _jnp_to_numpy_dtype,
    _linear,
    _MappedSafetensorsReader,
    _qwen_block_post_attn,
    _qwen_qkv_proj,
    _read_indexed_tensor,
    _rms_norm,
    _split_tensor_shard,
    configure_xla_compilation_cache,
    resolve_default_dtype,
    resolve_model_dir,
    resolve_tp_mesh,
    tpu,
)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> tuple[int, ...]:
    """Compute default uniformly spaced target layer indices for block-diffusion capture."""
    if num_target_layers <= 0 or num_draft_layers <= 0:
        raise ValueError("num_target_layers and num_draft_layers must be positive")
    if num_draft_layers == 1:
        return (num_target_layers // 2,)
    start = 1 if num_target_layers > 3 else 0
    end = max(start, num_target_layers - 3)
    span = end - start
    return tuple(
        round(start + (i * span) / (num_draft_layers - 1))
        for i in range(num_draft_layers)
    )


@dataclass(frozen=True)
class BlockDiffusionConfig:
    """Configuration for a parallel block-diffusion draft checkpoint (supports DFlash config schema)."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    is_causal_override: bool | None = None
    final_logit_softcapping: float | None = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0

    def effective_layer_types(self) -> tuple[str, ...]:
        """Return per-layer attention types ('full_attention' or 'sliding_attention')."""
        if self.layer_types:
            return self.layer_types
        return ("full_attention",) * self.num_hidden_layers

    @classmethod
    def from_dir(
        cls,
        model_dir: Path,
        target_config: Any | None = None,
    ) -> BlockDiffusionConfig:
        cfg_path = model_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        dcfg = raw.get("block_diffusion_config") or raw.get("dflash_config") or {}

        hidden_size = int(raw.get("hidden_size") or (target_config.hidden_size if target_config else 0))
        num_layers = int(raw["num_hidden_layers"])
        num_heads = int(raw["num_attention_heads"])
        num_kv = int(raw["num_key_value_heads"])
        intermediate_size = int(raw["intermediate_size"])
        vocab_size = int(raw.get("vocab_size") or (target_config.vocab_size if target_config else 151936))
        head_dim = int(raw.get("head_dim") or (hidden_size // num_heads))
        rms_norm_eps = float(raw.get("rms_norm_eps", 1e-6))

        rope_params = raw.get("rope_parameters") or {}
        if not isinstance(rope_params, dict):
            rope_params = {}
        raw_rope_theta = raw.get("rope_theta")
        if raw_rope_theta is None:
            raw_rope_theta = rope_params.get("rope_theta")
            if raw_rope_theta is None and isinstance(rope_params.get("full_attention"), dict):
                raw_rope_theta = rope_params["full_attention"].get("rope_theta")
        rope_theta = float(raw_rope_theta if raw_rope_theta is not None else 1_000_000.0)

        block_size = int(dcfg.get("block_size", raw.get("block_size", 16)))
        if block_size < 2:
            raise ValueError(f"BlockDiffusion block_size must be >= 2, got {block_size}")

        mask_raw = dcfg.get("mask_token_id", raw.get("mask_token_id"))
        if mask_raw is None:
            raise KeyError(f"Missing mask_token_id in BlockDiffusion config: {cfg_path}")
        mask_token_id = int(mask_raw)

        num_target_raw = raw.get(
            "num_target_layers",
            dcfg.get("num_target_layers", target_config.num_hidden_layers if target_config else None),
        )
        raw_layer_ids = dcfg.get("target_layer_ids", raw.get("target_layer_ids"))
        if raw_layer_ids is None:
            if num_target_raw is None:
                raise KeyError("Missing both target_layer_ids and num_target_layers in BlockDiffusion config")
            num_target_layers = int(num_target_raw)
            target_layer_ids = build_target_layer_ids(num_target_layers, num_layers)
        else:
            target_layer_ids = tuple(int(x) for x in raw_layer_ids)
            if not target_layer_ids:
                raise ValueError("BlockDiffusion target_layer_ids cannot be empty")
            num_target_layers = (
                int(num_target_raw) if num_target_raw is not None else (max(target_layer_ids) + 1)
            )

        for lid in target_layer_ids:
            if lid < 0 or lid >= num_target_layers:
                raise ValueError(
                    f"BlockDiffusion target_layer_id {lid} out of range [0, {num_target_layers})"
                )

        raw_lt = raw.get("layer_types")
        if raw_lt is not None:
            layer_types = tuple(str(x) for x in raw_lt)
            if len(layer_types) != num_layers:
                raise ValueError(
                    f"BlockDiffusion layer_types length ({len(layer_types)}) != num_hidden_layers ({num_layers})"
                )
        else:
            layer_types = ("full_attention",) * num_layers

        raw_sw = raw.get("sliding_window")
        sliding_window = int(raw_sw) if raw_sw is not None else None
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError(f"BlockDiffusion sliding_window must be positive, got {sliding_window}")

        raw_causal = dcfg.get("is_causal", raw.get("is_causal"))
        is_causal_override = bool(raw_causal) if raw_causal is not None else None

        raw_softcap = dcfg.get("final_logit_softcapping", raw.get("final_logit_softcapping"))
        final_logit_softcapping = float(raw_softcap) if raw_softcap is not None else None
        if final_logit_softcapping is not None and final_logit_softcapping <= 0.0:
            final_logit_softcapping = None

        input_embedding_scale = float(
            dcfg.get("input_embedding_scale", raw.get("input_embedding_scale", 1.0))
        )
        output_multiplier = float(
            dcfg.get("output_multiplier", raw.get("output_multiplier", 1.0))
        )

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            block_size=block_size,
            mask_token_id=mask_token_id,
            target_layer_ids=target_layer_ids,
            num_target_layers=num_target_layers,
            layer_types=layer_types,
            sliding_window=sliding_window,
            is_causal_override=is_causal_override,
            final_logit_softcapping=final_logit_softcapping,
            input_embedding_scale=input_embedding_scale,
            output_multiplier=output_multiplier,
        )


def build_dflash_layer_masks(
    dcfg: BlockDiffusionConfig,
    cur_prefix: jax.Array,
    max_seq: int,
) -> jax.Array:
    """Build per-layer [num_draft_layers, block_size, max_seq + block_size] additive masks."""
    block_size = dcfg.block_size
    q_pos = cur_prefix + jnp.arange(block_size, dtype=jnp.int32)[:, None]
    cache_kv_pos = jnp.arange(max_seq, dtype=jnp.int32)[None, :]
    noise_kv_pos = cur_prefix + jnp.arange(block_size, dtype=jnp.int32)[None, :]
    kv_pos = jnp.concatenate([cache_kv_pos, noise_kv_pos], axis=-1)

    valid_cache = cache_kv_pos < cur_prefix
    valid_noise = jnp.ones((1, block_size), dtype=bool)
    valid_key = jnp.broadcast_to(
        jnp.concatenate([valid_cache, valid_noise], axis=-1),
        (block_size, max_seq + block_size),
    )

    layer_masks = []
    for lt in dcfg.effective_layer_types():
        is_causal = (
            (lt == "sliding_attention")
            if dcfg.is_causal_override is None
            else dcfg.is_causal_override
        )
        sw = dcfg.sliding_window if lt == "sliding_attention" else None
        vis = valid_key
        if is_causal:
            vis = vis & (kv_pos <= q_pos)
        if sw is not None:
            vis = vis & ((q_pos - kv_pos) < int(sw))
            if not is_causal:
                vis = vis & ((kv_pos - q_pos) < int(sw))
        layer_masks.append(
            jnp.where(vis, jnp.float32(0.0), jnp.float32(_MASK_BLOCK_VAL))
        )
    return jnp.stack(layer_masks, axis=0)


def apply_dflash_output_head(
    normed: jax.Array,
    lm_head: jax.Array,
    lm_head_scale: jax.Array | None,
    dcfg: BlockDiffusionConfig,
    transpose_lm_head: bool = False,
) -> jax.Array:
    """Compute draft vocabulary logits with optional output_multiplier and logit softcapping."""
    if transpose_lm_head and lm_head_scale is None:
        if normed.dtype in (jnp.bfloat16, jnp.float16):
            logits = jax.lax.dot_general(
                normed,
                lm_head,
                (((normed.ndim - 1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            logits = jax.lax.dot_general(
                normed,
                lm_head,
                (((normed.ndim - 1,), (1,)), ((), ())),
                precision=jax.lax.Precision.HIGHEST,
            ).astype(jnp.float32)
    else:
        logits = _linear(normed, lm_head, w_scale=lm_head_scale, out_dtype=jnp.float32)
    if dcfg.output_multiplier != 1.0:
        logits = logits * jnp.float32(dcfg.output_multiplier)
    softcap = dcfg.final_logit_softcapping
    if softcap is not None and float(softcap) > 0.0:
        cap = jnp.float32(softcap)
        logits = jnp.tanh(logits / cap) * cap
    return logits


def commit_draft_context_kv(
    draft_wt: DraftWeightsTuple,
    dcfg: BlockDiffusionConfig,
    draft_k_cache: jax.Array,
    draft_v_cache: jax.Array,
    gathered_ctx: jax.Array,
    dest_positions: jax.Array,
    dest_slots: jax.Array,
    rep_sharding: NamedSharding | None,
    kv_head_sharding: NamedSharding | None,
    kv_cache_sharding: NamedSharding | None,
) -> tuple[jax.Array, jax.Array]:
    """Project accepted target hidden states once and scatter into draft K/V caches."""
    tile_size = int(gathered_ctx.shape[0])
    target_h = _rms_norm(
        _linear(gathered_ctx, draft_wt.fc, w_scale=draft_wt.fc_scale),
        draft_wt.hidden_norm,
        dcfg.rms_norm_eps,
    )
    if rep_sharding is not None:
        target_h = jax.lax.with_sharding_constraint(target_h, rep_sharding)

    dl = draft_wt.layers

    def draft_kv_body(
        _carry: None,
        xs: tuple[jax.Array | None, ...],
    ) -> tuple[None, tuple[jax.Array, jax.Array]]:
        wk_l, wk_s, bk_l, wv_l, wv_s, bv_l, kn_l, dkc, dvc = xs
        k_ctx = _linear(target_h, wk_l, bk_l, w_scale=wk_s).reshape(
            tile_size, dcfg.num_key_value_heads, dcfg.head_dim
        )
        if kn_l is not None:
            k_ctx = _rms_norm(k_ctx, kn_l, dcfg.rms_norm_eps)
        k_ctx = _apply_qwen_rope(k_ctx, dest_positions, dcfg.head_dim, dcfg.rope_theta)
        v_ctx = _linear(target_h, wv_l, bv_l, w_scale=wv_s).reshape(
            tile_size, dcfg.num_key_value_heads, dcfg.head_dim
        )
        if kv_head_sharding is not None:
            k_ctx = jax.lax.with_sharding_constraint(k_ctx, kv_head_sharding)
            v_ctx = jax.lax.with_sharding_constraint(v_ctx, kv_head_sharding)
        dkc = dkc.at[dest_slots].set(k_ctx)
        dvc = dvc.at[dest_slots].set(v_ctx)
        return None, (dkc, dvc)

    _, (dk_next, dv_next) = jax.lax.scan(
        draft_kv_body,
        None,
        (
            dl.wk,
            dl.wk_scale,
            dl.bk,
            dl.wv,
            dl.wv_scale,
            dl.bv,
            dl.k_norm,
            draft_k_cache,
            draft_v_cache,
        ),
    )
    if kv_cache_sharding is not None:
        dk_next = jax.lax.with_sharding_constraint(dk_next, kv_cache_sharding)
        dv_next = jax.lax.with_sharding_constraint(dv_next, kv_cache_sharding)
    return dk_next, dv_next


def run_dflash_block_step(
    raw_embed: jax.Array,
    lm_head: jax.Array,
    lm_head_scale: jax.Array | None,
    draft_wt: DraftWeightsTuple,
    dcfg: BlockDiffusionConfig,
    root_token_id: jax.Array,
    cur_prefix: jax.Array,
    draft_k_cache: jax.Array,
    draft_v_cache: jax.Array,
    max_seq: int,
    top_k: int,
    rep_sharding: NamedSharding | None,
    head_sharding: NamedSharding | None,
    kv_head_sharding: NamedSharding | None,
    transpose_lm_head: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run one parallel block-diffusion draft step and return (top_ids, top_probs)."""
    block_size = dcfg.block_size
    mask_fill = jnp.full((block_size - 1,), dcfg.mask_token_id, dtype=jnp.int32)
    block_ids = jnp.concatenate([root_token_id.reshape((1,)), mask_fill], axis=0)
    block_pos = cur_prefix + jnp.arange(block_size, dtype=jnp.int32)

    x0 = raw_embed[block_ids]
    if dcfg.input_embedding_scale != 1.0:
        x0 = x0 * jnp.asarray(dcfg.input_embedding_scale, dtype=x0.dtype)
    if rep_sharding is not None:
        x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)

    layer_masks = build_dflash_layer_masks(dcfg, cur_prefix, max_seq)

    def draft_body(
        x: jax.Array,
        scan_in: tuple[LayerScanWeights, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, None]:
        if rep_sharding is not None:
            x = jax.lax.with_sharding_constraint(x, rep_sharding)
        lw, dkc, dvc, layer_mask = scan_in
        q, k_noise, v_noise = _qwen_qkv_proj(
            x, lw, block_pos, dcfg, head_sharding, kv_head_sharding
        )
        k_all = jnp.concatenate([dkc[:max_seq], k_noise], axis=0)
        v_all = jnp.concatenate([dvc[:max_seq], v_noise], axis=0)
        attn_out = _gqa_attention(
            q,
            k_all,
            v_all,
            layer_mask,
            dcfg.num_attention_heads,
            dcfg.num_key_value_heads,
            dcfg.head_dim,
        )
        x_next = _qwen_block_post_attn(x, attn_out, lw, dcfg, rep_sharding)
        return x_next, None

    x_final, _ = jax.lax.scan(
        draft_body,
        x0,
        (draft_wt.layers, draft_k_cache, draft_v_cache, layer_masks),
    )
    normed = _rms_norm(x_final[1:], draft_wt.final_norm, dcfg.rms_norm_eps)
    logits = apply_dflash_output_head(
        normed, lm_head, lm_head_scale, dcfg, transpose_lm_head=transpose_lm_head
    )
    if rep_sharding is not None:
        logits = jax.lax.with_sharding_constraint(logits, rep_sharding)
    top_logits, top_ids = jax.lax.top_k(logits, top_k)
    top_probs = jax.nn.softmax(top_logits, axis=-1)
    if rep_sharding is not None:
        top_ids = jax.lax.with_sharding_constraint(top_ids, rep_sharding)
        top_probs = jax.lax.with_sharding_constraint(top_probs, rep_sharding)
    return top_ids.astype(jnp.int32), top_probs.astype(jnp.float32)


def _place_np_sharded(
    full_np: np.ndarray,
    spec: P,
    mesh: Mesh | None,
    dtype: jnp.dtype,
    np_dtype: Any,
) -> jax.Array:
    if mesh is None:
        return jax.device_put(jnp.asarray(full_np, dtype=dtype))
    sharding = NamedSharding(mesh, spec)
    tp_size = mesh.size
    local_devs = set(mesh.local_devices)
    all_devs = list(mesh.devices.flat)
    shards = []
    for dev in all_devs:
        if dev not in local_devs:
            continue
        rank = all_devs.index(dev)
        part = _split_tensor_shard(full_np, spec, rank, tp_size)
        shards.append(jax.device_put(jnp.asarray(part, dtype=np_dtype), dev))
    return jax.make_array_from_single_device_arrays(full_np.shape, sharding, shards)


@dataclass
class BlockDiffusionWeights:
    """Stacked (optionally TP-sharded and quantized) JAX arrays for a block-diffusion drafter."""

    fc: jax.Array
    hidden_norm: jax.Array
    final_norm: jax.Array
    in_norm: jax.Array
    post_norm: jax.Array
    wq: jax.Array
    bq: jax.Array | None
    wk: jax.Array
    bk: jax.Array | None
    wv: jax.Array
    bv: jax.Array | None
    wo: jax.Array
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array
    mesh: Mesh | None = None
    q_norm: jax.Array | None = None
    k_norm: jax.Array | None = None
    weight_quant: str = "none"
    fc_scale: jax.Array | None = None
    wq_scale: jax.Array | None = None
    wk_scale: jax.Array | None = None
    wv_scale: jax.Array | None = None
    wo_scale: jax.Array | None = None
    w_gate_scale: jax.Array | None = None
    w_up_scale: jax.Array | None = None
    w_down_scale: jax.Array | None = None

    def to_layer_scan_weights(self) -> LayerScanWeights:
        """Pack stacked per-layer drafter weights and optional scales for lax.scan."""
        return LayerScanWeights(
            in_norm=self.in_norm,
            post_norm=self.post_norm,
            wq=self.wq,
            wq_scale=self.wq_scale,
            bq=self.bq,
            wk=self.wk,
            wk_scale=self.wk_scale,
            bk=self.bk,
            wv=self.wv,
            wv_scale=self.wv_scale,
            bv=self.bv,
            wo=self.wo,
            wo_scale=self.wo_scale,
            w_gate=self.w_gate,
            w_gate_scale=self.w_gate_scale,
            w_up=self.w_up,
            w_up_scale=self.w_up_scale,
            w_down=self.w_down,
            w_down_scale=self.w_down_scale,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )

    def to_draft_weights_tuple(self) -> DraftWeightsTuple:
        """Pack drafter device arrays into an explicit constant-free pytree."""
        return DraftWeightsTuple(
            fc=self.fc,
            fc_scale=self.fc_scale,
            hidden_norm=self.hidden_norm,
            final_norm=self.final_norm,
            layers=self.to_layer_scan_weights(),
        )

    @classmethod
    def from_numpy_dict(
        cls,
        raw: dict[str, np.ndarray],
        config: BlockDiffusionConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        weight_quant: str = "none",
    ) -> BlockDiffusionWeights:
        norm_quant = normalize_weight_quant(weight_quant)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        top_prefix = "" if "fc.weight" in raw else ("model." if "model.fc.weight" in raw else "")
        layer_prefix = (
            ""
            if "layers.0.input_layernorm.weight" in raw
            else ("model." if "model.layers.0.input_layernorm.weight" in raw else top_prefix)
        )

        def get_top_vec(name: str) -> jax.Array:
            key = f"{top_prefix}{name}"
            if key not in raw:
                raise KeyError(f"Missing block-drafter tensor {key!r}")
            return _place_np_sharded(raw[key], P(), mesh, dtype, np_dtype)

        def get_fc() -> tuple[jax.Array, jax.Array | None]:
            key = f"{top_prefix}fc.weight"
            if key not in raw:
                raise KeyError(f"Missing block-drafter tensor {key!r}")
            return quantize_and_place_2d(
                np.ascontiguousarray(raw[key].T), P(), mesh, norm_quant, active_np_dtype=np_dtype
            )

        def stack_layer(suffix: str, spec: P | None = None) -> jax.Array:
            items = []
            for i in range(config.num_hidden_layers):
                key = f"{layer_prefix}layers.{i}.{suffix}"
                if key not in raw:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                items.append(raw[key])
            return _place_np_sharded(
                np.stack(items, axis=0), P() if spec is None else spec, mesh, dtype, np_dtype
            )

        def stack_layer_optional(suffix: str, spec: P | None = None) -> jax.Array | None:
            if f"{layer_prefix}layers.0.{suffix}" not in raw:
                return None
            return stack_layer(suffix, spec=spec)

        def stack_linear(suffix: str, spec: P) -> tuple[jax.Array, jax.Array | None]:
            if norm_quant == "none":
                items = []
                for i in range(config.num_hidden_layers):
                    key = f"{layer_prefix}layers.{i}.{suffix}"
                    if key not in raw:
                        raise KeyError(f"Missing block-drafter tensor {key!r}")
                    items.append(np.ascontiguousarray(raw[key].T))
                return _place_np_sharded(np.stack(items, axis=0), spec, mesh, dtype, np_dtype), None

            q_items = []
            s_items = []
            for i in range(config.num_hidden_layers):
                key = f"{layer_prefix}layers.{i}.{suffix}"
                if key not in raw:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                q_i, s_i = quantize_np_per_channel(np.ascontiguousarray(raw[key].T), norm_quant)
                q_items.append(q_i)
                s_items.append(s_i)
            return (
                place_np_sharded_preserved(np.stack(q_items, axis=0), spec, mesh),
                place_np_sharded_preserved(
                    np.stack(s_items, axis=0), scale_spec_for_weight(spec), mesh
                ),
            )

        fc, fc_scale = get_fc()
        wq, wq_scale = stack_linear("self_attn.q_proj.weight", P(None, None, "tp"))
        wk, wk_scale = stack_linear("self_attn.k_proj.weight", P(None, None, "tp"))
        wv, wv_scale = stack_linear("self_attn.v_proj.weight", P(None, None, "tp"))
        wo, wo_scale = stack_linear("self_attn.o_proj.weight", P(None, "tp", None))
        w_gate, w_gate_scale = stack_linear("mlp.gate_proj.weight", P(None, None, "tp"))
        w_up, w_up_scale = stack_linear("mlp.up_proj.weight", P(None, None, "tp"))
        w_down, w_down_scale = stack_linear("mlp.down_proj.weight", P(None, "tp", None))

        return cls(
            fc=fc,
            hidden_norm=get_top_vec("hidden_norm.weight"),
            final_norm=get_top_vec("norm.weight"),
            in_norm=stack_layer("input_layernorm.weight", spec=P()),
            post_norm=stack_layer("post_attention_layernorm.weight", spec=P()),
            wq=wq,
            bq=stack_layer_optional("self_attn.q_proj.bias", spec=P(None, "tp")),
            wk=wk,
            bk=stack_layer_optional("self_attn.k_proj.bias", spec=P(None, "tp")),
            wv=wv,
            bv=stack_layer_optional("self_attn.v_proj.bias", spec=P(None, "tp")),
            wo=wo,
            w_gate=w_gate,
            w_up=w_up,
            w_down=w_down,
            mesh=mesh,
            q_norm=stack_layer_optional("self_attn.q_norm.weight", spec=P()),
            k_norm=stack_layer_optional("self_attn.k_norm.weight", spec=P()),
            weight_quant=norm_quant,
            fc_scale=fc_scale,
            wq_scale=wq_scale,
            wk_scale=wk_scale,
            wv_scale=wv_scale,
            wo_scale=wo_scale,
            w_gate_scale=w_gate_scale,
            w_up_scale=w_up_scale,
            w_down_scale=w_down_scale,
        )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        config: BlockDiffusionConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        weight_quant: str = "none",
    ) -> BlockDiffusionWeights:
        norm_quant = normalize_weight_quant(weight_quant)
        tensor_loc = _iter_safetensors_index(model_dir)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        tp_size = mesh.size if mesh is not None else 1

        top_prefix = "" if "fc.weight" in tensor_loc else ("model." if "model.fc.weight" in tensor_loc else "")
        layer_prefix = (
            ""
            if "layers.0.input_layernorm.weight" in tensor_loc
            else ("model." if "model.layers.0.input_layernorm.weight" in tensor_loc else top_prefix)
        )

        with _MappedSafetensorsReader() as reader:
            def read_top_vec(name: str) -> jax.Array:
                key = f"{top_prefix}{name}"
                if key not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                return _place_np_sharded(arr, P(), mesh, dtype, np_dtype)

            def read_fc() -> tuple[jax.Array, jax.Array | None]:
                key = f"{top_prefix}fc.weight"
                if key not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                raw_fc = _read_indexed_tensor(
                    tensor_loc[key], None if norm_quant != "none" else np_dtype, reader=reader
                )
                return quantize_and_place_2d(
                    np.ascontiguousarray(raw_fc.T), P(), mesh, norm_quant, active_np_dtype=np_dtype
                )

            def stream_stacked_vec(suffix: str, spec: P) -> jax.Array:
                num_layers = config.num_hidden_layers
                key0 = f"{layer_prefix}layers.0.{suffix}"
                if key0 not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key0!r}")
                layer_shape = tensor_loc[key0][2]
                full_shape = (num_layers,) + layer_shape

                if mesh is None:
                    stacked = np.empty(full_shape, dtype=np_dtype)
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        stacked[i] = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                    out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
                    del stacked
                    return out

                sharding = NamedSharding(mesh, spec)
                local_devs = set(mesh.local_devices)
                all_devs = list(mesh.devices.flat)
                sample = _split_tensor_shard(
                    np.empty((1,) + layer_shape, dtype=np_dtype), spec, 0, tp_size
                )
                shard_layer_shape = sample.shape[1:]
                del sample

                device_arrays = []
                for dev in all_devs:
                    if dev not in local_devs:
                        continue
                    rank = all_devs.index(dev)
                    rank_buf = np.empty((num_layers,) + shard_layer_shape, dtype=np_dtype)
                    layer_spec = P(*spec[1:])
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                        rank_buf[i] = _split_tensor_shard(arr, layer_spec, rank, tp_size)
                        del arr
                    device_arrays.append(jax.device_put(jnp.asarray(rank_buf), dev))
                    del rank_buf
                return jax.make_array_from_single_device_arrays(full_shape, sharding, device_arrays)

            def stream_stacked_vec_optional(suffix: str, spec: P) -> jax.Array | None:
                if f"{layer_prefix}layers.0.{suffix}" not in tensor_loc:
                    return None
                return stream_stacked_vec(suffix, spec)

            def stream_stacked_linear(
                suffix: str,
                spec: P,
            ) -> tuple[jax.Array, jax.Array | None]:
                num_layers = config.num_hidden_layers
                key0 = f"{layer_prefix}layers.0.{suffix}"
                if key0 not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key0!r}")
                layer_shape = tuple(reversed(tensor_loc[key0][2]))
                full_shape = (num_layers,) + layer_shape

                if norm_quant == "none":
                    if mesh is None:
                        stacked = np.empty(full_shape, dtype=np_dtype)
                        for i in range(num_layers):
                            key = f"{layer_prefix}layers.{i}.{suffix}"
                            arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                            stacked[i] = np.ascontiguousarray(arr.T)
                            del arr
                        out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
                        del stacked
                        return out, None

                    sharding = NamedSharding(mesh, spec)
                    local_devs = set(mesh.local_devices)
                    all_devs = list(mesh.devices.flat)
                    sample = _split_tensor_shard(
                        np.empty((1,) + layer_shape, dtype=np_dtype), spec, 0, tp_size
                    )
                    shard_layer_shape = sample.shape[1:]
                    del sample
                    layer_spec = P(*spec[1:])

                    device_arrays = []
                    for dev in all_devs:
                        if dev not in local_devs:
                            continue
                        rank = all_devs.index(dev)
                        rank_buf = np.empty((num_layers,) + shard_layer_shape, dtype=np_dtype)
                        for i in range(num_layers):
                            key = f"{layer_prefix}layers.{i}.{suffix}"
                            arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                            rank_buf[i] = _split_tensor_shard(
                                np.ascontiguousarray(arr.T), layer_spec, rank, tp_size
                            )
                            del arr
                        device_arrays.append(jax.device_put(jnp.asarray(rank_buf), dev))
                        del rank_buf
                    return jax.make_array_from_single_device_arrays(
                        full_shape, sharding, device_arrays
                    ), None

                q_dtype = quant_np_dtype(norm_quant)
                scale_spec = scale_spec_for_weight(spec)
                full_scale_shape = (num_layers, layer_shape[-1])

                if mesh is None:
                    q_stacked = np.empty(full_shape, dtype=q_dtype)
                    s_stacked = np.empty(full_scale_shape, dtype=np.float32)
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        raw_w = _read_indexed_tensor(tensor_loc[key], None, reader=reader)
                        q_i, s_i = quantize_np_per_channel(np.ascontiguousarray(raw_w.T), norm_quant)
                        del raw_w
                        q_stacked[i] = q_i
                        s_stacked[i] = s_i
                    q_dev = jax.device_put(jnp.asarray(q_stacked))
                    s_dev = jax.device_put(jnp.asarray(s_stacked))
                    del q_stacked, s_stacked
                    return q_dev, s_dev

                w_sharding = NamedSharding(mesh, spec)
                s_sharding = NamedSharding(mesh, scale_spec)
                local_devs = set(mesh.local_devices)
                all_devs = list(mesh.devices.flat)
                layer_spec = P(*spec[1:])
                layer_scale_spec = scale_spec_for_weight(layer_spec)
                sample_w = _split_tensor_shard(
                    np.empty((1,) + layer_shape, dtype=q_dtype), spec, 0, tp_size
                )
                sample_s = _split_tensor_shard(
                    np.empty((1, layer_shape[-1]), dtype=np.float32), scale_spec, 0, tp_size
                )
                shard_w_shape = sample_w.shape[1:]
                shard_s_shape = sample_s.shape[1:]
                del sample_w, sample_s

                w_arrays = []
                s_arrays = []
                for dev in all_devs:
                    if dev not in local_devs:
                        continue
                    rank = all_devs.index(dev)
                    rank_w_buf = np.empty((num_layers,) + shard_w_shape, dtype=q_dtype)
                    rank_s_buf = np.empty((num_layers,) + shard_s_shape, dtype=np.float32)
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        raw_w = _read_indexed_tensor(tensor_loc[key], None, reader=reader)
                        q_i, s_i = quantize_np_per_channel(np.ascontiguousarray(raw_w.T), norm_quant)
                        del raw_w
                        rank_w_buf[i] = _split_tensor_shard(q_i, layer_spec, rank, tp_size)
                        rank_s_buf[i] = _split_tensor_shard(s_i, layer_scale_spec, rank, tp_size)
                        del q_i, s_i
                    w_arrays.append(jax.device_put(jnp.asarray(rank_w_buf), dev))
                    s_arrays.append(jax.device_put(jnp.asarray(rank_s_buf), dev))
                    del rank_w_buf, rank_s_buf
                return (
                    jax.make_array_from_single_device_arrays(full_shape, w_sharding, w_arrays),
                    jax.make_array_from_single_device_arrays(full_scale_shape, s_sharding, s_arrays),
                )

            fc, fc_scale = read_fc()
            wq, wq_scale = stream_stacked_linear("self_attn.q_proj.weight", P(None, None, "tp"))
            wk, wk_scale = stream_stacked_linear("self_attn.k_proj.weight", P(None, None, "tp"))
            wv, wv_scale = stream_stacked_linear("self_attn.v_proj.weight", P(None, None, "tp"))
            wo, wo_scale = stream_stacked_linear("self_attn.o_proj.weight", P(None, "tp", None))
            w_gate, w_gate_scale = stream_stacked_linear("mlp.gate_proj.weight", P(None, None, "tp"))
            w_up, w_up_scale = stream_stacked_linear("mlp.up_proj.weight", P(None, None, "tp"))
            w_down, w_down_scale = stream_stacked_linear("mlp.down_proj.weight", P(None, "tp", None))

            return cls(
                fc=fc,
                hidden_norm=read_top_vec("hidden_norm.weight"),
                final_norm=read_top_vec("norm.weight"),
                in_norm=stream_stacked_vec("input_layernorm.weight", P()),
                post_norm=stream_stacked_vec("post_attention_layernorm.weight", P()),
                wq=wq,
                bq=stream_stacked_vec_optional("self_attn.q_proj.bias", P(None, "tp")),
                wk=wk,
                bk=stream_stacked_vec_optional("self_attn.k_proj.bias", P(None, "tp")),
                wv=wv,
                bv=stream_stacked_vec_optional("self_attn.v_proj.bias", P(None, "tp")),
                wo=wo,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                mesh=mesh,
                q_norm=stream_stacked_vec_optional("self_attn.q_norm.weight", P()),
                k_norm=stream_stacked_vec_optional("self_attn.k_norm.weight", P()),
                weight_quant=norm_quant,
                fc_scale=fc_scale,
                wq_scale=wq_scale,
                wk_scale=wk_scale,
                wv_scale=wv_scale,
                wo_scale=wo_scale,
                w_gate_scale=w_gate_scale,
                w_up_scale=w_up_scale,
                w_down_scale=w_down_scale,
            )


class QwenBlockTreeRunner:
    """Runs block-diffusion parallel drafting packed with O(1) suffix continuations on systolic tiles."""

    def __init__(
        self,
        target_runner: QwenBenchmarkRunner,
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
        num_captured = len(draft_config.target_layer_ids)
        capture_slots_np = np.array(
            [layer_map.get(i, num_captured) for i in range(target_runner.config.num_hidden_layers)],
            dtype=np.int32,
        )
        self._capture_slots = target_runner._put_rep(capture_slots_np, dtype=jnp.int32)
        self._draft_weights_tuple: DraftWeightsTuple = draft_weights.to_draft_weights_tuple()

        self._compile_block_tree_kernels()

    def _compile_block_tree_kernels(self) -> None:
        tcfg = self.target.config
        dcfg = self.draft_config
        max_seq = self.max_seq_len
        tile_size = self.tile_size
        top_k = self.top_k
        num_captured = len(dcfg.target_layer_ids)

        rep_sharding = self.target.rep_sharding
        head_sharding = self.target.head_sharding
        kv_head_sharding = self.target.kv_head_sharding
        kv_cache_sharding = self.target.kv_cache_sharding
        dtype = self.dtype

        def _empty_draft_kv_cache() -> tuple[jax.Array, jax.Array]:
            shape = (
                dcfg.num_hidden_layers,
                max_seq + 1,
                dcfg.num_key_value_heads,
                dcfg.head_dim,
            )
            if kv_cache_sharding is not None:
                return (
                    jnp.zeros(shape, dtype=dtype, device=kv_cache_sharding),
                    jnp.zeros(shape, dtype=dtype, device=kv_cache_sharding),
                )
            return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)

        self._empty_draft_kv_cache = _empty_draft_kv_cache

        @jax.jit
        def target_tile_with_ctx_jit(
            target_wt: TargetWeightsTuple,
            capture_slots: jax.Array,
            valid_prefix_len: jax.Array,
            token_tile: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            self.target_with_ctx_compile_count += 1
            x0 = target_wt.embed[token_tile]
            if rep_sharding is not None:
                x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)

            prefix_allowed = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) < valid_prefix_len,
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)
            prefix_mask = jnp.broadcast_to(prefix_allowed[None, :], (tile_size, max_seq))
            full_mask = jnp.concatenate([prefix_mask, tile_mask], axis=-1)

            cap_init = jnp.zeros(
                (num_captured + 1, tile_size, tcfg.hidden_size), dtype=dtype
            )
            if rep_sharding is not None:
                cap_init = jax.lax.with_sharding_constraint(cap_init, rep_sharding)

            def body(
                carry: tuple[jax.Array, jax.Array],
                scan_in: tuple[LayerScanWeights, jax.Array, jax.Array, jax.Array],
            ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                x, cap_buf = carry
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                lw, kc, vc, slot = scan_in
                q, k_step, v_step = _qwen_qkv_proj(
                    x, lw, pos_tile, tcfg, head_sharding, kv_head_sharding
                )
                k_all = jnp.concatenate([kc[:max_seq], k_step], axis=0)
                v_all = jnp.concatenate([vc[:max_seq], v_step], axis=0)
                attn_out = _gqa_attention(
                    q,
                    k_all,
                    v_all,
                    full_mask,
                    tcfg.num_attention_heads,
                    tcfg.num_key_value_heads,
                    tcfg.head_dim,
                )
                x_next = _qwen_block_post_attn(x, attn_out, lw, tcfg, rep_sharding)
                if rep_sharding is not None:
                    x_next = jax.lax.with_sharding_constraint(x_next, rep_sharding)
                cap_buf = jax.lax.cond(
                    slot < num_captured,
                    lambda cb: cb.at[slot].set(x_next),
                    lambda cb: cb,
                    cap_buf,
                )
                return (x_next, cap_buf), (k_step, v_step)

            (x_final, cap_done), (step_k_out, step_v_out) = jax.lax.scan(
                body,
                (x0, cap_init),
                (target_wt.layers, k_cache, v_cache, capture_slots),
            )
            normed = _rms_norm(x_final, target_wt.final_norm, tcfg.rms_norm_eps)
            logits_tile = _linear(
                normed, target_wt.lm_head, w_scale=target_wt.lm_head_scale, out_dtype=jnp.float32
            )
            has_nan = jnp.any(jnp.isnan(logits_tile), axis=-1)
            argmax_tokens = jnp.where(
                has_nan,
                jnp.int32(-1),
                jnp.argmax(logits_tile, axis=-1).astype(jnp.int32),
            )
            ctx_features = jnp.transpose(cap_done[:num_captured], (1, 0, 2)).reshape(
                tile_size, num_captured * tcfg.hidden_size
            )
            if rep_sharding is not None:
                argmax_tokens = jax.lax.with_sharding_constraint(argmax_tokens, rep_sharding)
                ctx_features = jax.lax.with_sharding_constraint(ctx_features, rep_sharding)
            return argmax_tokens, step_k_out, step_v_out, ctx_features

        @jax.jit
        def commit_both_kv_jit(
            draft_wt: DraftWeightsTuple,
            k_cache: jax.Array,
            v_cache: jax.Array,
            step_k: jax.Array,
            step_v: jax.Array,
            draft_k_cache: jax.Array,
            draft_v_cache: jax.Array,
            step_ctx_features: jax.Array,
            prev_prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            self.commit_both_kv_compile_count += 1
            idx = jnp.arange(tile_size, dtype=jnp.int32)
            dest_positions = prev_prefix_len + idx
            dest_slots = jnp.where(idx < n_valid, dest_positions, max_seq)

            gathered_k = step_k[:, src_rows, :, :]
            gathered_v = step_v[:, src_rows, :, :]
            k_next = k_cache.at[:, dest_slots, :, :].set(gathered_k)
            v_next = v_cache.at[:, dest_slots, :, :].set(gathered_v)
            if kv_cache_sharding is not None:
                k_next = jax.lax.with_sharding_constraint(k_next, kv_cache_sharding)
                v_next = jax.lax.with_sharding_constraint(v_next, kv_cache_sharding)

            gathered_ctx = step_ctx_features[src_rows]
            dk_next, dv_next = commit_draft_context_kv(
                draft_wt,
                dcfg,
                draft_k_cache,
                draft_v_cache,
                gathered_ctx,
                dest_positions,
                dest_slots,
                rep_sharding,
                kv_head_sharding,
                kv_cache_sharding,
            )
            return k_next, v_next, dk_next, dv_next

        @jax.jit
        def block_draft_top_k_jit(
            embed: jax.Array,
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
                embed,
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
                rep_sharding,
                head_sharding,
                kv_head_sharding,
            )

        self._target_tile_with_ctx_jit = target_tile_with_ctx_jit
        self._commit_both_kv_jit = commit_both_kv_jit
        self._block_draft_top_k_jit = block_draft_top_k_jit

    def total_compile_count(self) -> int:
        return (
            self.target_with_ctx_compile_count
            + self.commit_both_kv_compile_count
            + self.draft_step_compile_count
        )

    def warmup(self) -> float:
        """Warm up all block-drafter and target JIT kernels once so steady-state runs have 0 compiles."""
        t0 = time.perf_counter()
        T = self.tile_size
        k_cache, v_cache = self.target._empty_kv_cache()
        dk_cache, dv_cache = self._empty_draft_kv_cache()

        tok_tile = self.target._put_rep(np.zeros((T,), dtype=np.int32))
        pos_tile = self.target._put_rep(np.zeros((T,), dtype=np.int32))
        mask_tile = self.target._put_rep(np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32))
        argmax_t, step_k, step_v, step_ctx = self._target_tile_with_ctx_jit(
            self.target._weights_tuple,
            self._capture_slots,
            self.target._put_rep(0, dtype=jnp.int32),
            tok_tile,
            pos_tile,
            mask_tile,
            k_cache,
            v_cache,
        )
        src_rows = self.target._put_rep(np.zeros((T,), dtype=np.int32))
        k_n, v_n, dk_n, dv_n = self._commit_both_kv_jit(
            self._draft_weights_tuple,
            k_cache,
            v_cache,
            step_k,
            step_v,
            dk_cache,
            dv_cache,
            step_ctx,
            self.target._put_rep(0, dtype=jnp.int32),
            src_rows,
            self.target._put_rep(1, dtype=jnp.int32),
        )
        t_ids, t_probs = self._block_draft_top_k_jit(
            self.target._weights_tuple.embed,
            self.target._weights_tuple.lm_head,
            self.target._weights_tuple.lm_head_scale,
            self._draft_weights_tuple,
            self.target._put_rep(0, dtype=jnp.int32),
            self.target._put_rep(1, dtype=jnp.int32),
            dk_n,
            dv_n,
        )
        jax.block_until_ready((argmax_t, k_n, v_n, dk_n, dv_n, t_ids, t_probs))
        return time.perf_counter() - t0

    def run_block_tree(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        use_suffix_fallback: bool = True,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        """Run native C++ ForgeEngine with block-diffusion drafting packed into 128/256 tile."""
        self.target._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.target.config.eos_token_ids)
        )
        compiles_before = self.total_compile_count()

        k_cache, v_cache = self.target._empty_kv_cache()
        dk_cache, dv_cache = self._empty_draft_kv_cache()
        jax.block_until_ready((k_cache, v_cache, dk_cache, dv_cache))

        state: dict[str, Any] = {
            "k_cache": k_cache,
            "v_cache": v_cache,
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
        lm_head = target_wt.lm_head
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
                state["k_cache"],
                state["v_cache"],
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
                state["k_cache"],
                state["v_cache"],
                state["step_k"],
                state["step_v"],
                state["dk_cache"],
                state["dv_cache"],
                state["step_ctx"],
                self.target._put_rep(prev_prefix_len, dtype=jnp.int32),
                self.target._put_rep(src_rows),
                self.target._put_rep(n_valid, dtype=jnp.int32),
            )
            state["k_cache"] = k_n
            state["v_cache"] = v_n
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
                lm_head,
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
            f"qwen-tpuforge-block-tree-{T}",
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
        jax.block_until_ready(
            (state["k_cache"], state["v_cache"], state["dk_cache"], state["dv_cache"])
        )
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
            },
        )


def load_qwen_block_tree_runner(
    model_id_or_path: str,
    draft_model_id_or_path: str,
    max_seq_len: int = 2048,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    top_k: int = 8,
    enable_xla_cache: bool = True,
    weight_quant: str = "none",
    draft_weight_quant: str = "none",
) -> tuple[QwenBlockTreeRunner, Path, Path]:
    """Resolve and load target Qwen + block-diffusion draft checkpoints concurrently onto TPU HBM."""
    from tokenizers import Tokenizer

    if enable_xla_cache:
        configure_xla_compilation_cache()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_target = pool.submit(resolve_model_dir, model_id_or_path)
        f_draft = pool.submit(resolve_model_dir, draft_model_id_or_path)
        target_dir = f_target.result()
        draft_dir = f_draft.result()

    target_config = QwenConfig.from_dir(target_dir)
    draft_config = BlockDiffusionConfig.from_dir(draft_dir, target_config=target_config)
    tok_path = target_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {target_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    dtype = resolve_default_dtype(dtype_name)
    _, mesh = resolve_tp_mesh(tp_size, config=target_config)
    target_weights = QwenWeights.from_model_dir(
        target_dir, target_config, dtype=dtype, mesh=mesh, weight_quant=weight_quant
    )
    draft_weights = BlockDiffusionWeights.from_model_dir(
        draft_dir, draft_config, dtype=dtype, mesh=mesh, weight_quant=draft_weight_quant
    )

    base_runner = QwenBenchmarkRunner(
        config=target_config,
        weights=target_weights,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tile_size=tile_size,
        mesh=mesh,
    )
    block_runner = QwenBlockTreeRunner(
        target_runner=base_runner,
        draft_config=draft_config,
        draft_weights=draft_weights,
        top_k=top_k,
    )
    return block_runner, target_dir, draft_dir

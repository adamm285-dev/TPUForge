"""Pure-JAX block-diffusion systolic tree drafter for TPUForge."""

from __future__ import annotations

import concurrent.futures
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

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

    @classmethod
    def from_dir(
        cls,
        model_dir: Path,
        target_config: QwenConfig | None = None,
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
        rope_theta = float(raw.get("rope_theta", 1_000_000.0))

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
        )


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
    """Stacked (optionally TP-sharded) JAX device arrays for a block-diffusion drafter."""

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

    @classmethod
    def from_numpy_dict(
        cls,
        raw: dict[str, np.ndarray],
        config: BlockDiffusionConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
    ) -> BlockDiffusionWeights:
        np_dtype = _jnp_to_numpy_dtype(dtype)
        top_prefix = "" if "fc.weight" in raw else ("model." if "model.fc.weight" in raw else "")
        layer_prefix = (
            ""
            if "layers.0.input_layernorm.weight" in raw
            else ("model." if "model.layers.0.input_layernorm.weight" in raw else top_prefix)
        )

        def get_top(name: str, transpose: bool = False) -> jax.Array:
            key = f"{top_prefix}{name}"
            if key not in raw:
                raise KeyError(f"Missing block-drafter tensor {key!r}")
            arr = raw[key]
            if transpose:
                arr = np.ascontiguousarray(arr.T)
            return _place_np_sharded(arr, P(), mesh, dtype, np_dtype)

        def stack_layer(suffix: str, transpose: bool = False, spec: P | None = None) -> jax.Array:
            items = []
            for i in range(config.num_hidden_layers):
                key = f"{layer_prefix}layers.{i}.{suffix}"
                if key not in raw:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                arr = raw[key]
                if transpose:
                    arr = np.ascontiguousarray(arr.T)
                items.append(arr)
            return _place_np_sharded(
                np.stack(items, axis=0), P() if spec is None else spec, mesh, dtype, np_dtype
            )

        def stack_layer_optional(
            suffix: str, transpose: bool = False, spec: P | None = None
        ) -> jax.Array | None:
            if f"{layer_prefix}layers.0.{suffix}" not in raw:
                return None
            return stack_layer(suffix, transpose=transpose, spec=spec)

        return cls(
            fc=get_top("fc.weight", transpose=True),
            hidden_norm=get_top("hidden_norm.weight", transpose=False),
            final_norm=get_top("norm.weight", transpose=False),
            in_norm=stack_layer("input_layernorm.weight", spec=P()),
            post_norm=stack_layer("post_attention_layernorm.weight", spec=P()),
            wq=stack_layer("self_attn.q_proj.weight", transpose=True, spec=P(None, None, "tp")),
            bq=stack_layer_optional("self_attn.q_proj.bias", spec=P(None, "tp")),
            wk=stack_layer("self_attn.k_proj.weight", transpose=True, spec=P(None, None, "tp")),
            bk=stack_layer_optional("self_attn.k_proj.bias", spec=P(None, "tp")),
            wv=stack_layer("self_attn.v_proj.weight", transpose=True, spec=P(None, None, "tp")),
            bv=stack_layer_optional("self_attn.v_proj.bias", spec=P(None, "tp")),
            wo=stack_layer("self_attn.o_proj.weight", transpose=True, spec=P(None, "tp", None)),
            w_gate=stack_layer("mlp.gate_proj.weight", transpose=True, spec=P(None, None, "tp")),
            w_up=stack_layer("mlp.up_proj.weight", transpose=True, spec=P(None, None, "tp")),
            w_down=stack_layer("mlp.down_proj.weight", transpose=True, spec=P(None, "tp", None)),
            mesh=mesh,
            q_norm=stack_layer_optional("self_attn.q_norm.weight", spec=P()),
            k_norm=stack_layer_optional("self_attn.k_norm.weight", spec=P()),
        )

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        config: BlockDiffusionConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
    ) -> BlockDiffusionWeights:
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
            def read_top(name: str, transpose: bool = False) -> jax.Array:
                key = f"{top_prefix}{name}"
                if key not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key!r}")
                arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                if transpose:
                    arr = np.ascontiguousarray(arr.T)
                return _place_np_sharded(arr, P(), mesh, dtype, np_dtype)

            def stream_stacked_layer(suffix: str, transpose: bool, spec: P) -> jax.Array:
                num_layers = config.num_hidden_layers
                key0 = f"{layer_prefix}layers.0.{suffix}"
                if key0 not in tensor_loc:
                    raise KeyError(f"Missing block-drafter tensor {key0!r}")
                raw_shape0 = tensor_loc[key0][2]
                layer_shape = tuple(reversed(raw_shape0)) if transpose else raw_shape0
                full_shape = (num_layers,) + layer_shape

                if mesh is None:
                    stacked = np.empty(full_shape, dtype=np_dtype)
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                        if transpose:
                            arr = np.ascontiguousarray(arr.T)
                        stacked[i] = arr
                    out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
                    del stacked
                    return out

                sharding = NamedSharding(mesh, spec)
                local_devs = set(mesh.local_devices)
                all_devs = list(mesh.devices.flat)
                sample = _split_tensor_shard(
                    np.empty(full_shape, dtype=np_dtype)
                    if tp_size == 1
                    else np.empty((1,) + layer_shape, dtype=np_dtype),
                    spec,
                    0,
                    tp_size,
                )
                shard_layer_shape = sample.shape[1:]
                del sample

                device_arrays = []
                for dev in all_devs:
                    if dev not in local_devs:
                        continue
                    rank = all_devs.index(dev)
                    rank_buf = np.empty((num_layers,) + shard_layer_shape, dtype=np_dtype)
                    for i in range(num_layers):
                        key = f"{layer_prefix}layers.{i}.{suffix}"
                        arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                        if transpose:
                            arr = np.ascontiguousarray(arr.T)
                        rank_buf[i] = _split_tensor_shard(arr, P(*spec[1:]), rank, tp_size)
                        del arr
                    device_arrays.append(jax.device_put(jnp.asarray(rank_buf, dtype=np_dtype), dev))
                    del rank_buf
                return jax.make_array_from_single_device_arrays(full_shape, sharding, device_arrays)

            def stream_stacked_layer_optional(
                suffix: str, transpose: bool, spec: P
            ) -> jax.Array | None:
                if f"{layer_prefix}layers.0.{suffix}" not in tensor_loc:
                    return None
                return stream_stacked_layer(suffix, transpose, spec)

            return cls(
                fc=read_top("fc.weight", transpose=True),
                hidden_norm=read_top("hidden_norm.weight", transpose=False),
                final_norm=read_top("norm.weight", transpose=False),
                in_norm=stream_stacked_layer("input_layernorm.weight", False, P()),
                post_norm=stream_stacked_layer("post_attention_layernorm.weight", False, P()),
                wq=stream_stacked_layer("self_attn.q_proj.weight", True, P(None, None, "tp")),
                bq=stream_stacked_layer_optional("self_attn.q_proj.bias", False, P(None, "tp")),
                wk=stream_stacked_layer("self_attn.k_proj.weight", True, P(None, None, "tp")),
                bk=stream_stacked_layer_optional("self_attn.k_proj.bias", False, P(None, "tp")),
                wv=stream_stacked_layer("self_attn.v_proj.weight", True, P(None, None, "tp")),
                bv=stream_stacked_layer_optional("self_attn.v_proj.bias", False, P(None, "tp")),
                wo=stream_stacked_layer("self_attn.o_proj.weight", True, P(None, "tp", None)),
                w_gate=stream_stacked_layer("mlp.gate_proj.weight", True, P(None, None, "tp")),
                w_up=stream_stacked_layer("mlp.up_proj.weight", True, P(None, None, "tp")),
                w_down=stream_stacked_layer("mlp.down_proj.weight", True, P(None, "tp", None)),
                mesh=mesh,
                q_norm=stream_stacked_layer_optional("self_attn.q_norm.weight", False, P()),
                k_norm=stream_stacked_layer_optional("self_attn.k_norm.weight", False, P()),
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

        dw = draft_weights
        self._draft_weights_tuple: tuple[jax.Array | None, ...] = (
            dw.fc,
            dw.hidden_norm,
            dw.final_norm,
            dw.in_norm,
            dw.post_norm,
            dw.wq,
            dw.bq,
            dw.wk,
            dw.bk,
            dw.wv,
            dw.bv,
            dw.wo,
            dw.w_gate,
            dw.w_up,
            dw.w_down,
            dw.q_norm,
            dw.k_norm,
        )

        self._compile_block_tree_kernels()

    def _compile_block_tree_kernels(self) -> None:
        tcfg = self.target.config
        dcfg = self.draft_config
        max_seq = self.max_seq_len
        tile_size = self.tile_size
        block_size = dcfg.block_size
        mask_token_id = dcfg.mask_token_id
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
            target_wt: tuple[jax.Array | None, ...],
            capture_slots: jax.Array,
            valid_prefix_len: jax.Array,
            token_tile: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            self.target_with_ctx_compile_count += 1
            embed, final_norm, lm_head, *layer_weights = target_wt
            x0 = embed[token_tile]
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
                lw_and_slot: tuple[Any, ...],
            ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
                x, cap_buf = carry
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                lw = lw_and_slot[:-1]
                slot = lw_and_slot[-1]
                kc, vc = lw[-2], lw[-1]
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
                (*layer_weights, k_cache, v_cache, capture_slots),
            )
            normed = _rms_norm(x_final, final_norm, tcfg.rms_norm_eps)
            logits_tile = _linear(normed, lm_head, out_dtype=jnp.float32)
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
            draft_wt: tuple[jax.Array | None, ...],
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
            fc, hidden_norm, _, _, _, _, _, wk, bk, wv, bv, _, _, _, _, _, k_norm = draft_wt

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
            # Project and normalize target hidden features once before feeding draft layer K/V.
            target_h = _rms_norm(_linear(gathered_ctx, fc), hidden_norm, dcfg.rms_norm_eps)
            if rep_sharding is not None:
                target_h = jax.lax.with_sharding_constraint(target_h, rep_sharding)

            def draft_kv_body(
                _carry: None,
                xs: tuple[jax.Array | None, ...],
            ) -> tuple[None, tuple[jax.Array, jax.Array]]:
                wk_l, bk_l, wv_l, bv_l, kn_l, dkc, dvc = xs
                k_ctx = _linear(target_h, wk_l, bk_l).reshape(
                    tile_size, dcfg.num_key_value_heads, dcfg.head_dim
                )
                if kn_l is not None:
                    k_ctx = _rms_norm(k_ctx, kn_l, dcfg.rms_norm_eps)
                k_ctx = _apply_qwen_rope(k_ctx, dest_positions, dcfg.head_dim, dcfg.rope_theta)
                v_ctx = _linear(target_h, wv_l, bv_l).reshape(
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
                (wk, bk, wv, bv, k_norm, draft_k_cache, draft_v_cache),
            )
            if kv_cache_sharding is not None:
                dk_next = jax.lax.with_sharding_constraint(dk_next, kv_cache_sharding)
                dv_next = jax.lax.with_sharding_constraint(dv_next, kv_cache_sharding)
            return k_next, v_next, dk_next, dv_next

        @jax.jit
        def block_draft_top_k_jit(
            embed: jax.Array,
            lm_head: jax.Array,
            draft_wt: tuple[jax.Array | None, ...],
            root_token_id: jax.Array,
            cur_prefix: jax.Array,
            draft_k_cache: jax.Array,
            draft_v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self.draft_step_compile_count += 1
            _, _, draft_final_norm, *draft_layer_weights = draft_wt
            mask_fill = jnp.full((block_size - 1,), mask_token_id, dtype=jnp.int32)
            block_ids = jnp.concatenate([root_token_id.reshape((1,)), mask_fill], axis=0)
            block_pos = cur_prefix + jnp.arange(block_size, dtype=jnp.int32)

            x0 = embed[block_ids]
            if rep_sharding is not None:
                x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)

            prefix_allowed = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) < cur_prefix,
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)
            prefix_mask = jnp.broadcast_to(prefix_allowed[None, :], (block_size, max_seq))
            # Block-diffusion self-attention is bidirectional (is_causal=False) across the B slots.
            block_mask = jnp.zeros((block_size, block_size), dtype=jnp.float32)
            full_mask = jnp.concatenate([prefix_mask, block_mask], axis=-1)

            def draft_body(
                x: jax.Array,
                lw: tuple[jax.Array | None, ...],
            ) -> tuple[jax.Array, None]:
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                dkc, dvc = lw[-2], lw[-1]
                q, k_noise, v_noise = _qwen_qkv_proj(
                    x, lw, block_pos, dcfg, head_sharding, kv_head_sharding
                )
                k_all = jnp.concatenate([dkc[:max_seq], k_noise], axis=0)
                v_all = jnp.concatenate([dvc[:max_seq], v_noise], axis=0)
                attn_out = _gqa_attention(
                    q,
                    k_all,
                    v_all,
                    full_mask,
                    dcfg.num_attention_heads,
                    dcfg.num_key_value_heads,
                    dcfg.head_dim,
                )
                x_next = _qwen_block_post_attn(x, attn_out, lw, dcfg, rep_sharding)
                return x_next, None

            x_final, _ = jax.lax.scan(
                draft_body,
                x0,
                (*draft_layer_weights, draft_k_cache, draft_v_cache),
            )
            normed = _rms_norm(x_final[1:], draft_final_norm, dcfg.rms_norm_eps)
            logits = _linear(normed, lm_head, out_dtype=jnp.float32)
            if rep_sharding is not None:
                logits = jax.lax.with_sharding_constraint(logits, rep_sharding)
            top_logits, top_ids = jax.lax.top_k(logits, top_k)
            top_probs = jax.nn.softmax(top_logits, axis=-1)
            if rep_sharding is not None:
                top_ids = jax.lax.with_sharding_constraint(top_ids, rep_sharding)
                top_probs = jax.lax.with_sharding_constraint(top_probs, rep_sharding)
            return top_ids.astype(jnp.int32), top_probs.astype(jnp.float32)

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
            self.target._weights_tuple[0],
            self.target._weights_tuple[2],
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
        embed = target_wt[0]
        lm_head = target_wt[2]

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
        req = tpu.GenerationRequest()
        req.prompt_tokens = list(prompt_tokens)
        req.eos_token_ids = eos_list
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
    target_weights = QwenWeights.from_model_dir(target_dir, target_config, dtype=dtype, mesh=mesh)
    draft_weights = BlockDiffusionWeights.from_model_dir(draft_dir, draft_config, dtype=dtype, mesh=mesh)

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

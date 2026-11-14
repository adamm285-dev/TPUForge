"""Shared INT8/FP8 weight-only quantization, mixed-precision systolic matmul, and scan pytrees."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

VALID_WEIGHT_QUANT_MODES = ("none", "int8", "fp8")


class LayerScanWeights(NamedTuple):
    """Stacked per-layer weights (and optional per-channel scales) scanned by jax.lax.scan."""

    in_norm: jax.Array
    post_norm: jax.Array
    wq: jax.Array
    wq_scale: jax.Array | None
    bq: jax.Array | None
    wk: jax.Array
    wk_scale: jax.Array | None
    bk: jax.Array | None
    wv: jax.Array
    wv_scale: jax.Array | None
    bv: jax.Array | None
    wo: jax.Array
    wo_scale: jax.Array | None
    w_gate: jax.Array
    w_gate_scale: jax.Array | None
    w_up: jax.Array
    w_up_scale: jax.Array | None
    w_down: jax.Array
    w_down_scale: jax.Array | None
    q_norm: jax.Array | None
    k_norm: jax.Array | None


class TargetWeightsTuple(NamedTuple):
    """Top-level explicit device-array bundle passed into target JAX JIT kernels."""

    embed: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array
    lm_head_scale: jax.Array | None
    layers: LayerScanWeights


class DraftWeightsTuple(NamedTuple):
    """Top-level explicit device-array bundle passed into block-diffusion draft JIT kernels."""

    fc: jax.Array
    fc_scale: jax.Array | None
    hidden_norm: jax.Array
    final_norm: jax.Array
    layers: LayerScanWeights


def normalize_weight_quant(mode: str | None) -> str:
    """Normalize weight quantization alias into 'none', 'int8', or 'fp8'."""
    if isinstance(mode, bool):
        raise ValueError("weight_quant cannot be bool")  # noqa: TRY004
    if mode is None:
        return "none"
    if not isinstance(mode, str):
        raise TypeError(f"weight_quant must be str or None, got {type(mode).__name__}")
    norm = mode.strip().lower()
    if norm in ("", "none"):
        return "none"
    if norm in ("int8", "int8_weight_only"):
        return "int8"
    if norm in ("fp8", "fp8_e4m3", "float8_e4m3fn"):
        return "fp8"
    raise ValueError(f"Unsupported weight_quant={mode!r}; expected 'none', 'int8', or 'fp8'")


def quant_np_dtype(mode: str) -> Any:
    """Return the host numpy/ml_dtypes storage dtype for a quantized mode."""
    norm = normalize_weight_quant(mode)
    if norm == "int8":
        return np.int8
    if norm == "fp8":
        return ml_dtypes.float8_e4m3fn
    raise ValueError(f"weight_quant={mode!r} does not use a quantized storage dtype")


def scale_spec_for_weight(weight_spec: P) -> P:
    """Derive PartitionSpec for per-output-channel scales from weight PartitionSpec [..., in, out]."""
    ndim = len(weight_spec)
    if ndim == 0:
        return P()
    if ndim == 2:
        return P(weight_spec[1]) if weight_spec[1] is not None else P()
    if ndim == 3:
        if weight_spec[0] is not None or weight_spec[2] is not None:
            return P(weight_spec[0], weight_spec[2])
        return P()
    raise ValueError(f"Unsupported weight PartitionSpec rank {ndim}: {weight_spec}")


def quantize_np_per_channel(weight_np: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrically quantize [..., in_features, out_features] along axis=-2 with FP32 scales."""
    norm = normalize_weight_quant(mode)
    if norm == "none":
        raise ValueError("Cannot quantize tensor when weight_quant='none'")
    if weight_np.ndim < 2:
        raise ValueError(f"Expected >= 2D weight tensor, got shape {weight_np.shape}")

    w_f32 = np.asarray(weight_np, dtype=np.float32)
    abs_max = np.max(np.abs(w_f32), axis=-2)

    if norm == "int8":
        scale = np.where(abs_max > 0.0, abs_max / np.float32(127.0), np.float32(1.0)).astype(
            np.float32
        )
        inv_scale = np.expand_dims(scale, axis=-2)
        q = np.clip(np.rint(w_f32 / inv_scale), -127.0, 127.0).astype(np.int8)
    else:
        scale = np.where(abs_max > 0.0, abs_max / np.float32(448.0), np.float32(1.0)).astype(
            np.float32
        )
        inv_scale = np.expand_dims(scale, axis=-2)
        scaled = np.clip(w_f32 / inv_scale, -448.0, 448.0)
        q = np.asarray(scaled, dtype=ml_dtypes.float8_e4m3fn)

    return np.ascontiguousarray(q), np.ascontiguousarray(scale)


def split_np_shard(arr: np.ndarray, spec: P, rank: int, tp_size: int) -> np.ndarray:
    """Slice local Megatron TP shard along the unique 'tp' axis annotated in PartitionSpec."""
    if tp_size <= 1:
        return np.ascontiguousarray(arr)
    tp_axes = [ax for ax, name in enumerate(spec) if name == "tp"]
    if not tp_axes:
        return np.ascontiguousarray(arr)
    if len(tp_axes) != 1:
        raise ValueError(f"Expected at most 1 'tp' axis in PartitionSpec {spec}")
    ax = tp_axes[0]
    dim_size = arr.shape[ax]
    if dim_size % tp_size != 0:
        raise ValueError(f"Axis {ax} size {dim_size} not divisible by tp_size={tp_size}")
    chunk = dim_size // tp_size
    sl = [slice(None)] * arr.ndim
    sl[ax] = slice(rank * chunk, (rank + 1) * chunk)
    return np.ascontiguousarray(arr[tuple(sl)])


def place_np_sharded_preserved(
    full_np: np.ndarray,
    spec: P,
    mesh: Mesh | None,
) -> jax.Array:
    """Place a numpy/ml_dtypes array onto single or multi-device mesh preserving exact dtype."""
    if mesh is None:
        return jax.device_put(jnp.asarray(full_np))
    sharding = NamedSharding(mesh, spec)
    tp_size = mesh.size
    local_devs = set(mesh.local_devices)
    all_devs = list(mesh.devices.flat)
    shards = []
    for dev in all_devs:
        if dev not in local_devs:
            continue
        rank = all_devs.index(dev)
        part = split_np_shard(full_np, spec, rank, tp_size)
        shards.append(jax.device_put(jnp.asarray(part), dev))
    return jax.make_array_from_single_device_arrays(full_np.shape, sharding, shards)


def quantize_and_place_2d(
    arr_2d: np.ndarray,
    spec: P,
    mesh: Mesh | None,
    weight_quant: str,
    active_np_dtype: Any | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Quantize a 2D [in, out] weight (if enabled) before TP slicing and place weight + scale."""
    norm = normalize_weight_quant(weight_quant)
    if norm == "none":
        prepared = arr_2d if active_np_dtype is None else np.asarray(arr_2d, dtype=active_np_dtype)
        return place_np_sharded_preserved(prepared, spec, mesh), None
    q_np, s_np = quantize_np_per_channel(arr_2d, norm)
    q_dev = place_np_sharded_preserved(q_np, spec, mesh)
    s_dev = place_np_sharded_preserved(s_np, scale_spec_for_weight(spec), mesh)
    return q_dev, s_dev


def systolic_linear(
    x: jax.Array,
    w: jax.Array,
    b: jax.Array | None = None,
    w_scale: jax.Array | None = None,
    out_dtype: Any | None = None,
) -> jax.Array:
    """Compute x @ w (+ b) supporting native mixed-precision INT8/FP8 dot_general + scale."""
    target_dtype = x.dtype if out_dtype is None else out_dtype
    if w_scale is not None:
        out = jax.lax.dot_general(
            x,
            w,
            (((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        out = out * w_scale.astype(jnp.float32)
        if b is not None:
            out = out + b.astype(jnp.float32)
        return out.astype(target_dtype)

    if x.dtype in (jnp.bfloat16, jnp.float16):
        out = jnp.matmul(x, w, preferred_element_type=jnp.float32)
        if b is not None:
            out = out + b.astype(jnp.float32)
        return out.astype(target_dtype)

    out = jnp.matmul(x, w, precision=jax.lax.Precision.HIGHEST)
    if b is not None:
        out = out + b
    return out.astype(target_dtype) if out.dtype != target_dtype else out


def batched_expert_systolic_linear(
    x_routed: jax.Array,
    w_experts: jax.Array,
    w_experts_scale: jax.Array | None = None,
    out_dtype: Any | None = None,
) -> jax.Array:
    """Batched MoE expert matmul [E, K, in_dim] x [E, in_dim, out_dim] -> [E, K, out_dim]."""
    target_dtype = x_routed.dtype if out_dtype is None else out_dtype
    if w_experts_scale is not None:
        out = jax.lax.dot_general(
            x_routed,
            w_experts,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        out = out * w_experts_scale[:, None, :].astype(jnp.float32)
        return out.astype(target_dtype)

    if x_routed.dtype in (jnp.bfloat16, jnp.float16):
        out = jax.lax.dot_general(
            x_routed,
            w_experts,
            (((2,), (1,)), ((0,), (0,))),
            preferred_element_type=jnp.float32,
        )
        return out.astype(target_dtype)

    out = jax.lax.dot_general(
        x_routed,
        w_experts,
        (((2,), (1,)), ((0,), (0,))),
        precision=jax.lax.Precision.HIGHEST,
    )
    return out.astype(target_dtype) if out.dtype != target_dtype else out


def pytree_nbytes(tree: Any) -> int:
    """Return total device byte footprint across all JAX/NumPy array leaves in a pytree."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(tree):
        if leaf is None:
            continue
        if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
            total += int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
    return total

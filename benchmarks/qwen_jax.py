"""Pure-JAX + zero-copy safetensors Qwen2.5-Coder runner for Vanilla vs TPUForge benchmarks."""

from __future__ import annotations

import contextlib
import json
import math
import mmap
import os
import struct
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

# Ensure build/ and MSYS2 UCRT64 runtime DLLs are discoverable for _tpuforge_ext
_ROOT = Path(__file__).resolve().parent.parent
_BUILD_DIR = _ROOT / "build"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))
if sys.platform == "win32":
    _UCRT_BIN = Path(r"C:\msys64\ucrt64\bin")
    if _UCRT_BIN.is_dir():
        os.add_dll_directory(str(_UCRT_BIN))

import _tpuforge_ext as tpu

from benchmarks.quantization_jax import (
    LayerScanWeights,
    TargetWeightsTuple,
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

TILE_SIZE = 128
VALID_TILE_SIZES = (128, 256)
_MASK_BLOCK_VAL = -1e9


def configure_xla_compilation_cache(cache_dir: str | Path | None = None) -> Path | None:
    """Enable persistent OpenXLA compilation caching across restarts unless explicitly disabled."""
    if cache_dir is None:
        env_dir = os.environ.get("TPUFORGE_XLA_CACHE_DIR", "").strip()
        if env_dir in ("0", "off", "OFF", "false", "FALSE", "disabled"):
            return None
        target_dir = Path(env_dir) if env_dir else (Path.home() / ".cache" / "tpuforge_xla_cache")
    else:
        target_dir = Path(cache_dir)

    resolved = target_dir.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(resolved))
    for flag_name, flag_val in (
        ("jax_persistent_cache_min_entry_size_bytes", -1),
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
    ):
        with contextlib.suppress(Exception):
            jax.config.update(flag_name, flag_val)
    return resolved


def resolve_tile_size(tile_size: int | str = "auto") -> int:
    """Resolve systolic tile width (128 for v2..v5e/CPU, 256 for v6e Trillium/v7/TPU 8x)."""
    if isinstance(tile_size, bool):
        raise ValueError("tile_size cannot be bool; expected 'auto', 128, or 256")  # noqa: TRY004
    if isinstance(tile_size, str):
        norm = tile_size.strip().lower()
        if norm == "auto":
            devs = jax.devices()
            device_kind = str(getattr(devs[0], "device_kind", "")) if devs else ""
            inferred = int(tpu.infer_mxu_dim(device_kind))
            return inferred if inferred in VALID_TILE_SIZES else TILE_SIZE
        if norm in ("128", "256"):
            return int(norm)
        raise ValueError(f"Unsupported tile_size={tile_size!r}; expected 'auto', 128, or 256")
    if isinstance(tile_size, int):
        if tile_size == 0:
            return resolve_tile_size("auto")
        if tile_size in VALID_TILE_SIZES:
            return int(tile_size)
        raise ValueError(f"Unsupported tile_size={tile_size!r}; expected 128 or 256")
    raise TypeError(f"tile_size must be int or str, got {type(tile_size).__name__}")


def resolve_tp_mesh(
    tp_size: int | str = "auto",
    config: QwenConfig | None = None,
) -> tuple[int, Mesh | None]:
    """Resolve Megatron Tensor Parallelism degree and build a 1D ('tp',) JAX Mesh."""
    devs = jax.devices()
    num_devs = len(devs)

    if isinstance(tp_size, bool):
        raise ValueError("tp_size cannot be bool")  # noqa: TRY004

    if isinstance(tp_size, str):
        norm = tp_size.strip().lower()
        if norm == "auto":
            if jax.default_backend() != "tpu" or num_devs <= 1:
                return 1, None
            # Prefer largest power-of-two (or num_devs) that evenly splits heads and MLP.
            candidates = [c for c in (16, 8, 4, 2, 1) if c <= num_devs]
            if num_devs not in candidates:
                candidates.insert(0, num_devs)
            resolved_tp = 1
            for cand in sorted(candidates, reverse=True):
                if config is None or (
                    config.num_attention_heads % cand == 0
                    and config.num_key_value_heads % cand == 0
                    and config.intermediate_size % cand == 0
                ):
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
    if (
        config is not None
        and resolved_tp > 1
        and (
            config.num_attention_heads % resolved_tp != 0
            or config.num_key_value_heads % resolved_tp != 0
            or config.intermediate_size % resolved_tp != 0
        )
    ):
        raise ValueError(
            f"tp_size={resolved_tp} must evenly divide num_attention_heads="
            f"{config.num_attention_heads}, num_key_value_heads={config.num_key_value_heads}, "
            f"and intermediate_size={config.intermediate_size}"
        )
    if resolved_tp == 1:
        return 1, None
    mesh = Mesh(np.array(devs[:resolved_tp]), axis_names=("tp",))
    return resolved_tp, mesh

_SAFETENSORS_DTYPES: dict[str, Any] = {
    "BF16": ml_dtypes.bfloat16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "U8": np.uint8,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "BOOL": np.bool_,
}


@dataclass(frozen=True)
class QwenConfig:
    """Architecture hyperparameters parsed from Hugging Face Qwen2 config.json."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    eos_token_ids: tuple[int, ...] = (151645, 151643)

    @classmethod
    def from_dir(cls, model_dir: Path) -> QwenConfig:
        cfg_path = model_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

        eos_ids: list[int] = []
        raw_eos = raw.get("eos_token_id")
        if isinstance(raw_eos, int):
            eos_ids.append(raw_eos)
        elif isinstance(raw_eos, list):
            eos_ids.extend(int(x) for x in raw_eos)

        gen_cfg_path = model_dir / "generation_config.json"
        if gen_cfg_path.is_file():
            gen_raw = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
            gen_eos = gen_raw.get("eos_token_id")
            if isinstance(gen_eos, int):
                eos_ids.append(gen_eos)
            elif isinstance(gen_eos, list):
                eos_ids.extend(int(x) for x in gen_eos)

        dedup_eos: list[int] = []
        for eid in eos_ids:
            if eid not in dedup_eos:
                dedup_eos.append(eid)
        if not dedup_eos:
            dedup_eos = [151645, 151643]

        hidden_size = int(raw["hidden_size"])
        num_heads = int(raw["num_attention_heads"])
        head_dim = int(raw.get("head_dim") or (hidden_size // num_heads))
        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(raw["num_key_value_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            head_dim=head_dim,
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(raw.get("rope_theta", 1_000_000.0)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
            eos_token_ids=tuple(dedup_eos),
        )


def read_safetensors_file(path: Path) -> dict[str, np.ndarray]:
    """Read tensors from a .safetensors file via mmap, natively supporting BF16."""
    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"Truncated safetensors header length in {path}")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"Truncated safetensors JSON header in {path}")
        header = json.loads(header_raw.decode("utf-8"))
        data_base = 8 + header_len

        f.seek(0, 2)
        file_size = f.tell()
        if file_size == data_base:
            return tensors

        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            view = memoryview(mm)
            try:
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    dtype_str = meta["dtype"]
                    if dtype_str not in _SAFETENSORS_DTYPES:
                        raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {name}")
                    np_dtype = _SAFETENSORS_DTYPES[dtype_str]
                    shape = tuple(int(dim) for dim in meta["shape"])
                    start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                    raw_slice = view[data_base + start : data_base + end]
                    try:
                        tensors[name] = np.frombuffer(raw_slice, dtype=np_dtype).reshape(shape).copy()
                    finally:
                        raw_slice.release()
            finally:
                view.release()
    return tensors


def load_safetensors_dir(model_dir: Path) -> dict[str, np.ndarray]:
    """Load single-file or sharded Hugging Face safetensors checkpoints from a directory."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        shard_files = sorted(set(index_data["weight_map"].values()))
        merged: dict[str, np.ndarray] = {}
        for shard_name in shard_files:
            shard_path = model_dir / shard_name
            if not shard_path.is_file():
                raise FileNotFoundError(f"Missing safetensors shard: {shard_path}")
            merged.update(read_safetensors_file(shard_path))
        return merged

    single_path = model_dir / "model.safetensors"
    if single_path.is_file():
        return read_safetensors_file(single_path)

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if safetensor_files:
        merged = {}
        for sf in safetensor_files:
            merged.update(read_safetensors_file(sf))
        return merged

    raise FileNotFoundError(f"No .safetensors checkpoints found in {model_dir}")


def _iter_safetensors_index(
    model_dir: Path,
) -> dict[str, tuple[Path, str, tuple[int, ...], int, int]]:
    """Index tensor locations in single or sharded .safetensors files without mapping payload."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        shard_files = [model_dir / name for name in sorted(set(index_data["weight_map"].values()))]
    elif (model_dir / "model.safetensors").is_file():
        shard_files = [model_dir / "model.safetensors"]
    else:
        shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors checkpoints found in {model_dir}")

    tensor_loc: dict[str, tuple[Path, str, tuple[int, ...], int, int]] = {}
    for sf in shard_files:
        if not sf.is_file():
            raise FileNotFoundError(f"Missing safetensors shard: {sf}")
        with sf.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                raise ValueError(f"Truncated safetensors header length in {sf}")
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_base = 8 + header_len
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                dtype_str = meta["dtype"]
                if dtype_str not in _SAFETENSORS_DTYPES:
                    raise ValueError(f"Unsupported safetensors dtype {dtype_str!r} for {name}")
                shape = tuple(int(dim) for dim in meta["shape"])
                start, end = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
                tensor_loc[name] = (sf, dtype_str, shape, data_base + start, data_base + end)
    return tensor_loc


class _MappedSafetensorsReader:
    """Keeps persistent read-only mmap handles per shard to avoid repeated open/seek overhead."""

    def __init__(self) -> None:
        self._stack = contextlib.ExitStack()
        self._maps: dict[Path, mmap.mmap] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        self._maps.clear()
        self._stack.close()

    def get_map(self, path: Path) -> mmap.mmap:
        mm = self._maps.get(path)
        if mm is not None:
            return mm
        f = self._stack.enter_context(path.open("rb"))
        if os.fstat(f.fileno()).st_size <= 0:
            raise ValueError(f"Empty safetensors file: {path}")
        mm = self._stack.enter_context(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        self._maps[path] = mm
        return mm

    def read(
        self,
        loc: tuple[Path, str, tuple[int, ...], int, int],
        target_np_dtype: Any | None = None,
    ) -> np.ndarray:
        path, dtype_str, shape, start_byte, end_byte = loc
        src_dtype = _SAFETENSORS_DTYPES[dtype_str]
        n_elems = math.prod(shape) if shape else 1
        expected_bytes = n_elems * np.dtype(src_dtype).itemsize
        if end_byte - start_byte < expected_bytes:
            raise ValueError(f"Truncated tensor payload in {path}")
        mm = self.get_map(path)
        arr = np.frombuffer(mm, dtype=src_dtype, count=n_elems, offset=start_byte).reshape(shape)
        if target_np_dtype is not None and arr.dtype != target_np_dtype:
            return np.asarray(arr, dtype=target_np_dtype)
        return arr.copy()


def _read_indexed_tensor(
    loc: tuple[Path, str, tuple[int, ...], int, int],
    target_np_dtype: Any | None = None,
    reader: _MappedSafetensorsReader | None = None,
) -> np.ndarray:
    """Read a single tensor slice via cached mmap handle (or one-shot file read if unmapped)."""
    if reader is not None:
        return reader.read(loc, target_np_dtype=target_np_dtype)
    path, dtype_str, shape, start_byte, end_byte = loc
    src_dtype = _SAFETENSORS_DTYPES[dtype_str]
    nbytes = end_byte - start_byte
    with path.open("rb") as f:
        f.seek(start_byte)
        buf = f.read(nbytes)
    if len(buf) != nbytes:
        raise ValueError(f"Truncated tensor payload in {path}")
    arr = np.frombuffer(buf, dtype=src_dtype).reshape(shape)
    if target_np_dtype is not None and arr.dtype != target_np_dtype:
        return np.asarray(arr, dtype=target_np_dtype)
    return arr.copy()


_split_tensor_shard = split_np_shard


def _jnp_to_numpy_dtype(dtype: jnp.dtype) -> Any:
    if dtype == jnp.bfloat16:
        return ml_dtypes.bfloat16
    if dtype == jnp.float16:
        return np.float16
    return np.float32


@dataclass
class QwenWeights:
    """Stacked (optionally TP-sharded and INT8/FP8-quantized) JAX device arrays for lax.scan."""

    embed: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array
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
    lm_head_scale: jax.Array | None = None
    wq_scale: jax.Array | None = None
    wk_scale: jax.Array | None = None
    wv_scale: jax.Array | None = None
    wo_scale: jax.Array | None = None
    w_gate_scale: jax.Array | None = None
    w_up_scale: jax.Array | None = None
    w_down_scale: jax.Array | None = None

    def to_layer_scan_weights(self) -> LayerScanWeights:
        """Pack stacked per-layer arrays and optional scales for jax.lax.scan."""
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

    def to_target_weights_tuple(self) -> TargetWeightsTuple:
        """Pack all target device arrays into an explicit constant-free pytree."""
        return TargetWeightsTuple(
            embed=self.embed,
            final_norm=self.final_norm,
            lm_head=self.lm_head,
            lm_head_scale=self.lm_head_scale,
            layers=self.to_layer_scan_weights(),
        )

    def total_bytes(self) -> int:
        """Return total byte footprint across all target weights and scales."""
        return pytree_nbytes(self.to_target_weights_tuple())

    def linear_weight_bytes(self) -> int:
        """Return byte footprint of projection/lm_head matrices and per-channel scales."""
        return pytree_nbytes(
            (
                self.lm_head,
                self.lm_head_scale,
                self.wq,
                self.wq_scale,
                self.wk,
                self.wk_scale,
                self.wv,
                self.wv_scale,
                self.wo,
                self.wo_scale,
                self.w_gate,
                self.w_gate_scale,
                self.w_up,
                self.w_up_scale,
                self.w_down,
                self.w_down_scale,
            )
        )

    @classmethod
    def from_numpy_dict(
        cls,
        raw: dict[str, np.ndarray],
        config: QwenConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        weight_quant: str = "none",
    ) -> QwenWeights:
        norm_quant = normalize_weight_quant(weight_quant)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        tp_size = mesh.size if mesh is not None else 1

        def to_jnp(arr: np.ndarray, spec: P | None = None) -> jax.Array:
            casted = np.asarray(arr, dtype=np_dtype)
            return place_np_sharded_preserved(casted, P() if spec is None else spec, mesh)

        def stack_layer(suffix: str, spec: P | None = None) -> jax.Array:
            items = []
            for i in range(config.num_hidden_layers):
                key = f"model.layers.{i}.{suffix}"
                if key not in raw:
                    raise KeyError(f"Missing tensor {key!r} in safetensors checkpoint")
                items.append(raw[key])
            return to_jnp(np.stack(items, axis=0), spec=spec)

        def stack_layer_optional(suffix: str, spec: P | None = None) -> jax.Array | None:
            if f"model.layers.0.{suffix}" not in raw:
                return None
            return stack_layer(suffix, spec=spec)

        def stack_linear(suffix: str, spec: P) -> tuple[jax.Array, jax.Array | None]:
            if norm_quant == "none":
                items = []
                for i in range(config.num_hidden_layers):
                    key = f"model.layers.{i}.{suffix}"
                    if key not in raw:
                        raise KeyError(f"Missing tensor {key!r} in safetensors checkpoint")
                    items.append(np.ascontiguousarray(raw[key].T))
                return to_jnp(np.stack(items, axis=0), spec=spec), None

            q_items = []
            s_items = []
            for i in range(config.num_hidden_layers):
                key = f"model.layers.{i}.{suffix}"
                if key not in raw:
                    raise KeyError(f"Missing tensor {key!r} in safetensors checkpoint")
                q_i, s_i = quantize_np_per_channel(np.ascontiguousarray(raw[key].T), norm_quant)
                q_items.append(q_i)
                s_items.append(s_i)
            q_dev = place_np_sharded_preserved(np.stack(q_items, axis=0), spec, mesh)
            s_dev = place_np_sharded_preserved(
                np.stack(s_items, axis=0), scale_spec_for_weight(spec), mesh
            )
            return q_dev, s_dev

        lm_spec = P(None, "tp") if (tp_size > 1 and config.vocab_size % tp_size == 0) else P()
        embed_np = raw["model.embed_tokens.weight"]
        embed = to_jnp(embed_np, spec=P())
        final_norm = to_jnp(raw["model.norm.weight"], spec=P())
        if "lm_head.weight" in raw:
            lm_head, lm_head_scale = quantize_and_place_2d(
                np.ascontiguousarray(raw["lm_head.weight"].T),
                lm_spec,
                mesh,
                norm_quant,
                active_np_dtype=np_dtype,
            )
        elif config.tie_word_embeddings:
            lm_head, lm_head_scale = quantize_and_place_2d(
                np.ascontiguousarray(embed_np.T),
                lm_spec,
                mesh,
                norm_quant,
                active_np_dtype=np_dtype,
            )
        else:
            raise KeyError("lm_head.weight missing and tie_word_embeddings is False")

        wq, wq_scale = stack_linear("self_attn.q_proj.weight", P(None, None, "tp"))
        wk, wk_scale = stack_linear("self_attn.k_proj.weight", P(None, None, "tp"))
        wv, wv_scale = stack_linear("self_attn.v_proj.weight", P(None, None, "tp"))
        wo, wo_scale = stack_linear("self_attn.o_proj.weight", P(None, "tp", None))
        w_gate, w_gate_scale = stack_linear("mlp.gate_proj.weight", P(None, None, "tp"))
        w_up, w_up_scale = stack_linear("mlp.up_proj.weight", P(None, None, "tp"))
        w_down, w_down_scale = stack_linear("mlp.down_proj.weight", P(None, "tp", None))

        return cls(
            embed=embed,
            final_norm=final_norm,
            lm_head=lm_head,
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
            lm_head_scale=lm_head_scale,
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
        config: QwenConfig,
        dtype: jnp.dtype,
        mesh: Mesh | None = None,
        weight_quant: str = "none",
    ) -> QwenWeights:
        """Stream tensors shard-by-shard from disk directly onto TPU HBM with low host RAM."""
        norm_quant = normalize_weight_quant(weight_quant)
        tensor_loc = _iter_safetensors_index(model_dir)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        tp_size = mesh.size if mesh is not None else 1

        with _MappedSafetensorsReader() as reader:
            def stream_stacked_vec(suffix: str, spec: P) -> jax.Array:
                num_layers = config.num_hidden_layers
                key0 = f"model.layers.0.{suffix}"
                if key0 not in tensor_loc:
                    raise KeyError(f"Missing tensor {key0!r} in safetensors checkpoint")
                layer_shape = tensor_loc[key0][2]
                full_shape = (num_layers,) + layer_shape

                if mesh is None:
                    stacked = np.empty(full_shape, dtype=np_dtype)
                    for i in range(num_layers):
                        key = f"model.layers.{i}.{suffix}"
                        stacked[i] = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                    out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
                    del stacked
                    return out

                sharding = NamedSharding(mesh, spec)
                local_devs = set(mesh.local_devices)
                all_devs = list(mesh.devices.flat)
                sample_shard = split_np_shard(np.empty((1,) + layer_shape, dtype=np_dtype), spec, 0, tp_size)
                shard_layer_shape = sample_shard.shape[1:]
                del sample_shard

                device_arrays = []
                for dev in all_devs:
                    if dev not in local_devs:
                        continue
                    rank = all_devs.index(dev)
                    rank_buf = np.empty((num_layers,) + shard_layer_shape, dtype=np_dtype)
                    layer_spec = P(*spec[1:])
                    for i in range(num_layers):
                        key = f"model.layers.{i}.{suffix}"
                        arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                        rank_buf[i] = split_np_shard(arr, layer_spec, rank, tp_size)
                        del arr
                    device_arrays.append(jax.device_put(jnp.asarray(rank_buf), dev))
                    del rank_buf
                return jax.make_array_from_single_device_arrays(full_shape, sharding, device_arrays)

            def stream_stacked_vec_optional(suffix: str, spec: P) -> jax.Array | None:
                if f"model.layers.0.{suffix}" not in tensor_loc:
                    return None
                return stream_stacked_vec(suffix, spec)

            def stream_stacked_linear(
                suffix: str,
                spec: P,
            ) -> tuple[jax.Array, jax.Array | None]:
                num_layers = config.num_hidden_layers
                key0 = f"model.layers.0.{suffix}"
                if key0 not in tensor_loc:
                    raise KeyError(f"Missing tensor {key0!r} in safetensors checkpoint")
                layer_shape = tuple(reversed(tensor_loc[key0][2]))
                full_shape = (num_layers,) + layer_shape

                if norm_quant == "none":
                    if mesh is None:
                        stacked = np.empty(full_shape, dtype=np_dtype)
                        for i in range(num_layers):
                            key = f"model.layers.{i}.{suffix}"
                            arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                            stacked[i] = np.ascontiguousarray(arr.T)
                            del arr
                        out = jax.device_put(jnp.asarray(stacked, dtype=dtype))
                        del stacked
                        return out, None

                    sharding = NamedSharding(mesh, spec)
                    local_devs = set(mesh.local_devices)
                    all_devs = list(mesh.devices.flat)
                    sample_shard = split_np_shard(
                        np.empty((1,) + layer_shape, dtype=np_dtype), spec, 0, tp_size
                    )
                    shard_layer_shape = sample_shard.shape[1:]
                    del sample_shard
                    layer_spec = P(*spec[1:])

                    device_arrays = []
                    for dev in all_devs:
                        if dev not in local_devs:
                            continue
                        rank = all_devs.index(dev)
                        rank_buf = np.empty((num_layers,) + shard_layer_shape, dtype=np_dtype)
                        for i in range(num_layers):
                            key = f"model.layers.{i}.{suffix}"
                            arr = _read_indexed_tensor(tensor_loc[key], np_dtype, reader=reader)
                            rank_buf[i] = split_np_shard(
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
                        key = f"model.layers.{i}.{suffix}"
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
                sample_w = split_np_shard(np.empty((1,) + layer_shape, dtype=q_dtype), spec, 0, tp_size)
                sample_s = split_np_shard(
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
                        key = f"model.layers.{i}.{suffix}"
                        raw_w = _read_indexed_tensor(tensor_loc[key], None, reader=reader)
                        q_i, s_i = quantize_np_per_channel(np.ascontiguousarray(raw_w.T), norm_quant)
                        del raw_w
                        rank_w_buf[i] = split_np_shard(q_i, layer_spec, rank, tp_size)
                        rank_s_buf[i] = split_np_shard(s_i, layer_scale_spec, rank, tp_size)
                        del q_i, s_i
                    w_arrays.append(jax.device_put(jnp.asarray(rank_w_buf), dev))
                    s_arrays.append(jax.device_put(jnp.asarray(rank_s_buf), dev))
                    del rank_w_buf, rank_s_buf
                return (
                    jax.make_array_from_single_device_arrays(full_shape, w_sharding, w_arrays),
                    jax.make_array_from_single_device_arrays(full_scale_shape, s_sharding, s_arrays),
                )

            lm_spec = P(None, "tp") if (tp_size > 1 and config.vocab_size % tp_size == 0) else P()
            embed_raw_dtype = None if norm_quant != "none" else np_dtype
            embed_np = _read_indexed_tensor(
                tensor_loc["model.embed_tokens.weight"], embed_raw_dtype, reader=reader
            )
            embed = place_np_sharded_preserved(np.asarray(embed_np, dtype=np_dtype), P(), mesh)
            if "lm_head.weight" in tensor_loc:
                del embed_np
                lm_np = _read_indexed_tensor(
                    tensor_loc["lm_head.weight"], embed_raw_dtype, reader=reader
                )
                lm_head, lm_head_scale = quantize_and_place_2d(
                    np.ascontiguousarray(lm_np.T),
                    lm_spec,
                    mesh,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )
                del lm_np
            elif config.tie_word_embeddings:
                lm_head, lm_head_scale = quantize_and_place_2d(
                    np.ascontiguousarray(embed_np.T),
                    lm_spec,
                    mesh,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )
                del embed_np
            else:
                del embed_np
                raise KeyError("lm_head.weight missing and tie_word_embeddings is False")

            final_norm_np = _read_indexed_tensor(
                tensor_loc["model.norm.weight"], np_dtype, reader=reader
            )
            final_norm = place_np_sharded_preserved(final_norm_np, P(), mesh)
            del final_norm_np

            wq, wq_scale = stream_stacked_linear("self_attn.q_proj.weight", P(None, None, "tp"))
            wk, wk_scale = stream_stacked_linear("self_attn.k_proj.weight", P(None, None, "tp"))
            wv, wv_scale = stream_stacked_linear("self_attn.v_proj.weight", P(None, None, "tp"))
            wo, wo_scale = stream_stacked_linear("self_attn.o_proj.weight", P(None, "tp", None))
            w_gate, w_gate_scale = stream_stacked_linear("mlp.gate_proj.weight", P(None, None, "tp"))
            w_up, w_up_scale = stream_stacked_linear("mlp.up_proj.weight", P(None, None, "tp"))
            w_down, w_down_scale = stream_stacked_linear("mlp.down_proj.weight", P(None, "tp", None))

            return cls(
                embed=embed,
                final_norm=final_norm,
                lm_head=lm_head,
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
                lm_head_scale=lm_head_scale,
                wq_scale=wq_scale,
                wk_scale=wk_scale,
                wv_scale=wv_scale,
                wo_scale=wo_scale,
                w_gate_scale=w_gate_scale,
                w_up_scale=w_up_scale,
                w_down_scale=w_down_scale,
            )


def resolve_default_dtype(dtype_name: str = "auto") -> jnp.dtype:
    """Resolve compute dtype: float32 on CPU for numerical precision, bfloat16 on TPU/GPU."""
    if dtype_name == "auto":
        backend = jax.default_backend()
        return jnp.bfloat16 if backend in ("tpu", "gpu") else jnp.float32
    mapping = {
        "float32": jnp.float32,
        "fp32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "bf16": jnp.bfloat16,
        "float16": jnp.float16,
        "fp16": jnp.float16,
    }
    if dtype_name.lower() not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}")
    return mapping[dtype_name.lower()]


def encode_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool | None = None,
) -> list[int]:
    """Encode chat turns using the Qwen ChatML template (with optional Qwen3 thinking tag)."""
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if enable_thinking is False:
        parts.append("<think>\n\n</think>\n\n")
    elif enable_thinking is True:
        parts.append("<think>\n")
    prompt_str = "".join(parts)
    encoded = tokenizer.encode(prompt_str, add_special_tokens=False)
    return [int(t) for t in encoded.ids]


_STATIC_LOCAL_MODEL_ROOTS: tuple[Path, ...] = (
    Path("H:/Caches/huggingface/hub"),
    Path("F:/Caches/huggingface/hub"),
    Path("C:/Caches/huggingface/hub"),
    Path("C:/AI_Models"),
    Path("F:/AI_Models"),
    Path("H:/AI_Models"),
)


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def _is_complete_safetensors_dir(path: Path) -> bool:
    """Return True only when config.json and every required .safetensors shard exist on disk."""
    if not _safe_is_dir(path) or not (path / "config.json").is_file():
        return False

    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        if not isinstance(index_data, dict):
            return False
        weight_map = index_data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        try:
            root_resolved = path.resolve()
            for shard_name in set(weight_map.values()):
                if not isinstance(shard_name, str) or not shard_name.strip():
                    return False
                shard_resolved = (path / shard_name).resolve()
                if not shard_resolved.is_relative_to(root_resolved) or not shard_resolved.is_file():
                    return False
        except (OSError, ValueError):
            return False
        return True

    if (path / "model.safetensors").is_file():
        return True
    try:
        return any(path.glob("*.safetensors"))
    except OSError:
        return False


def default_local_model_roots() -> list[Path]:
    """Return deduplicated existing local Hugging Face / checkpoint roots across configured drives."""
    raw_candidates: list[Path] = []
    env_dirs = os.environ.get("TPUFORGE_MODEL_DIRS", "").strip()
    if env_dirs:
        for entry in env_dirs.split(os.pathsep):
            cleaned = entry.strip()
            if cleaned:
                raw_candidates.append(Path(cleaned).expanduser())
    for env_key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        val = os.environ.get(env_key, "").strip()
        if val:
            raw_candidates.append(Path(val).expanduser())
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        raw_candidates.append(Path(hf_home).expanduser() / "hub")
    raw_candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    raw_candidates.extend(_STATIC_LOCAL_MODEL_ROOTS)

    roots: list[Path] = []
    seen: set[Path] = set()
    for cand in raw_candidates:
        if not _safe_is_dir(cand):
            continue
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            roots.append(cand)
    return roots


def _pick_hf_snapshot(hub_model_dir: Path) -> Path | None:
    """Return the preferred complete snapshot directory inside an HF hub cache models--* folder."""
    snapshots_dir = hub_model_dir / "snapshots"
    if not _safe_is_dir(snapshots_dir):
        return None
    ref_main = hub_model_dir / "refs" / "main"
    try:
        if ref_main.is_file():
            rev = ref_main.read_text(encoding="utf-8").strip()
            if rev:
                snap = snapshots_dir / rev
                if _is_complete_safetensors_dir(snap):
                    return snap
    except OSError:
        pass

    try:
        candidates = [p for p in snapshots_dir.iterdir() if _safe_is_dir(p)]
    except OSError:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for snap in candidates:
        if _is_complete_safetensors_dir(snap):
            return snap
    return None


def _classify_checkpoint_config(model_dir: Path) -> dict[str, Any]:
    """Read compact architecture metadata from config.json for local discovery tables."""
    try:
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {
            "arch": "unknown",
            "model_type": "unknown",
            "is_dflash_drafter": False,
            "hidden_size": None,
            "num_hidden_layers": None,
        }
    if not isinstance(raw, dict):
        raw = {}
    text_cfg = raw.get("text_config") if isinstance(raw.get("text_config"), dict) else {}
    model_type = str(raw.get("model_type") or text_cfg.get("model_type") or "unknown")
    is_drafter = bool("dflash_config" in raw or "target_layer_ids" in raw)
    if is_drafter:
        arch = "dflash"
    elif model_type.startswith("gemma4"):
        arch = "gemma4"
    else:
        arch = model_type
    hidden_size = raw.get("hidden_size") if "hidden_size" in raw else text_cfg.get("hidden_size")
    num_layers = (
        raw.get("num_hidden_layers")
        if "num_hidden_layers" in raw
        else text_cfg.get("num_hidden_layers")
    )
    return {
        "arch": arch,
        "model_type": model_type,
        "is_dflash_drafter": is_drafter,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
    }


def discover_local_models(
    search_roots: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    """Scan local model directories and HF hub caches for complete safetensors checkpoints."""
    roots = (
        [Path(r).expanduser() for r in search_roots]
        if search_roots is not None
        else default_local_model_roots()
    )
    discovered: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()

    def _record(repo_id: str, model_dir: Path, source: str) -> None:
        try:
            resolved = model_dir.resolve()
        except OSError:
            return
        if resolved in seen_dirs:
            return
        seen_dirs.add(resolved)
        meta = _classify_checkpoint_config(model_dir)
        discovered.append(
            {
                "repo_id": repo_id,
                "local_dir": str(model_dir),
                "source": source,
                **meta,
            }
        )

    for root in roots:
        if not _safe_is_dir(root):
            continue
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for child in children:
            if not _safe_is_dir(child):
                continue
            if child.name.startswith("models--") and "--" in child.name[len("models--") :]:
                snap = _pick_hf_snapshot(child)
                if snap is not None:
                    repo_id = child.name[len("models--") :].replace("--", "/", 1)
                    _record(repo_id, snap, "hf_cache")
                continue
            if _is_complete_safetensors_dir(child):
                _record(child.name, child, "direct")
                continue
            if child.name.startswith(".") or child.name in ("snapshots", "refs", "blobs"):
                continue
            try:
                subdirs = sorted(child.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for sub in subdirs:
                if _is_complete_safetensors_dir(sub):
                    _record(f"{child.name}/{sub.name}", sub, "direct")

    return discovered


def resolve_model_dir(
    model_id_or_path: str | Path,
    *,
    search_roots: Sequence[str | Path] | None = None,
    allow_download: bool = True,
) -> Path:
    """Resolve a local/HF model directory after validating complete safetensors shards."""
    raw = str(model_id_or_path).strip()
    if not raw:
        raise ValueError("model_id_or_path must be a non-empty repo ID or directory path")

    candidate = Path(raw).expanduser()
    if _safe_is_dir(candidate):
        if _is_complete_safetensors_dir(candidate):
            return candidate
        raise FileNotFoundError(
            f"Local model directory is incomplete (missing config.json or .safetensors shards): {candidate}"
        )

    roots = (
        [Path(r).expanduser() for r in search_roots]
        if search_roots is not None
        else default_local_model_roots()
    )
    for root in roots:
        if not _safe_is_dir(root):
            continue
        for direct in (root / raw, root / Path(raw).name):
            if _is_complete_safetensors_dir(direct):
                return direct
        if "/" in raw:
            hub_dir = root / ("models--" + raw.replace("/", "--"))
            snap = _pick_hf_snapshot(hub_dir)
            if snap is not None:
                return snap

    if not allow_download:
        raise FileNotFoundError(
            f"Model {raw!r} was not found in local model roots ({[str(r) for r in roots]}) "
            "and network downloads are disabled (--local-only)."
        )

    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=raw,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt"],
    )
    return Path(downloaded)


def _rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    x_f32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(var + eps)
    return normed.astype(weight.dtype) * weight


def _apply_qwen_rope(
    x: jax.Array,
    positions: jax.Array,
    head_dim: int,
    rope_theta: float,
) -> jax.Array:
    """Apply split-half Rotary Position Embeddings matching Hugging Face Qwen2ForCausalLM."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (jnp.arange(0, half, dtype=jnp.float32) / float(half))
    )
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos_a = jnp.cos(angles)[:, None, :]
    sin_a = jnp.sin(angles)[:, None, :]
    x_f32 = x.astype(jnp.float32)
    x1 = x_f32[..., :half]
    x2 = x_f32[..., half:]
    out = jnp.concatenate([x1 * cos_a - x2 * sin_a, x2 * cos_a + x1 * sin_a], axis=-1)
    return out.astype(x.dtype)


def _gqa_attention(
    q: jax.Array,
    k_all: jax.Array,
    v_all: jax.Array,
    mask_sm: jax.Array,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> jax.Array:
    """Compute Grouped-Query Attention for query [S, H_q, D] against [M, H_kv, D]."""
    seq_q = q.shape[0]
    num_groups = num_q_heads // num_kv_heads
    scale = 1.0 / math.sqrt(float(head_dim))
    if q.dtype in (jnp.bfloat16, jnp.float16):
        # Single-pass native BF16/FP16 MXU accumulation into FP32 avoids multi-pass emulation.
        q_grouped = q.reshape(seq_q, num_kv_heads, num_groups, head_dim)
        scores = (
            jnp.einsum(
                "skgd,mkd->kgsm",
                q_grouped,
                k_all,
                preferred_element_type=jnp.float32,
            )
            * scale
        )
        scores = scores + mask_sm[None, None, :, :]
        probs = jax.nn.softmax(scores, axis=-1).astype(v_all.dtype)
        ctx = jnp.einsum(
            "kgsm,mkd->skgd",
            probs,
            v_all,
            preferred_element_type=jnp.float32,
        )
        return ctx.reshape(seq_q, num_q_heads * head_dim).astype(q.dtype)

    q_grouped = q.reshape(seq_q, num_kv_heads, num_groups, head_dim).astype(jnp.float32)
    k_f32 = k_all.astype(jnp.float32)
    v_f32 = v_all.astype(jnp.float32)

    scores = (
        jnp.einsum(
            "skgd,mkd->kgsm",
            q_grouped,
            k_f32,
            precision=jax.lax.Precision.HIGHEST,
        )
        * scale
    )
    scores = scores + mask_sm[None, None, :, :]
    probs = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum(
        "kgsm,mkd->skgd",
        probs,
        v_f32,
        precision=jax.lax.Precision.HIGHEST,
    )
    return ctx.reshape(seq_q, num_q_heads * head_dim).astype(q.dtype)


def _linear(
    x: jax.Array,
    w: jax.Array,
    b: jax.Array | None = None,
    w_scale: jax.Array | None = None,
    out_dtype: Any | None = None,
) -> jax.Array:
    return systolic_linear(x, w, b=b, w_scale=w_scale, out_dtype=out_dtype)


def _qwen_qkv_proj(
    x: jax.Array,
    lw: LayerScanWeights,
    positions: jax.Array,
    cfg: QwenConfig,
    head_sharding: NamedSharding | None = None,
    kv_head_sharding: NamedSharding | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    seq_q = x.shape[0]
    h = _rms_norm(x, lw.in_norm, cfg.rms_norm_eps)
    q_proj = _linear(h, lw.wq, lw.bq, w_scale=lw.wq_scale).reshape(
        seq_q, cfg.num_attention_heads, cfg.head_dim
    )
    if lw.q_norm is not None:
        q_proj = _rms_norm(q_proj, lw.q_norm, cfg.rms_norm_eps)
    q = _apply_qwen_rope(
        q_proj,
        positions,
        cfg.head_dim,
        cfg.rope_theta,
    )
    k_proj = _linear(h, lw.wk, lw.bk, w_scale=lw.wk_scale).reshape(
        seq_q, cfg.num_key_value_heads, cfg.head_dim
    )
    if lw.k_norm is not None:
        k_proj = _rms_norm(k_proj, lw.k_norm, cfg.rms_norm_eps)
    k_new = _apply_qwen_rope(
        k_proj,
        positions,
        cfg.head_dim,
        cfg.rope_theta,
    )
    v_new = _linear(h, lw.wv, lw.bv, w_scale=lw.wv_scale).reshape(
        seq_q, cfg.num_key_value_heads, cfg.head_dim
    )
    if head_sharding is not None:
        q = jax.lax.with_sharding_constraint(q, head_sharding)
    if kv_head_sharding is not None:
        k_new = jax.lax.with_sharding_constraint(k_new, kv_head_sharding)
        v_new = jax.lax.with_sharding_constraint(v_new, kv_head_sharding)
    return q, k_new, v_new


def _qwen_block_post_attn(
    x: jax.Array,
    attn_out: jax.Array,
    lw: LayerScanWeights,
    cfg: QwenConfig,
    rep_sharding: NamedSharding | None = None,
) -> jax.Array:
    proj = _linear(attn_out, lw.wo, w_scale=lw.wo_scale)
    if rep_sharding is not None:
        proj = jax.lax.with_sharding_constraint(proj, rep_sharding)
    x = x + proj
    h2 = _rms_norm(x, lw.post_norm, cfg.rms_norm_eps)
    gate = _linear(h2, lw.w_gate, w_scale=lw.w_gate_scale)
    up = _linear(h2, lw.w_up, w_scale=lw.w_up_scale)
    mlp_out = _linear(jax.nn.silu(gate) * up, lw.w_down, w_scale=lw.w_down_scale)
    if rep_sharding is not None:
        mlp_out = jax.lax.with_sharding_constraint(mlp_out, rep_sharding)
    return x + mlp_out


@dataclass
class RunMetrics:
    """Structured telemetry captured from a single benchmark execution."""

    mode: str
    token_ids: list[int]
    text: str
    prompt_len: int
    generated_len: int
    compile_count: int
    prefill_s: float
    decode_s: float
    total_s: float
    decode_tok_per_s: float
    forward_passes: int
    decode_passes: int
    draft_tokens_accepted: int
    mean_tokens_per_decode_pass: float
    extra: dict[str, Any] = field(default_factory=dict)


class QwenBenchmarkRunner:
    """Runs Vanilla JAX, StaticArena (M-002), and TPUForge Tree (M-002 + M-003) on one Qwen checkpoint."""

    def __init__(
        self,
        config: QwenConfig,
        weights: QwenWeights,
        tokenizer: Any = None,
        max_seq_len: int = 1024,
        tile_size: int | str = "auto",
        mesh: Mesh | None = None,
    ) -> None:
        resolved_tile = resolve_tile_size(tile_size)
        if max_seq_len < 2 * resolved_tile:
            raise ValueError(f"max_seq_len ({max_seq_len}) must be >= {2 * resolved_tile}")
        resolved_mesh = mesh if mesh is not None else weights.mesh
        self.config = config
        self.weights = weights
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.tile_size = resolved_tile
        self.mesh = resolved_mesh
        self.tp_size = resolved_mesh.size if resolved_mesh is not None else 1
        self.dtype = weights.embed.dtype
        self.weight_quant = weights.weight_quant
        self.model_hbm_bytes = weights.total_bytes()

        if resolved_mesh is not None:
            self.rep_sharding: NamedSharding | None = NamedSharding(resolved_mesh, P())
            self.head_sharding: NamedSharding | None = NamedSharding(resolved_mesh, P(None, "tp", None))
            self.kv_head_sharding: NamedSharding | None = NamedSharding(resolved_mesh, P(None, "tp", None))
            self.kv_cache_sharding: NamedSharding | None = NamedSharding(
                resolved_mesh, P(None, None, "tp", None)
            )
        else:
            self.rep_sharding = None
            self.head_sharding = None
            self.kv_head_sharding = None
            self.kv_cache_sharding = None

        self.vanilla_prefill_compile_count = 0
        self.vanilla_decode_compile_count = 0
        self.static_step_compile_count = 0
        self.commit_kv_compile_count = 0

        self._compile_kernels()

    def _put_rep(self, arr: Any, dtype: Any | None = None) -> jax.Array:
        jarr = jnp.asarray(arr, dtype=dtype) if dtype is not None else jnp.asarray(arr)
        if self.rep_sharding is not None:
            return jax.device_put(jarr, self.rep_sharding)
        return jarr

    def _compile_kernels(self) -> None:
        cfg = self.config
        w = self.weights
        max_seq = self.max_seq_len
        tile_size = self.tile_size
        num_q = cfg.num_attention_heads
        num_kv = cfg.num_key_value_heads
        head_dim = cfg.head_dim
        rep_sharding = self.rep_sharding
        head_sharding = self.head_sharding
        kv_head_sharding = self.kv_head_sharding
        kv_cache_sharding = self.kv_cache_sharding

        self._weights_tuple: TargetWeightsTuple = w.to_target_weights_tuple()
        del w

        def _empty_kv_cache() -> tuple[jax.Array, jax.Array]:
            shape = (cfg.num_hidden_layers, max_seq + 1, num_kv, head_dim)
            if kv_cache_sharding is not None:
                return (
                    jnp.zeros(shape, dtype=self.dtype, device=kv_cache_sharding),
                    jnp.zeros(shape, dtype=self.dtype, device=kv_cache_sharding),
                )
            return jnp.zeros(shape, dtype=self.dtype), jnp.zeros(shape, dtype=self.dtype)

        self._empty_kv_cache = _empty_kv_cache

        @jax.jit
        def vanilla_prefill_jit(
            wt: TargetWeightsTuple,
            prompt_ids: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.vanilla_prefill_compile_count += 1
            seq_p = prompt_ids.shape[0]
            positions = jnp.arange(seq_p, dtype=jnp.int32)
            x0 = wt.embed[prompt_ids]
            if rep_sharding is not None:
                x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)
            causal = jnp.where(
                jnp.arange(seq_p)[:, None] >= jnp.arange(seq_p)[None, :],
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)

            def body(x: jax.Array, scan_in: tuple[LayerScanWeights, jax.Array, jax.Array]):
                lw, kc, vc = scan_in
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                q, k_new, v_new = _qwen_qkv_proj(
                    x, lw, positions, cfg, head_sharding, kv_head_sharding
                )
                kc = kc.at[:seq_p].set(k_new)
                vc = vc.at[:seq_p].set(v_new)
                attn_out = _gqa_attention(
                    q, k_new, v_new, causal, num_q, num_kv, head_dim
                )
                return _qwen_block_post_attn(x, attn_out, lw, cfg, rep_sharding), (kc, vc)

            x_final, (k_out, v_out) = jax.lax.scan(body, x0, (wt.layers, k_cache, v_cache))
            last_h = _rms_norm(x_final[-1:], wt.final_norm, cfg.rms_norm_eps)
            logits = _linear(
                last_h, wt.lm_head, w_scale=wt.lm_head_scale, out_dtype=jnp.float32
            )[0]
            if rep_sharding is not None:
                logits = jax.lax.with_sharding_constraint(logits, rep_sharding)
            return logits, k_out, v_out

        @jax.jit
        def vanilla_decode_step_jit(
            wt: TargetWeightsTuple,
            token_id: jax.Array,
            pos: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.vanilla_decode_compile_count += 1
            x0 = wt.embed[token_id].reshape(1, cfg.hidden_size)
            if rep_sharding is not None:
                x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)
            pos_arr = jnp.asarray([pos], dtype=jnp.int32)
            valid_mask = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) <= pos,
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)[None, :]

            def body(x: jax.Array, scan_in: tuple[LayerScanWeights, jax.Array, jax.Array]):
                lw, kc, vc = scan_in
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                q, k_new, v_new = _qwen_qkv_proj(
                    x, lw, pos_arr, cfg, head_sharding, kv_head_sharding
                )
                kc = kc.at[pos].set(k_new[0])
                vc = vc.at[pos].set(v_new[0])
                attn_out = _gqa_attention(
                    q, kc[:max_seq], vc[:max_seq], valid_mask, num_q, num_kv, head_dim
                )
                return _qwen_block_post_attn(x, attn_out, lw, cfg, rep_sharding), (kc, vc)

            x_final, (k_out, v_out) = jax.lax.scan(body, x0, (wt.layers, k_cache, v_cache))
            last_h = _rms_norm(x_final, wt.final_norm, cfg.rms_norm_eps)
            logits = _linear(
                last_h, wt.lm_head, w_scale=wt.lm_head_scale, out_dtype=jnp.float32
            )[0]
            if rep_sharding is not None:
                logits = jax.lax.with_sharding_constraint(logits, rep_sharding)
            return logits, k_out, v_out

        @jax.jit
        def static_systolic_tile_step_jit(
            wt: TargetWeightsTuple,
            valid_prefix_len: jax.Array,
            token_tile: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
            k_cache: jax.Array,
            v_cache: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.static_step_compile_count += 1
            x0 = wt.embed[token_tile]
            if rep_sharding is not None:
                x0 = jax.lax.with_sharding_constraint(x0, rep_sharding)
            prefix_allowed = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) < valid_prefix_len,
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)
            prefix_mask = jnp.broadcast_to(prefix_allowed[None, :], (tile_size, max_seq))
            full_mask = jnp.concatenate([prefix_mask, tile_mask], axis=-1)

            def body(x: jax.Array, scan_in: tuple[LayerScanWeights, jax.Array, jax.Array]):
                lw, kc, vc = scan_in
                if rep_sharding is not None:
                    x = jax.lax.with_sharding_constraint(x, rep_sharding)
                q, k_step, v_step = _qwen_qkv_proj(
                    x, lw, pos_tile, cfg, head_sharding, kv_head_sharding
                )
                k_all = jnp.concatenate([kc[:max_seq], k_step], axis=0)
                v_all = jnp.concatenate([vc[:max_seq], v_step], axis=0)
                attn_out = _gqa_attention(
                    q, k_all, v_all, full_mask, num_q, num_kv, head_dim
                )
                return _qwen_block_post_attn(x, attn_out, lw, cfg, rep_sharding), (k_step, v_step)

            x_final, (step_k_out, step_v_out) = jax.lax.scan(
                body, x0, (wt.layers, k_cache, v_cache)
            )
            normed = _rms_norm(x_final, wt.final_norm, cfg.rms_norm_eps)
            logits_tile = _linear(
                normed, wt.lm_head, w_scale=wt.lm_head_scale, out_dtype=jnp.float32
            )
            has_nan = jnp.any(jnp.isnan(logits_tile), axis=-1)
            argmax_tokens = jnp.where(
                has_nan,
                jnp.int32(-1),
                jnp.argmax(logits_tile, axis=-1).astype(jnp.int32),
            )
            if rep_sharding is not None:
                argmax_tokens = jax.lax.with_sharding_constraint(argmax_tokens, rep_sharding)
            return argmax_tokens, step_k_out, step_v_out

        @jax.jit
        def commit_kv_jit(
            k_cache: jax.Array,
            v_cache: jax.Array,
            step_k: jax.Array,
            step_v: jax.Array,
            prev_prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            self.commit_kv_compile_count += 1
            gathered_k = step_k[:, src_rows, :, :]
            gathered_v = step_v[:, src_rows, :, :]
            idx = jnp.arange(tile_size, dtype=jnp.int32)
            dest_slots = jnp.where(idx < n_valid, prev_prefix_len + idx, max_seq)
            k_next = k_cache.at[:, dest_slots, :, :].set(gathered_k)
            v_next = v_cache.at[:, dest_slots, :, :].set(gathered_v)
            if kv_cache_sharding is not None:
                k_next = jax.lax.with_sharding_constraint(k_next, kv_cache_sharding)
                v_next = jax.lax.with_sharding_constraint(v_next, kv_cache_sharding)
            return k_next, v_next

        self._vanilla_prefill_jit = vanilla_prefill_jit
        self._vanilla_decode_step_jit = vanilla_decode_step_jit
        self._static_systolic_tile_step_jit = static_systolic_tile_step_jit
        self._commit_kv_jit = commit_kv_jit

    def _decode_text(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
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

        k_cache, v_cache = self._empty_kv_cache()
        jax.block_until_ready((k_cache, v_cache))

        t0 = time.perf_counter()
        prompt_arr = self._put_rep(prompt_tokens, dtype=jnp.int32)
        logits, k_cache, v_cache = self._vanilla_prefill_jit(
            self._weights_tuple, prompt_arr, k_cache, v_cache
        )
        next_tok = int(jnp.argmax(logits))
        jax.block_until_ready((logits, k_cache, v_cache))
        t1 = time.perf_counter()

        generated: list[int] = []
        decode_passes = 0
        t_dec0 = time.perf_counter()
        if next_tok not in eos_set:
            generated.append(next_tok)
            cur_pos = len(prompt_tokens)
            while len(generated) < max_new_tokens:
                logits, k_cache, v_cache = self._vanilla_decode_step_jit(
                    self._weights_tuple,
                    self._put_rep(next_tok, dtype=jnp.int32),
                    self._put_rep(cur_pos, dtype=jnp.int32),
                    k_cache,
                    v_cache,
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
        total_s = prefill_s + decode_s
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
            total_s=total_s,
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
            },
        )

    def run_static_arena(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        """Mode 2 (Lever 1 M-002): Fixed-tile OpenXLA step, zero recompilations, serial decode."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.static_step_compile_count

        k_cache, v_cache = self._empty_kv_cache()
        jax.block_until_ready((k_cache, v_cache))

        t0 = time.perf_counter()
        prefill_len = len(prompt_tokens) - 1
        forward_passes = 0
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

            _, step_k, step_v = self._static_systolic_tile_step_jit(
                self._weights_tuple,
                self._put_rep(offset, dtype=jnp.int32),
                self._put_rep(tok_np),
                self._put_rep(pos_np),
                self._put_rep(mask_np),
                k_cache,
                v_cache,
            )
            src_rows = np.zeros((T,), dtype=np.int32)
            src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
            k_cache, v_cache = self._commit_kv_jit(
                k_cache,
                v_cache,
                step_k,
                step_v,
                self._put_rep(offset, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(chunk, dtype=jnp.int32),
            )
            forward_passes += 1
        jax.block_until_ready((k_cache, v_cache))
        t1 = time.perf_counter()

        decode_mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
        np.fill_diagonal(decode_mask_np, 0.0)
        decode_mask_jnp = self._put_rep(decode_mask_np)
        src_row_0 = self._put_rep(np.zeros((T,), dtype=np.int32))
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

            argmax_tokens, step_k, step_v = self._static_systolic_tile_step_jit(
                self._weights_tuple,
                self._put_rep(cur_prefix, dtype=jnp.int32),
                self._put_rep(tok_np),
                self._put_rep(pos_np),
                decode_mask_jnp,
                k_cache,
                v_cache,
            )
            forward_passes += 1
            decode_passes += 1

            argmax_np = np.asarray(jax.device_get(argmax_tokens), dtype=np.int32)
            next_tok = int(argmax_np[0])
            if next_tok < 0:
                raise ValueError("NaN detected in target logits")
            if next_tok in eos_set:
                break

            k_cache, v_cache = self._commit_kv_jit(
                k_cache,
                v_cache,
                step_k,
                step_v,
                self._put_rep(cur_prefix, dtype=jnp.int32),
                src_row_0,
                one_jnp,
            )
            generated.append(next_tok)
            cur_prefix += 1
            cur_tok = next_tok
        jax.block_until_ready((k_cache, v_cache))
        t_dec1 = time.perf_counter()

        compiles_delta = self.static_step_compile_count - compiles_before
        prefill_s = t1 - t0
        decode_s = t_dec1 - t_dec0
        total_s = prefill_s + decode_s
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
            total_s=total_s,
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
            },
        )

    def run_tpuforge_tree(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        """Mode 3 (Levers 1 + 2 M-002 + M-003): Native C++ ForgeEngine systolic tree."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.config.eos_token_ids)
        )
        compiles_before = self.static_step_compile_count

        k_cache, v_cache = self._empty_kv_cache()
        jax.block_until_ready((k_cache, v_cache))
        state: dict[str, Any] = {
            "k_cache": k_cache,
            "v_cache": v_cache,
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
                state["k_cache"],
                state["v_cache"],
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
                state["k_cache"],
                state["v_cache"],
                state["step_k"],
                state["step_v"],
                self._put_rep(prev_prefix_len, dtype=jnp.int32),
                self._put_rep(src_rows),
                self._put_rep(n_valid, dtype=jnp.int32),
            )
            state["k_cache"] = k_next
            state["v_cache"] = v_next
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
            f"qwen-tpuforge-{T}",
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
        jax.block_until_ready((state["k_cache"], state["v_cache"]))
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
            },
        )


def load_qwen_runner(
    model_id_or_path: str,
    max_seq_len: int = 1024,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    tp_size: int | str = "auto",
    enable_xla_cache: bool = True,
    weight_quant: str = "none",
) -> tuple[QwenBenchmarkRunner, Path]:
    """Load config, tokenizer, and streaming safetensors weights into QwenBenchmarkRunner."""
    from tokenizers import Tokenizer

    if enable_xla_cache:
        configure_xla_compilation_cache()

    model_dir = resolve_model_dir(model_id_or_path)
    config = QwenConfig.from_dir(model_dir)
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    dtype = resolve_default_dtype(dtype_name)
    _, mesh = resolve_tp_mesh(tp_size, config=config)
    weights = QwenWeights.from_model_dir(
        model_dir, config, dtype=dtype, mesh=mesh, weight_quant=weight_quant
    )
    runner = QwenBenchmarkRunner(
        config=config,
        weights=weights,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tile_size=tile_size,
        mesh=mesh,
    )
    return runner, model_dir

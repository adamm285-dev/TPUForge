"""Host-DRAM / mmap sparse MoE expert streaming with fixed-shape TPU systolic tree verification."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Self

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import PartitionSpec as P

from benchmarks.quantization_jax import (
    batched_expert_systolic_linear,
    normalize_weight_quant,
    place_np_sharded_preserved,
    pytree_nbytes,
    quant_np_dtype,
    quantize_and_place_2d,
    quantize_np_per_channel,
    systolic_linear,
)
from benchmarks.qwen_jax import (
    _MASK_BLOCK_VAL,
    RunMetrics,
    _apply_qwen_rope,
    _gqa_attention,
    _iter_safetensors_index,
    _jnp_to_numpy_dtype,
    _MappedSafetensorsReader,
    _read_indexed_tensor,
    _rms_norm,
    configure_xla_compilation_cache,
    resolve_default_dtype,
    resolve_model_dir,
    resolve_tile_size,
    tpu,
)

VALID_HOST_STORE_MODES = ("mmap", "dram")


def normalize_host_store_mode(mode: str | None) -> str:
    """Validate and normalize MoE host backing store mode ('mmap' or 'dram')."""
    if isinstance(mode, bool):
        raise ValueError("host_store_mode cannot be bool")  # noqa: TRY004
    if mode is None:
        return "mmap"
    if not isinstance(mode, str):
        raise TypeError(f"host_store_mode must be str, got {type(mode).__name__}")
    norm = mode.strip().lower()
    if norm in VALID_HOST_STORE_MODES:
        return norm
    raise ValueError(f"Unsupported host_store_mode={mode!r}; expected 'mmap' or 'dram'")


@dataclass(frozen=True)
class MoeConfig:
    """Architecture hyperparameters for Hugging Face OLMoE and Qwen2/3-MoE checkpoints."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()
    shared_expert_intermediate_size: int = 0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    eos_token_ids: tuple[int, ...] = (151645, 151643)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return whether decoder layer `layer_idx` uses sparse routed MoE experts."""
        return (layer_idx not in self.mlp_only_layers) and (
            (layer_idx + 1) % self.decoder_sparse_step == 0
        )

    @classmethod
    def from_dir(cls, model_dir: Path) -> MoeConfig:
        """Parse Hugging Face OLMoE / Qwen3-MoE / Qwen2-MoE config.json + generation_config.json."""
        cfg_path = model_dir / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing config.json in {model_dir}")
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

        model_type = str(raw.get("model_type", "qwen3_moe")).lower()
        hidden_size = int(raw["hidden_size"])
        num_heads = int(raw["num_attention_heads"])
        num_kv_heads = int(raw.get("num_key_value_heads", num_heads))
        head_dim = int(raw.get("head_dim") or (hidden_size // num_heads))
        num_experts = int(raw.get("num_experts") or raw.get("num_local_experts") or 0)
        num_experts_per_tok = int(raw.get("num_experts_per_tok", 2))
        decoder_sparse_step = int(raw.get("decoder_sparse_step", 1))
        if num_experts < 2:
            raise ValueError(f"Expected sparse MoE num_experts >= 2, got {num_experts}")
        if num_experts_per_tok < 1 or num_experts_per_tok > num_experts:
            raise ValueError(
                f"Invalid num_experts_per_tok={num_experts_per_tok} for num_experts={num_experts}"
            )
        if decoder_sparse_step < 1:
            raise ValueError(f"decoder_sparse_step must be >= 1, got {decoder_sparse_step}")

        intermediate_size = int(raw.get("intermediate_size", hidden_size * 4))
        moe_intermediate_size = int(raw.get("moe_intermediate_size") or intermediate_size)
        is_olmoe = "olmoe" in model_type
        norm_topk_prob = (
            bool(raw["norm_topk_prob"]) if "norm_topk_prob" in raw else (not is_olmoe)
        )
        rms_norm_eps = float(raw.get("rms_norm_eps", 1e-5 if is_olmoe else 1e-6))

        eos_ids: list[int] = []
        for src_cfg in (raw,):
            raw_eos = src_cfg.get("eos_token_id")
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

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=int(raw["vocab_size"]),
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=tuple(int(x) for x in raw.get("mlp_only_layers", [])),
            shared_expert_intermediate_size=int(raw.get("shared_expert_intermediate_size", 0)),
            rms_norm_eps=rms_norm_eps,
            rope_theta=float(raw.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            eos_token_ids=tuple(dedup_eos),
        )


class MoeLayerTrunkWeights(NamedTuple):
    """HBM-pinned per-layer attention, router, and optional dense/shared MLP weights."""

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
    q_norm: jax.Array | None
    k_norm: jax.Array | None
    router_w: jax.Array | None
    shared_gate: jax.Array | None
    shared_gate_scale: jax.Array | None
    shared_up: jax.Array | None
    shared_up_scale: jax.Array | None
    shared_down: jax.Array | None
    shared_down_scale: jax.Array | None
    shared_router_w: jax.Array | None


@dataclass
class MoeTrunkWeights:
    """HBM-pinned non-expert trunk weights (embed, norms, attention, routers, lm_head)."""

    embed: jax.Array
    final_norm: jax.Array
    lm_head: jax.Array
    lm_head_scale: jax.Array | None
    layers: tuple[MoeLayerTrunkWeights, ...]
    weight_quant: str = "none"

    def trunk_bytes(self) -> int:
        """Return total HBM byte footprint of all pinned non-expert trunk weights."""
        return pytree_nbytes((self.embed, self.final_norm, self.lm_head, self.lm_head_scale, self.layers))

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path,
        config: MoeConfig,
        dtype: jnp.dtype,
        weight_quant: str = "none",
    ) -> MoeTrunkWeights:
        """Stream only trunk weights (attention, norms, routers, head) onto device HBM."""
        norm_quant = normalize_weight_quant(weight_quant)
        np_dtype = _jnp_to_numpy_dtype(dtype)
        lin_read_dtype = None if norm_quant != "none" else np_dtype
        tensor_loc = _iter_safetensors_index(model_dir)

        with _MappedSafetensorsReader() as reader:
            def read_vec(name: str, target_np_dtype: Any = np_dtype) -> jax.Array:
                if name not in tensor_loc:
                    raise KeyError(f"Missing MoE trunk tensor {name!r}")
                arr = _read_indexed_tensor(tensor_loc[name], target_np_dtype, reader=reader)
                return jax.device_put(jnp.asarray(arr))

            def read_vec_optional(name: str, target_np_dtype: Any = np_dtype) -> jax.Array | None:
                if name not in tensor_loc:
                    return None
                return read_vec(name, target_np_dtype=target_np_dtype)

            def read_linear(name: str) -> tuple[jax.Array, jax.Array | None]:
                if name not in tensor_loc:
                    raise KeyError(f"Missing MoE linear tensor {name!r}")
                arr = _read_indexed_tensor(tensor_loc[name], lin_read_dtype, reader=reader)
                return quantize_and_place_2d(
                    np.ascontiguousarray(arr.T),
                    P(),
                    None,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )

            def read_linear_optional(name: str) -> tuple[jax.Array | None, jax.Array | None]:
                if name not in tensor_loc:
                    return None, None
                return read_linear(name)

            embed = read_vec("model.embed_tokens.weight")
            final_norm = read_vec("model.norm.weight")
            if "lm_head.weight" in tensor_loc:
                lm_head, lm_head_scale = read_linear("lm_head.weight")
            elif config.tie_word_embeddings:
                arr = _read_indexed_tensor(
                    tensor_loc["model.embed_tokens.weight"], lin_read_dtype, reader=reader
                )
                lm_head, lm_head_scale = quantize_and_place_2d(
                    np.ascontiguousarray(arr.T),
                    P(),
                    None,
                    norm_quant,
                    active_np_dtype=np_dtype,
                )
            else:
                raise KeyError("lm_head.weight missing and tie_word_embeddings is False")

            layers: list[MoeLayerTrunkWeights] = []
            for i in range(config.num_hidden_layers):
                p = f"model.layers.{i}"
                wq, wq_scale = read_linear(f"{p}.self_attn.q_proj.weight")
                wk, wk_scale = read_linear(f"{p}.self_attn.k_proj.weight")
                wv, wv_scale = read_linear(f"{p}.self_attn.v_proj.weight")
                wo, wo_scale = read_linear(f"{p}.self_attn.o_proj.weight")

                if config.is_moe_layer(i):
                    router_arr = _read_indexed_tensor(
                        tensor_loc[f"{p}.mlp.gate.weight"], np.float32, reader=reader
                    )
                    router_w: jax.Array | None = place_np_sharded_preserved(
                        np.ascontiguousarray(router_arr.T), P(), None
                    )
                    sg, sg_s = read_linear_optional(f"{p}.mlp.shared_expert.gate_proj.weight")
                    su, su_s = read_linear_optional(f"{p}.mlp.shared_expert.up_proj.weight")
                    sd, sd_s = read_linear_optional(f"{p}.mlp.shared_expert.down_proj.weight")
                    sr_key = f"{p}.mlp.shared_expert_gate.weight"
                    if sr_key in tensor_loc:
                        sr_arr = _read_indexed_tensor(tensor_loc[sr_key], np.float32, reader=reader)
                        shared_router_w: jax.Array | None = place_np_sharded_preserved(
                            np.ascontiguousarray(sr_arr.T), P(), None
                        )
                    else:
                        shared_router_w = None
                else:
                    router_w = None
                    sg, sg_s = read_linear(f"{p}.mlp.gate_proj.weight")
                    su, su_s = read_linear(f"{p}.mlp.up_proj.weight")
                    sd, sd_s = read_linear(f"{p}.mlp.down_proj.weight")
                    shared_router_w = None

                layers.append(
                    MoeLayerTrunkWeights(
                        in_norm=read_vec(f"{p}.input_layernorm.weight"),
                        post_norm=read_vec(f"{p}.post_attention_layernorm.weight"),
                        wq=wq,
                        wq_scale=wq_scale,
                        bq=read_vec_optional(f"{p}.self_attn.q_proj.bias"),
                        wk=wk,
                        wk_scale=wk_scale,
                        bk=read_vec_optional(f"{p}.self_attn.k_proj.bias"),
                        wv=wv,
                        wv_scale=wv_scale,
                        bv=read_vec_optional(f"{p}.self_attn.v_proj.bias"),
                        wo=wo,
                        wo_scale=wo_scale,
                        q_norm=read_vec_optional(f"{p}.self_attn.q_norm.weight"),
                        k_norm=read_vec_optional(f"{p}.self_attn.k_norm.weight"),
                        router_w=router_w,
                        shared_gate=sg,
                        shared_gate_scale=sg_s,
                        shared_up=su,
                        shared_up_scale=su_s,
                        shared_down=sd,
                        shared_down_scale=sd_s,
                        shared_router_w=shared_router_w,
                    )
                )

        return cls(
            embed=embed,
            final_norm=final_norm,
            lm_head=lm_head,
            lm_head_scale=lm_head_scale,
            layers=tuple(layers),
            weight_quant=norm_quant,
        )


_ExpertEntry = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]

_SAFETENSORS_DTYPE_NAMES: dict[str, Any] = {
    "BF16": ml_dtypes.bfloat16,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "U8": np.uint8,
    "F8_E4M3": ml_dtypes.float8_e4m3fn,
}


class ExpertHostStore:
    """Host-side sparse MoE expert store backed by OS mmap safetensors slices or Host DRAM."""

    def __init__(
        self,
        model_dir: Path,
        config: MoeConfig,
        dtype: jnp.dtype,
        weight_quant: str = "none",
        host_store_mode: str = "mmap",
        lru_cache_experts: int = 64,
        layer_prefix: str = "model.",
        experts_subpath: str = "mlp.experts",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = config
        self.dtype = dtype
        self.weight_quant = normalize_weight_quant(weight_quant)
        self.host_store_mode = normalize_host_store_mode(host_store_mode)
        self.lru_cache_experts = max(0, int(lru_cache_experts))

        self.active_np_dtype = _jnp_to_numpy_dtype(dtype)
        self.storage_np_dtype = (
            quant_np_dtype(self.weight_quant)
            if self.weight_quant != "none"
            else self.active_np_dtype
        )

        self.layer_prefix = layer_prefix
        self.experts_subpath = experts_subpath.strip(".")
        self._tensor_loc = _iter_safetensors_index(self.model_dir)
        self._layer_layout: dict[int, str] = {}
        self._moe_layers: tuple[int, ...] = tuple(
            i for i in range(config.num_hidden_layers) if config.is_moe_layer(i)
        )
        for i in self._moe_layers:
            base = f"{self.layer_prefix}layers.{i}.{self.experts_subpath}"
            key_2d = f"{base}.0.gate_proj.weight"
            key_3d = f"{base}.gate_up_proj"
            if key_2d in self._tensor_loc:
                self._layer_layout[i] = "2d"
            elif key_3d in self._tensor_loc and f"{base}.down_proj" in self._tensor_loc:
                self._layer_layout[i] = "3d_stacked"
            else:
                raise KeyError(
                    f"Missing sparse MoE expert tensors for layer {i} in {self.model_dir}"
                )

        self._reader: _MappedSafetensorsReader | None = None
        self._lru: OrderedDict[tuple[int, int], _ExpertEntry] = OrderedDict()
        self._dram_bank: dict[int, list[_ExpertEntry]] | None = None

        reader = _MappedSafetensorsReader()
        if self.host_store_mode == "dram":
            try:
                bank: dict[int, list[_ExpertEntry]] = {}
                for l_idx in self._moe_layers:
                    bank[l_idx] = [
                        self._load_expert_from_reader(reader, l_idx, e_idx)
                        for e_idx in range(config.num_experts)
                    ]
                self._dram_bank = bank
            finally:
                reader.close()
        else:
            self._reader = reader

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Release open mmap file handles and cached host buffers immediately."""
        self._lru.clear()
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def single_expert_bytes(self) -> int:
        """Return byte size of one expert triplet (and scales if quantized) in host storage."""
        h = self.config.hidden_size
        m = self.config.moe_intermediate_size
        w_bytes = 3 * h * m * int(np.dtype(self.storage_np_dtype).itemsize)
        scale_bytes = ((2 * m + h) * 4) if self.weight_quant != "none" else 0
        return w_bytes + scale_bytes

    def total_source_bytes(self) -> int:
        """Return total byte size across all MoE layers and all experts."""
        return len(self._moe_layers) * self.config.num_experts * self.single_expert_bytes()

    def staging_buffer_bytes(self, stage_capacity: int) -> int:
        """Return HBM byte size of the fixed-shape [E_stage] expert staging buffer."""
        return int(stage_capacity) * self.single_expert_bytes()

    def _read_stacked_slice(
        self,
        reader: _MappedSafetensorsReader,
        loc: tuple[Path, str, tuple[int, ...], int, int],
        expert_idx: int,
    ) -> np.ndarray:
        path, dtype_str, shape, start_byte, end_byte = loc
        if len(shape) != 3 or expert_idx < 0 or expert_idx >= shape[0]:
            raise ValueError(f"Invalid 3D stacked expert slice shape={shape} idx={expert_idx}")
        src_dtype = _SAFETENSORS_DTYPE_NAMES[dtype_str]
        slice_elems = math.prod(shape[1:])
        slice_bytes = slice_elems * int(np.dtype(src_dtype).itemsize)
        offset = start_byte + expert_idx * slice_bytes
        if offset + slice_bytes > end_byte:
            raise ValueError(f"Truncated 3D stacked expert payload in {path}")
        mm = reader.get_map(path)
        arr = np.frombuffer(mm, dtype=src_dtype, count=slice_elems, offset=offset).reshape(shape[1:])
        read_dtype = np.float32 if self.weight_quant != "none" else self.active_np_dtype
        if arr.dtype != read_dtype:
            return np.asarray(arr, dtype=read_dtype)
        return arr.copy()

    def _prepare_expert_matrix(self, w_in_out: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        w_contig = np.ascontiguousarray(w_in_out)
        if self.weight_quant == "none":
            return np.asarray(w_contig, dtype=self.active_np_dtype), None
        return quantize_np_per_channel(w_contig, self.weight_quant)

    def _load_expert_from_reader(
        self,
        reader: _MappedSafetensorsReader,
        layer_idx: int,
        expert_idx: int,
    ) -> _ExpertEntry:
        layout = self._layer_layout[layer_idx]
        read_dtype = np.float32 if self.weight_quant != "none" else self.active_np_dtype
        moe_inter = self.config.moe_intermediate_size
        base = f"{self.layer_prefix}layers.{layer_idx}.{self.experts_subpath}"
        if layout == "2d":
            p = f"{base}.{expert_idx}"
            raw_g = _read_indexed_tensor(
                self._tensor_loc[f"{p}.gate_proj.weight"], read_dtype, reader=reader
            )
            raw_u = _read_indexed_tensor(
                self._tensor_loc[f"{p}.up_proj.weight"], read_dtype, reader=reader
            )
            raw_d = _read_indexed_tensor(
                self._tensor_loc[f"{p}.down_proj.weight"], read_dtype, reader=reader
            )
        else:
            p = base
            gu = self._read_stacked_slice(reader, self._tensor_loc[f"{p}.gate_up_proj"], expert_idx)
            if gu.shape[0] != 2 * moe_inter:
                raise ValueError(
                    f"Expected stacked gate_up_proj slice axis 0 == {2 * moe_inter}, got {gu.shape}"
                )
            raw_g = gu[:moe_inter, :]
            raw_u = gu[moe_inter:, :]
            raw_d = self._read_stacked_slice(reader, self._tensor_loc[f"{p}.down_proj"], expert_idx)

        wg, sg = self._prepare_expert_matrix(raw_g.T)
        wu, su = self._prepare_expert_matrix(raw_u.T)
        wd, sd = self._prepare_expert_matrix(raw_d.T)
        return wg, wu, wd, sg, su, sd

    def get_expert(self, layer_idx: int, expert_idx: int) -> _ExpertEntry:
        """Fetch one transposed (and optionally quantized) expert triplet from DRAM or mmap."""
        if self._dram_bank is not None:
            return self._dram_bank[layer_idx][expert_idx]

        key = (layer_idx, expert_idx)
        if self.lru_cache_experts > 0 and key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]

        if self._reader is None:
            raise RuntimeError("ExpertHostStore is closed; cannot stream experts after close()")
        entry = self._load_expert_from_reader(self._reader, layer_idx, expert_idx)
        if self.lru_cache_experts > 0:
            self._lru[key] = entry
            if len(self._lru) > self.lru_cache_experts:
                self._lru.popitem(last=False)
        return entry

    def iter_staged_chunks(
        self,
        layer_idx: int,
        top_idx_np: np.ndarray,
        top_vals_np: np.ndarray,
        valid_row_mask: np.ndarray,
        stage_capacity: int,
    ) -> Iterator[
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """Group active-row routed experts into fixed [E_stage] host staging chunks."""
        if not np.any(valid_row_mask):
            return

        active_experts = np.unique(top_idx_np[valid_row_mask])
        e_stage = int(stage_capacity)
        hidden = self.config.hidden_size
        moe_inter = self.config.moe_intermediate_size
        has_scales = self.weight_quant != "none"

        lut = np.empty((self.config.num_experts,), dtype=np.int32)
        for start in range(0, len(active_experts), e_stage):
            chunk = active_experts[start : start + e_stage]
            chunk_len = len(chunk)
            wg_buf = np.zeros((e_stage, hidden, moe_inter), dtype=self.storage_np_dtype)
            wu_buf = np.zeros((e_stage, hidden, moe_inter), dtype=self.storage_np_dtype)
            wd_buf = np.zeros((e_stage, moe_inter, hidden), dtype=self.storage_np_dtype)
            sg_buf = np.ones((e_stage, moe_inter), dtype=np.float32) if has_scales else None
            su_buf = np.ones((e_stage, moe_inter), dtype=np.float32) if has_scales else None
            sd_buf = np.ones((e_stage, hidden), dtype=np.float32) if has_scales else None

            lut.fill(-1)
            for slot, raw_eid in enumerate(chunk):
                eid = int(raw_eid)
                lut[eid] = slot
                wg_e, wu_e, wd_e, sg_e, su_e, sd_e = self.get_expert(layer_idx, eid)
                wg_buf[slot] = wg_e
                wu_buf[slot] = wu_e
                wd_buf[slot] = wd_e
                if has_scales:
                    assert sg_buf is not None and su_buf is not None and sd_buf is not None
                    assert sg_e is not None and su_e is not None and sd_e is not None
                    sg_buf[slot] = sg_e
                    su_buf[slot] = su_e
                    sd_buf[slot] = sd_e

            slot_idx = lut[top_idx_np]
            slot_idx = np.where(valid_row_mask[:, None], slot_idx, np.int32(-1))
            slot_weights = np.where(slot_idx >= 0, top_vals_np, np.float32(0.0)).astype(
                np.float32
            )
            yield (
                chunk_len,
                wg_buf,
                wu_buf,
                wd_buf,
                sg_buf,
                su_buf,
                sd_buf,
                slot_idx,
                slot_weights,
            )


def _apply_qk_norm(
    proj: jax.Array,
    norm_w: jax.Array | None,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    eps: float,
) -> jax.Array:
    """Apply either OLMoE flattened [num_heads * head_dim] or Qwen3 per-head [head_dim] RMSNorm."""
    if norm_w is None:
        return proj.reshape(seq_len, num_heads, head_dim)
    if norm_w.shape[-1] == num_heads * head_dim:
        return _rms_norm(proj, norm_w, eps).reshape(seq_len, num_heads, head_dim)
    if norm_w.shape[-1] == head_dim:
        return _rms_norm(proj.reshape(seq_len, num_heads, head_dim), norm_w, eps)
    raise ValueError(
        f"Incompatible q/k_norm shape {norm_w.shape} for num_heads={num_heads}, head_dim={head_dim}"
    )


def _layer_attn_and_norms(
    x: jax.Array,
    lw: MoeLayerTrunkWeights,
    positions: jax.Array,
    attn_mask: jax.Array,
    k_cache_l: jax.Array,
    v_cache_l: jax.Array,
    max_prefix: int | None,
    cfg: MoeConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run GQA attention against prefix KV cache and return (x_base, h_mlp, k_step, v_step)."""
    seq_q = x.shape[0]
    h = _rms_norm(x, lw.in_norm, cfg.rms_norm_eps)
    q_proj = systolic_linear(h, lw.wq, lw.bq, w_scale=lw.wq_scale)
    q_normed = _apply_qk_norm(
        q_proj, lw.q_norm, seq_q, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
    )
    q = _apply_qwen_rope(q_normed, positions, cfg.head_dim, cfg.rope_theta)

    k_proj = systolic_linear(h, lw.wk, lw.bk, w_scale=lw.wk_scale)
    k_normed = _apply_qk_norm(
        k_proj, lw.k_norm, seq_q, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps
    )
    k_step = _apply_qwen_rope(k_normed, positions, cfg.head_dim, cfg.rope_theta)
    v_step = systolic_linear(h, lw.wv, lw.bv, w_scale=lw.wv_scale).reshape(
        seq_q, cfg.num_key_value_heads, cfg.head_dim
    )

    if max_prefix is not None:
        k_all = jnp.concatenate([k_cache_l[:max_prefix], k_step], axis=0)
        v_all = jnp.concatenate([v_cache_l[:max_prefix], v_step], axis=0)
    else:
        k_all = k_step
        v_all = v_step

    attn_out = _gqa_attention(
        q,
        k_all,
        v_all,
        attn_mask,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
    )
    x_post_attn = x + systolic_linear(attn_out, lw.wo, w_scale=lw.wo_scale)
    h_mlp = _rms_norm(x_post_attn, lw.post_norm, cfg.rms_norm_eps)
    return x_post_attn, h_mlp, k_step, v_step


def _add_shared_expert_residual(
    x_base: jax.Array,
    h_mlp: jax.Array,
    lw: MoeLayerTrunkWeights,
) -> jax.Array:
    """Compute optional dense/shared SwiGLU residual with optional sigmoid shared-gate."""
    if lw.shared_gate is None or lw.shared_up is None or lw.shared_down is None:
        return x_base
    gate = systolic_linear(h_mlp, lw.shared_gate, w_scale=lw.shared_gate_scale)
    up = systolic_linear(h_mlp, lw.shared_up, w_scale=lw.shared_up_scale)
    act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(h_mlp.dtype)
    shared_out = systolic_linear(act, lw.shared_down, w_scale=lw.shared_down_scale)
    if lw.shared_router_w is not None:
        gate_scale = jax.nn.sigmoid(
            jnp.matmul(h_mlp.astype(jnp.float32), lw.shared_router_w.astype(jnp.float32))
        ).astype(shared_out.dtype)
        shared_out = shared_out * gate_scale
    return x_base + shared_out


def _route_tokens(
    h_mlp: jax.Array,
    router_w: jax.Array,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[jax.Array, jax.Array]:
    """Compute top-k router expert IDs and gating weights in FP32."""
    router_logits = jnp.matmul(
        h_mlp.astype(jnp.float32), router_w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
    )
    probs = jax.nn.softmax(router_logits, axis=-1)
    top_vals, top_idx = jax.lax.top_k(probs, top_k)
    if norm_topk_prob:
        denom = jnp.maximum(jnp.sum(top_vals, axis=-1, keepdims=True), jnp.float32(1e-12))
        top_vals = top_vals / denom
    return top_idx.astype(jnp.int32), top_vals.astype(jnp.float32)


class MoeStreamingRunner:
    """Executes Vanilla, StaticArena, and TPUForge Tree modes over host-streamed MoE experts."""

    def __init__(
        self,
        config: MoeConfig,
        trunk: MoeTrunkWeights,
        expert_store: ExpertHostStore,
        tokenizer: Any = None,
        max_seq_len: int = 1024,
        tile_size: int | str = "auto",
        stage_experts: int = 16,
    ) -> None:
        resolved_tile = resolve_tile_size(tile_size)
        if max_seq_len < 2 * resolved_tile:
            raise ValueError(f"max_seq_len={max_seq_len} must be >= {2 * resolved_tile}")

        self.config = config
        self.trunk = trunk
        self.expert_store = expert_store
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.tile_size = resolved_tile
        self.dtype = trunk.embed.dtype
        self.weight_quant = trunk.weight_quant
        self.host_store_mode = expert_store.host_store_mode

        self.stage_capacity = min(config.num_experts, max(1, int(stage_experts)))
        self.vanilla_stage_capacity = min(self.stage_capacity, config.num_experts_per_tok)

        self.trunk_hbm_bytes = trunk.trunk_bytes()
        self.staging_hbm_bytes = expert_store.staging_buffer_bytes(self.stage_capacity)
        self.total_expert_source_bytes = expert_store.total_source_bytes()

        self.vanilla_prefill_compile_count = 0
        self.vanilla_decode_compile_count = 0
        self.static_attn_router_compile_count = 0
        self.static_expert_stage_compile_count = 0
        self.static_dense_layer_compile_count = 0
        self.static_final_compile_count = 0
        self.static_commit_kv_compile_count = 0

        self._compile_kernels()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close backing mmap handles held by the host expert store."""
        self.expert_store.close()

    def static_compile_count(self) -> int:
        """Return cumulative XLA compilations across static tiled MoE execution kernels."""
        return (
            self.static_attn_router_compile_count
            + self.static_expert_stage_compile_count
            + self.static_dense_layer_compile_count
            + self.static_final_compile_count
            + self.static_commit_kv_compile_count
        )

    def _empty_kv_caches(self) -> tuple[list[jax.Array], list[jax.Array]]:
        shape = (self.max_seq_len + 1, self.config.num_key_value_heads, self.config.head_dim)
        ks = [jnp.zeros(shape, dtype=self.dtype) for _ in range(self.config.num_hidden_layers)]
        vs = [jnp.zeros(shape, dtype=self.dtype) for _ in range(self.config.num_hidden_layers)]
        return ks, vs

    def _compile_kernels(self) -> None:
        cfg = self.config
        max_seq = self.max_seq_len
        tile_size = self.tile_size
        compute_dtype = self.dtype
        norm_top_k = cfg.norm_topk_prob
        top_k = cfg.num_experts_per_tok

        def _static_tile_attn_and_norms(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            valid_prefix_len: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            prefix_allowed = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) < valid_prefix_len,
                0.0,
                _MASK_BLOCK_VAL,
            ).astype(jnp.float32)
            prefix_mask = jnp.broadcast_to(prefix_allowed[None, :], (tile_size, max_seq))
            full_mask = jnp.concatenate([prefix_mask, tile_mask], axis=-1)
            return _layer_attn_and_norms(
                x, lw, pos_tile, full_mask, k_cache_l, v_cache_l, max_seq, cfg
            )

        def _expert_stage_impl(
            acc: jax.Array,
            h_mlp: jax.Array,
            wg_e: jax.Array,
            wu_e: jax.Array,
            wd_e: jax.Array,
            sg_e: jax.Array | None,
            su_e: jax.Array | None,
            sd_e: jax.Array | None,
            slot_idx: jax.Array,
            slot_weights: jax.Array,
        ) -> jax.Array:
            seq_len = h_mlp.shape[0]
            e_stage = wg_e.shape[0]
            x_routed = jnp.broadcast_to(h_mlp[None, :, :], (e_stage, seq_len, cfg.hidden_size))
            gate = batched_expert_systolic_linear(x_routed, wg_e, sg_e, out_dtype=compute_dtype)
            up = batched_expert_systolic_linear(x_routed, wu_e, su_e, out_dtype=compute_dtype)
            act = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(
                compute_dtype
            )
            expert_out = batched_expert_systolic_linear(act, wd_e, sd_e, out_dtype=jnp.float32)
            matches = jax.nn.one_hot(slot_idx, e_stage, dtype=jnp.float32)
            coeff = jnp.einsum("ske,sk->es", matches, slot_weights)
            routed_sum = jnp.einsum("es,esh->sh", coeff, expert_out)
            return (acc.astype(jnp.float32) + routed_sum).astype(acc.dtype)

        def _vanilla_decode_attn_and_norms(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            cur_pos: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            allowed = jnp.where(
                jnp.arange(max_seq, dtype=jnp.int32) <= cur_pos, 0.0, _MASK_BLOCK_VAL
            ).astype(jnp.float32)[None, :]
            positions = jnp.array([cur_pos], dtype=jnp.int32)
            h = _rms_norm(x, lw.in_norm, cfg.rms_norm_eps)
            q_proj = systolic_linear(h, lw.wq, lw.bq, w_scale=lw.wq_scale)
            q = _apply_qwen_rope(
                _apply_qk_norm(
                    q_proj, lw.q_norm, 1, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps
                ),
                positions,
                cfg.head_dim,
                cfg.rope_theta,
            )
            k_proj = systolic_linear(h, lw.wk, lw.bk, w_scale=lw.wk_scale)
            k_new = _apply_qwen_rope(
                _apply_qk_norm(
                    k_proj,
                    lw.k_norm,
                    1,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    cfg.rms_norm_eps,
                ),
                positions,
                cfg.head_dim,
                cfg.rope_theta,
            )
            v_new = systolic_linear(h, lw.wv, lw.bv, w_scale=lw.wv_scale).reshape(
                1, cfg.num_key_value_heads, cfg.head_dim
            )
            k_upd = k_cache_l.at[cur_pos].set(k_new[0])
            v_upd = v_cache_l.at[cur_pos].set(v_new[0])
            attn_out = _gqa_attention(
                q,
                k_upd[:max_seq],
                v_upd[:max_seq],
                allowed,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
            )
            x_post = x + systolic_linear(attn_out, lw.wo, w_scale=lw.wo_scale)
            h_mlp = _rms_norm(x_post, lw.post_norm, cfg.rms_norm_eps)
            return x_post, h_mlp, k_upd, v_upd

        @jax.jit
        def static_attn_router_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            valid_prefix_len: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            self.static_attn_router_compile_count += 1
            x_post, h_mlp, k_step, v_step = _static_tile_attn_and_norms(
                x, lw, k_cache_l, v_cache_l, valid_prefix_len, pos_tile, tile_mask
            )
            assert lw.router_w is not None
            top_idx, top_vals = _route_tokens(h_mlp, lw.router_w, top_k, norm_top_k)
            x_base = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_base, h_mlp, top_idx, top_vals, k_step, v_step

        @jax.jit
        def static_dense_layer_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            valid_prefix_len: jax.Array,
            pos_tile: jax.Array,
            tile_mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.static_dense_layer_compile_count += 1
            x_post, h_mlp, k_step, v_step = _static_tile_attn_and_norms(
                x, lw, k_cache_l, v_cache_l, valid_prefix_len, pos_tile, tile_mask
            )
            x_next = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_next, k_step, v_step

        @jax.jit
        def static_expert_stage_jit(
            acc: jax.Array,
            h_mlp: jax.Array,
            wg_e: jax.Array,
            wu_e: jax.Array,
            wd_e: jax.Array,
            sg_e: jax.Array | None,
            su_e: jax.Array | None,
            sd_e: jax.Array | None,
            slot_idx: jax.Array,
            slot_weights: jax.Array,
        ) -> jax.Array:
            self.static_expert_stage_compile_count += 1
            return _expert_stage_impl(
                acc, h_mlp, wg_e, wu_e, wd_e, sg_e, su_e, sd_e, slot_idx, slot_weights
            )

        @jax.jit
        def expert_stage_dynamic_jit(
            acc: jax.Array,
            h_mlp: jax.Array,
            wg_e: jax.Array,
            wu_e: jax.Array,
            wd_e: jax.Array,
            sg_e: jax.Array | None,
            su_e: jax.Array | None,
            sd_e: jax.Array | None,
            slot_idx: jax.Array,
            slot_weights: jax.Array,
        ) -> jax.Array:
            return _expert_stage_impl(
                acc, h_mlp, wg_e, wu_e, wd_e, sg_e, su_e, sd_e, slot_idx, slot_weights
            )

        @jax.jit
        def static_final_argmax_jit(
            x_final: jax.Array,
            final_norm: jax.Array,
            lm_head: jax.Array,
            lm_head_scale: jax.Array | None,
        ) -> jax.Array:
            self.static_final_compile_count += 1
            normed = _rms_norm(x_final, final_norm, cfg.rms_norm_eps)
            logits = systolic_linear(
                normed, lm_head, w_scale=lm_head_scale, out_dtype=jnp.float32
            )
            has_nan = jnp.any(jnp.isnan(logits), axis=-1)
            return jnp.where(
                has_nan,
                jnp.int32(-1),
                jnp.argmax(logits, axis=-1).astype(jnp.int32),
            )

        @jax.jit
        def static_commit_all_kv_jit(
            k_caches: tuple[jax.Array, ...],
            v_caches: tuple[jax.Array, ...],
            step_ks: tuple[jax.Array, ...],
            step_vs: tuple[jax.Array, ...],
            prev_prefix_len: jax.Array,
            src_rows: jax.Array,
            n_valid: jax.Array,
        ) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
            self.static_commit_kv_compile_count += 1
            idx = jnp.arange(tile_size, dtype=jnp.int32)
            dest_slots = jnp.where(idx < n_valid, prev_prefix_len + idx, max_seq)
            k_next = tuple(
                kc.at[dest_slots].set(sk[src_rows]) for kc, sk in zip(k_caches, step_ks, strict=True)
            )
            v_next = tuple(
                vc.at[dest_slots].set(sv[src_rows]) for vc, sv in zip(v_caches, step_vs, strict=True)
            )
            return k_next, v_next

        @jax.jit
        def vanilla_prefill_attn_router_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            positions: jax.Array,
            causal_mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            self.vanilla_prefill_compile_count += 1
            empty_kv = jnp.zeros((0, cfg.num_key_value_heads, cfg.head_dim), dtype=compute_dtype)
            x_post, h_mlp, k_new, v_new = _layer_attn_and_norms(
                x, lw, positions, causal_mask, empty_kv, empty_kv, None, cfg
            )
            assert lw.router_w is not None
            top_idx, top_vals = _route_tokens(h_mlp, lw.router_w, top_k, norm_top_k)
            x_base = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_base, h_mlp, top_idx, top_vals, k_new, v_new

        @jax.jit
        def vanilla_prefill_dense_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            positions: jax.Array,
            causal_mask: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.vanilla_prefill_compile_count += 1
            empty_kv = jnp.zeros((0, cfg.num_key_value_heads, cfg.head_dim), dtype=compute_dtype)
            x_post, h_mlp, k_new, v_new = _layer_attn_and_norms(
                x, lw, positions, causal_mask, empty_kv, empty_kv, None, cfg
            )
            x_next = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_next, k_new, v_new

        @jax.jit
        def vanilla_decode_attn_router_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            cur_pos: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            self.vanilla_decode_compile_count += 1
            x_post, h_mlp, k_upd, v_upd = _vanilla_decode_attn_and_norms(
                x, lw, k_cache_l, v_cache_l, cur_pos
            )
            assert lw.router_w is not None
            top_idx, top_vals = _route_tokens(h_mlp, lw.router_w, top_k, norm_top_k)
            x_base = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_base, h_mlp, top_idx, top_vals, k_upd, v_upd

        @jax.jit
        def vanilla_decode_dense_jit(
            x: jax.Array,
            lw: MoeLayerTrunkWeights,
            k_cache_l: jax.Array,
            v_cache_l: jax.Array,
            cur_pos: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            self.vanilla_decode_compile_count += 1
            x_post, h_mlp, k_upd, v_upd = _vanilla_decode_attn_and_norms(
                x, lw, k_cache_l, v_cache_l, cur_pos
            )
            x_next = _add_shared_expert_residual(x_post, h_mlp, lw)
            return x_next, k_upd, v_upd

        @jax.jit
        def final_head_logits_jit(
            last_x: jax.Array,
            final_norm: jax.Array,
            lm_head: jax.Array,
            lm_head_scale: jax.Array | None,
        ) -> jax.Array:
            normed = _rms_norm(last_x, final_norm, cfg.rms_norm_eps)
            return systolic_linear(normed, lm_head, w_scale=lm_head_scale, out_dtype=jnp.float32)

        self._static_attn_router_jit = static_attn_router_jit
        self._static_dense_layer_jit = static_dense_layer_jit
        self._static_expert_stage_jit = static_expert_stage_jit
        self._expert_stage_dynamic_jit = expert_stage_dynamic_jit
        self._static_final_argmax_jit = static_final_argmax_jit
        self._static_commit_all_kv_jit = static_commit_all_kv_jit
        self._vanilla_prefill_attn_router_jit = vanilla_prefill_attn_router_jit
        self._vanilla_prefill_dense_jit = vanilla_prefill_dense_jit
        self._vanilla_decode_attn_router_jit = vanilla_decode_attn_router_jit
        self._vanilla_decode_dense_jit = vanilla_decode_dense_jit
        self._final_head_logits_jit = final_head_logits_jit

    def _new_transfer_stats(self) -> dict[str, int]:
        return {
            "routed_expert_loads": 0,
            "h2d_chunk_transfers": 0,
            "decode_routed_expert_loads": 0,
            "decode_h2d_chunk_transfers": 0,
        }

    def _accumulate_staged_chunks(
        self,
        acc: jax.Array,
        h_mlp: jax.Array,
        layer_idx: int,
        top_idx_np: np.ndarray,
        top_vals_np: np.ndarray,
        valid_row_mask: np.ndarray,
        stage_capacity: int,
        stage_jit: Any,
        stats: dict[str, int],
        count_as_decode: bool,
    ) -> jax.Array:
        for (
            chunk_len,
            wg_np,
            wu_np,
            wd_np,
            sg_np,
            su_np,
            sd_np,
            slot_idx_np,
            slot_weights_np,
        ) in self.expert_store.iter_staged_chunks(
            layer_idx, top_idx_np, top_vals_np, valid_row_mask, stage_capacity
        ):
            stats["routed_expert_loads"] += chunk_len
            stats["h2d_chunk_transfers"] += 1
            if count_as_decode:
                stats["decode_routed_expert_loads"] += chunk_len
                stats["decode_h2d_chunk_transfers"] += 1

            acc = stage_jit(
                acc,
                h_mlp,
                jax.device_put(wg_np),
                jax.device_put(wu_np),
                jax.device_put(wd_np),
                jax.device_put(sg_np) if sg_np is not None else None,
                jax.device_put(su_np) if su_np is not None else None,
                jax.device_put(sd_np) if sd_np is not None else None,
                jax.device_put(slot_idx_np),
                jax.device_put(slot_weights_np),
            )
        return acc

    def _run_static_tile_step(
        self,
        valid_prefix_len: int,
        tok_tile: np.ndarray,
        pos_tile: np.ndarray,
        mask_tile: np.ndarray,
        k_caches: list[jax.Array],
        v_caches: list[jax.Array],
        stats: dict[str, int],
        count_as_decode: bool,
    ) -> tuple[jax.Array, tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        valid_prefix_jnp = jnp.asarray(valid_prefix_len, dtype=jnp.int32)
        tok_jnp = jnp.asarray(tok_tile, dtype=jnp.int32)
        pos_jnp = jnp.asarray(pos_tile, dtype=jnp.int32)
        mask_jnp = jnp.asarray(mask_tile, dtype=jnp.float32)
        valid_row_mask = np.asarray(mask_tile[:, 0] > -1.0, dtype=bool)

        x = self.trunk.embed[tok_jnp]
        step_ks: list[jax.Array] = []
        step_vs: list[jax.Array] = []

        for l_idx, lw in enumerate(self.trunk.layers):
            if lw.router_w is None:
                x, k_step, v_step = self._static_dense_layer_jit(
                    x, lw, k_caches[l_idx], v_caches[l_idx], valid_prefix_jnp, pos_jnp, mask_jnp
                )
            else:
                x_base, h_mlp, top_idx, top_vals, k_step, v_step = self._static_attn_router_jit(
                    x, lw, k_caches[l_idx], v_caches[l_idx], valid_prefix_jnp, pos_jnp, mask_jnp
                )
                top_idx_np, top_vals_np = jax.device_get((top_idx, top_vals))
                x = self._accumulate_staged_chunks(
                    x_base,
                    h_mlp,
                    l_idx,
                    np.asarray(top_idx_np, dtype=np.int32),
                    np.asarray(top_vals_np, dtype=np.float32),
                    valid_row_mask,
                    self.stage_capacity,
                    self._static_expert_stage_jit,
                    stats,
                    count_as_decode=count_as_decode,
                )
            step_ks.append(k_step)
            step_vs.append(v_step)

        argmax_tokens = self._static_final_argmax_jit(
            x, self.trunk.final_norm, self.trunk.lm_head, self.trunk.lm_head_scale
        )
        return argmax_tokens, tuple(step_ks), tuple(step_vs)

    def warmup(self) -> float:
        """Warm up static MoE XLA kernels once so steady-state prompt sweeps have 0 compiles."""
        t0 = time.perf_counter()
        T = self.tile_size
        k_caches, v_caches = self._empty_kv_caches()
        tok_tile = np.zeros((T,), dtype=np.int32)
        pos_tile = np.zeros((T,), dtype=np.int32)
        mask_tile = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
        mask_tile[0, 0] = 0.0
        for r in range(1, T):
            mask_tile[r, r] = 0.0

        argmax_tokens, step_ks, step_vs = self._run_static_tile_step(
            0, tok_tile, pos_tile, mask_tile, k_caches, v_caches, self._new_transfer_stats(), False
        )
        src_rows = np.zeros((T,), dtype=np.int32)
        k_next, v_next = self._static_commit_all_kv_jit(
            tuple(k_caches),
            tuple(v_caches),
            step_ks,
            step_vs,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(src_rows, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )
        jax.block_until_ready((argmax_tokens, k_next[0], v_next[0]))
        return time.perf_counter() - t0

    def compute_prefill_last_logits(self, prompt_tokens: list[int]) -> np.ndarray:
        """Compute full-vocab float32 logits for prompt_tokens[-1] via host-streamed MoE experts."""
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        seq_p = len(prompt_tokens)
        positions = jnp.arange(seq_p, dtype=jnp.int32)
        causal = jnp.where(
            jnp.arange(seq_p)[:, None] >= jnp.arange(seq_p)[None, :],
            0.0,
            _MASK_BLOCK_VAL,
        ).astype(jnp.float32)
        valid_rows = np.ones((seq_p,), dtype=bool)
        x = self.trunk.embed[jnp.asarray(prompt_tokens, dtype=jnp.int32)]
        stats = self._new_transfer_stats()
        for l_idx, lw in enumerate(self.trunk.layers):
            if lw.router_w is None:
                x, _, _ = self._vanilla_prefill_dense_jit(x, lw, positions, causal)
            else:
                x_base, h_mlp, top_idx, top_vals, _, _ = self._vanilla_prefill_attn_router_jit(
                    x, lw, positions, causal
                )
                top_idx_np, top_vals_np = jax.device_get((top_idx, top_vals))
                x = self._accumulate_staged_chunks(
                    x_base,
                    h_mlp,
                    l_idx,
                    np.asarray(top_idx_np, dtype=np.int32),
                    np.asarray(top_vals_np, dtype=np.float32),
                    valid_rows,
                    self.stage_capacity,
                    self._expert_stage_dynamic_jit,
                    stats,
                    count_as_decode=False,
                )
        logits = self._final_head_logits_jit(
            x[-1:], self.trunk.final_norm, self.trunk.lm_head, self.trunk.lm_head_scale
        )[0]
        return np.asarray(jax.device_get(logits), dtype=np.float32)

    def _validate_bounds(self, prompt_tokens: list[int], max_new_tokens: int) -> None:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if len(prompt_tokens) + max_new_tokens > self.max_seq_len - self.tile_size:
            raise ValueError(
                f"prompt({len(prompt_tokens)}) + max_new({max_new_tokens}) exceeds "
                f"max_seq_len - tile_size ({self.max_seq_len - self.tile_size})"
            )

    def _decode_text(self, token_ids: list[int]) -> str:
        if self.tokenizer is None:
            return " ".join(str(t) for t in token_ids)
        return str(self.tokenizer.decode(token_ids))

    def _extra_metrics(self, stats: dict[str, int], gen_len: int) -> dict[str, Any]:
        single_b = self.expert_store.single_expert_bytes()
        dec_loads = stats["decode_routed_expert_loads"]
        dec_bytes = dec_loads * single_b
        return {
            "tile_size": self.tile_size,
            "weight_quant": self.weight_quant,
            "host_store_mode": self.host_store_mode,
            "stage_capacity": self.stage_capacity,
            "trunk_hbm_bytes": self.trunk_hbm_bytes,
            "staging_hbm_bytes": self.staging_hbm_bytes,
            "total_expert_source_bytes": self.total_expert_source_bytes,
            "routed_expert_loads": stats["routed_expert_loads"],
            "h2d_chunk_transfers": stats["h2d_chunk_transfers"],
            "routed_expert_bytes_h2d": stats["routed_expert_loads"] * single_b,
            "decode_routed_expert_loads": dec_loads,
            "decode_h2d_chunk_transfers": stats["decode_h2d_chunk_transfers"],
            "decode_routed_expert_bytes": dec_bytes,
            "decode_experts_per_token": (dec_loads / gen_len) if gen_len > 0 else 0.0,
            "decode_h2d_bytes_per_token": (dec_bytes / gen_len) if gen_len > 0 else 0.0,
        }

    def run_vanilla_moe(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        """Baseline: dynamic-shape prefill + 1-token serial decode fetching K experts per step."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.vanilla_prefill_compile_count + self.vanilla_decode_compile_count
        stats = self._new_transfer_stats()
        k_caches, v_caches = self._empty_kv_caches()
        jax.block_until_ready((k_caches[0], v_caches[0]))

        seq_p = len(prompt_tokens)
        positions = jnp.arange(seq_p, dtype=jnp.int32)
        causal = jnp.where(
            jnp.arange(seq_p)[:, None] >= jnp.arange(seq_p)[None, :],
            0.0,
            _MASK_BLOCK_VAL,
        ).astype(jnp.float32)
        valid_prefill = np.ones((seq_p,), dtype=bool)
        valid_dec = np.ones((1,), dtype=bool)

        t0 = time.perf_counter()
        x = self.trunk.embed[jnp.asarray(prompt_tokens, dtype=jnp.int32)]
        for l_idx, lw in enumerate(self.trunk.layers):
            if lw.router_w is None:
                x, k_new, v_new = self._vanilla_prefill_dense_jit(x, lw, positions, causal)
            else:
                x_base, h_mlp, top_idx, top_vals, k_new, v_new = self._vanilla_prefill_attn_router_jit(
                    x, lw, positions, causal
                )
                top_idx_np, top_vals_np = jax.device_get((top_idx, top_vals))
                x = self._accumulate_staged_chunks(
                    x_base,
                    h_mlp,
                    l_idx,
                    np.asarray(top_idx_np, dtype=np.int32),
                    np.asarray(top_vals_np, dtype=np.float32),
                    valid_prefill,
                    self.stage_capacity,
                    self._expert_stage_dynamic_jit,
                    stats,
                    count_as_decode=False,
                )
            k_caches[l_idx] = k_caches[l_idx].at[:seq_p].set(k_new)
            v_caches[l_idx] = v_caches[l_idx].at[:seq_p].set(v_new)

        logits_last = self._final_head_logits_jit(
            x[-1:], self.trunk.final_norm, self.trunk.lm_head, self.trunk.lm_head_scale
        )[0]
        next_tok = int(np.asarray(jax.device_get(jnp.argmax(logits_last))))
        t1 = time.perf_counter()

        generated = [next_tok]
        decode_passes = 0
        t_dec0 = time.perf_counter()
        cur_pos = seq_p
        while len(generated) < max_new_tokens and generated[-1] not in eos_set:
            pos_jnp = jnp.asarray(cur_pos, dtype=jnp.int32)
            x_dec = self.trunk.embed[jnp.asarray([generated[-1]], dtype=jnp.int32)]
            for l_idx, lw in enumerate(self.trunk.layers):
                if lw.router_w is None:
                    x_dec, kc_u, vc_u = self._vanilla_decode_dense_jit(
                        x_dec, lw, k_caches[l_idx], v_caches[l_idx], pos_jnp
                    )
                else:
                    x_base, h_mlp, top_idx, top_vals, kc_u, vc_u = (
                        self._vanilla_decode_attn_router_jit(
                            x_dec, lw, k_caches[l_idx], v_caches[l_idx], pos_jnp
                        )
                    )
                    top_idx_np, top_vals_np = jax.device_get((top_idx, top_vals))
                    x_dec = self._accumulate_staged_chunks(
                        x_base,
                        h_mlp,
                        l_idx,
                        np.asarray(top_idx_np, dtype=np.int32),
                        np.asarray(top_vals_np, dtype=np.float32),
                        valid_dec,
                        self.vanilla_stage_capacity,
                        self._expert_stage_dynamic_jit,
                        stats,
                        count_as_decode=True,
                    )
                k_caches[l_idx] = kc_u
                v_caches[l_idx] = vc_u
            logits_step = self._final_head_logits_jit(
                x_dec, self.trunk.final_norm, self.trunk.lm_head, self.trunk.lm_head_scale
            )[0]
            next_tok = int(np.asarray(jax.device_get(jnp.argmax(logits_step))))
            generated.append(next_tok)
            cur_pos += 1
            decode_passes += 1
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
            extra=self._extra_metrics(stats, gen_len),
        )

    def run_static_arena_moe(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
    ) -> RunMetrics:
        """Lever 1 (M-002) + M-008: fixed-tile zero-recompilation host-streamed MoE decode."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_set = set(eos_token_ids) if eos_token_ids is not None else set(self.config.eos_token_ids)
        compiles_before = self.static_compile_count()
        stats = self._new_transfer_stats()

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

            tok_np[:chunk] = prompt_tokens[offset : offset + chunk]
            pos_np[:chunk] = np.arange(offset, offset + chunk, dtype=np.int32)
            for r in range(chunk):
                mask_np[r, : r + 1] = 0.0
            for r in range(chunk, T):
                mask_np[r, r] = 0.0

            _, step_ks, step_vs = self._run_static_tile_step(
                offset, tok_np, pos_np, mask_np, k_caches, v_caches, stats, count_as_decode=False
            )
            src_rows = np.zeros((T,), dtype=np.int32)
            src_rows[:chunk] = np.arange(chunk, dtype=np.int32)
            k_tup, v_tup = self._static_commit_all_kv_jit(
                tuple(k_caches),
                tuple(v_caches),
                step_ks,
                step_vs,
                jnp.asarray(offset, dtype=jnp.int32),
                jnp.asarray(src_rows, dtype=jnp.int32),
                jnp.asarray(chunk, dtype=jnp.int32),
            )
            k_caches, v_caches = list(k_tup), list(v_tup)
            forward_passes += 1
        jax.block_until_ready((k_caches[0], v_caches[0]))
        t1 = time.perf_counter()

        decode_mask_np = np.full((T, T), _MASK_BLOCK_VAL, dtype=np.float32)
        np.fill_diagonal(decode_mask_np, 0.0)
        src_row_0 = np.zeros((T,), dtype=np.int32)
        src_row_0_jnp = jnp.asarray(src_row_0, dtype=jnp.int32)
        one_jnp = jnp.asarray(1, dtype=jnp.int32)

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

            argmax_dev, step_ks, step_vs = self._run_static_tile_step(
                cur_prefix,
                tok_np,
                pos_np,
                decode_mask_np,
                k_caches,
                v_caches,
                stats,
                count_as_decode=True,
            )
            k_tup, v_tup = self._static_commit_all_kv_jit(
                tuple(k_caches),
                tuple(v_caches),
                step_ks,
                step_vs,
                jnp.asarray(cur_prefix, dtype=jnp.int32),
                src_row_0_jnp,
                one_jnp,
            )
            k_caches, v_caches = list(k_tup), list(v_tup)
            next_tok = int(np.asarray(jax.device_get(argmax_dev))[0])
            generated.append(next_tok)
            cur_tok = next_tok
            cur_prefix += 1
            decode_passes += 1
            forward_passes += 1
            if next_tok in eos_set:
                break
        jax.block_until_ready((k_caches[0], v_caches[0]))
        t_dec1 = time.perf_counter()

        compiles_delta = self.static_compile_count() - compiles_before
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

    def run_tpuforge_tree_moe(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        eos_token_ids: Iterable[int] | None = None,
        seed_token_sequences: Sequence[Sequence[int]] | None = None,
    ) -> RunMetrics:
        """M-002 + M-003 + M-008: amortize host expert H2D transfers across 128/256 tree nodes."""
        self._validate_bounds(prompt_tokens, max_new_tokens)
        T = self.tile_size
        eos_list = (
            [int(x) for x in eos_token_ids]
            if eos_token_ids is not None
            else list(self.config.eos_token_ids)
        )
        compiles_before = self.static_compile_count()
        stats = self._new_transfer_stats()
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

        def _dispatch_tile(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
            count_as_decode: bool,
        ) -> jax.Array:
            argmax_tokens, step_ks, step_vs = self._run_static_tile_step(
                valid_prefix_len,
                tok_tile,
                pos_tile,
                mask_tile,
                state["k_caches"],
                state["v_caches"],
                stats,
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
            _dispatch_tile(valid_prefix_len, tok_tile, pos_tile, mask_tile, count_as_decode=False)

        def greedy_step_fn(
            valid_prefix_len: int,
            tok_tile: np.ndarray,
            pos_tile: np.ndarray,
            mask_tile: np.ndarray,
        ) -> np.ndarray:
            argmax_tokens = _dispatch_tile(
                valid_prefix_len, tok_tile, pos_tile, mask_tile, count_as_decode=True
            )
            return np.asarray(jax.device_get(argmax_tokens), dtype=np.int32)

        def commit_kv_fn(prev_prefix_len: int, accepted_row_indices: list[int]) -> None:
            n_valid = len(accepted_row_indices)
            src_rows = np.zeros((T,), dtype=np.int32)
            if n_valid > 0:
                src_rows[:n_valid] = np.asarray(accepted_row_indices, dtype=np.int32)
            k_tup, v_tup = self._static_commit_all_kv_jit(
                tuple(state["k_caches"]),
                tuple(state["v_caches"]),
                state["step_ks"],
                state["step_vs"],
                jnp.asarray(prev_prefix_len, dtype=jnp.int32),
                jnp.asarray(src_rows, dtype=jnp.int32),
                jnp.asarray(n_valid, dtype=jnp.int32),
            )
            state["k_caches"] = list(k_tup)
            state["v_caches"] = list(v_tup)
            if state["step_idx"] == prefill_tiles and state["t_prefill_done"] is None:
                jax.block_until_ready((k_tup[0], v_tup[0]))
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
            f"tpuforge-moe-{T}",
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

        compiles_delta = self.static_compile_count() - compiles_before
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


def load_moe_streaming_runner(
    model_id_or_path: str,
    max_seq_len: int = 1024,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    host_store_mode: str = "mmap",
    stage_experts: int = 16,
    weight_quant: str = "none",
    lru_cache_experts: int = 64,
    enable_xla_cache: bool = True,
) -> tuple[MoeStreamingRunner, Path]:
    """Load an HF MoE checkpoint trunk to HBM and stream routed experts from mmap or DRAM."""
    from tokenizers import Tokenizer

    if enable_xla_cache:
        configure_xla_compilation_cache()

    model_dir = resolve_model_dir(model_id_or_path)
    config = MoeConfig.from_dir(model_dir)
    tok_path = model_dir / "tokenizer.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer.json in {model_dir}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    dtype = resolve_default_dtype(dtype_name)
    norm_quant = normalize_weight_quant(weight_quant)
    norm_store = normalize_host_store_mode(host_store_mode)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_trunk = pool.submit(
            MoeTrunkWeights.from_model_dir, model_dir, config, dtype, norm_quant
        )
        f_store = pool.submit(
            ExpertHostStore,
            model_dir,
            config,
            dtype,
            norm_quant,
            norm_store,
            lru_cache_experts,
        )
        trunk = f_trunk.result()
        store = f_store.result()

    runner = MoeStreamingRunner(
        config=config,
        trunk=trunk,
        expert_store=store,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        tile_size=tile_size,
        stage_experts=stage_experts,
    )
    return runner, model_dir


def run_moe_benchmark_suite(
    model_id: str,
    max_seq_len: int = 1024,
    max_new_tokens: int = 32,
    dtype_name: str = "auto",
    tile_size: int | str = "auto",
    host_store_mode: str = "mmap",
    stage_experts: int = 16,
    weight_quant: str = "none",
    lru_cache_experts: int = 64,
    prompt_tokens: list[int] | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Run Vanilla vs StaticArena vs TPUForge Tree benchmark with host-streamed MoE experts."""
    t_load0 = time.perf_counter()
    runner, model_dir = load_moe_streaming_runner(
        model_id_or_path=model_id,
        max_seq_len=max_seq_len,
        dtype_name=dtype_name,
        tile_size=tile_size,
        host_store_mode=host_store_mode,
        stage_experts=stage_experts,
        weight_quant=weight_quant,
        lru_cache_experts=lru_cache_experts,
    )
    load_s = time.perf_counter() - t_load0
    try:
        warmup_s = runner.warmup()
        if prompt_tokens is None:
            if runner.tokenizer is not None:
                sample = (
                    "def fibonacci(n):\n    if n <= 1:\n        return n\n"
                    "    return fibonacci(n - 1) + fibonacci(n - 2)\n"
                ) * 4
                enc = runner.tokenizer.encode(sample, add_special_tokens=False)
                prompt = [int(t) for t in enc.ids]
            else:
                prompt = [10, 11, 12, 13] * 8 + [10]
        else:
            prompt = list(prompt_tokens)

        m_vanilla = runner.run_vanilla_moe(prompt, max_new_tokens=max_new_tokens)
        m_static = runner.run_static_arena_moe(prompt, max_new_tokens=max_new_tokens)
        m_tree = runner.run_tpuforge_tree_moe(prompt, max_new_tokens=max_new_tokens)

        report: dict[str, Any] = {
            "model_id": model_id,
            "model_dir": str(model_dir),
            "model_type": runner.config.model_type,
            "backend": jax.default_backend(),
            "tile_size": runner.tile_size,
            "dtype": str(runner.dtype),
            "weight_quant": runner.weight_quant,
            "host_store_mode": runner.host_store_mode,
            "stage_capacity": runner.stage_capacity,
            "trunk_hbm_bytes": runner.trunk_hbm_bytes,
            "staging_hbm_bytes": runner.staging_hbm_bytes,
            "total_expert_source_bytes": runner.total_expert_source_bytes,
            "load_seconds": round(load_s, 3),
            "warmup_seconds": round(warmup_s, 3),
            "exact_parity": m_vanilla.token_ids == m_static.token_ids == m_tree.token_ids,
            "vanilla_moe": {
                "token_ids": m_vanilla.token_ids,
                "decode_tok_per_s": m_vanilla.decode_tok_per_s,
                "mean_tokens_per_decode_pass": m_vanilla.mean_tokens_per_decode_pass,
                "compile_count": m_vanilla.compile_count,
                "extra": m_vanilla.extra,
            },
            "static_arena_moe": {
                "token_ids": m_static.token_ids,
                "decode_tok_per_s": m_static.decode_tok_per_s,
                "mean_tokens_per_decode_pass": m_static.mean_tokens_per_decode_pass,
                "compile_count": m_static.compile_count,
                "extra": m_static.extra,
            },
            "tpuforge_tree_moe": {
                "token_ids": m_tree.token_ids,
                "decode_tok_per_s": m_tree.decode_tok_per_s,
                "mean_tokens_per_decode_pass": m_tree.mean_tokens_per_decode_pass,
                "compile_count": m_tree.compile_count,
                "extra": m_tree.extra,
            },
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        runner.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run host-DRAM / mmap sparse MoE expert streaming TPUForge benchmark."
    )
    parser.add_argument("--model", required=True, help="HF repo ID or local MoE directory.")
    parser.add_argument("--host-store", default="mmap", choices=list(VALID_HOST_STORE_MODES))
    parser.add_argument("--stage-experts", type=int, default=16)
    parser.add_argument("--weight-quant", default="none", choices=["none", "int8", "fp8"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--tile-size", default="auto", choices=["auto", "128", "256"])
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-json", default="benchmarks/results/latest_moe_bench.json")
    args = parser.parse_args()

    out_path = Path(args.output_json) if args.output_json else None
    report = run_moe_benchmark_suite(
        model_id=args.model,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        dtype_name=args.dtype,
        tile_size=args.tile_size,
        host_store_mode=args.host_store,
        stage_experts=args.stage_experts,
        weight_quant=args.weight_quant,
        output_json=out_path,
    )
    print(
        f"TPUForge MoE ({report['model_type']}, {report['host_store_mode']}, "
        f"quant={report['weight_quant']}, E_stage={report['stage_capacity']}): "
        f"exact_parity={report['exact_parity']} | "
        f"tree_tok_per_pass={report['tpuforge_tree_moe']['mean_tokens_per_decode_pass']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

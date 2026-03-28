from __future__ import annotations

import json
import logging
import math
import queue
import random
import threading
import time
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .analysis import summarize_bundles
from .capture_io import build_batched_capture_payload, split_pass_capture, trim_pass_capture
from .config import ExperimentSpec
from .prepend import (
    apply_rotary_pos_emb,
    attention_forward,
    configure_attention_runtime,
    make_prepend_summary_kv,
    matched_norm_random_like,
)
from .prompts import PromptExample, build_prompt_examples, calibration_manifest, prompt_affixes, sample_calibration_ids
from .runtime import import_torch, load_model_and_tokenizer, runtime_stack_snapshot, verify_model_contract
from .types import CaptureCondition, ExampleCaptureBundle, LayerCapture, PassCapture, RopeMode

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _cached_base_causal_mask(seq_len: int, dtype_name: str, device_type: str, device_index: int | None):
    torch = import_torch()
    dtype = getattr(torch, dtype_name.split(".")[-1], None)
    if dtype is None:
        dtype = getattr(torch, dtype_name.replace("torch.", ""))
    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), neg_inf, device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def _storage_dtype(role: str, calibration: bool):
    torch = import_torch()
    if role == "top_k_indices":
        return torch.int8
    if role in {"beta", "attention_weights"}:
        return torch.bfloat16
    if calibration:
        return torch.bfloat16
    return getattr(torch, "float8_e4m3fn", torch.bfloat16)


def _to_cpu(tensor, dtype=None):
    if tensor is None:
        return None
    torch = import_torch()
    if isinstance(tensor, torch.Tensor):
        out = tensor.detach().cpu()
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out
    return tensor


def _storage_metadata(tensor, storage_dtype) -> dict[str, Any]:
    if tensor is None:
        return {}
    torch = import_torch()
    if isinstance(tensor, torch.Tensor):
        return {
            "source_dtype": str(tensor.dtype).replace("torch.", ""),
            "storage_dtype": str(storage_dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }
    return {"storage_dtype": str(storage_dtype)}


def _causal_attention_mask(attention_mask, batch_size: int, seq_len: int, device, dtype):
    torch = import_torch()
    base = _cached_base_causal_mask(seq_len, str(dtype), device.type, device.index)
    mask = base.expand(batch_size, -1, -1, -1).clone()
    if attention_mask is not None:
        neg_inf = torch.finfo(dtype).min
        attention_mask = attention_mask.to(device=device)
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * neg_inf
        mask = mask + pad
    return mask


def _last_token_positions(attention_mask):
    torch = import_torch()
    token_counts = attention_mask.sum(dim=1).to(dtype=torch.long)
    return torch.clamp(token_counts - 1, min=0)


def _gather_last_hidden(hidden_states, last_positions):
    torch = import_torch()
    batch_index = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_index, last_positions, :]


def _gather_last_kv(seq_tensor, last_positions):
    index = last_positions.view(-1, 1, 1, 1).expand(-1, seq_tensor.shape[1], 1, seq_tensor.shape[3])
    return seq_tensor.gather(2, index).squeeze(2)


class AsyncTransferSession:
    def __init__(self):
        self._copy_stream = None
        self._device = None
        self.staged_bytes = 0
        self.staged_tensors = 0
        self.staged_groups = 0

    def _normalize_tensor(self, tensor, dtype=None):
        torch = import_torch()
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return tensor
        src = tensor.detach()
        if dtype is not None and src.dtype != dtype:
            src = src.to(dtype=dtype)
        return src

    def stage_tensor_group(self, tensors: dict[str, tuple[Any, Any | None]]):
        if not tensors:
            return {}
        torch = import_torch()
        normalized: dict[str, Any] = {}
        grouped: dict[tuple[str, int | None, str], list[tuple[str, Any]]] = {}
        for name, (tensor, dtype) in tensors.items():
            src = self._normalize_tensor(tensor, dtype=dtype)
            normalized[name] = src
            if src is None or not isinstance(src, torch.Tensor):
                continue
            key = (src.device.type, src.device.index, str(src.dtype))
            grouped.setdefault(key, []).append((name, src))
        outputs = dict(normalized)
        for (_, _, _), group_items in grouped.items():
            first = group_items[0][1]
            if first.device.type != "cuda":
                for name, src in group_items:
                    outputs[name] = src.cpu()
                continue
            if self._copy_stream is None:
                self._device = first.device
                self._copy_stream = torch.cuda.Stream(device=first.device)
            current_stream = torch.cuda.current_stream(device=first.device)
            self._copy_stream.wait_stream(current_stream)
            flat_tensors = []
            total_numel = 0
            for name, src in group_items:
                flat = src.contiguous().view(-1)
                flat_tensors.append((name, src, flat, total_numel))
                total_numel += int(flat.numel())
            with torch.cuda.stream(self._copy_stream):
                packed = torch.empty(total_numel, dtype=first.dtype, device="cpu", pin_memory=True)
                for _, _, flat, offset in flat_tensors:
                    packed[offset : offset + flat.numel()].copy_(flat, non_blocking=True)
            for name, src, flat, offset in flat_tensors:
                outputs[name] = packed[offset : offset + flat.numel()].view(src.shape)
                self.staged_tensors += 1
                self.staged_bytes += int(src.numel() * src.element_size())
            self.staged_groups += 1
        return outputs

    def stage_tensor(self, tensor, dtype=None):
        if tensor is None:
            return None
        return self.stage_tensor_group({"value": (tensor, dtype)})["value"]

    def finalize_event(self):
        torch = import_torch()
        if self._copy_stream is None:
            return None
        event = torch.cuda.Event()
        event.record(self._copy_stream)
        return event

    def synchronize(self):
        event = self.finalize_event()
        if event is not None:
            event.synchronize()
        return event


class CollectionWriter:
    def __init__(self, root: str | Path, queue_size: int = 4):
        self.root = Path(root)
        self._queue: queue.Queue[tuple[Path, dict[str, Any], Any | None, dict[str, Any]] | None] = queue.Queue(
            maxsize=max(1, queue_size)
        )
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._writer_loop, name="kv-prepend-writer", daemon=True)
        self._thread.start()

    def _writer_loop(self):
        torch = import_torch()
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            path, payload, transfer_event, write_metadata = item
            try:
                wait_start = time.perf_counter()
                if transfer_event is not None:
                    transfer_event.synchronize()
                wait_s = time.perf_counter() - wait_start
                path.parent.mkdir(parents=True, exist_ok=True)
                save_start = time.perf_counter()
                torch.save(payload, path)
                save_s = time.perf_counter() - save_start
                logger.info(
                    "writer_saved batch_id=%s path=%s bundles=%s wait_s=%.3f save_s=%.3f staged_tensors=%s staged_groups=%s staged_bytes=%s",
                    write_metadata.get("batch_id"),
                    path,
                    write_metadata.get("bundle_count"),
                    wait_s,
                    save_s,
                    write_metadata.get("staged_tensors"),
                    write_metadata.get("staged_groups"),
                    write_metadata.get("staged_bytes"),
                )
            except BaseException as exc:  # pragma: no cover - async path
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_error(self):
        if self._error is not None:
            raise RuntimeError("Background capture writer failed.") from self._error

    def write_payload(
        self,
        batch_id: str,
        payload: dict[str, Any],
        *,
        row_count: int,
        target_subdir: str = "captures",
        transfer_event=None,
        staged_tensors: int = 0,
        staged_groups: int = 0,
        staged_bytes: int = 0,
    ):
        self._raise_if_error()
        path = self.root / target_subdir / f"{batch_id}.pt"
        self._queue.put(
            (
                path,
                payload,
                transfer_event,
                {
                    "batch_id": batch_id,
                    "bundle_count": row_count,
                    "staged_tensors": staged_tensors,
                    "staged_groups": staged_groups,
                    "staged_bytes": staged_bytes,
                },
            )
        )
        return path

    def write_batch(
        self,
        batch_id: str,
        bundles: list[ExampleCaptureBundle],
        extra_metadata: dict[str, Any] | None = None,
        target_subdir: str = "captures",
        transfer_event=None,
        staged_tensors: int = 0,
        staged_groups: int = 0,
        staged_bytes: int = 0,
    ):
        payload = {
            "schema_version": 2,
            "batch_id": batch_id,
            "metadata": extra_metadata or {},
            "bundles": bundles,
        }
        return self.write_payload(
            batch_id,
            payload,
            row_count=len(bundles),
            target_subdir=target_subdir,
            transfer_event=transfer_event,
            staged_tensors=staged_tensors,
            staged_groups=staged_groups,
            staged_bytes=staged_bytes,
        )

    def flush(self):
        start = time.perf_counter()
        self._queue.join()
        self._raise_if_error()
        logger.info("writer_flushed wait_s=%.3f", time.perf_counter() - start)

    def close(self):
        self.flush()
        self._queue.put(None)
        self._thread.join()


class InstrumentedQwen3MoeExperiment:
    def __init__(self, spec: ExperimentSpec, experiment_root: str | Path):
        self.spec = spec
        self.root = Path(experiment_root)
        self.model = None
        self.tokenizer = None
        self.config = None
        self.contract = None
        self.writer = CollectionWriter(self.root, queue_size=self.spec.collection.writer_queue_size)
        self._main_dense_layers = set(int(layer) for layer in self.spec.collection.main_dense_layers)
        self._main_router_layers = set(int(layer) for layer in self.spec.collection.main_router_layers)
        self._main_capture_signals = set(self.spec.collection.main_capture_signals)
        self._attention_weight_layers = set(int(layer) for layer in self.spec.collection.attention_weight_layers)
        self._prompt_affix_cache: dict[str, tuple[list[int], list[int]]] = {}
        self._sequence_length_buckets = sorted(int(bucket) for bucket in self.spec.collection.sequence_length_buckets)
        self._transfer_session: AsyncTransferSession | None = None

    def load(self):
        start = time.perf_counter()
        logger.info(
            "model_load_start model=%s dtype=%s quantization=%s device_map=%s attn_impl=%s",
            self.spec.model.model_name,
            self.spec.model.torch_dtype,
            self.spec.model.quantization,
            self.spec.model.device_map,
            self.spec.model.attn_implementation,
        )
        self.config, self.model, self.tokenizer = load_model_and_tokenizer(self.spec.model)
        self.contract = verify_model_contract(self.config, self.model)
        self.runtime_stack = runtime_stack_snapshot()
        configure_attention_runtime(
            enabled=self.spec.collection.enable_attention_compile,
            mode=self.spec.collection.attention_compile_mode,
            fullgraph=self.spec.collection.attention_compile_fullgraph,
            backend=self.spec.collection.attention_backend,
        )
        logger.info(
            "model_load_done seconds=%.3f runtime_backend=%s attention_backend=%s",
            time.perf_counter() - start,
            self.spec.collection.runtime_backend,
            self.spec.collection.attention_backend,
        )
        return self

    def _ensure_loaded(self):
        if self.model is None or self.tokenizer is None:
            self.load()

    def _model_input_device(self):
        self._ensure_loaded()
        return self.model.model.embed_tokens.weight.device

    def flush_writes(self):
        logger.info("flush_writes_start")
        self.writer.flush()
        logger.info("flush_writes_done")

    def _begin_transfer_session(self):
        self._transfer_session = AsyncTransferSession()

    def _finalize_transfer_session(self):
        session = self._transfer_session
        self._transfer_session = None
        if session is None:
            return None, 0, 0, 0
        return session.finalize_event(), session.staged_tensors, session.staged_groups, session.staged_bytes

    def _synchronize_transfer_session(self):
        session = self._transfer_session
        self._transfer_session = None
        if session is None:
            return 0, 0
        session.synchronize()
        return session.staged_tensors, session.staged_bytes

    def _stage_tensor(self, tensor, dtype=None):
        if self._transfer_session is None:
            return _to_cpu(tensor, dtype=dtype)
        return self._transfer_session.stage_tensor(tensor, dtype=dtype)

    def annotate_examples(self, examples: list[PromptExample]):
        self._ensure_loaded()
        if not examples:
            return examples
        text_ids_batch = self.tokenizer(
            [example.text for example in examples],
            add_special_tokens=False,
        )["input_ids"]
        for example, text_ids in zip(examples, text_ids_batch):
            if example.kind not in self._prompt_affix_cache:
                prefix, suffix = prompt_affixes(example.kind, self.spec.prompts)
                self._prompt_affix_cache[example.kind] = (
                    self.tokenizer(prefix, add_special_tokens=False)["input_ids"],
                    self.tokenizer(suffix, add_special_tokens=False)["input_ids"],
                )
            prefix_ids, suffix_ids = self._prompt_affix_cache[example.kind]
            max_text_tokens = max(0, self.spec.model.max_length - len(prefix_ids) - len(suffix_ids))
            text_ids = list(text_ids[:max_text_tokens])
            full_ids = prefix_ids + text_ids + suffix_ids
            full_ids = full_ids[: self.spec.model.max_length]
            example.content_token_mask = [0] * len(prefix_ids) + [1] * len(text_ids) + [0] * len(suffix_ids)
            example.prompt_token_ids = full_ids
            example.token_count = len(full_ids)
            if example.content_token_mask is None:
                example.content_token_mask = [1] * int(example.token_count)
            if len(example.content_token_mask) < int(example.token_count):
                example.content_token_mask.extend([0] * (int(example.token_count) - len(example.content_token_mask)))
            elif len(example.content_token_mask) > int(example.token_count):
                example.content_token_mask = example.content_token_mask[: int(example.token_count)]
        return examples

    def _main_dense_layer_set(self) -> set[int]:
        return self._main_dense_layers

    def _main_router_layer_set(self) -> set[int]:
        return self._main_router_layers

    def _main_capture_signal_set(self) -> set[str]:
        return self._main_capture_signals

    def _signal_enabled_for_layer(self, signal_name: str, layer_idx: int, calibration: bool) -> bool:
        if calibration:
            return True
        if signal_name not in self._main_capture_signal_set():
            return False
        if signal_name in {"router_logits", "top_k_binary"}:
            router_layers = self._main_router_layer_set()
            return not router_layers or layer_idx in router_layers
        dense_layers = self._main_dense_layer_set()
        return not dense_layers or layer_idx in dense_layers

    def _should_capture_layer(self, layer_idx: int, calibration: bool) -> bool:
        if calibration:
            return True
        for signal_name in self._main_capture_signal_set():
            if self._signal_enabled_for_layer(signal_name, layer_idx, calibration):
                return True
        return False

    def _should_store_attention(self, layer_idx: int, calibration: bool) -> bool:
        if not calibration:
            return False
        if self.spec.collection.capture_attention_weights_for_all_layers:
            return True
        if calibration and layer_idx in self._attention_weight_layers:
            return True
        return layer_idx in self._attention_weight_layers

    def _bucket_sequence_length(self, seq_len: int) -> int:
        for bucket in self._sequence_length_buckets:
            if seq_len <= bucket:
                return bucket
        return seq_len

    def tokenize_examples(self, examples: list[PromptExample], *, bucket_for_main: bool = False, pad_batch_for_main: bool = False):
        self._ensure_loaded()
        torch = import_torch()
        pad_token_id = int(self.tokenizer.pad_token_id)
        real_batch_size = len(examples)
        max_len = max(int(example.token_count or 0) for example in examples)
        target_len = self._bucket_sequence_length(max_len) if bucket_for_main else max_len
        target_batch_size = (
            max(real_batch_size, int(self.spec.collection.streaming_batch_size))
            if pad_batch_for_main
            else real_batch_size
        )
        input_ids = torch.full((target_batch_size, target_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((target_batch_size, target_len), dtype=torch.long)
        for row_idx, example in enumerate(examples):
            token_ids = list(example.prompt_token_ids or [])
            length = len(token_ids)
            if length:
                input_ids[row_idx, :length] = torch.tensor(token_ids, dtype=torch.long)
                attention_mask[row_idx, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "real_batch_size": real_batch_size}

    def prepare_examples(self, records: list[dict[str, Any]]):
        start = time.perf_counter()
        provisional = build_prompt_examples(records, self.spec.prompts)
        calibration_ids = sample_calibration_ids(
            provisional,
            calibration_subset_size=self.spec.collection.calibration_subset_size,
            seed=self.spec.collection.random_seed,
        )
        self._last_calibration_manifest = calibration_manifest(
            provisional,
            calibration_ids=calibration_ids,
            seed=self.spec.collection.random_seed,
        )
        examples = build_prompt_examples(records, self.spec.prompts, calibration_ids=calibration_ids)
        logger.info(
            "prepare_examples_done records=%s examples=%s calibration_selected=%s seconds=%.3f",
            len(records),
            len(examples),
            len(calibration_ids),
            time.perf_counter() - start,
        )
        return examples

    def _restrict_to_calibration_subset(self) -> bool:
        return self.spec.collection.runtime_backend == "hf" and int(self.spec.collection.calibration_subset_size) > 0

    def _write_calibration_manifest(self, dataset_name: str):
        manifest = getattr(self, "_last_calibration_manifest", None)
        if manifest is None:
            return None
        artifacts_dir = self.root / self.spec.output.artifacts_dir / "calibration_manifests"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / f"{dataset_name}.json"
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _project_qkv(self, attention_module, hidden_states, position_embeddings):
        torch = import_torch()

        batch_size, seq_len, _ = hidden_states.shape
        head_dim = attention_module.head_dim
        hidden_shape = (batch_size, seq_len, -1, head_dim)
        q_pre = attention_module.q_norm(attention_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k_pre = attention_module.k_norm(attention_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v_raw = attention_module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_pre, k_pre, cos, sin)
        return q_pre, k_pre, q_rot, k_rot, v_raw, cos, sin

    def _run_moe(self, mlp_module, hidden_states):
        torch = import_torch()
        F = torch.nn.functional
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hasattr(mlp_module, "gate") and hasattr(mlp_module, "experts"):
            flat = hidden_states.reshape(-1, hidden_dim)
            router_logits = F.linear(flat, mlp_module.gate.weight)
            router_probs = F.softmax(router_logits.float(), dim=-1)
            router_scores, top_k_indices = torch.topk(router_probs, mlp_module.gate.top_k, dim=-1)
            if mlp_module.gate.norm_topk_prob:
                router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
            router_scores = router_scores.to(router_logits.dtype)
            mlp_output = mlp_module.experts(flat, top_k_indices, router_scores).reshape(batch_size, seq_len, hidden_dim)
            return (
                mlp_output,
                router_logits.reshape(batch_size, seq_len, -1),
                top_k_indices.reshape(batch_size, seq_len, -1).to(dtype=torch.int8),
            )
        return mlp_module(hidden_states), None, None

    def _run_router_only(self, mlp_module, hidden_states):
        torch = import_torch()
        F = torch.nn.functional
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hasattr(mlp_module, "gate"):
            flat = hidden_states.reshape(-1, hidden_dim)
            router_logits = F.linear(flat, mlp_module.gate.weight)
            router_probs = F.softmax(router_logits.float(), dim=-1)
            top_k_indices = torch.topk(router_probs, mlp_module.gate.top_k, dim=-1).indices
            return (
                router_logits.reshape(batch_size, seq_len, -1),
                top_k_indices.reshape(batch_size, seq_len, -1).to(dtype=torch.int8),
            )
        return None, None

    def _iter_example_batches(self, examples: list[PromptExample]):
        ordered = list(examples)
        if self.spec.collection.sort_by_length:
            ordered.sort(key=lambda ex: int(ex.token_count or 0), reverse=True)
        current: list[PromptExample] = []
        current_max = 0
        for example in ordered:
            if current and bool(current[0].calibration) != bool(example.calibration):
                yield current
                current = []
                current_max = 0
            proposed_max = max(current_max, int(example.token_count or 0))
            proposed_size = len(current) + 1
            exceeds_size = proposed_size > self.spec.collection.streaming_batch_size
            exceeds_tokens = proposed_size * proposed_max > self.spec.collection.max_batch_tokens
            if current and (exceeds_size or exceeds_tokens):
                yield current
                current = [example]
                current_max = int(example.token_count or 0)
            else:
                current.append(example)
                current_max = proposed_max
        if current:
            yield current

    def _slice_batch_value(self, value, row_idx: int, batch_size: int):
        if value is None:
            return None
        try:
            import torch

            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= batch_size and row_idx < value.shape[0]:
                return value[row_idx : row_idx + 1]
        except ModuleNotFoundError:  # pragma: no cover
            pass
        return value

    def _slice_layer_capture(self, capture: LayerCapture, row_idx: int, batch_size: int) -> LayerCapture:
        return LayerCapture(
            layer_idx=capture.layer_idx,
            condition=capture.condition,
            resid_pre_attn=self._slice_batch_value(capture.resid_pre_attn, row_idx, batch_size),
            q_pre_rope=self._slice_batch_value(capture.q_pre_rope, row_idx, batch_size),
            v_raw=self._slice_batch_value(capture.v_raw, row_idx, batch_size),
            attention_weights=self._slice_batch_value(capture.attention_weights, row_idx, batch_size),
            beta=self._slice_batch_value(capture.beta, row_idx, batch_size),
            z_attn=self._slice_batch_value(capture.z_attn, row_idx, batch_size),
            h_pre_moe=self._slice_batch_value(capture.h_pre_moe, row_idx, batch_size),
            router_logits_pre_softmax=self._slice_batch_value(capture.router_logits_pre_softmax, row_idx, batch_size),
            top_k_indices=self._slice_batch_value(capture.top_k_indices, row_idx, batch_size),
            final_token_k_rot=self._slice_batch_value(capture.final_token_k_rot, row_idx, batch_size),
            final_token_v=self._slice_batch_value(capture.final_token_v, row_idx, batch_size),
            final_token_k_raw=self._slice_batch_value(capture.final_token_k_raw, row_idx, batch_size),
            position_ids=self._slice_batch_value(capture.position_ids, row_idx, batch_size),
            metadata=dict(capture.metadata),
        )

    def _split_pass_capture(self, pass_capture: PassCapture, batch_examples: list[PromptExample]) -> list[PassCapture]:
        return split_pass_capture(pass_capture, len(batch_examples))

    def _split_summary_batch(self, summaries: dict[int, dict[str, Any]], batch_examples: list[PromptExample]):
        batch_size = len(batch_examples)
        outputs = [dict() for _ in batch_examples]
        for layer_idx, layer_summary in summaries.items():
            for row_idx in range(batch_size):
                outputs[row_idx][layer_idx] = {
                    key: self._slice_batch_value(value, row_idx, batch_size)
                    for key, value in layer_summary.items()
                }
        return outputs

    def _build_layer_capture(
        self,
        *,
        layer_idx: int,
        condition: CaptureCondition,
        resid_pre_attn,
        q_pre,
        v_raw,
        attn_weights,
        beta,
        z_attn,
        h_pre_moe,
        router_logits,
        top_k_indices,
        final_token_k_rot,
        final_token_v,
        final_token_k_raw,
        position_ids,
        calibration: bool,
        extra_metadata: dict[str, Any] | None = None,
    ) -> LayerCapture:
        metadata = {"calibration": calibration}
        if extra_metadata:
            metadata.update(extra_metadata)
        staged = self._transfer_session.stage_tensor_group(
            {
                "resid_pre_attn": (
                    resid_pre_attn if self._signal_enabled_for_layer("resid_pre_attn", layer_idx, calibration) else None,
                    _storage_dtype("resid_pre_attn", calibration),
                ),
                "q_pre_rope": (
                    (
                        q_pre
                        if calibration
                        and self.spec.collection.capture_q_vectors
                        and self._signal_enabled_for_layer("q_pre_rope", layer_idx, calibration)
                        else None
                    ),
                    _storage_dtype("q_pre_rope", calibration),
                ),
                "v_raw": (
                    v_raw if self._signal_enabled_for_layer("v_raw", layer_idx, calibration) else None,
                    _storage_dtype("v_raw", calibration),
                ),
                "attention_weights": (
                    attn_weights if self._signal_enabled_for_layer("attention_weights", layer_idx, calibration) else None,
                    _storage_dtype("attention_weights", calibration),
                ),
                "beta": (
                    beta if self._signal_enabled_for_layer("beta", layer_idx, calibration) else None,
                    _storage_dtype("beta", calibration),
                ),
                "z_attn": (
                    z_attn if self._signal_enabled_for_layer("attention_output", layer_idx, calibration) else None,
                    _storage_dtype("z_attn", calibration),
                ),
                "h_pre_moe": (
                    h_pre_moe if self._signal_enabled_for_layer("pre_moe", layer_idx, calibration) else None,
                    _storage_dtype("h_pre_moe", calibration),
                ),
                "router_logits_pre_softmax": (
                    router_logits if self._signal_enabled_for_layer("router_logits", layer_idx, calibration) else None,
                    _storage_dtype("router_logits_pre_softmax", calibration),
                ),
                "top_k_indices": (
                    top_k_indices if self._signal_enabled_for_layer("top_k_binary", layer_idx, calibration) else None,
                    _storage_dtype("top_k_indices", calibration),
                ),
                "final_token_k_rot": (
                    final_token_k_rot if self._signal_enabled_for_layer("final_token_k_rot", layer_idx, calibration) else None,
                    _storage_dtype("final_token_k_rot", calibration),
                ),
                "final_token_v": (
                    final_token_v if self._signal_enabled_for_layer("final_token_v", layer_idx, calibration) else None,
                    _storage_dtype("final_token_v", calibration),
                ),
                "final_token_k_raw": (
                    final_token_k_raw if self._signal_enabled_for_layer("final_token_k_raw", layer_idx, calibration) else None,
                    _storage_dtype("final_token_k_raw", calibration),
                ),
                "position_ids": (
                    position_ids if self._signal_enabled_for_layer("position_ids", layer_idx, calibration) else None,
                    None,
                ),
            }
        )
        return LayerCapture(
            layer_idx=layer_idx,
            condition=condition.value,
            resid_pre_attn=staged["resid_pre_attn"],
            q_pre_rope=staged["q_pre_rope"],
            v_raw=staged["v_raw"],
            attention_weights=staged["attention_weights"],
            beta=staged["beta"],
            z_attn=staged["z_attn"],
            h_pre_moe=staged["h_pre_moe"],
            router_logits_pre_softmax=staged["router_logits_pre_softmax"],
            top_k_indices=staged["top_k_indices"],
            final_token_k_rot=staged["final_token_k_rot"],
            final_token_v=staged["final_token_v"],
            final_token_k_raw=staged["final_token_k_raw"],
            position_ids=staged["position_ids"],
            metadata=metadata,
        )

    def _choose_summary_slot(
        self,
        mode: str | None,
        k_pre,
        k_rot,
        v_raw,
        example_index: int,
        batch_rng: random.Random,
        external_summary: dict[str, Any] | None = None,
    ):
        import torch

        if external_summary is not None:
            return (
                external_summary["final_token_k_rot"].to(k_rot.device),
                external_summary["final_token_v"].to(v_raw.device),
            )
        if mode is None:
            return make_prepend_summary_kv(k_pre, k_rot, v_raw, self._current_cos, self._current_sin, self.spec.collection.default_rope_mode)
        if mode == "first_token":
            return k_rot[..., :1, :], v_raw[..., :1, :]
        if mode == "random_token":
            token_index = batch_rng.randrange(k_rot.shape[2])
            return k_rot[..., token_index : token_index + 1, :], v_raw[..., token_index : token_index + 1, :]
        if mode == "k_only":
            summary_k, summary_v = make_prepend_summary_kv(
                k_pre,
                k_rot,
                v_raw,
                self._current_cos,
                self._current_sin,
                self.spec.collection.default_rope_mode,
            )
            return summary_k, matched_norm_random_like(summary_v)
        if mode == "k_only_zero":
            summary_k, summary_v = make_prepend_summary_kv(
                k_pre,
                k_rot,
                v_raw,
                self._current_cos,
                self._current_sin,
                self.spec.collection.default_rope_mode,
            )
            return summary_k, summary_v.new_zeros(summary_v.shape)
        if mode == "v_only":
            summary_k, summary_v = make_prepend_summary_kv(
                k_pre,
                k_rot,
                v_raw,
                self._current_cos,
                self._current_sin,
                self.spec.collection.default_rope_mode,
            )
            return matched_norm_random_like(summary_k), summary_v
        if mode == "v_only_zero":
            summary_k, summary_v = make_prepend_summary_kv(
                k_pre,
                k_rot,
                v_raw,
                self._current_cos,
                self._current_sin,
                self.spec.collection.default_rope_mode,
            )
            return summary_k.new_zeros(summary_k.shape), summary_v
        raise ValueError(f"Unsupported control summary mode: {mode}")

    def _bias_values(self):
        torch = import_torch()
        return torch.linspace(
            float(self.spec.collection.bias_sweep_min),
            float(self.spec.collection.bias_sweep_max),
            int(self.spec.collection.bias_sweep_points),
            device=self._model_input_device(),
            dtype=torch.float32,
        )

    def _compute_bias_sweep_metadata(
        self,
        *,
        layer,
        layer_idx: int,
        causal_headwise,
        prepend_result,
        resid_pre_attn,
    ) -> dict[str, Any]:
        torch = import_torch()
        import torch.nn.functional as F

        start = time.perf_counter()
        try:
            summary_value = prepend_result.summary_value
            summary_headwise = summary_value.repeat_interleave(layer.self_attn.num_key_value_groups, dim=1).expand(
                -1, -1, causal_headwise.shape[2], -1
            )
            beta0 = prepend_result.beta.squeeze(-1).clamp(1e-6, 1 - 1e-6)
            margin0 = torch.log(beta0 / (1.0 - beta0))
            bias_values = self._bias_values()
            bias_chunk_size = 8
            router_rows = []
            topk_rows = []
            gate_weight = layer.mlp.gate.weight
            hidden_dim = resid_pre_attn.shape[-1]

            for start_idx in range(0, bias_values.shape[0], bias_chunk_size):
                bias_chunk = bias_values[start_idx : start_idx + bias_chunk_size]
                beta_chunk = torch.sigmoid(
                    margin0.unsqueeze(1) + bias_chunk.view(1, -1, 1, 1)
                ).unsqueeze(-1)
                mixed_headwise = (
                    (1.0 - beta_chunk) * causal_headwise.unsqueeze(1)
                    + beta_chunk * summary_headwise.unsqueeze(1)
                )
                concat = mixed_headwise.permute(0, 1, 3, 2, 4).reshape(-1, hidden_dim)
                z_b = layer.self_attn.o_proj(concat).reshape(
                    resid_pre_attn.shape[0],
                    bias_chunk.shape[0],
                    resid_pre_attn.shape[1],
                    hidden_dim,
                )
                h_base = resid_pre_attn.unsqueeze(1) + z_b
                h_pre_moe = layer.post_attention_layernorm(
                    h_base.reshape(-1, hidden_dim)
                ).reshape_as(h_base)
                router_logits = F.linear(
                    h_pre_moe.reshape(-1, hidden_dim),
                    gate_weight,
                ).reshape(
                    resid_pre_attn.shape[0],
                    bias_chunk.shape[0],
                    resid_pre_attn.shape[1],
                    -1,
                )
                probs = F.softmax(router_logits.float(), dim=-1)
                topk = torch.topk(probs, layer.mlp.gate.top_k, dim=-1).indices
                router_rows.append(router_logits[0])
                topk_rows.append(topk[0])

            router_curve = torch.cat(router_rows, dim=0).permute(1, 0, 2)
            topk_curve = torch.cat(topk_rows, dim=0).permute(1, 0, 2)
            probs_curve = torch.softmax(router_curve.float(), dim=-1)
            top1_prob = probs_curve.max(dim=-1).values
            grad = torch.gradient(top1_prob, spacing=(bias_values.float(),), dim=1)[0]
            curvature = torch.gradient(grad, spacing=(bias_values.float(),), dim=1)[0]
            baseline_probs = probs_curve[:, bias_values.shape[0] // 2, :]
            sorted_probs = torch.sort(baseline_probs, dim=-1, descending=True).values
            topk_width = layer.mlp.gate.top_k
            margin = sorted_probs[:, topk_width - 1] - sorted_probs[:, topk_width]
            kl = (baseline_probs * (baseline_probs.clamp_min(1e-6).log() - probs_curve[:, -1, :].clamp_min(1e-6).log())).sum(dim=-1)

            transitions = []
            first_transition = torch.full((topk_curve.shape[0],), float("nan"), device=topk_curve.device)
            flip_count = torch.zeros((topk_curve.shape[0],), dtype=torch.int32, device=topk_curve.device)
            baseline_index = bias_values.shape[0] // 2
            baseline_topk = topk_curve[:, baseline_index, :]
            for bias_index in range(topk_curve.shape[1]):
                if bias_index == baseline_index:
                    continue
                current = topk_curve[:, bias_index, :]
                changed = (current != baseline_topk).any(dim=-1)
                newly_changed = changed & torch.isnan(first_transition)
                first_transition[newly_changed] = bias_values[bias_index]
                flip_count = flip_count + changed.to(torch.int32)
                for token_index in torch.nonzero(changed, as_tuple=False).flatten().tolist():
                    prev = set(baseline_topk[token_index].tolist())
                    curr = set(current[token_index].tolist())
                    outs = sorted(prev - curr)
                    ins = sorted(curr - prev)
                    for out_expert, in_expert in zip(outs, ins):
                        rank_position = int((current[token_index] == in_expert).nonzero(as_tuple=False)[0].item())
                        transitions.append(
                            {
                                "token_index": int(token_index),
                                "b_critical": float(bias_values[bias_index].item()),
                                "expert_out": int(out_expert),
                                "expert_in": int(in_expert),
                                "rank_position": rank_position,
                            }
                        )

            signature = {
                "bias_values": [float(v) for v in bias_values.detach().cpu().tolist()],
                "sensitivity": grad[:, bias_values.shape[0] // 2].detach().cpu().tolist(),
                "curvature": curvature[:, bias_values.shape[0] // 2].detach().cpu().tolist(),
                "first_phase_transition": first_transition.detach().cpu().tolist(),
                "flip_count": flip_count.detach().cpu().tolist(),
                "margin": margin.detach().cpu().tolist(),
                "kl_terminal": kl.detach().cpu().tolist(),
                "beta_margin": margin0.mean(dim=0).detach().cpu().tolist(),
            }
            logger.info(
                "bias_sweep_done layer=%s tokens=%s bias_points=%s seconds=%.3f",
                layer_idx,
                resid_pre_attn.shape[1],
                bias_values.shape[0],
                time.perf_counter() - start,
            )
            return {"bias_spectrum_signature": signature, "bias_spectrum_transitions": transitions}
        except Exception as exc:  # pragma: no cover - runtime/model specific
            return {"bias_spectrum_error": str(exc)}

    def _layer_forward_variants(
        self,
        *,
        layer,
        layer_idx: int,
        hidden_states,
        attention_mask,
        token_counts,
        position_embeddings,
        position_ids,
        calibration: bool,
        main_condition: CaptureCondition,
        alt_condition: CaptureCondition | None,
        main_uses_prepend: bool,
        control_summary_mode: str | None = None,
        external_summary: dict[str, Any] | None = None,
        batch_rng: random.Random,
    ):
        residual = hidden_states
        normed_hidden = layer.input_layernorm(hidden_states)
        q_pre, k_pre, q_rot, k_rot, v_raw, cos, sin = self._project_qkv(layer.self_attn, normed_hidden, position_embeddings)
        self._current_cos = cos
        self._current_sin = sin
        store_attn = self._should_store_attention(layer_idx, calibration)
        need_causal_weights = bool(store_attn)
        final_token_k_rot = k_rot[..., -1, :]
        final_token_v = v_raw[..., -1, :]
        final_token_k_raw = k_pre[..., -1, :]

        causal = attention_forward(
            layer.self_attn,
            q_rot,
            k_rot,
            v_raw,
            attention_mask,
            prepend_mode=None,
            token_counts=token_counts,
            return_weights=need_causal_weights,
        )
        causal_z = layer.self_attn.o_proj(causal.attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
        causal_hidden = residual + causal_z
        causal_pre_moe = layer.post_attention_layernorm(causal_hidden)
        if main_uses_prepend:
            causal_router_logits, causal_topk = self._run_router_only(layer.mlp, causal_pre_moe)
            causal_next = None
        else:
            causal_mlp_out, causal_router_logits, causal_topk = self._run_moe(layer.mlp, causal_pre_moe)
            causal_next = causal_hidden + causal_mlp_out

        prepend_capture = None
        prepend_next = None
        prepend_pre_moe = None
        prepend_router_logits = None
        prepend_topk = None
        prepend_z = None
        prepend_result = None
        bias_metadata: dict[str, Any] | None = None

        if main_uses_prepend or alt_condition == CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE:
            summary_k, summary_v = self._choose_summary_slot(
                control_summary_mode,
                k_pre,
                k_rot,
                v_raw,
                0,
                batch_rng,
                external_summary=external_summary,
            )
            prepend_result = attention_forward(
                layer.self_attn,
                q_rot,
                k_rot,
                v_raw,
                attention_mask,
                prepend_mode=self.spec.collection.default_rope_mode,
                cos=cos,
                sin=sin,
                key_pre_rope=k_pre,
                summary_key=summary_k,
                summary_value=summary_v,
                token_counts=token_counts,
                return_weights=False,
                return_beta_only=calibration,
            )
            prepend_z = layer.self_attn.o_proj(prepend_result.attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
            prepend_hidden = residual + prepend_z
            prepend_pre_moe = layer.post_attention_layernorm(prepend_hidden)
            if main_uses_prepend:
                prepend_mlp_out, prepend_router_logits, prepend_topk = self._run_moe(layer.mlp, prepend_pre_moe)
                prepend_next = prepend_hidden + prepend_mlp_out
            else:
                prepend_router_logits, prepend_topk = self._run_router_only(layer.mlp, prepend_pre_moe)
                prepend_next = None
            if calibration:
                bias_metadata = self._compute_bias_sweep_metadata(
                    layer=layer,
                    layer_idx=layer_idx,
                    causal_headwise=causal.attn_output.transpose(1, 2),
                    prepend_result=prepend_result,
                    resid_pre_attn=residual,
                )

        captures: dict[str, LayerCapture] = {}
        should_capture = self._should_capture_layer(layer_idx, calibration)
        if should_capture:
            captures[CaptureCondition.CAUSAL.value] = self._build_layer_capture(
                layer_idx=layer_idx,
                condition=CaptureCondition.CAUSAL,
                resid_pre_attn=residual,
                q_pre=q_pre,
                v_raw=v_raw,
                attn_weights=causal.attn_weights if store_attn else None,
                beta=None,
                z_attn=causal_z,
                h_pre_moe=causal_pre_moe,
                router_logits=causal_router_logits,
                top_k_indices=causal_topk,
                final_token_k_rot=final_token_k_rot,
                final_token_v=final_token_v,
                final_token_k_raw=final_token_k_raw,
                position_ids=position_ids,
                calibration=calibration,
            )

        if should_capture and alt_condition == CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE and prepend_result is not None:
            captures[alt_condition.value] = self._build_layer_capture(
                layer_idx=layer_idx,
                condition=alt_condition,
                resid_pre_attn=residual,
                q_pre=q_pre,
                v_raw=v_raw,
                attn_weights=None,
                beta=prepend_result.beta,
                z_attn=prepend_z,
                h_pre_moe=prepend_pre_moe,
                router_logits=prepend_router_logits,
                top_k_indices=prepend_topk,
                final_token_k_rot=final_token_k_rot,
                final_token_v=final_token_v,
                final_token_k_raw=final_token_k_raw,
                position_ids=position_ids,
                calibration=calibration,
                extra_metadata=bias_metadata,
            )

        if should_capture and main_condition == CaptureCondition.PROPAGATED and prepend_result is not None:
            captures[CaptureCondition.PROPAGATED.value] = self._build_layer_capture(
                layer_idx=layer_idx,
                condition=CaptureCondition.PROPAGATED,
                resid_pre_attn=residual,
                q_pre=q_pre,
                v_raw=v_raw,
                attn_weights=None,
                beta=prepend_result.beta,
                z_attn=prepend_z,
                h_pre_moe=prepend_pre_moe,
                router_logits=prepend_router_logits,
                top_k_indices=prepend_topk,
                final_token_k_rot=final_token_k_rot,
                final_token_v=final_token_v,
                final_token_k_raw=final_token_k_raw,
                position_ids=position_ids,
                calibration=calibration,
                extra_metadata=bias_metadata,
            )
            captures[CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE.value] = self._build_layer_capture(
                layer_idx=layer_idx,
                condition=CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE,
                resid_pre_attn=residual,
                q_pre=q_pre,
                v_raw=v_raw,
                attn_weights=causal.attn_weights if store_attn else None,
                beta=None,
                z_attn=causal_z,
                h_pre_moe=causal_pre_moe,
                router_logits=causal_router_logits,
                top_k_indices=causal_topk,
                final_token_k_rot=final_token_k_rot,
                final_token_v=final_token_v,
                final_token_k_raw=final_token_k_raw,
                position_ids=position_ids,
                calibration=calibration,
            )

        next_hidden = prepend_next if main_uses_prepend and prepend_next is not None else causal_next
        return next_hidden, captures

    def _run_pass(
        self,
        input_ids,
        attention_mask,
        calibration: bool,
        pass_name: str,
        control_summary_mode: str | None = None,
        external_summary_by_layer: dict[int, dict[str, Any]] | None = None,
        start_layer: int = 0,
        initial_hidden_states=None,
        initial_captures_by_condition: dict[str, list[LayerCapture]] | None = None,
    ):
        torch = import_torch()
        with torch.inference_mode():
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            batch_size, seq_len = input_ids.shape
            model_core = self.model.model
            if initial_hidden_states is None:
                hidden_states = model_core.embed_tokens(input_ids.to(self._model_input_device()))
            else:
                hidden_states = initial_hidden_states
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            token_counts = attention_mask.sum(dim=1).to(device=hidden_states.device, dtype=torch.int32)
            uses_dense_attention_mask = bool(
                calibration or self.spec.collection.attention_backend == "sdpa"
            )
            causal_mask = (
                _causal_attention_mask(
                    attention_mask=attention_mask,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                if uses_dense_attention_mask
                else None
            )
            position_embeddings = model_core.rotary_emb(hidden_states, position_ids=position_ids)
            batch_rng = random.Random(self.spec.collection.random_seed)
            captures_by_condition: dict[str, list[LayerCapture]] = {
                condition: list(captures)
                for condition, captures in (initial_captures_by_condition or {}).items()
            }
            propagate_start = int(self.spec.collection.propagate_from_layer)
            propagate_hidden_states = hidden_states if start_layer >= propagate_start else None

            for layer_idx, layer in enumerate(model_core.layers[start_layer:], start=start_layer):
                if pass_name == "pass1" or layer_idx < self.spec.collection.propagate_from_layer:
                    hidden_states, layer_captures = self._layer_forward_variants(
                        layer=layer,
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        token_counts=token_counts,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        calibration=calibration,
                        main_condition=CaptureCondition.CAUSAL,
                        alt_condition=CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE,
                        main_uses_prepend=False,
                        control_summary_mode=control_summary_mode,
                        external_summary=None if external_summary_by_layer is None else external_summary_by_layer.get(layer_idx),
                        batch_rng=batch_rng,
                    )
                else:
                    hidden_states, layer_captures = self._layer_forward_variants(
                        layer=layer,
                        layer_idx=layer_idx,
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        token_counts=token_counts,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        calibration=calibration,
                        main_condition=CaptureCondition.PROPAGATED,
                        alt_condition=CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE,
                        main_uses_prepend=True,
                        control_summary_mode=control_summary_mode,
                        external_summary=None if external_summary_by_layer is None else external_summary_by_layer.get(layer_idx),
                        batch_rng=batch_rng,
                    )
                for condition, capture in layer_captures.items():
                    captures_by_condition.setdefault(condition, []).append(capture)
                if propagate_hidden_states is None and layer_idx + 1 == propagate_start:
                    propagate_hidden_states = hidden_states

            return (
                PassCapture(
                    pass_name=pass_name,
                    rope_mode=self.spec.collection.default_rope_mode,
                    captures_by_condition=captures_by_condition,
                ),
                propagate_hidden_states,
            )

    def _forward_summary_only(self, input_ids, attention_mask):
        torch = import_torch()
        start = time.perf_counter()
        with torch.inference_mode():
            if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            batch_size, seq_len = input_ids.shape
            model_core = self.model.model
            hidden_states = model_core.embed_tokens(input_ids.to(self._model_input_device()))
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            causal_mask = _causal_attention_mask(attention_mask, batch_size, seq_len, hidden_states.device, hidden_states.dtype)
            position_embeddings = model_core.rotary_emb(hidden_states, position_ids=position_ids)
            summaries = {}
            last_positions = _last_token_positions(attention_mask.to(device=hidden_states.device))
            token_counts = (last_positions + 1).to(dtype=torch.int32)
            for layer_idx, layer in enumerate(model_core.layers):
                residual = hidden_states
                normed_hidden = layer.input_layernorm(hidden_states)
                q_pre, k_pre, q_rot, k_rot, v_raw, _, _ = self._project_qkv(layer.self_attn, normed_hidden, position_embeddings)
                summaries[layer_idx] = self._transfer_session.stage_tensor_group(
                    {
                        "final_token_k_rot": (
                            _gather_last_kv(k_rot, last_positions),
                            _storage_dtype("final_token_k_rot", calibration=False),
                        ),
                        "final_token_k_raw": (
                            _gather_last_kv(k_pre, last_positions),
                            _storage_dtype("final_token_k_raw", calibration=False),
                        ),
                        "final_token_v": (
                            _gather_last_kv(v_raw, last_positions),
                            _storage_dtype("final_token_v", calibration=False),
                        ),
                    }
                )
                causal = attention_forward(
                    layer.self_attn,
                    q_rot,
                    k_rot,
                    v_raw,
                    causal_mask,
                    key_pre_rope=k_pre,
                    token_counts=token_counts,
                    return_weights=False,
                )
                causal_z = layer.self_attn.o_proj(causal.attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
                post = residual + causal_z
                mlp_out, _, _ = self._run_moe(layer.mlp, layer.post_attention_layernorm(post))
                hidden_states = post + mlp_out
            hidden_states = model_core.norm(hidden_states)
            logger.info(
                "summary_only_done batch_size=%s seq_len=%s seconds=%.3f",
                batch_size,
                seq_len,
                time.perf_counter() - start,
            )
            return summaries, _gather_last_hidden(hidden_states, last_positions)

    def collect_multi_slot_summaries_batch(self, batch_examples: list[PromptExample]):
        self._ensure_loaded()
        torch = import_torch()
        if not batch_examples:
            return []
        start = time.perf_counter()
        for example in batch_examples:
            if example.prompt_token_ids is None:
                self.annotate_examples([example])
        sequences = [list(example.prompt_token_ids or []) for example in batch_examples]
        pad_token_id = int(self.tokenizer.pad_token_id)
        all_slots = [dict() for _ in batch_examples]
        for _ in range(1 + self.spec.collection.multi_slot_decode_steps):
            max_len = max(len(tokens) for tokens in sequences)
            input_ids = torch.full((len(sequences), max_len), pad_token_id, device=self._model_input_device(), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for row_idx, token_ids in enumerate(sequences):
                if token_ids:
                    length = len(token_ids)
                    input_ids[row_idx, :length] = torch.tensor(token_ids, device=input_ids.device, dtype=torch.long)
                    attention_mask[row_idx, :length] = 1
            summaries, last_hidden = self._forward_summary_only(input_ids, attention_mask)
            split_summaries = self._split_summary_batch(summaries, batch_examples)
            for row_idx, per_example_summary in enumerate(split_summaries):
                for layer_idx, layer_summary in per_example_summary.items():
                    all_slots[row_idx].setdefault(layer_idx, {"k": [], "k_pre": [], "v": []})
                    all_slots[row_idx][layer_idx]["k"].append(layer_summary["final_token_k_rot"])
                    all_slots[row_idx][layer_idx]["k_pre"].append(layer_summary["final_token_k_raw"])
                    all_slots[row_idx][layer_idx]["v"].append(layer_summary["final_token_v"])
            next_logits = self.model.lm_head(last_hidden.clone())
            next_tokens = next_logits.argmax(dim=-1).tolist()
            for token_ids, next_token in zip(sequences, next_tokens):
                token_ids.append(int(next_token))
        logger.info(
            "multi_slot_batch_done examples=%s decode_steps=%s seconds=%.3f",
            len(batch_examples),
            1 + self.spec.collection.multi_slot_decode_steps,
            time.perf_counter() - start,
        )
        return all_slots

    def collect_multi_slot_summaries(self, example: PromptExample):
        batch_outputs = self.collect_multi_slot_summaries_batch([example])
        return batch_outputs[0] if batch_outputs else {}

    def _shuffle_text(self, text: str) -> str:
        words = text.split()
        rng = random.Random(self.spec.collection.random_seed)
        rng.shuffle(words)
        return " ".join(words)

    def _compute_external_summaries(self, records: list[dict[str, Any]], dataset_name: str = "control_summaries"):
        self._ensure_loaded()
        examples = self.prepare_examples(records)
        self._write_calibration_manifest(dataset_name)
        if self._restrict_to_calibration_subset():
            examples = [example for example in examples if example.calibration]
        self.annotate_examples(examples)
        summary_by_id: dict[str, dict[int, dict[str, Any]]] = {}
        for batch_examples in self._iter_example_batches(examples):
            encoded = self.tokenize_examples(batch_examples)
            input_ids = encoded["input_ids"].to(self._model_input_device())
            attention_mask = encoded["attention_mask"].to(self._model_input_device())
            batch_summary, _ = self._forward_summary_only(input_ids, attention_mask)
            for example, summary in zip(batch_examples, self._split_summary_batch(batch_summary, batch_examples)):
                summary_by_id[example.text_id] = summary
        return [summary_by_id[example.text_id] for example in examples]

    def _compute_prompt_continuation_summaries(self, records: list[dict[str, Any]]):
        self._ensure_loaded()
        examples = self.prepare_examples(records)
        if self._restrict_to_calibration_subset():
            examples = [example for example in examples if example.calibration]
        outputs = []
        for example in examples:
            multi_slot = self.collect_multi_slot_summaries(example)
            per_layer = {}
            for layer_idx, slots in multi_slot.items():
                slot_index = 1 if len(slots["k"]) > 1 else 0
                per_layer[layer_idx] = {
                    "final_token_k_rot": slots["k"][slot_index],
                    "final_token_v": slots["v"][slot_index],
                }
            outputs.append(per_layer)
        return outputs

    def collect_examples(
        self,
        records: list[dict[str, Any]],
        dataset_name: str = "custom",
        control_summary_mode: str | None = None,
        external_summaries: list[dict[int, dict[str, Any]]] | None = None,
        target_subdir: str | None = None,
        write_batches: bool = True,
        retain_bundles: bool = True,
    ):
        self._ensure_loaded()
        collect_start = time.perf_counter()
        examples = self.prepare_examples(records)
        self._write_calibration_manifest(dataset_name)
        if self._restrict_to_calibration_subset():
            examples = [example for example in examples if example.calibration]
            logger.info("collect_examples_restricted_to_calibration_subset dataset=%s examples=%s", dataset_name, len(examples))
        self.annotate_examples(examples)
        path_list: list[Path] = []
        all_bundles: list[ExampleCaptureBundle] = []
        target = target_subdir or self.spec.output.captures_dir
        external_lookup = None
        if external_summaries is not None:
            external_lookup = {example.text_id: summary for example, summary in zip(examples, external_summaries)}
        batched_examples = (
            [[example] for example in examples]
            if external_lookup is not None
            else list(self._iter_example_batches(examples))
        )
        logger.info(
            "collect_examples_start dataset=%s records=%s examples=%s batches=%s target_subdir=%s write_batches=%s retain_bundles=%s external=%s control=%s",
            dataset_name,
            len(records),
            len(examples),
            len(batched_examples),
            target,
            write_batches,
            retain_bundles,
            external_lookup is not None,
            control_summary_mode,
        )

        for batch_index, batch_examples in enumerate(batched_examples):
            batch_start = time.perf_counter()
            self._begin_transfer_session()
            is_lean_main_batch = (
                external_lookup is None
                and control_summary_mode is None
                and not any(example.calibration for example in batch_examples)
                and self.spec.collection.runtime_backend == "hf_teacher_forcing"
            )
            logger.info(
                "batch_start dataset=%s batch_index=%s batch_size=%s max_tokens=%s lean_main=%s calibration_batch=%s",
                dataset_name,
                batch_index,
                len(batch_examples),
                max(int(example.token_count or 0) for example in batch_examples),
                is_lean_main_batch,
                all(example.calibration for example in batch_examples),
            )
            encode_start = time.perf_counter()
            encoded = self.tokenize_examples(
                batch_examples,
                bucket_for_main=is_lean_main_batch,
                pad_batch_for_main=is_lean_main_batch and self.spec.collection.pad_main_batches_to_streaming_size,
            )
            logger.info(
                "batch_tokenized dataset=%s batch_index=%s seconds=%.3f padded_shape=%s real_batch=%s",
                dataset_name,
                batch_index,
                time.perf_counter() - encode_start,
                tuple(encoded["input_ids"].shape),
                encoded["real_batch_size"],
            )
            input_ids = encoded["input_ids"].to(self._model_input_device())
            attention_mask = encoded["attention_mask"].to(self._model_input_device())
            example_external = None if external_lookup is None else external_lookup[batch_examples[0].text_id]
            pass1_start = time.perf_counter()
            pass1_batched, propagate_hidden_states = self._run_pass(
                input_ids,
                attention_mask,
                calibration=all(example.calibration for example in batch_examples),
                pass_name="pass1",
                control_summary_mode=control_summary_mode,
                external_summary_by_layer=example_external,
            )
            logger.info(
                "batch_pass_done dataset=%s batch_index=%s pass=pass1 seconds=%.3f",
                dataset_name,
                batch_index,
                time.perf_counter() - pass1_start,
            )
            propagate_from_layer = int(self.spec.collection.propagate_from_layer)
            pass2_prefix_captures: dict[str, list[LayerCapture]] = {}
            for condition in (CaptureCondition.CAUSAL.value, CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value):
                prefix_captures = [
                    capture
                    for capture in pass1_batched.captures_by_condition.get(condition, [])
                    if capture.layer_idx < propagate_from_layer
                ]
                if prefix_captures:
                    pass2_prefix_captures[condition] = prefix_captures
            pass2_start = time.perf_counter()
            pass2_batched, _ = self._run_pass(
                input_ids,
                attention_mask,
                calibration=all(example.calibration for example in batch_examples),
                pass_name="pass2",
                control_summary_mode=control_summary_mode,
                external_summary_by_layer=example_external,
                start_layer=propagate_from_layer,
                initial_hidden_states=propagate_hidden_states,
                initial_captures_by_condition=pass2_prefix_captures,
            )
            logger.info(
                "batch_pass_done dataset=%s batch_index=%s pass=pass2 seconds=%.3f",
                dataset_name,
                batch_index,
                time.perf_counter() - pass2_start,
            )
            postprocess_start = time.perf_counter()
            batch_multi_slot = {}
            calibration_examples = [example for example in batch_examples if example.calibration]
            if calibration_examples:
                for example, multi_slot in zip(calibration_examples, self.collect_multi_slot_summaries_batch(calibration_examples)):
                    batch_multi_slot[example.text_id] = multi_slot
            needs_bundle_split = retain_bundles or not write_batches
            if needs_bundle_split:
                pass1_split = self._split_pass_capture(pass1_batched, batch_examples)
                pass2_split = self._split_pass_capture(pass2_batched, batch_examples)
            logger.info(
                "batch_postprocess_done dataset=%s batch_index=%s seconds=%.3f calibration_examples=%s split_bundles=%s",
                dataset_name,
                batch_index,
                time.perf_counter() - postprocess_start,
                len(calibration_examples),
                needs_bundle_split,
            )
            batch_bundles: list[ExampleCaptureBundle] = []
            if needs_bundle_split:
                for row_idx, example in enumerate(batch_examples):
                    bundle = ExampleCaptureBundle(
                        text_id=example.text_id,
                        dataset_name=dataset_name,
                        kind=example.kind,
                        prompt=example.prompt,
                        token_ids=list(example.prompt_token_ids or []),
                        content_token_mask=list(example.content_token_mask or []),
                        passes=[pass1_split[row_idx], pass2_split[row_idx]],
                        metadata={"calibration": example.calibration, "tags": list(example.tags)},
                    )
                    if example.calibration:
                        bundle.metadata["multi_slot_summaries"] = batch_multi_slot[example.text_id]
                    batch_bundles.append(bundle)
            batch_id = f"{dataset_name}_{control_summary_mode or 'main'}_{batch_index}_{len(batch_bundles) or len(batch_examples)}_{uuid.uuid4().hex[:8]}"
            transfer_event, staged_tensors, staged_groups, staged_bytes = self._finalize_transfer_session()
            if write_batches:
                extra_metadata = {
                    "dataset_name": dataset_name,
                    "control": control_summary_mode,
                    "runtime_stack": getattr(self, "runtime_stack", {}),
                    "model_contract": None if self.contract is None else self.contract.__dict__,
                    "batch_index": batch_index,
                    "batch_size": len(batch_examples),
                }
                if needs_bundle_split:
                    path = self.writer.write_batch(
                        batch_id,
                        batch_bundles,
                        extra_metadata=extra_metadata,
                        target_subdir=target,
                        transfer_event=transfer_event,
                        staged_tensors=staged_tensors,
                        staged_groups=staged_groups,
                        staged_bytes=staged_bytes,
                    )
                else:
                    payload = build_batched_capture_payload(
                        batch_id=batch_id,
                        dataset_name=dataset_name,
                        batch_examples=batch_examples,
                        passes=[
                            trim_pass_capture(pass1_batched, len(batch_examples)),
                            trim_pass_capture(pass2_batched, len(batch_examples)),
                        ],
                        extra_metadata=extra_metadata,
                        multi_slot_by_text_id=batch_multi_slot,
                    )
                    path = self.writer.write_payload(
                        batch_id,
                        payload,
                        row_count=len(batch_examples),
                        target_subdir=target,
                        transfer_event=transfer_event,
                        staged_tensors=staged_tensors,
                        staged_groups=staged_groups,
                        staged_bytes=staged_bytes,
                    )
                path_list.append(path)
            if not write_batches or retain_bundles:
                sync_start = time.perf_counter()
                if transfer_event is not None:
                    transfer_event.synchronize()
                logger.info(
                    "batch_transfer_synced dataset=%s batch_index=%s seconds=%.3f staged_tensors=%s staged_groups=%s staged_bytes=%s",
                    dataset_name,
                    batch_index,
                    time.perf_counter() - sync_start,
                    staged_tensors,
                    staged_groups,
                    staged_bytes,
                )
            if retain_bundles:
                all_bundles.extend(batch_bundles)
            logger.info(
                "batch_done dataset=%s batch_index=%s total_seconds=%.3f staged_tensors=%s staged_groups=%s staged_bytes=%s wrote_batch=%s",
                dataset_name,
                batch_index,
                time.perf_counter() - batch_start,
                staged_tensors,
                staged_groups,
                staged_bytes,
                write_batches,
            )
        logger.info(
            "collect_examples_done dataset=%s seconds=%.3f wrote_batches=%s retained_bundles=%s",
            dataset_name,
            time.perf_counter() - collect_start,
            len(path_list),
            len(all_bundles),
        )
        return path_list, all_bundles

    def run_controls(self, records: list[dict[str, Any]], dataset_name: str = "custom_controls"):
        outputs = {}
        summary_mode = self.spec.collection.controls_storage_mode == "summary_only"
        controls_root = self.root / self.spec.output.controls_dir
        controls_root.mkdir(parents=True, exist_ok=True)

        def _emit_control_result(control_name: str, paths: list[Path], bundles: list[ExampleCaptureBundle]):
            if summary_mode:
                summary_path = controls_root / f"{control_name}_summary.json"
                summary = summarize_bundles(bundles)
                summary["control_name"] = control_name
                summary["dataset_name"] = dataset_name
                summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                outputs[control_name] = {"summary": str(summary_path), "storage_mode": "summary_only"}
            else:
                outputs[control_name] = {"captures": [str(path) for path in paths], "storage_mode": "full"}

        for control_mode in ("first_token", "random_token", "k_only", "v_only", "k_only_zero", "v_only_zero"):
            paths, bundles = self.collect_examples(
                records,
                dataset_name=dataset_name,
                control_summary_mode=control_mode,
                target_subdir=self.spec.output.controls_dir,
                write_batches=not summary_mode,
            )
            _emit_control_result(control_mode, paths, bundles)
        shuffled_records = [{**record, "text": self._shuffle_text(str(record["text"]))} for record in records]
        shuffled_summaries = self._compute_external_summaries(
            shuffled_records,
            dataset_name=f"{dataset_name}_shuffled_sentence_source",
        )
        paths, bundles = self.collect_examples(
            records,
            dataset_name=f"{dataset_name}_shuffled_sentence",
            control_summary_mode=None,
            external_summaries=shuffled_summaries,
            target_subdir=self.spec.output.controls_dir,
            write_batches=not summary_mode,
        )
        _emit_control_result("shuffled_sentence", paths, bundles)
        base_summaries = self._compute_external_summaries(
            records,
            dataset_name=f"{dataset_name}_cross_example_source",
        )
        cross_example_summaries = base_summaries[1:] + base_summaries[:1] if base_summaries else []
        paths, bundles = self.collect_examples(
            records,
            dataset_name=f"{dataset_name}_cross_example",
            control_summary_mode=None,
            external_summaries=cross_example_summaries,
            target_subdir=self.spec.output.controls_dir,
            write_batches=not summary_mode,
        )
        _emit_control_result("cross_example", paths, bundles)
        continuation_summaries = self._compute_prompt_continuation_summaries(records)
        paths, bundles = self.collect_examples(
            records,
            dataset_name=f"{dataset_name}_prompt_continuation",
            control_summary_mode=None,
            external_summaries=continuation_summaries,
            target_subdir=self.spec.output.controls_dir,
            write_batches=not summary_mode,
        )
        _emit_control_result("prompt_continuation", paths, bundles)
        return outputs

    def run_bridge_echo(self, records: list[dict[str, Any]], dataset_name: str = "bridge_echo"):
        summary_mode = self.spec.collection.bridge_storage_mode == "summary_only"
        bridge_root = self.root / self.spec.output.bridge_dir
        bridge_root.mkdir(parents=True, exist_ok=True)
        bridge_records = []
        for record in records:
            kind = str(record["kind"])
            text = str(record["text"])
            echo_text = text + self.spec.prompts.bridge_separator + text
            bridge_records.append({**record, "text": echo_text, "kind": kind})
        paths, bundles = self.collect_examples(
            bridge_records,
            dataset_name=dataset_name,
            target_subdir=self.spec.output.bridge_dir,
            write_batches=not summary_mode,
        )
        if summary_mode:
            summary_path = bridge_root / f"{dataset_name}_summary.json"
            summary = summarize_bundles(bundles)
            summary["dataset_name"] = dataset_name
            summary["storage_mode"] = "summary_only"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            return [summary_path], bundles
        return paths, bundles

    def save_spec_snapshot(self):
        snapshot_path = self.root / "artifacts" / "experiment_spec.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(json.dumps(self.spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        (self.root / "artifacts" / "runtime_stack.json").write_text(
            json.dumps(getattr(self, "runtime_stack", {}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return snapshot_path

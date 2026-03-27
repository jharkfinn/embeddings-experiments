from __future__ import annotations

import hashlib
import json
import queue
import threading
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .prepend import attention_forward
from .types import CaptureCondition, ExampleCaptureBundle, LayerCapture, PassCapture


def _storage_dtype(role: str):
    if role == "top_k_indices":
        return torch.int8
    return getattr(torch, "float8_e4m3fn", torch.bfloat16)


def _to_cpu(tensor, dtype=None):
    if tensor is None:
        return None
    out = tensor.detach().cpu()
    if dtype is not None:
        out = out.to(dtype=dtype)
    return out


def _causal_attention_mask(attention_mask, batch_size: int, seq_len: int, device, dtype):
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((batch_size, 1, seq_len, seq_len), neg_inf, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)
        pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * neg_inf
        mask = mask + pad
    return mask


class _AsyncWriter:
    def __init__(self):
        self._queue: queue.Queue[tuple[Path, dict[str, Any]] | None] = queue.Queue(maxsize=4)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._writer_loop, name="kv-prepend-vllm-main-writer", daemon=True)
        self._thread.start()

    def _writer_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, path)
            except BaseException as exc:  # pragma: no cover - async path
                self._error = exc
            finally:
                self._queue.task_done()

    def write(self, path: Path, payload: dict[str, Any]):
        if self._error is not None:
            raise RuntimeError("Background vLLM writer failed.") from self._error
        self._queue.put((path, payload))

    def flush(self):
        self._queue.join()
        if self._error is not None:
            raise RuntimeError("Background vLLM writer failed.") from self._error

    def close(self):
        self.flush()
        self._queue.put(None)
        self._thread.join()


class VLLMMainCaptureExtension:
    @staticmethod
    def _get_text_model(model):
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            return model.language_model.model
        if hasattr(model, "model"):
            return model.model
        raise AttributeError(f"Unsupported vLLM model wrapper: {type(model)!r}")

    def setup_hooks(self, config_json: str):
        config = json.loads(config_json)
        self.runtime_config = config
        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._cached_segments = None
        self._cached_segment_signature = None
        self._handles = []
        self._hooks_enabled = True
        self._writer = _AsyncWriter()

        model = self.model_runner.model
        text_model = self._get_text_model(model)
        layers = list(text_model.layers)
        self._num_layers = len(layers)

        for layer_idx, layer in enumerate(layers):
            def _resid_pre_hook(li):
                def hook(_mod, inp):
                    if not getattr(self, "_hooks_enabled", True):
                        return None
                    hidden = inp[0]
                    segments = self._get_prompt_segments(hidden)
                    self._append_segments(segments, f"resid_{li}", hidden)
                    return None
                return hook

            self._handles.append(layer.input_layernorm.register_forward_pre_hook(_resid_pre_hook(layer_idx)))

        return f"Installed {len(self._handles)} vLLM main-run hooks"

    def get_architecture(self):
        model = self.model_runner.model
        text_model = self._get_text_model(model)
        config = text_model.config
        return json.dumps(
            {
                "num_layers": int(getattr(config, "num_hidden_layers")),
                "hidden_size": int(getattr(config, "hidden_size")),
                "num_attention_heads": int(getattr(config, "num_attention_heads")),
                "num_key_value_heads": int(getattr(config, "num_key_value_heads")),
                "num_experts": int(getattr(config, "num_experts")),
                "num_experts_per_tok": int(getattr(config, "num_experts_per_tok")),
            }
        )

    def clear_capture(self):
        self.capture.clear()
        self.req_prompt_token_ids.clear()
        self.req_capture_specs.clear()
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._cached_segments = None
        self._cached_segment_signature = None
        self._hooks_enabled = True
        return True

    @staticmethod
    def _prompt_key(token_ids):
        arr = torch.as_tensor(token_ids, dtype=torch.int32).cpu().numpy()
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def prepare_batch(self, batch_payload_json: str):
        batch_payload = json.loads(batch_payload_json)
        prompt_key_to_records = {}
        for record in batch_payload["records"]:
            prompt_key_to_records.setdefault(record["prompt_key"], deque()).append(dict(record))
        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._pending_batch_records = prompt_key_to_records
        self._active_batch_payload = batch_payload
        self._cached_segments = None
        self._cached_segment_signature = None
        return len(batch_payload["records"])

    @staticmethod
    def _buffer_to_numpy(buffer):
        if hasattr(buffer, "np"):
            return buffer.np
        if hasattr(buffer, "cpu"):
            return buffer.cpu()
        if isinstance(buffer, torch.Tensor):
            return buffer.detach().cpu().numpy()
        return buffer

    def _get_prompt_token_ids(self, req_index, prompt_len):
        token_ids = self.model_runner.input_batch.token_ids_cpu[req_index, :prompt_len]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        return token_ids.copy()

    def _get_prompt_segments(self, tensor):
        total_tokens = tensor.shape[0]
        input_batch = self.model_runner.input_batch
        req_ids = list(input_batch.req_ids)
        num_reqs = int(getattr(input_batch, "num_reqs", len(req_ids)))
        req_ids = req_ids[:num_reqs]
        query_start_loc = self._buffer_to_numpy(self.model_runner.query_start_loc)[: num_reqs + 1]
        expected_tokens = int(query_start_loc[-1]) if len(query_start_loc) else 0
        if total_tokens != expected_tokens:
            raise RuntimeError(f"Hook tensor token count mismatch: got {total_tokens}, expected {expected_tokens}")

        num_computed_tokens = self._buffer_to_numpy(input_batch.num_computed_tokens_cpu[:num_reqs])
        num_prompt_tokens = self._buffer_to_numpy(input_batch.num_prompt_tokens[:num_reqs])
        signature = (
            total_tokens,
            tuple(req_ids),
            tuple(query_start_loc.tolist()),
            tuple(num_computed_tokens.tolist()),
            tuple(num_prompt_tokens.tolist()),
        )
        if self._cached_segment_signature == signature:
            return self._cached_segments

        segments = []
        for req_index, req_id in enumerate(req_ids):
            if req_id is None:
                continue
            prompt_len = int(num_prompt_tokens[req_index])
            if req_id not in self.req_prompt_token_ids:
                prompt_ids = self._get_prompt_token_ids(req_index, prompt_len)
                self.req_prompt_token_ids[req_id] = prompt_ids
                prompt_key = self._prompt_key(prompt_ids)
                pending = self._pending_batch_records.get(prompt_key)
                if not pending:
                    raise RuntimeError(f"No prepared batch record matched prompt_key={prompt_key}")
                self.req_capture_specs[req_id] = pending.popleft()

            start = int(query_start_loc[req_index])
            end = int(query_start_loc[req_index + 1])
            scheduled_tokens = end - start
            prompt_tokens_done = int(num_computed_tokens[req_index])
            remaining_prompt = max(0, prompt_len - prompt_tokens_done)
            prompt_part_len = min(scheduled_tokens, remaining_prompt)
            if prompt_part_len <= 0:
                continue
            local_start = start
            local_end = start + prompt_part_len
            segments.append((req_id, local_start, local_end))

        self._cached_segments = segments
        self._cached_segment_signature = signature
        return segments

    def _append_segments(self, segments, field_name, tensor):
        for req_id, start, end in segments:
            req_capture = self.capture.setdefault(req_id, {})
            field_tensor = tensor[start:end].detach()
            field_entry = req_capture.get(field_name)
            if field_entry is None:
                req_info = self.req_capture_specs.get(req_id)
                if req_info is None:
                    raise RuntimeError(f"Missing capture metadata for req_id={req_id}")
                seq_len = int(req_info["prompt_len"])
                field_entry = {
                    "tensor": torch.empty((seq_len, *field_tensor.shape[1:]), dtype=field_tensor.dtype, device=field_tensor.device),
                    "offset": 0,
                }
                req_capture[field_name] = field_entry
            offset = int(field_entry["offset"])
            next_offset = offset + int(field_tensor.shape[0])
            if next_offset > field_entry["tensor"].shape[0]:
                raise RuntimeError(f"Capture overflow for req_id={req_id}, field={field_name}")
            field_entry["tensor"][offset:next_offset].copy_(field_tensor)
            field_entry["offset"] = next_offset

    def _snapshot_live_state(self, reset: bool):
        if self._active_batch_payload is None:
            raise RuntimeError("No prepared batch is active")
        snapshot = {
            "batch_id": self._active_batch_payload["batch_id"],
            "output_path": self._active_batch_payload["output_path"],
            "records": [dict(record) for record in self._active_batch_payload["records"]],
            "capture": self.capture,
            "req_prompt_token_ids": self.req_prompt_token_ids,
            "req_capture_specs": self.req_capture_specs,
        }
        if reset:
            self.capture = {}
            self.req_prompt_token_ids = {}
            self.req_capture_specs = {}
            self._pending_batch_records = {}
            self._active_batch_payload = None
            self._cached_segments = None
            self._cached_segment_signature = None
        return snapshot

    def _sorted_req_infos(self, snapshot):
        matched = sorted(snapshot["req_capture_specs"].items(), key=lambda item: int(item[1]["record_index"]))
        return [(req_id, info) for req_id, info in matched]

    def _pad_hidden_batch(self, tensors):
        seq_lens = [int(t.shape[0]) for t in tensors]
        max_len = max(seq_lens) if seq_lens else 0
        hidden_dim = int(tensors[0].shape[-1]) if tensors else 0
        if tensors:
            device = tensors[0].device
            dtype = tensors[0].dtype
        else:
            text_model = self._get_text_model(self.model_runner.model)
            device = text_model.embed_tokens.weight.device
            dtype = text_model.embed_tokens.weight.dtype
        batch = torch.zeros((len(tensors), max_len, hidden_dim), device=device, dtype=dtype)
        attention_mask = torch.zeros((len(tensors), max_len), device=device, dtype=torch.long)
        for idx, tensor in enumerate(tensors):
            seq_len = tensor.shape[0]
            batch[idx, :seq_len] = tensor
            attention_mask[idx, :seq_len] = 1
        return batch, attention_mask, seq_lens

    def _project_qkv(self, attention_module, hidden_states, position_embeddings):
        batch_size, seq_len, _ = hidden_states.shape
        qkv, _ = attention_module.qkv_proj(hidden_states)
        q, k, v = qkv.split([attention_module.q_size, attention_module.kv_size, attention_module.kv_size], dim=-1)

        q_by_head = q.view(batch_size, seq_len, attention_module.num_heads, attention_module.head_dim)
        q_by_head = attention_module.q_norm(q_by_head)
        q_normed = q_by_head.view(batch_size, seq_len, attention_module.q_size)

        k_by_head = k.view(batch_size, seq_len, attention_module.num_kv_heads, attention_module.head_dim)
        k_by_head = attention_module.k_norm(k_by_head)
        k_normed = k_by_head.view(batch_size, seq_len, attention_module.kv_size)

        q_rot_flat, k_rot_flat = attention_module.rotary_emb(position_embeddings, q_normed, k_normed)

        q_pre = q_by_head.transpose(1, 2).contiguous()
        k_pre = k_by_head.transpose(1, 2).contiguous()
        q_rot = q_rot_flat.view(batch_size, seq_len, attention_module.num_heads, attention_module.head_dim).transpose(1, 2).contiguous()
        k_rot = k_rot_flat.view(batch_size, seq_len, attention_module.num_kv_heads, attention_module.head_dim).transpose(1, 2).contiguous()
        v_raw = v.view(batch_size, seq_len, attention_module.num_kv_heads, attention_module.head_dim).transpose(1, 2).contiguous()
        return q_pre, k_pre, q_rot, k_rot, v_raw, q_rot_flat, k_rot_flat, v

    @staticmethod
    def _num_key_value_groups(attention_module) -> int:
        if hasattr(attention_module, "num_key_value_groups"):
            return int(attention_module.num_key_value_groups)
        return int(attention_module.num_heads // attention_module.num_kv_heads)

    @staticmethod
    def _o_proj(attention_module, attn_output):
        projected, _ = attention_module.o_proj(attn_output)
        return projected

    def _run_moe(self, mlp_module, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        if not hasattr(mlp_module, "gate") or not hasattr(mlp_module, "experts"):
            mlp_output = mlp_module(flat).reshape(batch_size, seq_len, hidden_dim)
            return mlp_output, None, None
        router_logits, _ = mlp_module.gate(flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k = int(getattr(mlp_module.experts, "top_k", 0) or 0)
        top_k_indices = torch.topk(router_probs, top_k, dim=-1).indices if top_k > 0 else None
        shared_out, fused_out = mlp_module.experts(hidden_states=flat, router_logits=router_logits)
        mlp_output = (shared_out + fused_out if shared_out is not None else fused_out).reshape(batch_size, seq_len, hidden_dim)
        router_logits = router_logits.reshape(batch_size, seq_len, -1)
        if top_k_indices is not None:
            top_k_indices = top_k_indices.reshape(batch_size, seq_len, -1).to(dtype=torch.int8)
        return mlp_output, router_logits, top_k_indices

    def _run_router_only(self, mlp_module, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_dim)
        if not hasattr(mlp_module, "gate"):
            return None, None
        router_logits, _ = mlp_module.gate(flat)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        top_k = int(getattr(mlp_module.experts, "top_k", 0) or 0) if hasattr(mlp_module, "experts") else 0
        top_k_indices = torch.topk(router_probs, top_k, dim=-1).indices if top_k > 0 else None
        router_logits = router_logits.reshape(batch_size, seq_len, -1)
        if top_k_indices is not None:
            top_k_indices = top_k_indices.reshape(batch_size, seq_len, -1).to(dtype=torch.int8)
        return router_logits, top_k_indices

    def _signal_enabled_for_layer(self, signal_name: str, layer_idx: int, config: dict[str, Any]) -> bool:
        signals = set(config["main_capture_signals"])
        if signal_name not in signals:
            return False
        if signal_name in {"router_logits", "top_k_binary"}:
            return layer_idx in set(config["main_router_layers"])
        return layer_idx in set(config["main_dense_layers"])

    def _build_layer_capture(self, layer_idx: int, condition: str, config: dict[str, Any], *, z_attn, h_pre_moe, router_logits, top_k_indices):
        return LayerCapture(
            layer_idx=layer_idx,
            condition=condition,
            z_attn=_to_cpu(
                z_attn if self._signal_enabled_for_layer("attention_output", layer_idx, config) else None,
                dtype=_storage_dtype("z_attn"),
            ),
            h_pre_moe=_to_cpu(
                h_pre_moe if self._signal_enabled_for_layer("pre_moe", layer_idx, config) else None,
                dtype=_storage_dtype("h_pre_moe"),
            ),
            router_logits_pre_softmax=_to_cpu(
                router_logits if self._signal_enabled_for_layer("router_logits", layer_idx, config) else None,
                dtype=_storage_dtype("router_logits_pre_softmax"),
            ),
            top_k_indices=_to_cpu(
                top_k_indices if self._signal_enabled_for_layer("top_k_binary", layer_idx, config) else None,
                dtype=_storage_dtype("top_k_indices"),
            ),
        )

    def _compute_pass1(self, layers, position_embeddings, causal_inputs, config):
        outputs = {
            CaptureCondition.CAUSAL.value: [],
            CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value: [],
        }
        for layer_idx, layer in enumerate(layers):
            hidden_states = causal_inputs[layer_idx]
            attention_mask = causal_inputs["attention_mask"]
            layer.self_attn.num_key_value_groups = self._num_key_value_groups(layer.self_attn)
            q_pre, k_pre, q_rot, k_rot, v_raw, q_rot_flat, k_rot_flat, v_flat = self._project_qkv(
                layer.self_attn,
                layer.input_layernorm(hidden_states),
                position_embeddings,
            )
            causal_attn = layer.self_attn.attn(q_rot_flat, k_rot_flat, v_flat)
            causal_z = self._o_proj(layer.self_attn, causal_attn)
            causal_hidden = hidden_states + causal_z
            causal_pre_moe = layer.post_attention_layernorm(causal_hidden)
            causal_router_logits, causal_topk = self._run_router_only(layer.mlp, causal_pre_moe)
            if any(self._signal_enabled_for_layer(sig, layer_idx, config) for sig in config["main_capture_signals"]):
                outputs[CaptureCondition.CAUSAL.value].append(
                    self._build_layer_capture(
                        layer_idx,
                        CaptureCondition.CAUSAL.value,
                        config,
                        z_attn=causal_z,
                        h_pre_moe=causal_pre_moe,
                        router_logits=causal_router_logits,
                        top_k_indices=causal_topk,
                    )
                )
                prepend = attention_forward(
                    layer.self_attn,
                    q_rot,
                    k_rot,
                    v_raw,
                    attention_mask,
                    prepend_mode=config["default_rope_mode"],
                    key_pre_rope=k_pre,
                    return_weights=False,
                )
                prepend_z = self._o_proj(layer.self_attn, prepend.attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
                prepend_hidden = hidden_states + prepend_z
                prepend_pre_moe = layer.post_attention_layernorm(prepend_hidden)
                prepend_router_logits, prepend_topk = self._run_router_only(layer.mlp, prepend_pre_moe)
                outputs[CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value].append(
                    self._build_layer_capture(
                        layer_idx,
                        CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value,
                        config,
                        z_attn=prepend_z,
                        h_pre_moe=prepend_pre_moe,
                        router_logits=prepend_router_logits,
                        top_k_indices=prepend_topk,
                    )
                )
        return outputs

    def _compute_pass2(self, layers, position_embeddings, hidden_states, attention_mask, config):
        outputs = {
            CaptureCondition.PROPAGATED.value: [],
            CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE.value: [],
        }
        for layer_idx in range(int(config["propagate_from_layer"]), len(layers)):
            layer = layers[layer_idx]
            layer.self_attn.num_key_value_groups = self._num_key_value_groups(layer.self_attn)
            q_pre, k_pre, q_rot, k_rot, v_raw, q_rot_flat, k_rot_flat, v_flat = self._project_qkv(
                layer.self_attn,
                layer.input_layernorm(hidden_states),
                position_embeddings,
            )
            prepend = attention_forward(
                layer.self_attn,
                q_rot,
                k_rot,
                v_raw,
                attention_mask,
                prepend_mode=config["default_rope_mode"],
                key_pre_rope=k_pre,
                return_weights=False,
            )
            prepend_z = self._o_proj(layer.self_attn, prepend.attn_output.reshape(*hidden_states.shape[:-1], -1).contiguous())
            prepend_hidden = hidden_states + prepend_z
            prepend_pre_moe = layer.post_attention_layernorm(prepend_hidden)
            prepend_mlp_out, prepend_router_logits, prepend_topk = self._run_moe(layer.mlp, prepend_pre_moe)
            propagated_next = prepend_hidden + prepend_mlp_out

            causal_attn = layer.self_attn.attn(q_rot_flat, k_rot_flat, v_flat)
            causal_z = self._o_proj(layer.self_attn, causal_attn)
            causal_hidden = hidden_states + causal_z
            causal_pre_moe = layer.post_attention_layernorm(causal_hidden)
            causal_router_logits, causal_topk = self._run_router_only(layer.mlp, causal_pre_moe)

            if any(self._signal_enabled_for_layer(sig, layer_idx, config) for sig in config["main_capture_signals"]):
                outputs[CaptureCondition.PROPAGATED.value].append(
                    self._build_layer_capture(
                        layer_idx,
                        CaptureCondition.PROPAGATED.value,
                        config,
                        z_attn=prepend_z,
                        h_pre_moe=prepend_pre_moe,
                        router_logits=prepend_router_logits,
                        top_k_indices=prepend_topk,
                    )
                )
                outputs[CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE.value].append(
                    self._build_layer_capture(
                        layer_idx,
                        CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE.value,
                        config,
                        z_attn=causal_z,
                        h_pre_moe=causal_pre_moe,
                        router_logits=causal_router_logits,
                        top_k_indices=causal_topk,
                    )
                )
            hidden_states = propagated_next
        return outputs

    def _build_bundles(self, snapshot):
        config = self.runtime_config
        text_model = self._get_text_model(self.model_runner.model)
        layers = list(text_model.layers)
        matched = self._sorted_req_infos(snapshot)
        req_ids = [req_id for req_id, _ in matched]
        infos = [info for _, info in matched]
        residual_batches = []
        for layer_idx in range(len(layers)):
            tensors = []
            for req_id in req_ids:
                entry = snapshot["capture"][req_id][f"resid_{layer_idx}"]
                tensors.append(entry["tensor"])
            residual_batches.append(tensors)

        input_batch, attention_mask, seq_lens = self._pad_hidden_batch(residual_batches[0])
        del input_batch  # shape only
        first_hidden = residual_batches[0][0]
        position_ids = torch.arange(attention_mask.shape[1], device=first_hidden.device).unsqueeze(0).expand(len(req_ids), -1)
        causal_mask = _causal_attention_mask(attention_mask, len(req_ids), attention_mask.shape[1], first_hidden.device, first_hidden.dtype)

        causal_inputs = {"attention_mask": causal_mask}
        for layer_idx in range(len(layers)):
            batch_hidden, _, _ = self._pad_hidden_batch(residual_batches[layer_idx])
            causal_inputs[layer_idx] = batch_hidden

        prev_hooks_enabled = getattr(self, "_hooks_enabled", True)
        self._hooks_enabled = False
        try:
            pass1_outputs = self._compute_pass1(layers, position_ids, causal_inputs, config)
            propagated_seed = causal_inputs[int(config["propagate_from_layer"])]
            pass2_outputs = self._compute_pass2(layers, position_ids, propagated_seed, causal_mask, config)
        finally:
            self._hooks_enabled = prev_hooks_enabled

        bundles = []
        for row_idx, info in enumerate(infos):
            seq_len = seq_lens[row_idx]
            def _slice_capture(capture: LayerCapture):
                return LayerCapture(
                    layer_idx=capture.layer_idx,
                    condition=capture.condition,
                    z_attn=None if capture.z_attn is None else capture.z_attn[row_idx : row_idx + 1, :seq_len],
                    h_pre_moe=None if capture.h_pre_moe is None else capture.h_pre_moe[row_idx : row_idx + 1, :seq_len],
                    router_logits_pre_softmax=None
                    if capture.router_logits_pre_softmax is None
                    else capture.router_logits_pre_softmax[row_idx : row_idx + 1, :seq_len],
                    top_k_indices=None if capture.top_k_indices is None else capture.top_k_indices[row_idx : row_idx + 1, :seq_len],
                )

            pass1 = PassCapture(
                pass_name="pass1",
                rope_mode=config["default_rope_mode"],
                captures_by_condition={
                    condition: [_slice_capture(capture) for capture in captures]
                    for condition, captures in pass1_outputs.items()
                },
            )
            pass2 = PassCapture(
                pass_name="pass2",
                rope_mode=config["default_rope_mode"],
                captures_by_condition={
                    condition: [_slice_capture(capture) for capture in captures]
                    for condition, captures in pass2_outputs.items()
                },
            )
            bundles.append(
                ExampleCaptureBundle(
                    text_id=str(info["text_id"]),
                    dataset_name=str(info["dataset_name"]),
                    kind=str(info["kind"]),
                    prompt=str(info["prompt"]),
                    token_ids=list(info["prompt_token_ids"]),
                    content_token_mask=list(info["content_token_mask"]),
                    passes=[pass1, pass2],
                    metadata={"calibration": False, "tags": list(info.get("tags", [])), "runtime_backend": "vllm"},
                )
            )
        return bundles

    def transfer_and_save_async(self):
        snapshot = self._snapshot_live_state(reset=True)
        bundles = self._build_bundles(snapshot)
        output_path = Path(snapshot["output_path"])
        payload = {
            "schema_version": 2,
            "batch_id": snapshot["batch_id"],
            "metadata": {"runtime_backend": "vllm", "num_bundles": len(bundles)},
            "bundles": bundles,
        }
        self._writer.write(output_path, payload)
        return len(bundles)

    def flush_saves(self):
        self._writer.flush()
        return True

    def remove_hooks(self):
        self._hooks_enabled = False
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._writer.close()
        return True

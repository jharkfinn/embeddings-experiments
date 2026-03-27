"""vLLM worker extension for notebook-style binary shard extraction."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import tempfile
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class SignalExtractorExtension:
    def setup_hooks(self, config_json):
        config = json.loads(config_json)
        if "arch" in config:
            self.arch = config["arch"]
            options = config.get("options", {})
        else:
            self.arch = config
            options = {}

        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._req_prompt_keys = {}
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._handles = []
        self._save_error = None
        self._save_error_lock = threading.Lock()
        self._cached_segments = None
        self._cached_segment_signature = None
        self._router_full_threshold = float(
            options.get("router_full_threshold", 1.0 / self.arch["num_experts"])
        )

        self._write_queue = queue.Queue(
            maxsize=max(1, int(options.get("write_queue_size", 4)))
        )
        self._writer_thread = threading.Thread(
            target=self._disk_writer_loop,
            daemon=True,
            name="binary-shard-writer",
        )
        self._writer_thread.start()

        model = self.model_runner.model
        text_model = model.language_model.model
        layers = list(text_model.layers)

        num_q_heads = self.arch["num_attention_heads"]
        num_kv_heads = self.arch["num_key_value_heads"]
        head_dim = self.arch["head_dim"]
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
                if gate is not None:

                    def _router_hook(li):
                        def hook(_mod, _inp, out):
                            logits = out[0] if isinstance(out, tuple) else out
                            probs = F.softmax(logits.float(), dim=-1)
                            ids = probs.topk(9, dim=-1).indices.to(torch.int16)
                            full_mask = (probs > self._router_full_threshold).to(torch.uint8)
                            self._append_segments(f"router_full_{li}", full_mask)
                            self._append_segments(f"router_topk_ids_{li}", ids)

                        return hook

                    self._handles.append(gate.register_forward_hook(_router_hook(layer_idx)))

            if layer_idx in self.arch["attn_layer_indices"]:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    qkv_proj = getattr(attn, "qkv_proj", None)
                    if qkv_proj is not None:

                        def _qkv_hook(li, qd, kvd, nhq, hd):
                            def hook(_mod, _inp, out):
                                tensor = out[0] if isinstance(out, tuple) else out
                                q_gate, k, v = tensor.split([qd * 2, kvd, kvd], dim=-1)
                                shape = q_gate.shape[:-1]
                                q_gate = q_gate.view(*shape, nhq, hd * 2)
                                q = q_gate[..., :hd].reshape(*shape, qd)
                                self._append_segments(f"q_{li}", q)
                                self._append_segments(f"k_{li}", k)
                                self._append_segments(f"v_{li}", v)

                            return hook

                        self._handles.append(
                            qkv_proj.register_forward_hook(
                                _qkv_hook(layer_idx, q_dim, kv_dim, num_q_heads, head_dim)
                            )
                        )

                    o_proj = getattr(attn, "o_proj", None)
                    if o_proj is not None:

                        def _oproj_hook(li):
                            def hook(_mod, _inp, out):
                                tensor = out[0] if isinstance(out, tuple) else out
                                self._append_segments(f"attn_out_{li}", tensor)

                            return hook

                        self._handles.append(o_proj.register_forward_hook(_oproj_hook(layer_idx)))

            if layer_idx == len(layers) - 1:

                def _final_hook(_li):
                    def hook(_mod, _inp, out):
                        hidden_states = out[0] if isinstance(out, tuple) else out
                        self._append_segments("hs_final", hidden_states)

                    return hook

                self._handles.append(layer.register_forward_hook(_final_hook(layer_idx)))

        return f"Installed {len(self._handles)} hooks on {len(layers)} layers"

    @staticmethod
    def _buffer_to_numpy(buffer):
        if buffer is None:
            return None
        if hasattr(buffer, "np"):
            return np.asarray(buffer.np)
        if hasattr(buffer, "cpu"):
            return np.asarray(buffer.cpu())
        if isinstance(buffer, torch.Tensor):
            return buffer.detach().cpu().numpy()
        return np.asarray(buffer)

    @staticmethod
    def _prompt_key(token_ids):
        arr = np.asarray(token_ids, dtype=np.int32)
        return hashlib.sha1(arr.tobytes()).hexdigest()

    def _get_prompt_token_ids(self, req_index, prompt_len):
        token_ids = self.model_runner.input_batch.token_ids_cpu[req_index, :prompt_len]
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().numpy()
        return np.asarray(token_ids, dtype=np.int32).copy()

    def _get_prompt_segments(self, tensor):
        total_tokens = tensor.shape[0]
        input_batch = self.model_runner.input_batch
        req_ids = list(input_batch.req_ids)
        num_reqs = int(getattr(input_batch, "num_reqs", len(req_ids)))
        req_ids = req_ids[:num_reqs]

        query_start_loc = getattr(self.model_runner, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(input_batch, "query_start_loc_cpu", None)
        if query_start_loc is None:
            query_start_loc = getattr(input_batch, "query_start_loc", None)
        query_start_loc = self._buffer_to_numpy(query_start_loc)
        if query_start_loc is None:
            raise RuntimeError("query_start_loc is unavailable on this vLLM build")
        query_start_loc = query_start_loc[: num_reqs + 1]
        expected_tokens = int(query_start_loc[-1]) if len(query_start_loc) else 0
        if expected_tokens and total_tokens != expected_tokens:
            raise RuntimeError(
                f"Hook tensor token count mismatch: got {total_tokens}, expected {expected_tokens}"
            )

        num_computed_tokens = getattr(input_batch, "num_computed_tokens_cpu", None)
        if num_computed_tokens is None:
            num_computed_tokens = getattr(input_batch, "num_computed_tokens", None)
        num_computed_tokens = self._buffer_to_numpy(num_computed_tokens)
        if num_computed_tokens is None:
            num_computed_tokens = np.zeros(num_reqs, dtype=np.int64)
        num_computed_tokens = np.asarray(num_computed_tokens[:num_reqs], dtype=np.int64)

        num_prompt_tokens = self._buffer_to_numpy(getattr(input_batch, "num_prompt_tokens", None))
        if num_prompt_tokens is None:
            raise RuntimeError("num_prompt_tokens is unavailable on this vLLM build")
        num_prompt_tokens = np.asarray(num_prompt_tokens[:num_reqs], dtype=np.int64)

        signature = (
            total_tokens,
            tuple(req_ids),
            tuple(np.asarray(query_start_loc, dtype=np.int64).tolist()),
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
                self._req_prompt_keys[req_id] = prompt_key
                pending = self._pending_batch_records.get(prompt_key)
                if not pending:
                    raise RuntimeError(
                        f"No prepared batch record matched prompt_key={prompt_key}"
                    )
                self.req_capture_specs[req_id] = pending.popleft()

            req_info = self.req_capture_specs.get(req_id)
            if req_info is None:
                raise RuntimeError(f"Missing capture metadata for req_id={req_id}")

            start = int(query_start_loc[req_index])
            end = int(query_start_loc[req_index + 1])
            scheduled_tokens = end - start
            prompt_tokens_done = int(num_computed_tokens[req_index])
            remaining_prompt = max(0, prompt_len - prompt_tokens_done)
            prompt_part_len = min(scheduled_tokens, remaining_prompt)
            if prompt_part_len <= 0:
                continue

            chunk_prompt_start = prompt_tokens_done
            echo_start = int(req_info["echo_start"])
            echo_end = int(req_info["echo_end"])
            capture_prompt_start = max(chunk_prompt_start, echo_start)
            capture_prompt_end = min(chunk_prompt_start + prompt_part_len, echo_end)
            if capture_prompt_start >= capture_prompt_end:
                continue

            local_start = start + (capture_prompt_start - chunk_prompt_start)
            local_end = local_start + (capture_prompt_end - capture_prompt_start)
            capture_offset = capture_prompt_start - echo_start
            segments.append((req_id, local_start, local_end, capture_offset))

        self._cached_segments = segments
        self._cached_segment_signature = signature
        return segments

    def _append_segments(self, field_name, tensor):
        segments = self._get_prompt_segments(tensor)
        for req_id, start, end, capture_offset in segments:
            req_capture = self.capture.setdefault(req_id, {})
            field_tensor = tensor[start:end].detach()
            field_entry = req_capture.get(field_name)
            if field_entry is None:
                req_info = self.req_capture_specs.get(req_id)
                if req_info is None:
                    raise RuntimeError(f"Missing capture metadata for req_id={req_id}")
                seq_len = int(req_info["echo_end"]) - int(req_info["echo_start"])
                field_entry = {
                    "tensor": torch.empty(
                        (seq_len, *field_tensor.shape[1:]),
                        dtype=field_tensor.dtype,
                        device=field_tensor.device,
                    ),
                    "filled": torch.zeros(seq_len, dtype=torch.bool, device=field_tensor.device),
                }
                req_capture[field_name] = field_entry

            next_offset = capture_offset + int(field_tensor.shape[0])
            if next_offset > field_entry["tensor"].shape[0]:
                raise RuntimeError(
                    f"Capture overflow for req_id={req_id}, field={field_name}: "
                    f"{next_offset} > {field_entry['tensor'].shape[0]}"
                )
            if bool(field_entry["filled"][capture_offset:next_offset].any().item()):
                raise RuntimeError(
                    f"Overlapping capture for req_id={req_id}, field={field_name}"
                )
            field_entry["tensor"][capture_offset:next_offset].copy_(field_tensor)
            field_entry["filled"][capture_offset:next_offset] = True

    def clear_capture(self):
        self.capture.clear()
        self.req_prompt_token_ids.clear()
        self.req_capture_specs.clear()
        self._req_prompt_keys.clear()
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._cached_segments = None
        self._cached_segment_signature = None
        return True

    def prepare_batch(self, batch_info_json):
        info = json.loads(batch_info_json)
        prompt_key_to_records = {}
        for record in info["records"]:
            prompt_key_to_records.setdefault(record["prompt_key"], deque()).append(dict(record))
        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._req_prompt_keys = {}
        self._pending_batch_records = prompt_key_to_records
        self._active_batch_payload = info
        self._cached_segments = None
        self._cached_segment_signature = None
        return len(info["records"])

    def _materialize_capture(self, req_id):
        result = {}
        capture = self.capture.get(req_id, {})
        for field_name, entry in capture.items():
            if not bool(entry["filled"].all().item()):
                missing = int((~entry["filled"]).sum().item())
                raise RuntimeError(
                    f"Incomplete capture for req_id={req_id}, field={field_name}: "
                    f"missing {missing} tokens"
                )
            result[field_name] = entry["tensor"]
        return result

    @staticmethod
    def _pack_sign_bits(tensor):
        sign = (tensor > 0).to(torch.uint8)
        if sign.is_cuda:
            sign = sign.cpu()
        return np.packbits(sign.numpy(), axis=1)

    @staticmethod
    def _pack_index_bits(tensor, num_bits):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        ids = tensor.numpy().astype(np.int32, copy=False)
        seq_len = ids.shape[0]
        bits = np.zeros((seq_len, num_bits), dtype=np.uint8)
        for col_idx in range(ids.shape[1]):
            col = ids[:, col_idx]
            valid = (col >= 0) & (col < num_bits)
            rows = np.arange(seq_len, dtype=np.int32)[valid]
            bits[rows, col[valid]] = 1
        return np.packbits(bits, axis=1)

    @staticmethod
    def _require_length(field_name, tensor, seq_len):
        if tensor is None:
            raise RuntimeError(f"Missing captured field: {field_name}")
        if int(tensor.shape[0]) != seq_len:
            raise RuntimeError(
                f"Field {field_name} has {tensor.shape[0]} tokens, expected {seq_len}"
            )
        return tensor

    @staticmethod
    def _to_numpy(tensor, dtype=None):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def _build_record_payload(self, req_id, record):
        materialized = self._materialize_capture(req_id)
        seq_len = int(record["echo_end"]) - int(record["echo_start"])
        payload = {}

        for layer_idx in range(self.arch["num_layers"]):
            full_key = f"router_full_{layer_idx}"
            topk_id_key = f"router_topk_ids_{layer_idx}"
            payload[full_key] = self._pack_sign_bits(
                self._require_length(full_key, materialized.get(full_key), seq_len)
            )
            payload[f"router_topk_{layer_idx}"] = self._pack_index_bits(
                self._require_length(topk_id_key, materialized.get(topk_id_key), seq_len),
                self.arch["num_experts"],
            )

        for layer_idx in self.arch["attn_layer_indices"]:
            for field_name in (
                f"q_{layer_idx}",
                f"k_{layer_idx}",
                f"v_{layer_idx}",
                f"attn_out_{layer_idx}",
            ):
                tensor = self._require_length(field_name, materialized.get(field_name), seq_len)
                payload[field_name] = self._pack_sign_bits(tensor)

        hidden_states = self._require_length("hs_final", materialized.get("hs_final"), seq_len)
        payload["hs_final"] = self._pack_sign_bits(hidden_states)
        return payload

    def _pack_columnar_shard(self, payloads, records):
        expected_fields = sorted(payloads[0].keys())
        for idx, payload in enumerate(payloads[1:], start=1):
            payload_fields = sorted(payload.keys())
            if payload_fields != expected_fields:
                missing = sorted(set(expected_fields) - set(payload_fields))
                extra = sorted(set(payload_fields) - set(expected_fields))
                raise RuntimeError(
                    f"Payload field mismatch for record {idx}: missing={missing}, extra={extra}"
                )

        shard_tensors = {}
        field_meta = {}
        for field_name in expected_fields:
            data = np.concatenate([payload[field_name] for payload in payloads], axis=0)
            shard_tensors[field_name] = np.asarray(data)
            field_meta[field_name] = {
                "encoding": "raw",
                "data_key": field_name,
                "data_dtype": str(shard_tensors[field_name].dtype),
                "shape": list(shard_tensors[field_name].shape),
            }

        record_meta = []
        offset = 0
        for record in records:
            seq_len = int(record["echo_end"]) - int(record["echo_start"])
            record_meta.append(
                {
                    "record_index": int(record["record_index"]),
                    "text_id": str(record["text_id"]),
                    "kind": str(record["kind"]),
                    "ordinal": int(record["ordinal"]),
                    "filename": str(record["filename"]),
                    "token_offset": offset,
                    "seq_len": seq_len,
                }
            )
            offset += seq_len

        meta = {
            "schema_version": 2,
            "format": "binary_feature_discovery_shard",
            "num_records": len(records),
            "fields": field_meta,
            "records": record_meta,
        }
        return shard_tensors, meta

    @staticmethod
    def _atomic_write_json(path, payload):
        directory = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".json.tmp")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _save_shard(self, data_path, meta_path, shard_tensors, meta):
        from safetensors.numpy import save_file

        directory = os.path.dirname(data_path)
        fd, tmp_data = tempfile.mkstemp(dir=directory, suffix=".safetensors.tmp")
        os.close(fd)
        try:
            save_file(shard_tensors, tmp_data)
            os.replace(tmp_data, data_path)
        except Exception:
            if os.path.exists(tmp_data):
                os.unlink(tmp_data)
            raise
        self._atomic_write_json(meta_path, meta)

    def _set_save_error(self, exc):
        with self._save_error_lock:
            if self._save_error is None:
                self._save_error = exc

    def _disk_writer_loop(self):
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                data_path, meta_path, shard_tensors, meta = item
                self._save_shard(data_path, meta_path, shard_tensors, meta)
            except Exception as exc:
                self._set_save_error(exc)
            finally:
                self._write_queue.task_done()

    def _raise_save_error_if_any(self):
        with self._save_error_lock:
            if self._save_error is None:
                return
            err = self._save_error
            self._save_error = None
        raise RuntimeError(f"Save failed: {err}")

    def transfer_and_save_async(self):
        self._raise_save_error_if_any()
        info = self._active_batch_payload
        if info is None:
            raise RuntimeError("No prepared batch is active")
        if len(self.req_capture_specs) != len(info["records"]):
            raise RuntimeError(
                f"Expected {len(info['records'])} captured requests, found "
                f"{len(self.req_capture_specs)}"
            )

        matched = sorted(self.req_capture_specs.items(), key=lambda item: int(item[1]["record_index"]))
        payloads = []
        records = []
        for req_id, record in matched:
            payloads.append(self._build_record_payload(req_id, record))
            records.append(record)

        if payloads:
            shard_tensors, meta = self._pack_columnar_shard(payloads, records)
            self._write_queue.put(
                (info["shard_data_path"], info["shard_meta_path"], shard_tensors, meta)
            )
        return len(payloads)

    def flush_saves(self):
        self._write_queue.join()
        self._raise_save_error_if_any()
        return True

    def get_captured_payload(self):
        if self._active_batch_payload is None:
            raise RuntimeError("No prepared batch is active")
        if len(self.req_capture_specs) != 1:
            raise RuntimeError(
                f"Expected exactly one captured request, found {len(self.req_capture_specs)}"
            )
        req_id, record = next(iter(self.req_capture_specs.items()))
        return self._build_record_payload(req_id, record)

    def get_architecture(self):
        model = self.model_runner.model
        text_model = model.language_model.model
        layers = list(text_model.layers)
        config = text_model.config
        layer_types = []
        attn_layer_indices = []
        for idx, layer in enumerate(layers):
            layer_type = getattr(layer, "layer_type", None)
            if layer_type is None and hasattr(config, "layer_types"):
                layer_type = config.layer_types[idx]
            layer_types.append(layer_type)
            if layer_type == "full_attention":
                attn_layer_indices.append(idx)
        return json.dumps(
            {
                "model_class": type(model).__name__,
                "num_layers": len(layers),
                "layer_types": layer_types,
                "attn_layer_indices": attn_layer_indices,
                "hidden_size": int(config.hidden_size),
                "num_attention_heads": int(config.num_attention_heads),
                "num_key_value_heads": int(config.num_key_value_heads),
                "head_dim": int(config.head_dim),
                "num_experts": int(config.num_experts),
                "num_experts_per_tok": int(config.num_experts_per_tok),
            }
        )

    def remove_hooks(self):
        self.flush_saves()
        self._write_queue.put(None)
        self._writer_thread.join()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_capture()
        return True

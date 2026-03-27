"""
vLLM worker extension v2 — request-tracked prompt capture.

Hooks slice prompt tokens by req_id on each scheduler step, so payload
construction no longer depends on batch execution order. Batched outputs
are written as columnar shards to reduce CPU serialization and filesystem
metadata overhead.
"""

import hashlib
import json
import math
import os
import queue
import tempfile
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class SignalExtractorExtension:

    def setup_hooks(self, arch_json):
        import json

        config = json.loads(arch_json)
        if "arch" in config:
            self.arch = config["arch"]
            options = config.get("options", {})
        else:
            self.arch = config
            options = {}

        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._handles = []
        self._save_error = None
        self._save_error_lock = threading.Lock()
        self._compute_prenorm_attn_dist = bool(
            options.get("compute_prenorm_attn_dist", False))
        self._shard_format = str(options.get("shard_format", "torch"))
        self._activation_quantization = str(
            options.get("activation_quantization", "int8"))
        self._gpu_packer_threads = []
        self._disk_writer_threads = []
        self._pack_queue = queue.Queue(
            maxsize=max(1, int(options.get("pack_queue_size", 3))))
        packed_queue_size = max(1, int(options.get("write_queue_size", 6)))
        self._packed_batch_queue = queue.Queue(maxsize=packed_queue_size)
        self._write_queue = queue.Queue(maxsize=packed_queue_size)
        self._gpu_packer_thread_count = max(
            1, int(options.get("gpu_packer_threads", 2)))
        self._disk_writer_thread_count = max(
            1, int(options.get("disk_writer_threads", 2)))
        self._pending_shards = {}
        self._pinned_pool = {}
        self._pinned_pool_lock = threading.Lock()

        for idx in range(self._gpu_packer_thread_count):
            thread = threading.Thread(
                target=self._gpu_packer_loop,
                name=f"signal-pack-{idx}",
                daemon=True,
            )
            thread.start()
            self._gpu_packer_threads.append(thread)

        self._aggregator_thread = threading.Thread(
            target=self._aggregate_shards_loop,
            name="signal-aggregate",
            daemon=True,
        )
        self._aggregator_thread.start()

        for idx in range(self._disk_writer_thread_count):
            thread = threading.Thread(
                target=self._disk_writer_loop,
                name=f"signal-write-{idx}",
                daemon=True,
            )
            thread.start()
            self._disk_writer_threads.append(thread)

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
                            vals, ids = probs.topk(9, dim=-1)
                            segments = self._get_prompt_segments(logits)
                            self._append_segments(segments, f"rtw_{li}", vals.half())
                            self._append_segments(segments, f"rti_{li}", ids.short())
                        return hook

                    self._handles.append(
                        gate.register_forward_hook(_router_hook(layer_idx)))

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
                                segments = self._get_prompt_segments(tensor)
                                self._append_segments(segments, f"q_{li}", q)
                                self._append_segments(segments, f"k_{li}", k)
                                self._append_segments(segments, f"v_{li}", v)
                            return hook

                        self._handles.append(
                            qkv_proj.register_forward_hook(
                                _qkv_hook(layer_idx, q_dim, kv_dim,
                                          num_q_heads, head_dim)))

                    o_proj = getattr(attn, "o_proj", None)
                    if o_proj is not None:
                        def _oproj_hook(li):
                            def hook(_mod, _inp, out):
                                tensor = out[0] if isinstance(out, tuple) else out
                                segments = self._get_prompt_segments(tensor)
                                self._append_segments(segments, f"ao_{li}", tensor)
                            return hook

                        self._handles.append(
                            o_proj.register_forward_hook(_oproj_hook(layer_idx)))

            if layer_idx == len(layers) - 1:
                def _final_hook(_li):
                    def hook(_mod, _inp, out):
                        hidden_states = out[0] if isinstance(out, tuple) else out
                        segments = self._get_prompt_segments(hidden_states)
                        self._append_segments(segments, "hs", hidden_states)
                    return hook

                self._handles.append(layer.register_forward_hook(_final_hook(layer_idx)))

        return (
            f"Installed {len(self._handles)} hooks on {len(layers)} layers "
            f"(gpu_packers={self._gpu_packer_thread_count}, "
            f"disk_writers={self._disk_writer_thread_count})"
        )

    def clear_capture(self):
        self.capture.clear()
        self.req_prompt_token_ids.clear()
        self.req_capture_specs.clear()
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._cached_segments = None
        self._cached_segment_signature = None
        return True

    def prepare_batch(self, batch_payload_json):
        self._raise_save_error_if_any()
        batch_payload = json.loads(batch_payload_json)
        prompt_key_to_records = {}
        for record in batch_payload["records"]:
            prompt_key_to_records.setdefault(
                record["prompt_key"], deque()).append(dict(record))
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
            return np.asarray(buffer.np)
        if hasattr(buffer, "cpu"):
            return np.asarray(buffer.cpu)
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
        """Get prompt segments, cached per forward step.

        Computes segments once per scheduler step, then reuses them for
        all hooks in that step.
        """
        total_tokens = tensor.shape[0]

        input_batch = self.model_runner.input_batch
        req_ids = list(input_batch.req_ids)
        num_reqs = int(getattr(input_batch, "num_reqs", len(req_ids)))
        req_ids = req_ids[:num_reqs]

        query_start_loc = self._buffer_to_numpy(self.model_runner.query_start_loc)
        query_start_loc = query_start_loc[:num_reqs + 1]
        expected_tokens = int(query_start_loc[-1]) if len(query_start_loc) else 0
        if total_tokens != expected_tokens:
            raise RuntimeError(
                f"Hook tensor token count mismatch: got {total_tokens}, "
                f"expected {expected_tokens} from query_start_loc"
            )

        num_computed_tokens = np.asarray(
            input_batch.num_computed_tokens_cpu[:num_reqs], dtype=np.int64)
        num_prompt_tokens = np.asarray(
            input_batch.num_prompt_tokens[:num_reqs], dtype=np.int64)

        signature = (
            total_tokens,
            tuple(req_ids),
            tuple(np.asarray(query_start_loc, dtype=np.int64).tolist()),
            tuple(num_computed_tokens.tolist()),
            tuple(num_prompt_tokens.tolist()),
        )
        if (hasattr(self, "_cached_segments")
                and self._cached_segment_signature == signature):
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
            chunk_prompt_end = chunk_prompt_start + prompt_part_len
            echo_start = int(req_info["echo_start"])
            echo_end = int(req_info["echo_end"])
            capture_prompt_start = max(chunk_prompt_start, echo_start)
            capture_prompt_end = min(chunk_prompt_end, echo_end)
            if capture_prompt_start >= capture_prompt_end:
                continue

            local_start = start + (capture_prompt_start - chunk_prompt_start)
            local_end = start + (capture_prompt_end - chunk_prompt_start)
            segments.append((req_id, local_start, local_end))

        # Cache for remaining hooks in this step
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
                seq_len = int(req_info["echo_end"]) - int(req_info["echo_start"])
                field_entry = {
                    "tensor": torch.empty(
                        (seq_len, *field_tensor.shape[1:]),
                        dtype=field_tensor.dtype,
                        device=field_tensor.device,
                    ),
                    "offset": 0,
                }
                req_capture[field_name] = field_entry
            offset = int(field_entry["offset"])
            next_offset = offset + int(field_tensor.shape[0])
            if next_offset > field_entry["tensor"].shape[0]:
                raise RuntimeError(
                    f"Capture overflow for req_id={req_id}, field={field_name}: "
                    f"{next_offset} > {field_entry['tensor'].shape[0]}"
                )
            field_entry["tensor"][offset:next_offset].copy_(field_tensor)
            field_entry["offset"] = next_offset

    @staticmethod
    def _dtype_name(dtype):
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _output_field_name(capture_key):
        if capture_key == "hs":
            return "hs_final"
        if capture_key.startswith("rtw_"):
            return f"router_topk_weights_{capture_key.split('_', 1)[1]}"
        if capture_key.startswith("rti_"):
            return f"router_topk_ids_{capture_key.split('_', 1)[1]}"
        if capture_key.startswith("ao_"):
            return f"attn_out_{capture_key.split('_', 1)[1]}"
        return capture_key

    def _should_quantize_field(self, capture_key, tensor):
        if self._activation_quantization != "int8":
            return False
        if tensor.ndim < 2:
            return False
        return (
            capture_key == "hs"
            or capture_key.startswith("q_")
            or capture_key.startswith("k_")
            or capture_key.startswith("v_")
            or capture_key.startswith("ao_")
        )

    @staticmethod
    def _quantize_rowwise_int8(tensor):
        work = tensor.float()
        max_abs = work.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / 127.0, torch.ones_like(max_abs))
        quantized = torch.clamp(torch.round(work / scale), -127, 127).to(torch.int8)
        return quantized.contiguous(), scale.squeeze(-1).to(torch.float16).contiguous()

    def _acquire_pinned_buffer(self, shape, dtype):
        key = (tuple(shape), str(dtype))
        with self._pinned_pool_lock:
            pool = self._pinned_pool.get(key)
            if pool:
                return pool.pop()
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)

    def _release_tensor_buffer(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return
        if tensor.device.type != "cpu" or not tensor.is_pinned():
            return
        key = (tuple(tensor.shape), str(tensor.dtype))
        with self._pinned_pool_lock:
            self._pinned_pool.setdefault(key, []).append(tensor)

    def _release_tensor_payload_buffers(self, tensor_payload):
        for tensor in tensor_payload.values():
            self._release_tensor_buffer(tensor)

    def _copy_tensor_to_cpu(self, tensor, transfer_stream=None):
        tensor = tensor.detach().contiguous()
        if not tensor.is_cuda:
            return tensor.cpu() if tensor.device.type != "cpu" else tensor
        if transfer_stream is None:
            return tensor.cpu()
        with torch.cuda.stream(transfer_stream):
            cpu_tensor = self._acquire_pinned_buffer(tensor.shape, tensor.dtype)
            cpu_tensor.copy_(tensor, non_blocking=True)
        return cpu_tensor

    def _transfer_field_slices(self, field_tensors, quantize_for_storage, retain_fields=None):
        cpu_tensors = {}
        field_specs = {}
        retained_tensors = {}
        retain_fields = set(retain_fields or ())
        transfer_stream = None
        if field_tensors:
            first_tensors = next(iter(field_tensors.values()))
            if first_tensors and first_tensors[0].is_cuda:
                transfer_stream = torch.cuda.Stream(device=first_tensors[0].device)

        for capture_key, tensors in field_tensors.items():
            cat = torch.cat(tensors, dim=0)
            if cat.dtype == torch.bfloat16:
                cat = cat.half()

            output_name = self._output_field_name(capture_key)
            if quantize_for_storage and self._should_quantize_field(capture_key, cat):
                quantized, scale = self._quantize_rowwise_int8(cat)
                data_key = f"{output_name}__data"
                scale_key = f"{output_name}__scale"
                cpu_tensors[data_key] = self._copy_tensor_to_cpu(
                    quantized, transfer_stream)
                cpu_tensors[scale_key] = self._copy_tensor_to_cpu(
                    scale, transfer_stream)
                field_specs[output_name] = {
                    "encoding": "int8_rowwise",
                    "shape": list(cat.shape),
                    "data_key": data_key,
                    "data_dtype": self._dtype_name(quantized.dtype),
                    "scale_key": scale_key,
                    "scale_dtype": self._dtype_name(scale.dtype),
                }
            else:
                cpu_tensors[output_name] = self._copy_tensor_to_cpu(
                    cat, transfer_stream)
                field_specs[output_name] = {
                    "encoding": "raw",
                    "shape": list(cat.shape),
                    "data_key": output_name,
                    "data_dtype": self._dtype_name(cat.dtype),
                }
            if output_name in retain_fields:
                retained_tensors[output_name] = cat

        if transfer_stream is not None:
            transfer_stream.synchronize()

        return cpu_tensors, field_specs, retained_tensors

    def _prepare_batch_capture(self, snapshot, quantize_for_storage, retain_fields=None):
        field_tensors = {}
        seq_lengths = []
        token_slices = []
        matched_infos = []
        req_capture_specs = snapshot["req_capture_specs"]
        if len(req_capture_specs) != len(snapshot["records"]):
            raise RuntimeError(
                f"Expected {len(snapshot['records'])} captured requests, "
                f"found {len(req_capture_specs)}"
            )

        matched = sorted(
            req_capture_specs.items(),
            key=lambda item: int(item[1]["record_index"]),
        )

        for req_id, info in matched:
            echo_start = int(info["echo_start"])
            echo_end = int(info["echo_end"])
            seq_len = echo_end - echo_start
            if seq_len <= 0:
                raise RuntimeError(
                    f"Invalid echo slice for req_id={req_id}: {echo_start}:{echo_end}"
                )

            prompt_ids = snapshot["req_prompt_token_ids"].get(req_id)
            if prompt_ids is None:
                raise RuntimeError(f"Missing prompt token ids for req_id={req_id}")
            if echo_end > len(prompt_ids):
                raise RuntimeError(
                    f"Echo slice {echo_start}:{echo_end} exceeds prompt length "
                    f"{len(prompt_ids)} for req_id={req_id}"
                )

            fields = snapshot["capture"].get(req_id, {})
            hs_entry = fields.get("hs")
            if hs_entry is None:
                raise RuntimeError(
                    f"Missing final hidden-state capture for req_id={req_id}")
            captured_seq_len = int(hs_entry["offset"])
            if captured_seq_len != seq_len:
                raise RuntimeError(
                    f"Echo capture length mismatch for req_id={req_id}: "
                    f"captured {captured_seq_len} tokens, expected {seq_len}"
                )

            seq_lengths.append(seq_len)
            token_slices.append(np.asarray(prompt_ids[echo_start:echo_end], dtype=np.int32).copy())
            matched_infos.append(info)
            for key, entry in fields.items():
                if entry is None:
                    continue
                total_parts_len = int(entry["offset"])
                if total_parts_len != seq_len:
                    raise RuntimeError(
                        f"Capture length mismatch for req_id={req_id}, field={key}: "
                        f"{total_parts_len} vs expected {seq_len}"
                    )
                field_tensors.setdefault(key, []).append(entry["tensor"])

        cpu_tensors, field_specs, retained_tensors = self._transfer_field_slices(
            field_tensors,
            quantize_for_storage=quantize_for_storage,
            retain_fields=retain_fields,
        )
        return (
            matched_infos,
            seq_lengths,
            token_slices,
            cpu_tensors,
            field_specs,
            retained_tensors,
        )

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

        import json
        return json.dumps({
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
        })

    def _set_save_error(self, exc):
        with self._save_error_lock:
            if self._save_error is None:
                self._save_error = exc

    def _raise_save_error_if_any(self):
        with self._save_error_lock:
            if self._save_error is None:
                return
            err = self._save_error
            self._save_error = None
        raise RuntimeError(f"Background save failed: {err}")

    def _wait_for_save(self):
        self._pack_queue.join()
        self._packed_batch_queue.join()
        self._write_queue.join()
        self._raise_save_error_if_any()

    def _snapshot_live_state(self, reset):
        if self._active_batch_payload is None:
            raise RuntimeError("No prepared batch is active")
        snapshot = {
            "shard_stem": self._active_batch_payload["shard_stem"],
            "shard_data_path": self._active_batch_payload["shard_data_path"],
            "shard_meta_path": self._active_batch_payload["shard_meta_path"],
            "shard_piece_index": int(self._active_batch_payload["shard_piece_index"]),
            "shard_piece_count": int(self._active_batch_payload["shard_piece_count"]),
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

    def _compute_prenorm_attn_from_retained(self, retained_tensors, seq_lengths):
        if not self._compute_prenorm_attn_dist:
            return {}

        num_q_heads = self.arch["num_attention_heads"]
        num_kv_heads = self.arch["num_key_value_heads"]
        head_dim = self.arch["head_dim"]
        attn_payload = {}

        for li in self.arch["attn_layer_indices"]:
            q_name = f"q_{li}"
            k_name = f"k_{li}"
            q_cat = retained_tensors.get(q_name)
            k_cat = retained_tensors.get(k_name)
            if q_cat is None or k_cat is None:
                continue

            offset = 0
            weights_per_seq = []
            for seq_len in seq_lengths:
                if seq_len <= 0:
                    continue
                q_seq = q_cat[offset:offset + seq_len].float()
                k_seq = k_cat[offset:offset + seq_len].float()
                offset += seq_len
                q_last = q_seq[-1].reshape(num_q_heads, head_dim)
                k_all = k_seq.reshape(seq_len, num_kv_heads, head_dim)
                heads_per_group = num_q_heads // num_kv_heads
                scale = 1.0 / math.sqrt(head_dim)
                q_grouped = q_last.reshape(num_kv_heads, heads_per_group, head_dim)
                k_transposed = k_all.permute(1, 0, 2)
                scores = torch.bmm(
                    q_grouped.reshape(num_kv_heads * heads_per_group, 1, head_dim),
                    k_transposed.repeat_interleave(heads_per_group, dim=0).transpose(1, 2),
                ).reshape(num_kv_heads, heads_per_group, seq_len) * scale
                weights = F.softmax(scores, dim=-1).mean(dim=(0, 1))
                weights_per_seq.append(weights.half())

            if weights_per_seq:
                attn_payload[f"prenorm_attn_dist_last_{li}"] = torch.cat(
                    weights_per_seq, dim=0).contiguous()
        return attn_payload

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

    def _write_tensor_payload(self, path, tensor_payload):
        directory = os.path.dirname(path)
        suffix = f"{os.path.splitext(path)[1]}.tmp"
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=suffix)
        os.close(fd)
        try:
            payload = {
                key: value.detach().cpu().contiguous()
                for key, value in tensor_payload.items()
            }
            if self._shard_format == "safetensors":
                from safetensors.torch import save_file

                save_file(
                    payload,
                    tmp_path,
                    metadata={
                        "schema_version": "3",
                        "format": "binary_embeddings_columnar_shard",
                    },
                )
            else:
                torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _save_shard(self, data_path, meta_path, tensor_payload, meta_payload):
        self._write_tensor_payload(data_path, tensor_payload)
        self._atomic_write_json(meta_path, meta_payload)

    def _gpu_packer_loop(self):
        while True:
            item = self._pack_queue.get()
            try:
                if item is None:
                    self._packed_batch_queue.put(None)
                    return
                self._packed_batch_queue.put(self._pack_batch_piece(item))
            except Exception as exc:
                self._set_save_error(exc)
            finally:
                self._pack_queue.task_done()

    def _merge_field_specs(self, piece_specs, final_tensors):
        merged = {}
        for specs in piece_specs:
            for field_name, spec in specs.items():
                if field_name in merged:
                    continue
                merged[field_name] = dict(spec)
        for field_name, spec in merged.items():
            data_key = spec["data_key"]
            if data_key in final_tensors:
                spec["shape"] = list(final_tensors[data_key].shape)
        return merged

    def _build_final_shard(self, shard_state):
        pieces = [shard_state["pieces"][idx] for idx in range(shard_state["piece_count"])]
        seq_lens_parts = []
        token_ids_parts = []
        field_parts = {}
        merged_records = []
        piece_specs = []
        token_offset = 0

        for piece in pieces:
            piece_tensors = piece["tensor_payload"]
            seq_lens_parts.append(piece_tensors["seq_lens"])
            token_ids_parts.append(piece_tensors["token_ids"])
            piece_specs.append(piece["field_specs"])

            for record in sorted(piece["records"], key=lambda item: int(item["record_index"])):
                merged_records.append({
                    "record_index": int(record["record_index"]),
                    "kind": str(record["kind"]),
                    "ordinal": int(record["ordinal"]),
                    "text_id": str(record["text_id"]),
                    "filename": str(record["filename"]),
                    "seq_len": int(record["seq_len"]),
                    "token_offset": token_offset,
                })
                token_offset += int(record["seq_len"])

            for key, tensor in piece_tensors.items():
                if key in ("seq_lens", "token_ids"):
                    continue
                field_parts.setdefault(key, []).append(tensor)

        final_seq_lens = (torch.cat(seq_lens_parts, dim=0)
                          if seq_lens_parts else torch.empty((0,), dtype=torch.int32))
        final_seq_offsets = torch.zeros(
            (final_seq_lens.shape[0] + 1,), dtype=torch.int64)
        if final_seq_lens.numel() > 0:
            final_seq_offsets[1:] = torch.cumsum(
                final_seq_lens.to(torch.int64), dim=0)
        final_tensors = {
            "schema_version": torch.tensor([3], dtype=torch.int32),
            "seq_lens": final_seq_lens.contiguous(),
            "seq_offsets": final_seq_offsets.contiguous(),
            "token_ids": (torch.cat(token_ids_parts, dim=0)
                          if token_ids_parts else torch.empty((0,), dtype=torch.int32)),
        }
        for key, parts in field_parts.items():
            final_tensors[key] = torch.cat(parts, dim=0).contiguous()

        for piece in pieces:
            self._release_tensor_payload_buffers(piece["tensor_payload"])

        field_specs = self._merge_field_specs(piece_specs, final_tensors)
        meta_payload = {
            "schema_version": 3,
            "format": "binary_embeddings_columnar_shard",
            "storage_format": self._shard_format,
            "activation_quantization": self._activation_quantization,
            "num_records": len(merged_records),
            "num_tokens": int(final_seq_offsets[-1].item()),
            "records": merged_records,
            "fields": field_specs,
        }
        return {
            "data_path": shard_state["data_path"],
            "meta_path": shard_state["meta_path"],
            "tensor_payload": final_tensors,
            "meta_payload": meta_payload,
        }

    def _aggregate_shards_loop(self):
        finished_packers = 0
        while True:
            item = self._packed_batch_queue.get()
            try:
                if item is None:
                    finished_packers += 1
                    if finished_packers == self._gpu_packer_thread_count:
                        if self._pending_shards:
                            raise RuntimeError(
                                f"Incomplete shard aggregation for {len(self._pending_shards)} shards"
                            )
                        for _ in range(self._disk_writer_thread_count):
                            self._write_queue.put(None)
                        return
                    continue

                key = (item["data_path"], item["meta_path"])
                shard_state = self._pending_shards.setdefault(
                    key,
                    {
                        "shard_stem": item["shard_stem"],
                        "data_path": item["data_path"],
                        "meta_path": item["meta_path"],
                        "piece_count": int(item["shard_piece_count"]),
                        "pieces": {},
                    },
                )
                if int(item["shard_piece_count"]) != shard_state["piece_count"]:
                    raise RuntimeError(
                        f"Shard piece-count mismatch for {item['shard_stem']}"
                    )
                piece_index = int(item["shard_piece_index"])
                if piece_index in shard_state["pieces"]:
                    raise RuntimeError(
                        f"Duplicate shard piece {piece_index} for {item['shard_stem']}"
                    )
                shard_state["pieces"][piece_index] = item
                if len(shard_state["pieces"]) == shard_state["piece_count"]:
                    complete_state = self._pending_shards.pop(key)
                    self._write_queue.put(self._build_final_shard(complete_state))
            except Exception as exc:
                self._set_save_error(exc)
                for _ in range(self._disk_writer_thread_count):
                    self._write_queue.put(None)
                return
            finally:
                self._packed_batch_queue.task_done()

    def _disk_writer_loop(self):
        while True:
            item = self._write_queue.get()
            try:
                if item is None:
                    return
                self._save_shard(
                    item["data_path"],
                    item["meta_path"],
                    item["tensor_payload"],
                    item["meta_payload"],
                )
            except Exception as exc:
                self._set_save_error(exc)
            finally:
                self._write_queue.task_done()

    def _match_request_batch(self, batch_info):
        prompt_key_to_req_ids = {}
        for req_id, prompt_ids in self.req_prompt_token_ids.items():
            prompt_key = self._prompt_key(prompt_ids)
            prompt_key_to_req_ids.setdefault(prompt_key, []).append(req_id)

        matched = []
        for info in batch_info:
            prompt_key = info["prompt_key"]
            req_ids = prompt_key_to_req_ids.get(prompt_key)
            if not req_ids:
                raise RuntimeError(f"No captured request matched prompt_key={prompt_key}")
            matched.append((req_ids.pop(), info))

        remaining = sum(len(req_ids) for req_ids in prompt_key_to_req_ids.values())
        if remaining:
            raise RuntimeError(
                f"Matched {len(matched)} payloads but {remaining} captured requests "
                "were left unmatched"
            )

        return matched

    def _pack_batch_piece(self, batch_payload):
        retain_fields = None
        if self._compute_prenorm_attn_dist:
            retain_fields = {
                f"{prefix}_{li}"
                for li in self.arch["attn_layer_indices"]
                for prefix in ("q", "k")
            }

        matched_infos, seq_lengths, token_slices, cpu_tensors, field_specs, retained_tensors = (
            self._prepare_batch_capture(
                batch_payload,
                quantize_for_storage=True,
                retain_fields=retain_fields,
            )
        )
        flat_token_ids = (
            np.concatenate(token_slices, axis=0).astype(np.int32, copy=False)
            if token_slices else np.empty((0,), dtype=np.int32)
        )
        tensor_payload = {
            "seq_lens": torch.as_tensor(seq_lengths, dtype=torch.int32),
            "token_ids": torch.from_numpy(flat_token_ids),
        }
        tensor_payload.update(cpu_tensors)

        prenorm_payload = self._compute_prenorm_attn_from_retained(
            retained_tensors, seq_lengths)
        for field_name, tensor in prenorm_payload.items():
            tensor_payload[field_name] = self._copy_tensor_to_cpu(tensor)
            field_specs[field_name] = {
                "encoding": "raw",
                "shape": list(tensor.shape),
                "data_key": field_name,
                "data_dtype": self._dtype_name(tensor.dtype),
            }

        records = []
        for idx, info in enumerate(matched_infos):
            records.append({
                "record_index": int(info.get("shard_record_index", idx)),
                "kind": str(info["kind"]),
                "ordinal": int(info["ordinal"]),
                "text_id": str(info["text_id"]),
                "filename": str(info["filename"]),
                "seq_len": int(seq_lengths[idx]),
            })

        return {
            "shard_stem": batch_payload["shard_stem"],
            "shard_piece_index": int(batch_payload["shard_piece_index"]),
            "shard_piece_count": int(batch_payload["shard_piece_count"]),
            "data_path": batch_payload["shard_data_path"],
            "meta_path": batch_payload["shard_meta_path"],
            "tensor_payload": tensor_payload,
            "field_specs": field_specs,
            "records": records,
        }

    def _bulk_transfer_and_build(self, snapshot):
        arch = self.arch
        attn_layers = arch["attn_layer_indices"]
        num_layers = arch["num_layers"]
        retain_fields = None
        if self._compute_prenorm_attn_dist:
            retain_fields = {
                f"{prefix}_{li}"
                for li in attn_layers
                for prefix in ("q", "k")
            }

        matched_infos, seq_lengths, token_slices, cpu_tensors, _field_specs, retained_tensors = (
            self._prepare_batch_capture(
                snapshot,
                quantize_for_storage=False,
                retain_fields=retain_fields,
            )
        )
        offsets = {key: 0 for key in cpu_tensors}
        payloads = []
        prenorm_offsets = None
        prenorm_cpu_tensors = {}
        if self._compute_prenorm_attn_dist:
            prenorm_payload = self._compute_prenorm_attn_from_retained(
                retained_tensors, seq_lengths)
            prenorm_cpu_tensors = {
                key: self._copy_tensor_to_cpu(value)
                for key, value in prenorm_payload.items()
            }
            prenorm_offsets = {key: 0 for key in prenorm_cpu_tensors}

        for seq_idx, info in enumerate(matched_infos):
            seq_len = seq_lengths[seq_idx]
            payload = {
                "schema_version": np.array(3, dtype=np.int32),
                "token_ids": token_slices[seq_idx],
                "seq_len": np.array(seq_len, dtype=np.int32),
                "text_id": np.asarray(info["text_id"]),
                "kind": np.asarray(info["kind"]),
            }

            for li in range(num_layers):
                weight_key = f"router_topk_weights_{li}"
                id_key = f"router_topk_ids_{li}"
                if weight_key in cpu_tensors and id_key in cpu_tensors:
                    weight_offset = offsets[weight_key]
                    id_offset = offsets[id_key]
                    payload[f"router_topk_weights_{li}"] = (
                        cpu_tensors[weight_key][weight_offset:weight_offset + seq_len].numpy())
                    payload[f"router_topk_ids_{li}"] = (
                        cpu_tensors[id_key][id_offset:id_offset + seq_len].numpy())
                    offsets[weight_key] = weight_offset + seq_len
                    offsets[id_key] = id_offset + seq_len

            for li in attn_layers:
                for short_name, full_name in [
                    ("q", "q"),
                    ("k", "k"),
                    ("v", "v"),
                    ("attn_out", "attn_out"),
                ]:
                    capture_key = f"{short_name}_{li}"
                    if capture_key in cpu_tensors:
                        offset = offsets[capture_key]
                        payload[f"{full_name}_{li}"] = (
                            cpu_tensors[capture_key][offset:offset + seq_len].numpy())
                        offsets[capture_key] = offset + seq_len

            if prenorm_offsets is not None:
                for li in attn_layers:
                    field_name = f"prenorm_attn_dist_last_{li}"
                    if field_name in prenorm_cpu_tensors:
                        offset = prenorm_offsets[field_name]
                        payload[field_name] = (
                            prenorm_cpu_tensors[field_name][offset:offset + seq_len].numpy())
                        prenorm_offsets[field_name] = offset + seq_len

            if "hs_final" in cpu_tensors:
                offset = offsets["hs_final"]
                payload["hs_final"] = (
                    cpu_tensors["hs_final"][offset:offset + seq_len].numpy())
                offsets["hs_final"] = offset + seq_len

            payloads.append((info.get("path", "/dev/null"), payload))

        return payloads

    def transfer_and_save_async(self):
        self._raise_save_error_if_any()
        snapshot = self._snapshot_live_state(reset=True)
        self._pack_queue.put(snapshot)
        return len(snapshot["records"])

    def flush_saves(self):
        self._wait_for_save()
        return True

    def get_captured_payload(self, echo_start=None, echo_end=None, token_ids_list=None):
        if self._active_batch_payload is None:
            raise RuntimeError("No prepared batch is active")
        if len(self.req_capture_specs) != 1:
            raise RuntimeError(
                f"Expected exactly one captured request, found {len(self.req_capture_specs)}"
            )
        snapshot = self._snapshot_live_state(reset=False)
        payloads = self._bulk_transfer_and_build(snapshot)
        return payloads[0][1]

    def remove_hooks(self):
        self._wait_for_save()
        for _ in self._gpu_packer_threads:
            self._pack_queue.put(None)
        for thread in self._gpu_packer_threads:
            thread.join()
        self._gpu_packer_threads.clear()
        self._aggregator_thread.join()
        for thread in self._disk_writer_threads:
            thread.join()
        self._disk_writer_threads.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.capture.clear()
        self.req_prompt_token_ids.clear()
        self.req_capture_specs.clear()
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._pending_shards = {}
        return True

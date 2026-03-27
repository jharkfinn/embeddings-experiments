"""vLLM worker extension for dense last-token router/HS extraction."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F


class DenseRouterLastTokenExtension:
    def setup_hooks(self, config_json):
        config = json.loads(config_json)
        if "arch" in config:
            self.arch = config["arch"]
        else:
            self.arch = config

        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._req_prompt_keys = {}
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._handles = []
        self._cached_targets = None
        self._cached_target_signature = None

        model = self.model_runner.model
        text_model = model.language_model.model
        layers = list(text_model.layers)

        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
                if gate is not None:

                    def _router_hook(li):
                        def hook(_mod, _inp, out):
                            logits = out[0] if isinstance(out, tuple) else out
                            probs = F.softmax(logits.float(), dim=-1)
                            for req_id, local_index in self._get_target_positions(int(probs.shape[0])):
                                self._store_vector(
                                    req_id,
                                    f"router_rw_last_{li}",
                                    probs[local_index].to(torch.float16),
                                )

                        return hook

                    self._handles.append(gate.register_forward_hook(_router_hook(layer_idx)))

            if layer_idx == len(layers) - 1:

                def _final_hook(_li):
                    def hook(_mod, _inp, out):
                        hidden_states = out[0] if isinstance(out, tuple) else out
                        for req_id, local_index in self._get_target_positions(int(hidden_states.shape[0])):
                            self._store_vector(
                                req_id,
                                "hs_last",
                                hidden_states[local_index].to(torch.float16),
                            )

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

    def _get_target_positions(self, total_tokens):
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
        query_start_loc = np.asarray(query_start_loc[: num_reqs + 1], dtype=np.int64)

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
            tuple(query_start_loc.tolist()),
            tuple(num_computed_tokens.tolist()),
            tuple(num_prompt_tokens.tolist()),
        )
        if self._cached_target_signature == signature:
            return self._cached_targets

        targets = []
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
                self.req_capture_specs[req_id] = pending.pop(0)

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
            target_index = int(req_info["target_index"])
            if target_index < chunk_prompt_start or target_index >= chunk_prompt_end:
                continue

            local_index = start + (target_index - chunk_prompt_start)
            targets.append((req_id, local_index))

        self._cached_targets = targets
        self._cached_target_signature = signature
        return targets

    def _store_vector(self, req_id, field_name, tensor):
        req_capture = self.capture.setdefault(req_id, {})
        if field_name in req_capture:
            return
        req_capture[field_name] = tensor.detach().cpu().numpy().astype(np.float16, copy=True)

    def clear_capture(self):
        self.capture.clear()
        self.req_prompt_token_ids.clear()
        self.req_capture_specs.clear()
        self._req_prompt_keys.clear()
        self._pending_batch_records = {}
        self._active_batch_payload = None
        self._cached_targets = None
        self._cached_target_signature = None
        return True

    def prepare_batch(self, batch_info_json):
        info = json.loads(batch_info_json)
        prompt_key_to_records = {}
        for record in info["records"]:
            prompt_key_to_records.setdefault(record["prompt_key"], []).append(dict(record))
        self.capture = {}
        self.req_prompt_token_ids = {}
        self.req_capture_specs = {}
        self._req_prompt_keys = {}
        self._pending_batch_records = prompt_key_to_records
        self._active_batch_payload = info
        self._cached_targets = None
        self._cached_target_signature = None
        return len(info["records"])

    def collect_batch_vectors(self):
        info = self._active_batch_payload
        if info is None:
            raise RuntimeError("No prepared batch is active")
        if len(self.req_capture_specs) != len(info["records"]):
            raise RuntimeError(
                f"Expected {len(info['records'])} matched requests, found {len(self.req_capture_specs)}"
            )

        matched = sorted(self.req_capture_specs.items(), key=lambda item: int(item[1]["record_index"]))
        num_rows = len(matched)
        num_layers = int(self.arch["num_layers"])
        num_experts = int(self.arch["num_experts"])
        hidden_size = int(self.arch["hidden_size"])

        router_rw_last = np.empty((num_rows, num_layers * num_experts), dtype=np.float16)
        hs_last = np.empty((num_rows, hidden_size), dtype=np.float16)
        ordered_records = []

        for row_index, (req_id, record) in enumerate(matched):
            capture = self.capture.get(req_id, {})
            layer_parts = []
            for layer_idx in range(num_layers):
                key = f"router_rw_last_{layer_idx}"
                value = capture.get(key)
                if value is None or value.shape != (num_experts,):
                    raise RuntimeError(
                        f"Missing or malformed capture for req_id={req_id}, field={key}, "
                        f"shape={None if value is None else value.shape}"
                    )
                layer_parts.append(value)
            rw_vector = np.concatenate(layer_parts, axis=0)
            if rw_vector.shape != (num_layers * num_experts,):
                raise RuntimeError(f"Unexpected RW shape for req_id={req_id}: {rw_vector.shape}")
            router_rw_last[row_index] = rw_vector

            hs_vector = capture.get("hs_last")
            if hs_vector is None or hs_vector.shape != (hidden_size,):
                raise RuntimeError(
                    f"Missing or malformed capture for req_id={req_id}, field=hs_last, "
                    f"shape={None if hs_vector is None else hs_vector.shape}"
                )
            hs_last[row_index] = hs_vector

            ordered_records.append(
                {
                    "record_index": int(record["record_index"]),
                    "kind": str(record["kind"]),
                    "ordinal": int(record["ordinal"]),
                    "text_id": str(record["text_id"]),
                }
            )

        return {
            "router_rw_last": router_rw_last,
            "hs_last": hs_last,
            "records": ordered_records,
        }

    def get_architecture(self):
        return json.dumps(self.arch)

    def remove_hooks(self):
        for handle in getattr(self, "_handles", []):
            handle.remove()
        self._handles = []
        return True

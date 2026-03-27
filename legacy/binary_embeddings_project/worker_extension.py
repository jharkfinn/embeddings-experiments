"""
vLLM worker extension for signal extraction.

Hooks store GPU tensors. Async CPU transfer overlaps with next batch's compute.
"""

import math
import sys
import threading
import torch
import torch.nn.functional as F
import numpy as np


class SignalExtractorExtension:

    def setup_hooks(self, arch_json):
        import json
        self.arch = json.loads(arch_json)
        self.capture = {}
        self._handles = []
        self._save_thread = None
        self._save_error = None

        model = self.model_runner.model
        text_model = model.language_model.model
        layers = list(text_model.layers)

        num_q_heads = self.arch["num_attention_heads"]
        num_kv_heads = self.arch["num_key_value_heads"]
        head_dim = self.arch["head_dim"]
        q_dim = num_q_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        for layer_idx, layer in enumerate(layers):
            # Hook A: Router gate
            mlp = getattr(layer, "mlp", None)
            if mlp is not None:
                gate = getattr(mlp, "gate", None) or getattr(mlp, "router", None)
                if gate is not None:
                    def _router_hook(li):
                        def hook(_mod, _inp, out):
                            logits = out[0] if isinstance(out, tuple) else out
                            probs = F.softmax(logits.float(), dim=-1).half()
                            self.capture[f"router_{li}"] = probs.detach()
                        return hook
                    self._handles.append(
                        gate.register_forward_hook(_router_hook(layer_idx)))

            # Hooks B & C: fused qkv_proj + o_proj
            if layer_idx in self.arch["attn_layer_indices"]:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    qkv_proj = getattr(attn, "qkv_proj", None)
                    if qkv_proj is not None:
                        def _qkv_hook(li, qd, kvd, nhq, hd):
                            def hook(_mod, _inp, out):
                                t = out[0] if isinstance(out, tuple) else out
                                q_gate, k, v = t.split(
                                    [qd * 2, kvd, kvd], dim=-1)
                                orig_shape = q_gate.shape[:-1]
                                q_gate = q_gate.view(*orig_shape, nhq, hd * 2)
                                q = q_gate[..., :hd].reshape(*orig_shape, qd)
                                self.capture[f"q_proj_{li}"] = q.detach()
                                self.capture[f"k_proj_{li}"] = k.detach()
                                self.capture[f"v_proj_{li}"] = v.detach()
                            return hook
                        self._handles.append(
                            qkv_proj.register_forward_hook(
                                _qkv_hook(layer_idx, q_dim, kv_dim,
                                          num_q_heads, head_dim)))

                    o_proj = getattr(attn, "o_proj", None)
                    if o_proj is not None:
                        def _oproj_hook(li):
                            def hook(_mod, _inp, out):
                                t = out[0] if isinstance(out, tuple) else out
                                self.capture[f"o_proj_{li}"] = t.detach()
                            return hook
                        self._handles.append(
                            o_proj.register_forward_hook(_oproj_hook(layer_idx)))

            # Hook D: Final layer
            if layer_idx == len(layers) - 1:
                def _final_hook(li):
                    def hook(_mod, _inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        self.capture["hs_final"] = hs.detach()
                    return hook
                self._handles.append(
                    layer.register_forward_hook(_final_hook(layer_idx)))

        return f"Installed {len(self._handles)} hooks on {len(layers)} layers"

    def clear_capture(self):
        self.capture.clear()
        return True

    def get_architecture(self):
        model = self.model_runner.model
        text_model = model.language_model.model
        layers = list(text_model.layers)
        config = text_model.config

        layer_types = []
        attn_layer_indices = []
        for idx, layer in enumerate(layers):
            lt = getattr(layer, "layer_type", None)
            if lt is None and hasattr(config, "layer_types"):
                lt = config.layer_types[idx]
            layer_types.append(lt)
            if lt == "full_attention":
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

    def _wait_for_save(self):
        """Wait for any in-flight background save to finish."""
        if self._save_thread is not None:
            self._save_thread.join()
            self._save_thread = None
            if self._save_error is not None:
                err = self._save_error
                self._save_error = None
                raise RuntimeError(f"Background save failed: {err}")

    def transfer_and_save_async(self, batch_info_json):
        """Transfer captured GPU tensors to CPU and save in background thread.

        This does the GPU->CPU transfer synchronously (must happen before
        GPU tensors are overwritten by next batch), then hands off the
        CPU tensors to a background thread for numpy conversion + disk write.
        """
        import json
        self._wait_for_save()  # ensure previous save finished

        batch_info = json.loads(batch_info_json)
        arch = self.arch
        num_q_heads = arch["num_attention_heads"]
        num_kv_heads = arch["num_key_value_heads"]
        head_dim = arch["head_dim"]

        # Transfer all needed slices to CPU NOW (synchronous)
        # This is the expensive part but must happen before next forward pass
        cpu_payloads = []
        for info in batch_info:
            echo_start, echo_end, token_ids, path, text_id, kind = info
            seq_len = echo_end - echo_start

            payload = {}
            payload["token_ids"] = np.array(token_ids, dtype=np.int32)
            payload["seq_len"] = np.array(seq_len, dtype=np.int32)
            payload["text_id"] = np.asarray(text_id)
            payload["kind"] = np.asarray(kind)

            # Router — only save topk ids and weights (not full softmax)
            for layer_idx in range(arch["num_layers"]):
                key = f"router_{layer_idx}"
                if key in self.capture:
                    probs = self.capture[key][echo_start:echo_end]
                    topk_vals, topk_ids = probs.float().topk(9, dim=-1)
                    payload[f"router_topk_weights_{layer_idx}"] = (
                        topk_vals.half().cpu().numpy())
                    payload[f"router_topk_ids_{layer_idx}"] = (
                        topk_ids.short().cpu().numpy())

            # Q, K, V, attn_out
            for layer_idx in arch["attn_layer_indices"]:
                for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    key = f"{proj}_{layer_idx}"
                    if key in self.capture:
                        save_key = proj.replace("_proj", "")
                        if proj == "o_proj":
                            save_key = "attn_out"
                        payload[f"{save_key}_{layer_idx}"] = (
                            self.capture[key][echo_start:echo_end]
                            .half().cpu().numpy())

            # Last-token attention weights from Q and K (on GPU, fast)
            for layer_idx in arch["attn_layer_indices"]:
                q_key = f"q_proj_{layer_idx}"
                k_key = f"k_proj_{layer_idx}"
                if q_key in self.capture and k_key in self.capture:
                    q_echo = self.capture[q_key][echo_start:echo_end].float()
                    k_echo = self.capture[k_key][echo_start:echo_end].float()
                    q_last = q_echo[-1].reshape(num_q_heads, head_dim)
                    k_all = k_echo.reshape(seq_len, num_kv_heads, head_dim)
                    hpg = num_q_heads // num_kv_heads
                    scale = 1.0 / math.sqrt(head_dim)
                    q_g = q_last.reshape(num_kv_heads, hpg, head_dim)
                    k_t = k_all.permute(1, 0, 2)
                    scores = torch.bmm(
                        q_g.reshape(num_kv_heads * hpg, 1, head_dim),
                        k_t.repeat_interleave(hpg, dim=0).transpose(1, 2),
                    ).reshape(num_kv_heads, hpg, seq_len) * scale
                    weights = F.softmax(scores, dim=-1).mean(dim=(0, 1))
                    payload[f"raw_qk_similarity_last_{layer_idx}"] = (
                        weights.half().cpu().numpy())

            # Final hidden states
            if "hs_final" in self.capture:
                payload["hs_final"] = (
                    self.capture["hs_final"][echo_start:echo_end]
                    .half().cpu().numpy())

            cpu_payloads.append((path, payload))

        # Now save to disk in background thread
        def _save_worker(payloads):
            try:
                for path, payload in payloads:
                    torch.save(payload, path)
            except Exception as e:
                self._save_error = e

        self._save_thread = threading.Thread(
            target=_save_worker, args=(cpu_payloads,))
        self._save_thread.start()

        return len(cpu_payloads)

    def flush_saves(self):
        """Wait for any pending background save to complete."""
        self._wait_for_save()
        return True

    def get_captured_payload(self, echo_start, echo_end, token_ids_list):
        """Synchronous single-payload build for test pass."""
        arch = self.arch
        seq_len = echo_end - echo_start
        num_q_heads = arch["num_attention_heads"]
        num_kv_heads = arch["num_key_value_heads"]
        head_dim = arch["head_dim"]

        payload = {}
        payload["token_ids"] = np.array(token_ids_list, dtype=np.int32)
        payload["seq_len"] = np.array(seq_len, dtype=np.int32)

        for layer_idx in range(arch["num_layers"]):
            key = f"router_{layer_idx}"
            if key in self.capture:
                probs = self.capture[key][echo_start:echo_end]
                topk_vals, topk_ids = probs.float().topk(9, dim=-1)
                payload[f"router_topk_weights_{layer_idx}"] = (
                    topk_vals.half().cpu().numpy())
                payload[f"router_topk_ids_{layer_idx}"] = (
                    topk_ids.short().cpu().numpy())

        for layer_idx in arch["attn_layer_indices"]:
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"{proj}_{layer_idx}"
                if key in self.capture:
                    save_key = proj.replace("_proj", "")
                    if proj == "o_proj":
                        save_key = "attn_out"
                    payload[f"{save_key}_{layer_idx}"] = (
                        self.capture[key][echo_start:echo_end]
                        .half().cpu().numpy())

        if "hs_final" in self.capture:
            payload["hs_final"] = (
                self.capture["hs_final"][echo_start:echo_end]
                .half().cpu().numpy())

        return payload

    def remove_hooks(self):
        self._wait_for_save()
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.capture.clear()
        return True

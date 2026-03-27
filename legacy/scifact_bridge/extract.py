from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import queue
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .runtime_bootstrap import bootstrap_workspace_env
    from .experiment_utils import (
        EPS,
        MAX_LENGTH,
        MODEL_NAME,
        SHARED_EXPERT_INDEX,
        TextRecord,
        architecture_path,
        build_records,
        ensure_project_dirs,
        human_bytes,
        human_duration,
        load_scifact,
        make_filename,
        write_json,
        write_manifest,
    )
except ImportError:
    from runtime_bootstrap import bootstrap_workspace_env
    from experiment_utils import (
        EPS,
        MAX_LENGTH,
        MODEL_NAME,
        SHARED_EXPERT_INDEX,
        TextRecord,
        architecture_path,
        build_records,
        ensure_project_dirs,
        human_bytes,
        human_duration,
        load_scifact,
        make_filename,
        write_json,
        write_manifest,
    )

bootstrap_workspace_env()

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers.models.bamba.modeling_bamba import apply_rotary_pos_emb
except ImportError:  # pragma: no cover - fallback for future package refactors.
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract causal and KV-rerouted Qwen3.5 MoE signals on SciFact.")
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/kv_moee_experiment"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--reroute-bias", type=float, default=1.0)
    parser.add_argument("--torch-dtype", type=str, default="float16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--writer-queue-size", type=int, default=4)
    parser.add_argument("--paper-id-sample-size", type=int, default=1000)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported --torch-dtype value: {name}")
    return mapping[name]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_heads * n_rep, seq_len, head_dim)


def prepend_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim != 4:
        raise ValueError(f"Expected a 4D additive attention mask, got shape {tuple(attention_mask.shape)}")
    zero_column = torch.zeros(
        attention_mask.shape[0],
        attention_mask.shape[1],
        attention_mask.shape[2],
        1,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([zero_column, attention_mask], dim=-1)


def validate_payload_finite(payload: dict[str, np.ndarray], text_id: str, pass_name: str) -> None:
    for key, value in payload.items():
        array = np.asarray(value)
        if array.dtype.kind not in {"f", "c"}:
            continue
        if not np.isfinite(array).all():
            raise RuntimeError(
                f"Non-finite values detected in pass={pass_name} text_id={text_id} field={key} "
                f"shape={array.shape} dtype={array.dtype}"
            )


def get_input_device(text_model: torch.nn.Module) -> torch.device:
    return text_model.embed_tokens.weight.device


def resolve_text_model(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    candidates = [
        ("model.language_model", getattr(getattr(model, "model", None), "language_model", None)),
        ("language_model", getattr(model, "language_model", None)),
        ("model", getattr(model, "model", None)),
    ]
    for path, candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens"):
            return candidate, path
    raise RuntimeError("Could not resolve the Qwen text backbone from the loaded model.")


@dataclass
class PassRuntimeState:
    name: str
    num_layers: int
    attn_layer_indices: list[int]
    reroute_mode: str = "none"
    reroute_bias: float = 1.0
    reroute_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    reroute_layer_indices: set[int] | None = None

    def __post_init__(self) -> None:
        if self.reroute_mode not in {"none", "replay", "self"}:
            raise ValueError(f"Unsupported reroute_mode: {self.reroute_mode}")
        if self.reroute_layer_indices is None:
            if self.reroute_mode == "none":
                self.reroute_layer_indices = set()
            else:
                self.reroute_layer_indices = set(self.attn_layer_indices)
        else:
            self.reroute_layer_indices = set(self.reroute_layer_indices)
        self.sequence_lengths: list[int] = []
        self.last_token_positions: torch.Tensor | None = None
        self.token_mask: torch.Tensor | None = None
        self.hs_last_token: list[torch.Tensor | None] = [None] * self.num_layers
        self.hs_mean: list[torch.Tensor | None] = [None] * self.num_layers
        self.hs_final_all_tokens: torch.Tensor | None = None
        self.routing_indices: list[torch.Tensor | None] = [None] * self.num_layers
        self.routing_weights: list[torch.Tensor | None] = [None] * self.num_layers
        self.routing_full_last: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_out_pool: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_out_mask: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_out_counts: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_ffn_pool: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_ffn_mask: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_ffn_counts: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_contrib_pool: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_contrib_mask: list[torch.Tensor | None] = [None] * self.num_layers
        self.expert_contrib_counts: list[torch.Tensor | None] = [None] * self.num_layers
        self.va_all_tokens: dict[int, torch.Tensor] = {}
        self.attn_weights_last: dict[int, torch.Tensor] = {}
        self.attn_weights_reroute: dict[int, torch.Tensor] = {}
        self.last_token_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def should_reroute_layer(self, layer_idx: int) -> bool:
        return self.reroute_mode != "none" and layer_idx in self.reroute_layer_indices

    def set_batch_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        self.token_mask = attention_mask.to(dtype=torch.bool)
        lengths = attention_mask.sum(dim=1, dtype=torch.int64)
        self.sequence_lengths = [int(value) for value in lengths.detach().cpu().tolist()]
        self.last_token_positions = lengths.to(device=input_ids.device) - 1

    def finalize_batch(
        self,
        input_ids: torch.Tensor,
        text_ids: list[str],
        architecture: dict[str, Any],
    ) -> list[dict[str, np.ndarray]]:
        if self.last_token_positions is None or self.token_mask is None or not self.sequence_lengths:
            raise RuntimeError("Batch metadata was not initialized before finalize_batch().")

        batch_size = input_ids.shape[0]
        if len(text_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} text ids, received {len(text_ids)}")

        hs_last = torch.stack([tensor for tensor in self.hs_last_token], dim=0).to(torch.float16)
        hs_mean = torch.stack([tensor for tensor in self.hs_mean], dim=0).to(torch.float16)
        hs_final_all = self.hs_final_all_tokens.to(torch.float16)

        routing_indices = torch.stack([tensor for tensor in self.routing_indices], dim=0).to(torch.int16)
        routing_weights = torch.stack([tensor for tensor in self.routing_weights], dim=0).to(torch.float16)
        routing_full_last = torch.stack([tensor for tensor in self.routing_full_last], dim=0).to(torch.float32)

        expert_out_pool = torch.stack([tensor for tensor in self.expert_out_pool], dim=0).to(torch.float16)
        expert_out_mask = torch.stack([tensor for tensor in self.expert_out_mask], dim=0).to(torch.bool)
        expert_out_counts = torch.stack([tensor for tensor in self.expert_out_counts], dim=0).to(torch.int16)
        expert_ffn_pool = torch.stack([tensor for tensor in self.expert_ffn_pool], dim=0).to(torch.float32)
        expert_ffn_mask = torch.stack([tensor for tensor in self.expert_ffn_mask], dim=0).to(torch.bool)
        expert_ffn_counts = torch.stack([tensor for tensor in self.expert_ffn_counts], dim=0).to(torch.int16)
        expert_contrib_pool = torch.stack([tensor for tensor in self.expert_contrib_pool], dim=0).to(torch.float32)
        expert_contrib_mask = torch.stack([tensor for tensor in self.expert_contrib_mask], dim=0).to(torch.bool)
        expert_contrib_counts = torch.stack([tensor for tensor in self.expert_contrib_counts], dim=0).to(torch.int16)

        va_all_list = [self.va_all_tokens[layer_idx].to(torch.float16) for layer_idx in self.attn_layer_indices]
        va_all_tokens = torch.stack(va_all_list, dim=0)

        attn_last = torch.stack([self.attn_weights_last[layer_idx] for layer_idx in self.attn_layer_indices], dim=0)
        attn_last = attn_last.to(torch.float16)
        attn_reroute = torch.stack(
            [self.attn_weights_reroute[layer_idx] for layer_idx in self.attn_layer_indices], dim=0
        ).to(torch.float16)

        payloads: list[dict[str, np.ndarray]] = []
        for batch_idx, text_id in enumerate(text_ids):
            seq_len = self.sequence_lengths[batch_idx]
            sample_routing_indices = routing_indices[:, batch_idx, :seq_len]
            sample_va_all_tokens = va_all_tokens[:, batch_idx, :seq_len]
            sample_va_mean = sample_va_all_tokens.float().mean(dim=1).to(torch.float16)

            va_expert_pool = []
            va_expert_mask = []
            va_expert_counts = []
            for attn_offset, layer_idx in enumerate(self.attn_layer_indices):
                layer_values = sample_va_all_tokens[attn_offset].float()
                layer_indices = sample_routing_indices[layer_idx].cpu().numpy()
                active_experts = layer_indices[-1]

                pool = torch.zeros((9, layer_values.shape[-1]), dtype=torch.float32)
                counts = torch.zeros(9, dtype=torch.int16)
                mask = torch.zeros(9, dtype=torch.bool)

                for slot, expert_id in enumerate(active_experts.tolist()):
                    if expert_id == SHARED_EXPERT_INDEX:
                        token_mask = np.ones(seq_len, dtype=bool)
                    else:
                        token_mask = (layer_indices[:, :8] == expert_id).any(axis=1)
                    token_count = int(token_mask.sum())
                    counts[slot] = token_count
                    mask[slot] = token_count > 0
                    if token_count > 0:
                        token_selector = torch.from_numpy(token_mask).to(layer_values.device)
                        pool[slot] = layer_values[token_selector].mean(dim=0)

                va_expert_pool.append(pool.to(torch.float16))
                va_expert_mask.append(mask)
                va_expert_counts.append(counts)

            payloads.append(
                {
                    "token_ids": input_ids[batch_idx, :seq_len].detach().cpu().numpy().astype(np.int32, copy=False),
                    "seq_len": np.asarray(seq_len, dtype=np.int32),
                    "text_id": np.asarray(text_id),
                    "hs_last_token": hs_last[:, batch_idx].cpu().numpy(),
                    "hs_mean": hs_mean[:, batch_idx].cpu().numpy(),
                    "hs_final_all_tokens": hs_final_all[batch_idx, :seq_len].cpu().numpy(),
                    "routing_indices": sample_routing_indices.cpu().numpy(),
                    "routing_weights": routing_weights[:, batch_idx, :seq_len].cpu().numpy(),
                    "routing_full_last": routing_full_last[:, batch_idx].cpu().numpy(),
                    "va_mean": sample_va_mean.cpu().numpy(),
                    "va_all_tokens": sample_va_all_tokens.cpu().numpy(),
                    "va_expert_pool": torch.stack(va_expert_pool, dim=0).cpu().numpy(),
                    "va_expert_mask": torch.stack(va_expert_mask, dim=0).cpu().numpy(),
                    "va_expert_counts": torch.stack(va_expert_counts, dim=0).cpu().numpy(),
                    "expert_out_pool": expert_out_pool[:, batch_idx].cpu().numpy(),
                    "expert_out_mask": expert_out_mask[:, batch_idx].cpu().numpy(),
                    "expert_out_counts": expert_out_counts[:, batch_idx].cpu().numpy(),
                    "expert_ffn_pool": expert_ffn_pool[:, batch_idx].cpu().numpy(),
                    "expert_ffn_mask": expert_ffn_mask[:, batch_idx].cpu().numpy(),
                    "expert_ffn_counts": expert_ffn_counts[:, batch_idx].cpu().numpy(),
                    "expert_contrib_pool": expert_contrib_pool[:, batch_idx].cpu().numpy(),
                    "expert_contrib_mask": expert_contrib_mask[:, batch_idx].cpu().numpy(),
                    "expert_contrib_counts": expert_contrib_counts[:, batch_idx].cpu().numpy(),
                    "attn_weights_last": attn_last[:, batch_idx, :, :seq_len].cpu().numpy(),
                    "attn_weights_reroute": attn_reroute[:, batch_idx].cpu().numpy(),
                    "shared_expert_index": np.asarray(SHARED_EXPERT_INDEX, dtype=np.int16),
                    "value_dim": np.asarray(architecture["value_dim"], dtype=np.int32),
                    "hidden_size": np.asarray(architecture["hidden_size"], dtype=np.int32),
                }
            )
        return payloads


class InstrumentationContext:
    def __init__(self, text_model: torch.nn.Module, architecture: dict[str, Any], reroute_bias: float) -> None:
        self.text_model = text_model
        self.architecture = architecture
        self.reroute_bias = reroute_bias
        self.runtime: PassRuntimeState | None = None
        self._original_methods: dict[tuple[int, str], Any] = {}
        self._hooks: list[Any] = []

    def install(self) -> None:
        for layer_idx, layer in enumerate(self.text_model.layers):
            if getattr(layer, "layer_type", None) == "full_attention":
                attn = layer.self_attn
                self._original_methods[(id(attn), "forward")] = attn.forward
                attn.forward = types.MethodType(self._make_attention_forward(layer_idx), attn)

            mlp = layer.mlp
            if all(hasattr(mlp, name) for name in ("gate", "experts", "shared_expert", "shared_expert_gate")):
                self._original_methods[(id(mlp), "forward")] = mlp.forward
                mlp.forward = types.MethodType(self._make_mlp_forward(layer_idx), mlp)

            hook = layer.register_forward_hook(self._make_layer_hook(layer_idx))
            self._hooks.append(hook)

    def uninstall(self) -> None:
        for layer in self.text_model.layers:
            if getattr(layer, "layer_type", None) == "full_attention":
                attn = layer.self_attn
                original = self._original_methods.get((id(attn), "forward"))
                if original is not None:
                    attn.forward = original

            mlp = layer.mlp
            original = self._original_methods.get((id(mlp), "forward"))
            if original is not None:
                mlp.forward = original

        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_layer_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if self.runtime is None:
                return
            hidden_states = output[0] if isinstance(output, tuple) else output
            if self.runtime.last_token_positions is None or self.runtime.token_mask is None:
                raise RuntimeError("Batch metadata is missing for hidden state capture.")
            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_positions = self.runtime.last_token_positions.to(hidden_states.device)
            token_mask = self.runtime.token_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            token_counts = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
            self.runtime.hs_last_token[layer_idx] = hidden_states[batch_indices, last_positions].detach()
            self.runtime.hs_mean[layer_idx] = (
                (hidden_states * token_mask.unsqueeze(-1)).sum(dim=1) / token_counts
            ).detach()
            if layer_idx == self.architecture["num_layers"] - 1:
                self.runtime.hs_final_all_tokens = hidden_states.detach()

        return hook

    def _make_attention_forward(self, layer_idx: int):
        def forward(
            module,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values: Any = None,
            cache_position: torch.LongTensor | None = None,
            **_kwargs,
        ):
            runtime = self.runtime
            if runtime is None:
                raise RuntimeError("Attention wrapper invoked without an active runtime.")
            if past_key_values is not None:
                raise RuntimeError("This experiment expects full-sequence extraction with use_cache=False.")
            if runtime.last_token_positions is None:
                raise RuntimeError("Batch metadata is missing for attention capture.")

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)

            query_states, gate = torch.chunk(
                module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(*input_shape, -1)

            query_states = module.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_positions = runtime.last_token_positions.to(hidden_states.device)
            runtime.last_token_kv[layer_idx] = (
                key_states[batch_indices, :, last_positions, :].unsqueeze(2).detach(),
                value_states[batch_indices, :, last_positions, :].unsqueeze(2).detach(),
            )

            value_repeated = repeat_kv(value_states, module.num_key_value_groups)
            runtime.va_all_tokens[layer_idx] = value_repeated.transpose(1, 2).reshape(
                hidden_states.shape[0], hidden_states.shape[1], -1
            ).detach()

            kv_key = key_states
            kv_value = value_states
            kv_attention_mask = attention_mask
            reroute_layer = runtime.should_reroute_layer(layer_idx)
            if reroute_layer:
                if runtime.reroute_mode == "replay":
                    if runtime.reroute_kv is None or layer_idx not in runtime.reroute_kv:
                        raise RuntimeError(f"Missing reroute KV cache for attention layer {layer_idx}.")
                    prepend_k, prepend_v = runtime.reroute_kv[layer_idx]
                else:
                    prepend_k = key_states[batch_indices, :, last_positions, :].unsqueeze(2).detach()
                    prepend_v = value_states[batch_indices, :, last_positions, :].unsqueeze(2).detach()
                kv_key = torch.cat([prepend_k.to(key_states.device), key_states], dim=2)
                kv_value = torch.cat([prepend_v.to(value_states.device), value_states], dim=2)
                kv_attention_mask = prepend_attention_mask(attention_mask)

            key_repeated = repeat_kv(kv_key, module.num_key_value_groups)
            value_repeated_for_attn = repeat_kv(kv_value, module.num_key_value_groups)

            attn_logits = torch.matmul(query_states, key_repeated.transpose(2, 3)) * module.scaling
            if kv_attention_mask is not None:
                attn_logits = attn_logits + kv_attention_mask
            if reroute_layer and runtime.reroute_bias != 0.0:
                attn_logits[..., 0] += runtime.reroute_bias

            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_repeated_for_attn)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
            attn_output = attn_output * torch.sigmoid(gate)
            attn_output = module.o_proj(attn_output)

            last_weights = attn_weights[batch_indices, :, last_positions, :].detach()
            if reroute_layer:
                runtime.attn_weights_reroute[layer_idx] = last_weights[:, :, 0]
                runtime.attn_weights_last[layer_idx] = last_weights[:, :, 1:]
            else:
                runtime.attn_weights_reroute[layer_idx] = torch.zeros(
                    last_weights.shape[0],
                    last_weights.shape[1],
                    device=last_weights.device,
                    dtype=last_weights.dtype,
                )
                runtime.attn_weights_last[layer_idx] = last_weights

            return attn_output, attn_weights

        return forward

    def _make_mlp_forward(self, layer_idx: int):
        def forward(module, hidden_states: torch.Tensor):
            runtime = self.runtime
            if runtime is None:
                raise RuntimeError("MoE wrapper invoked without an active runtime.")
            if runtime.last_token_positions is None or runtime.token_mask is None:
                raise RuntimeError("Batch metadata is missing for MoE capture.")

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_flat = hidden_states.reshape(-1, hidden_dim)

            shared_expert_output = module.shared_expert(hidden_states_flat)
            router_full, routing_weights, selected_experts = module.gate(hidden_states_flat)
            expert_output = module.experts(hidden_states_flat, selected_experts, routing_weights)
            shared_gate = torch.sigmoid(module.shared_expert_gate(hidden_states_flat))
            shared_weighted = shared_gate * shared_expert_output

            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            last_positions = runtime.last_token_positions.to(hidden_states.device)
            token_mask = runtime.token_mask.to(hidden_states.device)

            pool_sums = torch.zeros((batch_size, 9, hidden_dim), device=hidden_states.device, dtype=torch.float32)
            counts = torch.zeros((batch_size, 9), device=hidden_states.device, dtype=torch.int16)
            expert_ffn_sums = torch.zeros((batch_size, 9, hidden_dim), device=hidden_states.device, dtype=torch.float32)
            expert_ffn_counts = torch.zeros((batch_size, 9), device=hidden_states.device, dtype=torch.int16)
            expert_contrib_sums = torch.zeros((batch_size, 9, hidden_dim), device=hidden_states.device, dtype=torch.float32)
            expert_contrib_counts = torch.zeros((batch_size, 9), device=hidden_states.device, dtype=torch.int16)

            expert_output_tokens = expert_output.reshape(batch_size, sequence_length, hidden_dim)
            selected_experts_tokens = selected_experts.reshape(batch_size, sequence_length, -1)
            routing_weights_tokens = routing_weights.reshape(batch_size, sequence_length, -1)
            shared_weighted_tokens = shared_weighted.reshape(batch_size, sequence_length, hidden_dim)
            shared_expert_tokens = shared_expert_output.reshape(batch_size, sequence_length, hidden_dim)
            last_token_experts = selected_experts_tokens[batch_indices, last_positions]

            for sample_idx in range(batch_size):
                sample_active_tokens = token_mask[sample_idx]
                sample_hidden_states = hidden_states[sample_idx]
                sample_selected = selected_experts_tokens[sample_idx]
                sample_weights = routing_weights_tokens[sample_idx]
                routed_slot_map = {
                    int(expert_id): slot for slot, expert_id in enumerate(last_token_experts[sample_idx].tolist())
                }
                for expert_idx, slot in routed_slot_map.items():
                    sample_token_mask = sample_active_tokens & (sample_selected == expert_idx).any(dim=-1)
                    token_count = int(sample_token_mask.sum().item())
                    counts[sample_idx, slot] = token_count
                    if token_count > 0:
                        pool_sums[sample_idx, slot] = expert_output_tokens[sample_idx, sample_token_mask].sum(dim=0).to(
                            pool_sums.dtype
                        )
                        selected_hidden = sample_hidden_states[sample_token_mask].to(torch.float32)
                        gate_up_weight = module.experts.gate_up_proj[expert_idx].to(dtype=torch.float32)
                        down_proj_weight = module.experts.down_proj[expert_idx].to(dtype=torch.float32)
                        gate, up = F.linear(selected_hidden, gate_up_weight).chunk(2, dim=-1)
                        raw_expert_output = F.linear(module.experts.act_fn(gate) * up, down_proj_weight)
                        expert_token_weights = (
                            sample_weights[sample_token_mask]
                            * (sample_selected[sample_token_mask] == expert_idx).to(sample_weights.dtype)
                        ).sum(dim=-1)
                        weighted_expert_output = raw_expert_output * expert_token_weights.to(torch.float32).unsqueeze(-1)
                        expert_ffn_sums[sample_idx, slot] = raw_expert_output.sum(dim=0).to(expert_ffn_sums.dtype)
                        expert_ffn_counts[sample_idx, slot] = token_count
                        expert_contrib_sums[sample_idx, slot] = weighted_expert_output.sum(dim=0).to(
                            expert_contrib_sums.dtype
                        )
                        expert_contrib_counts[sample_idx, slot] = token_count

            expert_output = expert_output + shared_weighted
            for sample_idx in range(batch_size):
                sample_active_tokens = token_mask[sample_idx]
                pool_sums[sample_idx, 8] = shared_weighted_tokens[sample_idx, sample_active_tokens].sum(dim=0).to(
                    pool_sums.dtype
                )
                counts[sample_idx, 8] = int(sample_active_tokens.sum().item())
                shared_token_count = int(sample_active_tokens.sum().item())
                if shared_token_count > 0:
                    expert_ffn_sums[sample_idx, 8] = shared_expert_tokens[sample_idx, sample_active_tokens].sum(dim=0).to(
                        expert_ffn_sums.dtype
                    )
                    expert_ffn_counts[sample_idx, 8] = shared_token_count
                    expert_contrib_sums[sample_idx, 8] = shared_weighted_tokens[sample_idx, sample_active_tokens].sum(
                        dim=0
                    ).to(expert_contrib_sums.dtype)
                    expert_contrib_counts[sample_idx, 8] = shared_token_count

            pooled = torch.zeros_like(pool_sums, dtype=torch.float16)
            mask = counts > 0
            if mask.any():
                pooled[mask] = (pool_sums[mask] / counts[mask].to(pool_sums.dtype).unsqueeze(-1)).to(torch.float16)
            expert_ffn_pooled = torch.zeros_like(expert_ffn_sums, dtype=torch.float32)
            expert_ffn_mask = expert_ffn_counts > 0
            if expert_ffn_mask.any():
                expert_ffn_pooled[expert_ffn_mask] = (
                    expert_ffn_sums[expert_ffn_mask]
                    / expert_ffn_counts[expert_ffn_mask].to(expert_ffn_sums.dtype).unsqueeze(-1)
                ).to(torch.float32)
            expert_contrib_pooled = torch.zeros_like(expert_contrib_sums, dtype=torch.float32)
            expert_contrib_mask = expert_contrib_counts > 0
            if expert_contrib_mask.any():
                expert_contrib_pooled[expert_contrib_mask] = (
                    expert_contrib_sums[expert_contrib_mask]
                    / expert_contrib_counts[expert_contrib_mask].to(expert_contrib_sums.dtype).unsqueeze(-1)
                ).to(torch.float32)

            selected_experts = selected_experts.reshape(batch_size, sequence_length, -1)
            routing_weights = routing_weights.reshape(batch_size, sequence_length, -1)
            router_full = router_full.reshape(batch_size, sequence_length, -1)
            shared_indices = torch.full(
                (batch_size, sequence_length, 1),
                SHARED_EXPERT_INDEX,
                dtype=selected_experts.dtype,
                device=selected_experts.device,
            )
            shared_scores = shared_gate.reshape(batch_size, sequence_length, 1)

            runtime.routing_indices[layer_idx] = torch.cat([selected_experts, shared_indices], dim=-1).detach()
            runtime.routing_weights[layer_idx] = torch.cat([routing_weights, shared_scores], dim=-1).detach()
            runtime.routing_full_last[layer_idx] = router_full[batch_indices, last_positions].detach().to(torch.float32)
            runtime.expert_out_pool[layer_idx] = pooled.detach()
            runtime.expert_out_mask[layer_idx] = mask.detach()
            runtime.expert_out_counts[layer_idx] = counts.detach()
            runtime.expert_ffn_pool[layer_idx] = expert_ffn_pooled.detach()
            runtime.expert_ffn_mask[layer_idx] = expert_ffn_mask.detach()
            runtime.expert_ffn_counts[layer_idx] = expert_ffn_counts.detach()
            runtime.expert_contrib_pool[layer_idx] = expert_contrib_pooled.detach()
            runtime.expert_contrib_mask[layer_idx] = expert_contrib_mask.detach()
            runtime.expert_contrib_counts[layer_idx] = expert_contrib_counts.detach()

            return expert_output.reshape(batch_size, sequence_length, hidden_dim)

        return forward


def inspect_model_architecture(
    model: torch.nn.Module,
    text_model: torch.nn.Module,
    model_path: str,
    project_root: Path,
    model_name: str,
) -> dict[str, Any]:
    config = text_model.config
    layers = list(text_model.layers)
    layer_types = [getattr(layer, "layer_type", config.layer_types[idx]) for idx, layer in enumerate(layers)]
    attn_layer_indices = [idx for idx, value in enumerate(layer_types) if value == "full_attention"]
    delta_layer_indices = [idx for idx, value in enumerate(layer_types) if value == "linear_attention"]

    first_delta = layers[delta_layer_indices[0]]
    first_attn = layers[attn_layer_indices[0]]
    value_dim = first_attn.self_attn.num_key_value_groups * config.num_key_value_heads * config.head_dim
    linear_value_dim = config.linear_num_value_heads * config.linear_value_head_dim

    architecture = {
        "model_name": model_name,
        "model_class": type(model).__name__,
        "text_model_class": type(text_model).__name__,
        "text_model_path": model_path,
        "num_layers": len(layers),
        "layer_types": layer_types,
        "attn_layer_indices": attn_layer_indices,
        "delta_layer_indices": delta_layer_indices,
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(config.head_dim),
        "value_dim": int(value_dim),
        "linear_value_dim": int(linear_value_dim),
        "num_experts": int(config.num_experts),
        "num_experts_per_tok": int(config.num_experts_per_tok),
        "shared_expert_index": SHARED_EXPERT_INDEX,
        "attention_module_attrs": sorted(name for name in dir(first_attn.self_attn) if not name.startswith("_")),
        "delta_module_attrs": sorted(name for name in dir(first_delta.linear_attn) if not name.startswith("_")),
        "moe_module_attrs": sorted(name for name in dir(first_attn.mlp) if not name.startswith("_")),
        "router_module_attrs": sorted(name for name in dir(first_attn.mlp.gate) if not name.startswith("_")),
        "experts_module_attrs": sorted(name for name in dir(first_attn.mlp.experts) if not name.startswith("_")),
    }

    text = [
        f"Model class: {type(model).__name__}",
        f"Text backbone path: {model_path}",
        f"Text model class: {type(text_model).__name__}",
        f"Number of layers: {len(layers)}",
        f"Layer types: {layer_types}",
        f"Attention layer indices: {attn_layer_indices}",
        f"DeltaNet layer indices: {delta_layer_indices}",
        f"Hidden size: {config.hidden_size}",
        f"Attention heads: {config.num_attention_heads}",
        f"KV heads: {config.num_key_value_heads}",
        f"Head dim: {config.head_dim}",
        f"Flattened value dim: {value_dim}",
        f"Linear attention flattened value dim: {linear_value_dim}",
        f"Number of routed experts: {config.num_experts}",
        f"Top-k routed experts per token: {config.num_experts_per_tok}",
        f"Shared expert index: {SHARED_EXPERT_INDEX}",
        "",
        f"First DeltaNet layer type: {type(first_delta).__name__}",
        f"First DeltaNet token mixer type: {type(first_delta.linear_attn).__name__}",
        f"First attention layer type: {type(first_attn).__name__}",
        f"First attention token mixer type: {type(first_attn.self_attn).__name__}",
        f"First MoE block type: {type(first_attn.mlp).__name__}",
        "",
        "First DeltaNet layer attrs:",
        ", ".join(architecture["delta_module_attrs"]),
        "",
        "First attention layer attrs:",
        ", ".join(architecture["attention_module_attrs"]),
        "",
        "First MoE block attrs:",
        ", ".join(architecture["moe_module_attrs"]),
        "",
        "First router attrs:",
        ", ".join(architecture["router_module_attrs"]),
        "",
        "First experts attrs:",
        ", ".join(architecture["experts_module_attrs"]),
        "",
        "Full model repr:",
        repr(model),
    ]

    inspection_path = project_root / "results" / "model_architecture.txt"
    inspection_path.write_text("\n".join(text), encoding="utf-8")
    write_json(architecture_path(project_root), architecture)
    return architecture


def save_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


@dataclass(frozen=True)
class RecordBatch:
    kind: str
    records: list[TextRecord]
    positions: list[int]


@dataclass(frozen=True)
class ExtractionPassSpec:
    name: str
    label: str
    output_dir_key: str
    use_paper_prompt: bool
    reroute_mode: str
    reroute_source: str | None
    selected_layers: set[int] | None = None


def paper_prompt_text(record: TextRecord) -> str:
    label = "Context" if record.kind == "doc" else "Query"
    return f"{label}: {record.text}\nCompress the {label} in one word:"


def texts_for_pass(records: list[TextRecord], use_paper_prompt: bool) -> list[str]:
    if not use_paper_prompt:
        return [record.text for record in records]
    return [paper_prompt_text(record) for record in records]


class AsyncPayloadWriter:
    def __init__(self, max_queue_size: int) -> None:
        self._queue: queue.Queue[tuple[Path, dict[str, np.ndarray]] | None] = queue.Queue(maxsize=max_queue_size)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="payload-writer", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            path, payload = item
            try:
                save_payload(path, payload)
            except BaseException as exc:  # pragma: no cover - surfaced to caller.
                self._error = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Background payload writer failed: {self._error}") from self._error

    def submit(self, path: Path, payload: dict[str, np.ndarray]) -> None:
        self._raise_if_failed()
        self._queue.put((path, payload))
        self._raise_if_failed()

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()
        self._raise_if_failed()


def build_pending_batches(
    records: list[TextRecord],
    output_dirs: list[Path],
    query_batch_size: int,
    query_length_keys: dict[str, int] | None = None,
) -> list[RecordBatch]:
    batches: list[RecordBatch] = []
    query_buffer: list[TextRecord] = []
    query_positions: list[int] = []
    query_length_key: int | None = None

    def flush_queries() -> None:
        nonlocal query_buffer, query_positions, query_length_key
        if query_buffer:
            batches.append(RecordBatch(kind="query", records=query_buffer, positions=query_positions))
            query_buffer = []
            query_positions = []
            query_length_key = None

    for position, record in enumerate(records, start=1):
        all_paths = [output_dir / record.filename for output_dir in output_dirs]
        if all(path.exists() for path in all_paths):
            print(f"Skipping {record.text_id}, already extracted")
            continue
        if any(path.exists() for path in all_paths):
            for path in all_paths:
                delete_if_exists(path)
            print(f"Resetting partial extraction for {record.text_id}")

        if record.kind == "query" and query_batch_size > 1:
            current_length_key = query_length_keys.get(record.text_id) if query_length_keys is not None else None
            if query_buffer and query_length_key is not None and current_length_key != query_length_key:
                flush_queries()
            query_buffer.append(record)
            query_positions.append(position)
            query_length_key = current_length_key
            if len(query_buffer) >= query_batch_size:
                flush_queries()
            continue

        flush_queries()
        batches.append(RecordBatch(kind=record.kind, records=[record], positions=[position]))

    flush_queries()
    return batches


def maybe_limit_records(records: list, limit_docs: int | None, limit_queries: int | None) -> list:
    docs = [record for record in records if record.kind == "doc"]
    queries = [record for record in records if record.kind == "query"]
    if limit_docs is not None:
        docs = docs[:limit_docs]
    if limit_queries is not None:
        queries = queries[:limit_queries]
    return docs + queries


def build_query_length_keys(
    records: list[TextRecord],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> dict[str, int]:
    query_length_keys: dict[str, int] = {}
    for record in records:
        if record.kind != "query":
            continue
        token_ids = tokenizer(
            paper_prompt_text(record),
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        query_length_keys[record.text_id] = len(token_ids)
    return query_length_keys


def delete_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def estimate_twonn_id(features: torch.Tensor, device: torch.device | None = None) -> float:
    if features.shape[0] < 3:
        return float("nan")
    work_device = device or features.device
    x = features.to(device=work_device, dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    distances = torch.cdist(x, x, p=2)
    distances.diagonal().fill_(float("inf"))
    nearest = torch.topk(distances, k=2, dim=1, largest=False).values.clamp_min(EPS)
    mu = nearest[:, 1] / nearest[:, 0]
    valid = torch.isfinite(mu) & (mu > 1.0 + 1e-6)
    mu = mu[valid]
    if mu.numel() < 3:
        return float("nan")
    mu, _ = torch.sort(mu)
    x_log = torch.log(mu)
    ranks = torch.arange(1, mu.numel() + 1, device=work_device, dtype=torch.float32)
    y_log = -torch.log1p(-(ranks / (mu.numel() + 1.0)))
    denom = torch.sum(x_log * x_log)
    if float(denom.item()) <= 0.0:
        return float("nan")
    slope = torch.sum(x_log * y_log) / denom
    return float(slope.item())


def fallback_paper_layer_selection(architecture: dict[str, Any], sample_size: int, reason: str) -> dict[str, Any]:
    attn_layers = list(architecture["attn_layer_indices"])
    midpoint = len(attn_layers) // 2
    selected_attn_layers = attn_layers[midpoint:] or attn_layers[-1:]
    selected_global_layers = selected_attn_layers.copy()
    return {
        "paper_prompt_doc_template": "Context: {text}\\nCompress the Context in one word:",
        "paper_prompt_query_template": "Query: {text}\\nCompress the Query in one word:",
        "paper_id_sample_size": sample_size,
        "paper_id_scores": [float("nan")] * architecture["num_layers"],
        "paper_id_exclusion_boundary": int(architecture["num_layers"] * 0.2),
        "paper_id_min_layer": selected_attn_layers[0],
        "paper_selected_global_layers": selected_global_layers,
        "paper_selected_attn_layer_indices": selected_attn_layers,
        "paper_layer_selection_strategy": f"fallback_second_half_attention:{reason}",
    }


def compute_paper_layer_selection(
    model: torch.nn.Module,
    tokenizer,
    records: list[TextRecord],
    architecture: dict[str, Any],
    input_device: torch.device,
    max_length: int,
    sample_size: int,
) -> dict[str, Any]:
    sample_records = [record for record in records if record.kind == "doc"][:sample_size]
    if not sample_records:
        sample_records = records[:sample_size]
    if not sample_records:
        raise RuntimeError("No records available for paper layer selection.")
    if len(sample_records) < 3:
        return fallback_paper_layer_selection(architecture, len(sample_records), reason="too_few_samples")

    features_by_layer: list[list[torch.Tensor]] = [[] for _ in range(architecture["num_layers"])]
    started_at = time.perf_counter()
    for index, record in enumerate(sample_records, start=1):
        encoded = tokenizer(
            [paper_prompt_text(record)],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        model_inputs = {key: value.to(input_device) for key, value in encoded.items()}
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            last_positions = torch.full(
                (model_inputs["input_ids"].shape[0],),
                model_inputs["input_ids"].shape[1] - 1,
                device=input_device,
                dtype=torch.long,
            )
        else:
            last_positions = attention_mask.sum(dim=1, dtype=torch.int64).to(device=input_device) - 1

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states for paper layer selection.")
        layer_hidden_states = hidden_states[1:] if len(hidden_states) == architecture["num_layers"] + 1 else hidden_states
        if len(layer_hidden_states) != architecture["num_layers"]:
            raise RuntimeError(
                f"Expected {architecture['num_layers']} hidden-state tensors, received {len(layer_hidden_states)}."
            )

        for layer_idx, layer_hidden in enumerate(layer_hidden_states):
            vector = layer_hidden[0, last_positions[0]].detach().to(device="cpu", dtype=torch.float32)
            features_by_layer[layer_idx].append(vector)

        del outputs, layer_hidden_states, hidden_states, model_inputs, encoded
        if index % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if index % 100 == 0 or index == len(sample_records):
            elapsed = time.perf_counter() - started_at
            print(
                f"[Paper-ID] Collected {index}/{len(sample_records)} samples "
                f"in {human_duration(elapsed)}"
            )

    id_device = input_device if input_device.type == "cuda" else torch.device("cpu")
    id_scores: list[float] = []
    for layer_idx, layer_vectors in enumerate(features_by_layer):
        layer_matrix = torch.stack(layer_vectors, dim=0)
        layer_id = estimate_twonn_id(layer_matrix, device=id_device)
        id_scores.append(layer_id)
        del layer_matrix
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    num_layers = architecture["num_layers"]
    exclusion_boundary = int(num_layers * 0.2)
    window_size = max(1, num_layers // 10)
    candidate_layers = [idx for idx in range(num_layers) if idx >= exclusion_boundary and np.isfinite(id_scores[idx])]
    if not candidate_layers:
        candidate_layers = [idx for idx in range(num_layers) if np.isfinite(id_scores[idx])]
    if not candidate_layers:
        return fallback_paper_layer_selection(architecture, len(sample_records), reason="no_finite_id_scores")

    min_layer = min(candidate_layers, key=lambda idx: id_scores[idx])
    selected_global_layers = list(range(min_layer, min(num_layers - 1, min_layer + window_size) + 1))
    selected_attn_layers = [idx for idx in architecture["attn_layer_indices"] if idx in selected_global_layers]
    if not selected_attn_layers:
        attn_candidates = [idx for idx in architecture["attn_layer_indices"] if idx in candidate_layers]
        if not attn_candidates:
            attn_candidates = list(architecture["attn_layer_indices"])
        best_attn = min(attn_candidates, key=lambda idx: id_scores[idx] if np.isfinite(id_scores[idx]) else float("inf"))
        selected_attn_layers = [best_attn]
        next_attn = [idx for idx in attn_candidates if idx > best_attn]
        if next_attn:
            selected_attn_layers.append(next_attn[0])

    selection = {
        "paper_prompt_doc_template": "Context: {text}\\nCompress the Context in one word:",
        "paper_prompt_query_template": "Query: {text}\\nCompress the Query in one word:",
        "paper_id_sample_size": len(sample_records),
        "paper_id_scores": id_scores,
        "paper_id_exclusion_boundary": exclusion_boundary,
        "paper_id_min_layer": min_layer,
        "paper_selected_global_layers": selected_global_layers,
        "paper_selected_attn_layer_indices": selected_attn_layers,
        "paper_layer_selection_strategy": "global_min_window_intersect_attention",
    }
    return selection

def is_fp8_checkpoint(config: Any) -> bool:
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        return quantization_config.get("quant_method") == "fp8"
    return getattr(quantization_config, "quant_method", None) == "fp8"


def strip_root_prefix(module_name: str) -> str:
    return module_name[6:] if module_name.startswith("model.") else module_name


def build_fp8_quantization_config(config: Any) -> Any:
    from transformers import FineGrainedFP8Config

    quantization_config = getattr(config, "quantization_config", None)
    if not isinstance(quantization_config, dict):
        raise RuntimeError("Expected an FP8 checkpoint with a dict quantization_config.")

    rewritten = copy.deepcopy(quantization_config)
    rewritten["modules_to_not_convert"] = [
        strip_root_prefix(module_name)
        for module_name in rewritten.get("modules_to_not_convert", [])
    ]
    return FineGrainedFP8Config.from_dict(rewritten)


def enable_cutlass_fp8_kernel() -> None:
    try:
        from transformers.integrations import finegrained_fp8
        from transformers.integrations.hub_kernels import get_kernel
    except Exception:
        return

    if os.environ.get("QWEN_FP8_USE_CUTLASS") != "1":
        finegrained_fp8._cutlass_kernel = None
        finegrained_fp8._cutlass_kernel_available = False
        return

    def _get_cutlass_kernel_allow_all():
        if finegrained_fp8._cutlass_kernel_available is None:
            try:
                finegrained_fp8._cutlass_kernel = get_kernel("RedHatAI/quantization", allow_all_kernels=True)
                finegrained_fp8._cutlass_kernel_available = True
            except Exception as exc:
                finegrained_fp8.logger.warning_once(
                    f"Failed to load CUTLASS quantization kernel: {exc}. Falling back to Triton."
                )
                finegrained_fp8._cutlass_kernel_available = False
        return finegrained_fp8._cutlass_kernel

    def _w8a8_fp8_matmul_allow_cutlass(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if finegrained_fp8._supports_cutlass(A, B, block_size, output_dtype):
            try:
                return finegrained_fp8.w8a8_block_fp8_matmul_cutlass(
                    A,
                    B,
                    As.to(torch.float32),
                    Bs.to(torch.float32),
                    output_dtype,
                )
            except Exception as exc:
                finegrained_fp8.logger.warning_once(
                    f"CUTLASS FP8 matmul failed at runtime: {exc}. Falling back to Triton."
                )

        torch.cuda.set_device(A.device)

        if block_size is None:
            N, K = B.shape
            n_needs_pad = (N % 128 != 0) and (N & (N - 1)) != 0
            k_needs_pad = (K % 128 != 0) and (K & (K - 1)) != 0
            if n_needs_pad or k_needs_pad:
                orig_N = N
                if n_needs_pad:
                    pad_n = ((N + 127) // 128 * 128) - N
                    B = F.pad(B, [0, 0, 0, pad_n])
                if k_needs_pad:
                    pad_k = ((K + 127) // 128 * 128) - K
                    B = F.pad(B, [0, pad_k])
                    A = F.pad(A, [0, pad_k])
                kernel = finegrained_fp8._get_triton_kernel()
                result = kernel.w8a8_fp8_matmul(A, B, As, Bs, None, output_dtype)
                return result[..., :orig_N]

        kernel = finegrained_fp8._get_triton_kernel()
        return kernel.w8a8_fp8_matmul(A, B, As, Bs, block_size, output_dtype)

    finegrained_fp8._get_cutlass_kernel = _get_cutlass_kernel_allow_all
    finegrained_fp8.w8a8_fp8_matmul = _w8a8_fp8_matmul_allow_cutlass


def load_checkpoint_tensor(
    snapshot_dir: Path,
    weight_map: dict[str, str],
    shard_cache: dict[str, dict[str, torch.Tensor]],
    tensor_name: str,
) -> torch.Tensor | None:
    shard_name = weight_map.get(tensor_name)
    if shard_name is None:
        return None
    if shard_name not in shard_cache:
        shard_cache[shard_name] = load_file(str(snapshot_dir / shard_name), device="cpu")
    return shard_cache[shard_name][tensor_name]


def repair_fp8_text_skip_modules(
    model: torch.nn.Module,
    model_name: str,
    restore_dtype: torch.dtype,
    skip_modules: list[str],
) -> list[str]:
    skip_modules = [
        module_name
        for module_name in skip_modules
        if module_name.startswith("language_model.")
    ]

    targets = []
    for module_name in skip_modules:
        try:
            module = model.get_submodule(module_name)
        except AttributeError:
            continue
        if type(module).__name__ == "FP8Linear":
            targets.append(module_name)

    if not targets:
        return []

    snapshot_dir = Path(snapshot_download(model_name, local_files_only=True))
    index_path = snapshot_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}
    repaired: list[str] = []

    with torch.no_grad():
        for module_name in targets:
            module = model.get_submodule(module_name)
            weight = load_checkpoint_tensor(
                snapshot_dir,
                weight_map,
                shard_cache,
                f"model.{module_name}.weight",
            )
            if weight is None:
                weight = load_checkpoint_tensor(snapshot_dir, weight_map, shard_cache, f"{module_name}.weight")
            if weight is None:
                raise KeyError(f"Could not find checkpoint tensor for {module_name}.weight")

            bias = load_checkpoint_tensor(
                snapshot_dir,
                weight_map,
                shard_cache,
                f"model.{module_name}.bias",
            )
            if bias is None:
                bias = load_checkpoint_tensor(snapshot_dir, weight_map, shard_cache, f"{module_name}.bias")

            device = module.weight.device
            restored = torch.nn.Linear(
                in_features=weight.shape[1],
                out_features=weight.shape[0],
                bias=bias is not None,
                device=device,
                dtype=restore_dtype,
            )
            restored.weight.copy_(weight.to(device=device, dtype=restore_dtype))
            if bias is not None:
                restored.bias.copy_(bias.to(device=device, dtype=restore_dtype))
            model.set_submodule(module_name, restored)
            repaired.append(module_name)

    return repaired


def load_model_and_tokenizer(model_name: str, torch_dtype: torch.dtype, device_map: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_name)

    if is_fp8_checkpoint(config):
        enable_cutlass_fp8_kernel()
        fp8_quantization_config = build_fp8_quantization_config(config)
        load_kwargs = {
            "device_map": device_map,
            "dtype": torch_dtype,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
            "quantization_config": fp8_quantization_config,
        }
        model = AutoModel.from_pretrained(model_name, **load_kwargs)
        text_model, _model_path = resolve_text_model(model)
        restore_dtype = text_model.embed_tokens.weight.dtype
        if restore_dtype != torch_dtype:
            print(
                f"[Load] FP8 checkpoint is running non-quantized weights as {restore_dtype}; "
                f"requested {torch_dtype}."
            )
        repaired = repair_fp8_text_skip_modules(
            model,
            model_name,
            restore_dtype,
            list(getattr(fp8_quantization_config, "modules_to_not_convert", [])),
        )
        if repaired:
            print(
                f"[Load] Restored {len(repaired)} FP8 skip modules to {restore_dtype}: "
                f"{', '.join(repaired[:4])}{'...' if len(repaired) > 4 else ''}"
            )
    else:
        load_kwargs = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception:
            model = AutoModel.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    dirs = ensure_project_dirs(args.project_root)
    torch_dtype = torch_dtype_from_name(args.torch_dtype)
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, tokenizer = load_model_and_tokenizer(args.model_name, torch_dtype, args.device_map)
    corpus, queries, _qrels = load_scifact(dirs["datasets"])
    records = build_records(corpus, queries)
    selected_records = maybe_limit_records(records, args.limit_docs, args.limit_queries)
    write_manifest(selected_records, dirs["results"])
    query_length_keys = build_query_length_keys(selected_records, tokenizer, args.max_length)
    pending_batches = build_pending_batches(
        selected_records,
        [dirs["rerouted"], dirs["paper_rerouted"]],
        args.query_batch_size,
        query_length_keys,
    )

    text_model, model_path = resolve_text_model(model)
    architecture = inspect_model_architecture(model, text_model, model_path, dirs["root"], args.model_name)
    paper_selection = compute_paper_layer_selection(
        model,
        tokenizer,
        selected_records,
        architecture,
        get_input_device(text_model),
        args.max_length,
        args.paper_id_sample_size,
    )
    architecture.update(paper_selection)
    write_json(architecture_path(dirs["root"]), architecture)
    print(
        "[Paper] Selected attention reroute layers: "
        f"{architecture['paper_selected_attn_layer_indices']} "
        f"(min-ID layer={architecture['paper_id_min_layer']})"
    )

    if args.inspect_only:
        print(json.dumps(architecture, indent=2))
        return

    instrumentation = InstrumentationContext(text_model, architecture, reroute_bias=args.reroute_bias)
    instrumentation.install()

    total_records = len(selected_records)
    pass_seconds = {
        "rerouted": 0.0,
        "paper_rerouted": 0.0,
    }
    processed = 0
    started_at = time.perf_counter()
    input_device = get_input_device(text_model)
    writer = AsyncPayloadWriter(max_queue_size=max(1, args.writer_queue_size))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pass_specs = [
        ExtractionPassSpec(
            name="rerouted",
            label="AllAttn-Rerouted",
            output_dir_key="rerouted",
            use_paper_prompt=True,
            reroute_mode="self",
            reroute_source=None,
            selected_layers=set(architecture["attn_layer_indices"]),
        ),
        ExtractionPassSpec(
            name="paper_rerouted",
            label="PaperLayers-Rerouted",
            output_dir_key="paper_rerouted",
            use_paper_prompt=True,
            reroute_mode="self",
            reroute_source=None,
            selected_layers=set(architecture["paper_selected_attn_layer_indices"]),
        ),
    ]

    try:
        for batch in pending_batches:
            first_position = batch.positions[0]
            last_position = batch.positions[-1]
            encoded_cache: dict[bool, tuple[dict[str, torch.Tensor], torch.Tensor | None, list[int]]] = {}
            for use_paper_prompt in {spec.use_paper_prompt for spec in pass_specs}:
                encoded = tokenizer(
                    texts_for_pass(batch.records, use_paper_prompt),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=True,
                )
                model_inputs = {key: value.to(input_device) for key, value in encoded.items()}
                attention_mask = model_inputs.get("attention_mask")
                if attention_mask is None:
                    seq_lens = [int(model_inputs["input_ids"].shape[1])] * len(batch.records)
                else:
                    seq_lens = [int(value) for value in attention_mask.sum(dim=1).detach().cpu().tolist()]
                encoded_cache[use_paper_prompt] = (model_inputs, attention_mask, seq_lens)

            pass_runtimes: dict[str, PassRuntimeState] = {}
            for pass_spec in pass_specs:
                model_inputs, attention_mask, seq_lens = encoded_cache[pass_spec.use_paper_prompt]
                if len(batch.records) == 1:
                    record = batch.records[0]
                    print(
                        f"[{pass_spec.label}] Processing {record.kind} {first_position}/{total_records}: "
                        f"{record.text_id} (seq_len={seq_lens[0]})"
                    )
                else:
                    print(
                        f"[{pass_spec.label}] Processing query batch {first_position}-{last_position}/{total_records} "
                        f"(batch_size={len(batch.records)}, max_seq_len={max(seq_lens)})"
                    )

                runtime = PassRuntimeState(
                    name=pass_spec.name,
                    num_layers=architecture["num_layers"],
                    attn_layer_indices=architecture["attn_layer_indices"],
                    reroute_mode=pass_spec.reroute_mode,
                    reroute_bias=args.reroute_bias,
                    reroute_kv=None,
                    reroute_layer_indices=pass_spec.selected_layers,
                )
                runtime.set_batch_inputs(model_inputs["input_ids"], attention_mask)
                instrumentation.runtime = runtime
                pass_started = time.perf_counter()
                with torch.no_grad():
                    _ = model(**model_inputs, use_cache=False)
                pass_seconds[pass_spec.name] += time.perf_counter() - pass_started
                payloads = runtime.finalize_batch(
                    model_inputs["input_ids"],
                    [record.text_id for record in batch.records],
                    architecture,
                )
                for record, payload in zip(batch.records, payloads, strict=True):
                    validate_payload_finite(payload, record.text_id, pass_spec.name)
                    writer.submit(dirs[pass_spec.output_dir_key] / record.filename, payload)
                pass_runtimes[pass_spec.name] = runtime
                instrumentation.runtime = None
                del payloads

            processed += len(batch.records)

            del pass_runtimes, encoded_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if processed > 0 and processed % 100 == 0:
                elapsed = time.perf_counter() - started_at
                avg_per_text = elapsed / processed
                remaining = total_records - processed
                eta = avg_per_text * remaining
                print(
                    f"[Progress] {processed} texts extracted in {human_duration(elapsed)} "
                    f"(avg {avg_per_text:.2f}s/text, ETA {human_duration(eta)})"
                )
    finally:
        instrumentation.runtime = None
        instrumentation.uninstall()
        writer.close()

    total_elapsed = time.perf_counter() - started_at
    gpu_peak = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    timing_lines = [
        f"Total wall-clock time for all-attention rerouted pass extraction: {pass_seconds['rerouted']:.3f}s",
        f"Total wall-clock time for paper rerouted pass extraction: {pass_seconds['paper_rerouted']:.3f}s",
        f"Average seconds per text for all-attention rerouted pass: {pass_seconds['rerouted'] / max(processed, 1):.3f}s",
        f"Average seconds per text for paper rerouted pass: {pass_seconds['paper_rerouted'] / max(processed, 1):.3f}s",
        f"GPU memory peak during extraction: {human_bytes(gpu_peak)}",
        f"Total end-to-end elapsed time: {human_duration(total_elapsed)}",
        f"Processed text count: {processed}",
    ]
    (dirs["results"] / "timing.txt").write_text("\n".join(timing_lines) + "\n", encoding="utf-8")
    print("\n".join(timing_lines))


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class PrependResult:
    attn_output: "torch.Tensor"
    attn_weights: "torch.Tensor"
    beta: "torch.Tensor | None"
    summary_key: "torch.Tensor"
    summary_value: "torch.Tensor"


_ATTENTION_COMPILE_ENABLED = False
_ATTENTION_COMPILE_MODE = "reduce-overhead"
_ATTENTION_COMPILE_FULLGRAPH = False


def configure_attention_compile(*, enabled: bool, mode: str = "reduce-overhead", fullgraph: bool = False):
    global _ATTENTION_COMPILE_ENABLED, _ATTENTION_COMPILE_MODE, _ATTENTION_COMPILE_FULLGRAPH
    _ATTENTION_COMPILE_ENABLED = bool(enabled)
    _ATTENTION_COMPILE_MODE = str(mode)
    _ATTENTION_COMPILE_FULLGRAPH = bool(fullgraph)
    _get_plain_sdpa_kernel.cache_clear()
    _get_prepend_sdpa_kernel.cache_clear()


def rotate_half(x):
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _plain_sdpa_no_weights(query_states, key_repeated, value_repeated, attn_mask):
    import torch.nn.functional as F

    return F.scaled_dot_product_attention(
        query_states,
        key_repeated,
        value_repeated,
        attn_mask=attn_mask,
        dropout_p=0.0,
    )


def _prepend_sdpa_no_weights(query_states, key_states, value_states, summary_key, summary_value, attn_mask, num_key_value_groups: int):
    import torch
    import torch.nn.functional as F

    used_key = torch.cat([summary_key, key_states], dim=2)
    used_value = torch.cat([summary_value, value_states], dim=2)
    key_repeated = repeat_kv(used_key, num_key_value_groups)
    value_repeated = repeat_kv(used_value, num_key_value_groups)
    return F.scaled_dot_product_attention(
        query_states,
        key_repeated,
        value_repeated,
        attn_mask=attn_mask,
        dropout_p=0.0,
    )


@lru_cache(maxsize=8)
def _get_plain_sdpa_kernel(enabled: bool, mode: str, fullgraph: bool):
    import torch

    if enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(_plain_sdpa_no_weights, mode=mode, fullgraph=fullgraph)
        except Exception:  # pragma: no cover - compile availability varies by build
            return _plain_sdpa_no_weights
    return _plain_sdpa_no_weights


@lru_cache(maxsize=8)
def _get_prepend_sdpa_kernel(enabled: bool, mode: str, fullgraph: bool):
    import torch

    if enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(_prepend_sdpa_no_weights, mode=mode, fullgraph=fullgraph)
        except Exception:  # pragma: no cover - compile availability varies by build
            return _prepend_sdpa_no_weights
    return _prepend_sdpa_no_weights


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    import torch

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def invert_rotary_pos_emb(k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (k * cos) - (rotate_half(k) * sin)


def matched_norm_random_like(tensor, generator=None):
    import torch

    noise = torch.randn_like(tensor, generator=generator)
    noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    tensor_norm = tensor.norm(dim=-1, keepdim=True)
    return noise / noise_norm * tensor_norm


def make_prepend_summary_kv(key_pre_rope, key_rotated, value_states, cos, sin, mode: str):
    if mode == "reuse_rotated":
        return key_rotated[..., -1:, :], value_states[..., -1:, :]
    base = invert_rotary_pos_emb(key_rotated[..., -1:, :], cos[:, -1:, :], sin[:, -1:, :])
    if mode == "derotate":
        return base, value_states[..., -1:, :]
    if mode != "rerotate_zero":
        raise ValueError(f"Unsupported rope mode: {mode}")
    zero_cos = cos[:, :1, :]
    zero_sin = sin[:, :1, :]
    _, rerotated = apply_rotary_pos_emb(base, base, zero_cos, zero_sin)
    return rerotated, value_states[..., -1:, :]


def build_prepend_attention_mask(attention_mask, query_len: int, num_slots: int):
    import torch

    if attention_mask is None:
        base = torch.full((1, 1, query_len, query_len), fill_value=0.0, device="cpu")
        base = torch.triu(base.fill_(float("-inf")), diagonal=1)
        attention_mask = base

    batch = attention_mask.shape[0]
    prefix = torch.zeros(
        (batch, attention_mask.shape[1], attention_mask.shape[2], num_slots),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([prefix, attention_mask], dim=-1)


def attention_forward(
    module,
    query_states,
    key_states,
    value_states,
    attention_mask,
    prepend_mode: str | None = None,
    cos=None,
    sin=None,
    key_pre_rope=None,
    summary_key=None,
    summary_value=None,
    return_weights: bool = True,
):
    import torch
    import torch.nn.functional as F

    provided_summary_key = summary_key
    provided_summary_value = summary_value
    summary_key = key_states[..., -1:, :] if provided_summary_key is None else provided_summary_key
    summary_value = value_states[..., -1:, :] if provided_summary_value is None else provided_summary_value
    beta = None
    used_key = key_states
    used_value = value_states
    used_mask = attention_mask

    if prepend_mode is not None:
        if provided_summary_key is None or provided_summary_value is None:
            summary_key, summary_value = make_prepend_summary_kv(
                key_pre_rope=key_pre_rope if key_pre_rope is not None else key_states,
                key_rotated=key_states,
                value_states=value_states,
                cos=cos,
                sin=sin,
                mode=prepend_mode,
            )
        used_key = torch.cat([summary_key, key_states], dim=2)
        used_value = torch.cat([summary_value, value_states], dim=2)
        used_mask = build_prepend_attention_mask(attention_mask, query_states.shape[2], num_slots=1)

    if return_weights:
        key_repeated = repeat_kv(used_key, module.num_key_value_groups)
        value_repeated = repeat_kv(used_value, module.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_repeated.transpose(2, 3)) * module.scaling
        if used_mask is not None:
            attn_weights = attn_weights + used_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_repeated)
    else:
        try:
            compile_enabled = bool(_ATTENTION_COMPILE_ENABLED and query_states.is_cuda)
            if prepend_mode is None:
                key_repeated = repeat_kv(used_key, module.num_key_value_groups)
                value_repeated = repeat_kv(used_value, module.num_key_value_groups)
                kernel = _get_plain_sdpa_kernel(
                    compile_enabled,
                    _ATTENTION_COMPILE_MODE,
                    _ATTENTION_COMPILE_FULLGRAPH,
                )
                attn_output = kernel(
                    query_states.contiguous(),
                    key_repeated.contiguous(),
                    value_repeated.contiguous(),
                    used_mask,
                )
            else:
                kernel = _get_prepend_sdpa_kernel(
                    compile_enabled,
                    _ATTENTION_COMPILE_MODE,
                    _ATTENTION_COMPILE_FULLGRAPH,
                )
                attn_output = kernel(
                    query_states.contiguous(),
                    key_states.contiguous(),
                    value_states.contiguous(),
                    summary_key.contiguous(),
                    summary_value.contiguous(),
                    used_mask,
                    int(module.num_key_value_groups),
                )
        except Exception:
            use_cpu_fallback = os.environ.get("KV_PREPEND_REPLAY_ATTENTION_CPU", "0") == "1"
            target_device = query_states.device
            query_f = query_states.contiguous().float()
            key_f = repeat_kv(used_key, module.num_key_value_groups).contiguous().float()
            value_f = repeat_kv(used_value, module.num_key_value_groups).contiguous().float()
            mask_f = used_mask
            if use_cpu_fallback:
                query_f = query_f.cpu()
                key_f = key_f.cpu()
                value_f = value_f.cpu()
                if mask_f is not None:
                    mask_f = mask_f.to(device="cpu", dtype=torch.float32)
            try:
                attn_scores = torch.einsum("bhqd,bhkd->bhqk", query_f, key_f) * float(module.scaling)
            except Exception as exc:  # pragma: no cover - debug path
                raise RuntimeError(
                    "Replay attention matmul failed with "
                    f"q={tuple(query_f.shape)}/{query_f.dtype}/{query_f.device}, "
                    f"k={tuple(key_f.shape)}/{key_f.dtype}/{key_f.device}, "
                    f"v={tuple(value_f.shape)}/{value_f.dtype}/{value_f.device}, "
                    f"mask={None if mask_f is None else (tuple(mask_f.shape), mask_f.dtype, mask_f.device)}, "
                    f"num_kv_groups={getattr(module, 'num_key_value_groups', None)}, "
                    f"num_heads={getattr(module, 'num_heads', None)}, "
                    f"num_kv_heads={getattr(module, 'num_kv_heads', None)}"
                ) from exc
            if mask_f is not None:
                attn_scores = attn_scores + mask_f.to(device=attn_scores.device, dtype=attn_scores.dtype)
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
            attn_output = torch.einsum("bhqk,bhkd->bhqd", attn_probs, value_f)
            if use_cpu_fallback:
                attn_output = attn_output.to(device=target_device)
            attn_output = attn_output.to(query_states.dtype)
        attn_weights = None
    attn_output = attn_output.transpose(1, 2).contiguous()

    if prepend_mode is not None and attn_weights is not None:
        beta = attn_weights[..., :1]

    return PrependResult(
        attn_output=attn_output,
        attn_weights=attn_weights,
        beta=beta,
        summary_key=summary_key,
        summary_value=summary_value,
    )

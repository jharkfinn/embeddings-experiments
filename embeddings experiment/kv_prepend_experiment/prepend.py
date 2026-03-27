from __future__ import annotations

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
_ATTENTION_COMPILE_MODE = "default"
_ATTENTION_COMPILE_FULLGRAPH = False
_ATTENTION_BACKEND = "sdpa"


def configure_attention_runtime(
    *,
    enabled: bool,
    mode: str = "default",
    fullgraph: bool = False,
    backend: str = "sdpa",
):
    global _ATTENTION_COMPILE_ENABLED, _ATTENTION_COMPILE_MODE, _ATTENTION_COMPILE_FULLGRAPH, _ATTENTION_BACKEND
    _ATTENTION_COMPILE_ENABLED = bool(enabled)
    _ATTENTION_COMPILE_MODE = str(mode)
    _ATTENTION_COMPILE_FULLGRAPH = bool(fullgraph)
    _ATTENTION_BACKEND = str(backend)
    _get_plain_sdpa_kernel.cache_clear()
    _get_prepend_sdpa_kernel.cache_clear()


def configure_attention_compile(*, enabled: bool, mode: str = "default", fullgraph: bool = False):
    configure_attention_runtime(enabled=enabled, mode=mode, fullgraph=fullgraph, backend=_ATTENTION_BACKEND)


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

    if enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("attention compile was requested, but torch.compile is unavailable.")
        return torch.compile(_plain_sdpa_no_weights, mode=mode, fullgraph=fullgraph)
    return _plain_sdpa_no_weights


@lru_cache(maxsize=8)
def _get_prepend_sdpa_kernel(enabled: bool, mode: str, fullgraph: bool):
    import torch

    if enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("attention compile was requested, but torch.compile is unavailable.")
        return torch.compile(_prepend_sdpa_no_weights, mode=mode, fullgraph=fullgraph)
    return _prepend_sdpa_no_weights


@lru_cache(maxsize=1)
def _get_flex_attention_ops():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    return create_block_mask, flex_attention


def _pack_sequences(tokens, lengths: list[int]):
    import torch

    parts = []
    for row_idx, length in enumerate(lengths):
        if length <= 0:
            continue
        parts.append(tokens[row_idx, :, :length, :])
    if not parts:
        return tokens.new_empty((1, tokens.shape[1], 0, tokens.shape[-1]))
    return torch.cat(parts, dim=1).unsqueeze(0)


def _unpack_sequences(output_packed, lengths: list[int], batch_size: int, num_heads: int, query_len: int, head_dim: int):
    out = output_packed.new_zeros((batch_size, num_heads, query_len, head_dim))
    offset = 0
    packed = output_packed[0]
    for row_idx, length in enumerate(lengths):
        if length <= 0:
            continue
        out[row_idx, :, :length, :] = packed[:, offset : offset + length, :]
        offset += length
    return out


@lru_cache(maxsize=128)
def _packed_metadata(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    device_type: str,
    device_index: int | None,
):
    import torch

    device = torch.device(device_type, device_index) if device_index is not None else torch.device(device_type)
    q_doc = []
    q_pos = []
    k_doc = []
    k_pos = []
    for doc_idx, (q_len, k_len) in enumerate(zip(q_lengths, k_lengths)):
        q_len_i = int(q_len)
        k_len_i = int(k_len)
        if q_len_i > 0:
            q_doc.extend([doc_idx] * q_len_i)
            q_pos.extend(range(q_len_i))
        if k_len_i > 0:
            k_doc.extend([doc_idx] * k_len_i)
            k_pos.extend(range(k_len_i))
    return (
        torch.tensor(q_doc, dtype=torch.int32, device=device),
        torch.tensor(q_pos, dtype=torch.int32, device=device),
        torch.tensor(k_doc, dtype=torch.int32, device=device),
        torch.tensor(k_pos, dtype=torch.int32, device=device),
        int(sum(q_lengths)),
        int(sum(k_lengths)),
    )


@lru_cache(maxsize=128)
def _cached_flex_block_mask(
    q_lengths: tuple[int, ...],
    k_lengths: tuple[int, ...],
    num_heads: int,
    device_type: str,
    device_index: int | None,
):
    q_doc_ids, q_positions, k_doc_ids, k_positions, total_q, total_k = _packed_metadata(
        q_lengths, k_lengths, device_type, device_index
    )
    if total_q <= 0 or total_k <= 0:
        return None
    create_block_mask, _ = _get_flex_attention_ops()
    prepend = bool(any(k_len > q_len for q_len, k_len in zip(q_lengths, k_lengths)))
    device = q_doc_ids.device

    def mask_mod(batch, head, q_idx, kv_idx):
        same_doc = q_doc_ids[q_idx] == k_doc_ids[kv_idx]
        if prepend:
            causal = k_positions[kv_idx] <= (q_positions[q_idx] + 1)
        else:
            causal = k_positions[kv_idx] <= q_positions[q_idx]
        return same_doc & causal

    return create_block_mask(
        mask_mod,
        B=1,
        H=num_heads,
        Q_LEN=total_q,
        KV_LEN=total_k,
        device=device,
        _compile=False,
    )


def _flex_attention_no_weights(query_states, key_states, value_states, q_lengths, k_lengths, num_key_value_groups: int):
    _, flex_attention = _get_flex_attention_ops()
    batch_size, num_heads, query_len, head_dim = query_states.shape
    q_lengths_tuple = tuple(int(length) for length in q_lengths)
    k_lengths_tuple = tuple(int(length) for length in k_lengths)
    q_packed = _pack_sequences(query_states, list(q_lengths_tuple)).contiguous()
    k_packed = _pack_sequences(key_states, list(k_lengths_tuple)).contiguous()
    v_packed = _pack_sequences(value_states, list(k_lengths_tuple)).contiguous()
    if q_packed.shape[2] == 0 or k_packed.shape[2] == 0:
        return query_states.new_zeros((batch_size, num_heads, query_len, head_dim))
    block_mask = _cached_flex_block_mask(
        q_lengths_tuple,
        k_lengths_tuple,
        int(num_heads),
        query_states.device.type,
        query_states.device.index,
    )
    return _unpack_sequences(
        flex_attention(
            q_packed,
            k_packed,
            v_packed,
            block_mask=block_mask,
            enable_gqa=bool(num_key_value_groups > 1),
        ),
        list(q_lengths_tuple),
        batch_size,
        num_heads,
        query_len,
        head_dim,
    )


@lru_cache(maxsize=8)
def _get_flex_packed_kernel(enabled: bool, mode: str, fullgraph: bool):
    import torch

    if enabled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("attention compile was requested, but torch.compile is unavailable.")
        return torch.compile(_flex_attention_no_weights, mode=mode, fullgraph=fullgraph)
    return _flex_attention_no_weights


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
    token_counts=None,
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
        used_mask = None

    if return_weights:
        if prepend_mode is not None:
            used_mask = build_prepend_attention_mask(attention_mask, query_states.shape[2], num_slots=1)
        key_repeated = repeat_kv(used_key, module.num_key_value_groups)
        value_repeated = repeat_kv(used_value, module.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_repeated.transpose(2, 3)) * module.scaling
        if used_mask is not None:
            attn_weights = attn_weights + used_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_repeated)
    else:
        compile_enabled = bool(_ATTENTION_COMPILE_ENABLED and query_states.is_cuda)
        backend = _ATTENTION_BACKEND
        if backend == "flex_packed":
            if token_counts is None:
                raise RuntimeError("flex_packed attention backend requires token_counts.")
            if not query_states.is_cuda:
                raise RuntimeError("flex_packed attention backend requires CUDA tensors.")
            q_lengths = tuple(int(length) for length in token_counts.tolist())
            k_lengths = tuple(length + 1 if prepend_mode is not None and length > 0 else length for length in q_lengths)
            kernel = _get_flex_packed_kernel(
                compile_enabled,
                _ATTENTION_COMPILE_MODE,
                _ATTENTION_COMPILE_FULLGRAPH,
            )
            attn_output = kernel(
                query_states.contiguous(),
                used_key.contiguous(),
                used_value.contiguous(),
                q_lengths,
                k_lengths,
                int(module.num_key_value_groups),
            )
        elif backend == "sdpa":
            if prepend_mode is None:
                key_repeated = repeat_kv(used_key, module.num_key_value_groups)
                value_repeated = repeat_kv(used_value, module.num_key_value_groups)
                kernel = _get_plain_sdpa_kernel(
                    compile_enabled,
                    _ATTENTION_COMPILE_MODE,
                    _ATTENTION_COMPILE_FULLGRAPH,
                )
                used_mask = attention_mask
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
                used_mask = build_prepend_attention_mask(attention_mask, query_states.shape[2], num_slots=1)
                attn_output = kernel(
                    query_states.contiguous(),
                    key_states.contiguous(),
                    value_states.contiguous(),
                    summary_key.contiguous(),
                    summary_value.contiguous(),
                    used_mask,
                    int(module.num_key_value_groups),
                )
        else:
            raise ValueError(f"Unsupported attention backend: {backend}")
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

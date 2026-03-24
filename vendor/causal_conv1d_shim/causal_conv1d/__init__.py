from __future__ import annotations

from functools import lru_cache

from kernels import get_kernel as _get_kernel

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "causal_conv1d_varlen",
    "causal_conv1d_varlen_states",
]


@lru_cache(maxsize=1)
def _kernel():
    return _get_kernel(
        "kernels-community/causal-conv1d",
        version=1,
        user_agent={"framework": "kv_moee_experiment"},
    )


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    activation=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
):
    return _kernel().causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        return_final_states=return_final_states,
        final_states_out=final_states_out,
        activation=activation,
    )


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    cache_seqlens=None,
):
    return _kernel().causal_conv1d_update(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
    )


def __getattr__(name: str):
    if name in {"causal_conv1d_varlen", "causal_conv1d_varlen_states"}:
        return getattr(_kernel(), name)
    raise AttributeError(name)

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


class MissingDependencyError(RuntimeError):
    pass


def import_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise MissingDependencyError("PyTorch is required for this experiment.") from exc
    return torch


def import_transformers():
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise MissingDependencyError(
            "transformers is required. Install it on the Thunder instance before running."
        ) from exc
    return transformers


def import_datasets():
    try:
        import datasets  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment specific
        raise MissingDependencyError(
            "datasets is required for NanoBEIR evaluation."
        ) from exc
    return datasets


@dataclass
class VerifiedModelContract:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    module_name: str


def runtime_stack_snapshot(transformers: Any | None = None) -> dict[str, Any]:
    torch = import_torch()
    if transformers is None:
        transformers = import_transformers()
    return {
        "python": inspect.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
    }


def resolve_qwen3_moe_module():
    transformers = import_transformers()
    module = __import__(
        "transformers.models.qwen3_moe.modeling_qwen3_moe",
        fromlist=["Qwen3MoeForCausalLM", "Qwen3MoeModel", "Qwen3MoeDecoderLayer", "Qwen3MoeSparseMoeBlock"],
    )
    return transformers, module


def build_fp8_quantization_config(transformers: Any):
    for attr in ("FbgemmFp8Config", "FBGEMMFP8Config"):
        if hasattr(transformers, attr):
            return getattr(transformers, attr)()
    return None


def load_model_and_tokenizer(model_spec):
    torch = import_torch()
    transformers, module = resolve_qwen3_moe_module()

    AutoConfig = transformers.AutoConfig
    AutoTokenizer = transformers.AutoTokenizer
    model_cls = getattr(module, "Qwen3MoeForCausalLM")

    config = AutoConfig.from_pretrained(model_spec.model_name, trust_remote_code=model_spec.trust_remote_code)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_spec.trust_remote_code,
        "attn_implementation": model_spec.attn_implementation,
        "device_map": model_spec.device_map,
    }
    if model_spec.torch_dtype:
        model_kwargs["torch_dtype"] = getattr(torch, model_spec.torch_dtype)
    if model_spec.quantization.lower() == "fp8":
        quantization_config = build_fp8_quantization_config(transformers)
        if quantization_config is None:
            raise RuntimeError(
                "No FP8 quantization config was found in this Transformers build. "
                "Install a build that exposes FBGEMM FP8 support on Thunder."
            )
        model_kwargs["quantization_config"] = quantization_config

    tokenizer_name = model_spec.tokenizer_name or model_spec.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=model_spec.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model_cls.from_pretrained(model_spec.model_name, **model_kwargs)
    model.eval()
    return config, model, tokenizer


def _assert_source_contains(source: str, needle: str, error: str) -> None:
    if needle not in source:
        raise AssertionError(error)


def verify_model_contract(config, model) -> VerifiedModelContract:
    _, module = resolve_qwen3_moe_module()
    if config.num_hidden_layers != 48:
        raise AssertionError(f"Expected 48 layers, found {config.num_hidden_layers}")
    if config.hidden_size != 2048:
        raise AssertionError(f"Expected hidden_size 2048, found {config.hidden_size}")
    if config.num_attention_heads != 32:
        raise AssertionError(f"Expected 32 Q heads, found {config.num_attention_heads}")
    if config.num_key_value_heads != 4:
        raise AssertionError(f"Expected 4 KV heads, found {config.num_key_value_heads}")
    if config.num_experts != 128:
        raise AssertionError(f"Expected 128 experts, found {config.num_experts}")
    if config.num_experts_per_tok != 8:
        raise AssertionError(f"Expected top-8 routing, found {config.num_experts_per_tok}")

    attention_source = inspect.getsource(module.Qwen3MoeAttention)
    decoder_source = inspect.getsource(module.Qwen3MoeDecoderLayer)
    router_source = inspect.getsource(module.Qwen3MoeTopKRouter)

    _assert_source_contains(
        attention_source,
        "self.q_norm(",
        "Qwen3MoeAttention no longer applies q_norm before RoPE; prepend projections must be updated.",
    )
    _assert_source_contains(
        attention_source,
        "self.k_norm(",
        "Qwen3MoeAttention no longer applies k_norm before RoPE; prepend projections must be updated.",
    )
    _assert_source_contains(
        attention_source,
        "apply_rotary_pos_emb(",
        "Qwen3MoeAttention no longer applies RoPE at the expected site; prepend summary construction is invalid.",
    )
    _assert_source_contains(
        attention_source,
        "ALL_ATTENTION_FUNCTIONS.get_interface(",
        "Qwen3MoeAttention no longer dispatches through ALL_ATTENTION_FUNCTIONS; GQA/attention assumptions changed.",
    )
    _assert_source_contains(
        attention_source,
        "self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads",
        "Qwen3MoeAttention no longer exposes the expected GQA grouping; prepend attention must be updated.",
    )
    _assert_source_contains(
        decoder_source,
        "hidden_states = residual + hidden_states",
        "Residual add before post-attention RMSNorm changed; hook points are invalid.",
    )
    _assert_source_contains(
        decoder_source,
        "hidden_states = self.post_attention_layernorm(hidden_states)",
        "Post-attention RMSNorm location changed; h_pre_moe hook point is invalid.",
    )
    _assert_source_contains(
        router_source,
        "softmax(router_logits",
        "Router no longer applies softmax before top-k; routing assumptions changed.",
    )

    return VerifiedModelContract(
        num_hidden_layers=config.num_hidden_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        module_name=module.__name__,
    )

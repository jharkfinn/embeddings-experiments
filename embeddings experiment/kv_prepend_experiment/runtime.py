from __future__ import annotations

import inspect
import importlib.metadata
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MissingDependencyError(RuntimeError):
    pass


def _is_native_fp8_checkpoint(model_name: str) -> bool:
    return model_name.rstrip("/").endswith("-FP8")


def _ensure_libcuda_on_link_path() -> None:
    candidates = [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/lib/wsl/lib/libcuda.so.1"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
    ]
    found = next((candidate for candidate in candidates if candidate.exists()), None)
    if found is None:
        return
    lib_dir = str(found.parent)
    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    if lib_dir not in current:
        os.environ["LD_LIBRARY_PATH"] = ":".join([lib_dir, *current]) if current else lib_dir
    os.environ.setdefault("TRITON_LIBCUDA_PATH", lib_dir)


def import_torch():
    _ensure_libcuda_on_link_path()
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
    try:
        torchao_version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        torchao_version = None
    return {
        "python": inspect.sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers_version": transformers.__version__,
        "torchao_version": torchao_version,
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


def build_torchao_quantization_config(transformers: Any, quantization_mode: str):
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TorchAO quantization was requested but torchao is not installed. "
            "Install torchao in the runtime environment before running this profile."
        ) from exc

    if not hasattr(transformers, "TorchAoConfig"):
        raise RuntimeError(
            "TorchAO quantization was requested but this Transformers build does not expose TorchAoConfig."
        )

    quantization_mode = quantization_mode.lower()
    if quantization_mode == "torchao_fp8_weight_only":
        quant_type = Float8WeightOnlyConfig()
    elif quantization_mode in {"torchao_fp8_dynamic", "torchao_fp8"}:
        quant_type = Float8DynamicActivationFloat8WeightConfig()
    else:
        raise ValueError(f"Unsupported TorchAO quantization mode: {quantization_mode}")
    return transformers.TorchAoConfig(quant_type=quant_type)


def _query_gpu_snapshot() -> dict[str, float | str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    used, total, util, power = [field.strip() for field in first_line[0].split(",")]
    return {
        "memory_used_mib": float(used),
        "memory_total_mib": float(total),
        "utilization_gpu_pct": float(util),
        "power_draw_watts": float(power),
    }


def _log_and_validate_gpu_preflight(model_spec) -> None:
    if not str(model_spec.device).startswith("cuda"):
        return
    snapshot = _query_gpu_snapshot()
    if snapshot is None:
        logger.warning("gpu_preflight_unavailable device=%s", model_spec.device)
        return
    logger.info(
        "gpu_preflight device=%s used_mib=%.0f total_mib=%.0f util_pct=%.0f power_watts=%.2f",
        model_spec.device,
        snapshot["memory_used_mib"],
        snapshot["memory_total_mib"],
        snapshot["utilization_gpu_pct"],
        snapshot["power_draw_watts"],
    )
    max_used_gib = model_spec.preflight_max_used_memory_gib
    if max_used_gib is None:
        return
    used_gib = float(snapshot["memory_used_mib"]) / 1024.0
    if used_gib > max_used_gib:
        raise RuntimeError(
            f"GPU preflight failed: found {used_gib:.2f} GiB already allocated on {model_spec.device} "
            f"before model load (limit {max_used_gib:.2f} GiB). Clear the device and retry."
        )


def load_model_and_tokenizer(model_spec):
    torch = import_torch()
    transformers, module = resolve_qwen3_moe_module()
    start = time.perf_counter()
    _log_and_validate_gpu_preflight(model_spec)

    AutoConfig = transformers.AutoConfig
    AutoTokenizer = transformers.AutoTokenizer
    model_cls = getattr(module, "Qwen3MoeForCausalLM")

    config = AutoConfig.from_pretrained(model_spec.model_name, trust_remote_code=model_spec.trust_remote_code)
    device_map = model_spec.device_map
    quantization_mode = model_spec.quantization.lower()
    if quantization_mode == "fp8":
        if device_map == "auto":
            raise ValueError(
                "FP8 runs must use an explicit CUDA-only device_map for this experiment. "
                "Set model.device_map to 'cuda:0'."
            )
        if device_map in {"cuda", "cuda:0"}:
            device_map = {"": 0}
    logger.info(
        "load_model_and_tokenizer_start model=%s dtype=%s quantization=%s device_map=%s attn_impl=%s",
        model_spec.model_name,
        model_spec.torch_dtype,
        model_spec.quantization,
        device_map,
        model_spec.attn_implementation,
    )
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_spec.trust_remote_code,
        "attn_implementation": model_spec.attn_implementation,
        "device_map": device_map,
    }
    if model_spec.torch_dtype:
        if str(model_spec.torch_dtype).lower() == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = getattr(torch, model_spec.torch_dtype)
    if quantization_mode == "fp8":
        if _is_native_fp8_checkpoint(model_spec.model_name):
            logger.info(
                "using_native_fp8_checkpoint model=%s checkpoint_quantization_config=true",
                model_spec.model_name,
            )
        else:
            quantization_config = build_fp8_quantization_config(transformers)
            if quantization_config is None:
                raise RuntimeError(
                    "No FP8 quantization config was found in this Transformers build. "
                    "Install a build that exposes FBGEMM FP8 support on Thunder."
                )
            model_kwargs["quantization_config"] = quantization_config
    elif quantization_mode.startswith("torchao_"):
        model_kwargs["quantization_config"] = build_torchao_quantization_config(
            transformers, quantization_mode
        )

    tokenizer_name = model_spec.tokenizer_name or model_spec.model_name
    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=model_spec.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("tokenizer_loaded name=%s seconds=%.3f", tokenizer_name, time.perf_counter() - tokenizer_start)

    model_start = time.perf_counter()
    try:
        model = model_cls.from_pretrained(model_spec.model_name, **model_kwargs)
    except Exception:
        snapshot = _query_gpu_snapshot()
        if snapshot is not None:
            logger.exception(
                "model_load_failed model=%s used_mib=%.0f total_mib=%.0f util_pct=%.0f power_watts=%.2f",
                model_spec.model_name,
                snapshot["memory_used_mib"],
                snapshot["memory_total_mib"],
                snapshot["utilization_gpu_pct"],
                snapshot["power_draw_watts"],
            )
        raise
    model.eval()
    logger.info(
        "model_loaded model=%s seconds=%.3f total_seconds=%.3f",
        model_spec.model_name,
        time.perf_counter() - model_start,
        time.perf_counter() - start,
    )
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

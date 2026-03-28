from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _nvidia_smi() -> dict[str, str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    line = result.stdout.strip().splitlines()[0]
    name, total, used, driver = [field.strip() for field in line.split(",")]
    return {
        "name": name,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "driver_version": driver,
    }


def gpu_smoke() -> dict[str, object]:
    import torch

    out: dict[str, object] = {
        "cwd": os.getcwd(),
        "python": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _nvidia_smi(),
    }
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        out["device_name"] = torch.cuda.get_device_name(0)
        out["device_capability"] = list(torch.cuda.get_device_capability(0))
        start = time.perf_counter()
        a = torch.randn((2048, 2048), device=device, dtype=torch.float16)
        b = torch.randn((2048, 2048), device=device, dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        out["matmul_seconds"] = round(time.perf_counter() - start, 4)
        out["result_norm"] = float(c.float().norm().item())
        out["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
    return out


def repo_smoke() -> dict[str, object]:
    from kv_prepend_experiment.config import load_experiment_spec

    spec_path = ROOT / "spec_main_hf_teacher_forcing_l40s_3tasks.json"
    spec = load_experiment_spec(spec_path)
    return {
        "spec_path": str(spec_path),
        "model_name": spec.model.model_name,
        "torch_dtype": spec.model.torch_dtype,
        "quantization": spec.model.quantization,
        "runtime_backend": spec.collection.runtime_backend,
        "streaming_batch_size": spec.collection.streaming_batch_size,
        "max_batch_tokens": spec.collection.max_batch_tokens,
        "main_capture_signals": list(spec.collection.main_capture_signals),
    }


def stack_smoke() -> dict[str, object]:
    import accelerate
    import datasets
    import torch
    import torchao
    import transformers

    from kv_prepend_experiment.runtime import (
        build_fp8_quantization_config,
        build_torchao_quantization_config,
        resolve_qwen3_moe_module,
        runtime_stack_snapshot,
    )

    _, module = resolve_qwen3_moe_module()
    fp8_config = build_fp8_quantization_config(transformers)
    torchao_weight_only = build_torchao_quantization_config(transformers, "torchao_fp8_weight_only")
    torchao_dynamic = build_torchao_quantization_config(transformers, "torchao_fp8_dynamic")
    return {
        "runtime": runtime_stack_snapshot(transformers),
        "accelerate_version": accelerate.__version__,
        "datasets_version": datasets.__version__,
        "torchao_version": torchao.__version__,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "has_fbgemm_fp8_config": fp8_config is not None,
        "fp8_config_class": None if fp8_config is None else fp8_config.__class__.__name__,
        "torchao_weight_only_config_class": torchao_weight_only.__class__.__name__,
        "torchao_dynamic_config_class": torchao_dynamic.__class__.__name__,
        "qwen3_moe_module": module.__name__,
    }


def model_load_smoke() -> dict[str, object]:
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.runtime import load_model_and_tokenizer, verify_model_contract

    spec_path = ROOT / "spec_main_hf_teacher_forcing_l40s_3tasks.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None
    start = time.perf_counter()
    config, model, tokenizer = load_model_and_tokenizer(spec.model)
    contract = verify_model_contract(config, model)
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "verified_contract": {
            "num_hidden_layers": contract.num_hidden_layers,
            "hidden_size": contract.hidden_size,
            "num_attention_heads": contract.num_attention_heads,
            "num_key_value_heads": contract.num_key_value_heads,
            "num_experts": contract.num_experts,
            "num_experts_per_tok": contract.num_experts_per_tok,
            "module_name": contract.module_name,
        },
        "nvidia_smi_after_load": _nvidia_smi(),
    }


@contextmanager
def _spoof_hopper_cc_for_fp8() -> object:
    import torch

    original = torch.cuda.get_device_capability

    def patched(device: int | None = None) -> tuple[int, int]:
        capability = original(device) if device is not None else original()
        if capability == (8, 9):
            return (9, 0)
        return capability

    torch.cuda.get_device_capability = patched
    try:
        yield
    finally:
        torch.cuda.get_device_capability = original


def model_load_smoke_force_fp8() -> dict[str, object]:
    import torch

    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.runtime import load_model_and_tokenizer, verify_model_contract

    spec_path = ROOT / "spec_main_hf_teacher_forcing_l40s_3tasks.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None
    start = time.perf_counter()
    with _spoof_hopper_cc_for_fp8():
        config, model, tokenizer = load_model_and_tokenizer(spec.model)
        contract = verify_model_contract(config, model)
        inputs = tokenizer(["hello world"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "verified_contract": {
            "num_hidden_layers": contract.num_hidden_layers,
            "hidden_size": contract.hidden_size,
            "num_attention_heads": contract.num_attention_heads,
            "num_key_value_heads": contract.num_key_value_heads,
            "num_experts": contract.num_experts,
            "num_experts_per_tok": contract.num_experts_per_tok,
            "module_name": contract.module_name,
        },
        "logits_shape": list(outputs.logits.shape),
        "nvidia_smi_after_load": _nvidia_smi(),
    }


def full_smoke() -> dict[str, object]:
    return {
        "gpu": gpu_smoke(),
        "repo": repo_smoke(),
        "stack": stack_smoke(),
    }

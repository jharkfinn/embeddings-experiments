from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    import transformers

    from kv_prepend_experiment.runtime import (
        build_fp8_quantization_config,
        resolve_qwen3_moe_module,
        runtime_stack_snapshot,
    )

    _, module = resolve_qwen3_moe_module()
    fp8_config = build_fp8_quantization_config(transformers)
    return {
        "runtime": runtime_stack_snapshot(transformers),
        "accelerate_version": accelerate.__version__,
        "datasets_version": datasets.__version__,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "has_fbgemm_fp8_config": fp8_config is not None,
        "fp8_config_class": None if fp8_config is None else fp8_config.__class__.__name__,
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


def full_smoke() -> dict[str, object]:
    return {
        "gpu": gpu_smoke(),
        "repo": repo_smoke(),
        "stack": stack_smoke(),
    }


if __name__ == "__main__":
    print(json.dumps(full_smoke(), indent=2, sort_keys=True))

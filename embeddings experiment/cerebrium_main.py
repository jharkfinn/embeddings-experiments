from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LOGGER = logging.getLogger(__name__)


def _read_proc_status() -> dict[str, str]:
    try:
        lines = (Path("/proc/self/status").read_text(encoding="utf-8")).splitlines()
    except OSError:
        return {}
    wanted = {"VmRSS", "VmHWM", "VmSize", "Threads"}
    parsed: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in wanted:
            parsed[key.lower()] = value.strip()
    return parsed


def _capture_dir_snapshot(run_root: Path | None) -> dict[str, object]:
    if run_root is None or not run_root.exists():
        return {}
    payload: dict[str, object] = {}
    capture_dirs = sorted(
        path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("captures_")
    )
    for capture_dir in capture_dirs:
        count = 0
        total_bytes = 0
        for entry in capture_dir.iterdir():
            if not entry.is_file():
                continue
            count += 1
            total_bytes += entry.stat().st_size
        payload[capture_dir.name] = {
            "files": count,
            "bytes": total_bytes,
        }
    return payload


def _resource_snapshot(run_root: Path | None = None) -> dict[str, object]:
    return {
        "proc": _read_proc_status(),
        "gpu": _nvidia_smi(),
        "captures": _capture_dir_snapshot(run_root),
    }


@contextmanager
def _resource_heartbeat(label: str, *, run_root: Path | None = None, interval_s: float = 10.0) -> object:
    stop_event = threading.Event()

    def emit(stage: str) -> None:
        LOGGER.info(
            "cerebrium_resource label=%s stage=%s snapshot=%s",
            label,
            stage,
            json.dumps(_resource_snapshot(run_root), sort_keys=True),
        )

    def loop() -> None:
        while not stop_event.wait(interval_s):
            emit("tick")

    emit("start")
    worker = threading.Thread(target=loop, name=f"{label}-heartbeat", daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        LOGGER.exception("cerebrium_run_failed label=%s", label)
        raise
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        emit("stop")


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


def _doc_text(row: dict[str, str]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def _load_nanobeir_records(dataset_names: list[str]) -> list[dict[str, str]]:
    from kv_prepend_experiment.nano_beir import load_nanobeir_task

    records: list[dict[str, str]] = []
    for dataset_name in dataset_names:
        task = load_nanobeir_task("zeta-alpha-ai/NanoBEIR", dataset_name)
        for doc_id, row in task.corpus.items():
            records.append(
                {
                    "text_id": str(doc_id),
                    "kind": "doc",
                    "dataset_name": task.dataset_name,
                    "text": _doc_text(row),
                }
            )
        for query_id, text in task.queries.items():
            records.append(
                {
                    "text_id": str(query_id),
                    "kind": "query",
                    "dataset_name": task.dataset_name,
                    "text": str(text),
                }
            )
    return records


def collect_smoke() -> dict[str, object]:
    from kv_prepend_experiment.collection import InstrumentedQwen3MoeExperiment
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.logging_utils import configure_logging

    spec_path = ROOT / "spec_main_hf_teacher_forcing_l40s_3tasks.json"
    records_path = ROOT / "smoke_records.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None

    run_root = ROOT / "cerebrium_smoke_output"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=run_root / "artifacts" / "logs" / "collect_smoke.log", level="INFO")
    LOGGER.info("collect_smoke_setup spec=%s records=%s", spec_path.name, records_path.name)

    records = json.loads(records_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    LOGGER.info("collect_smoke_records_loaded records=%s", len(records))
    with _resource_heartbeat("collect_smoke", run_root=run_root):
        experiment = InstrumentedQwen3MoeExperiment(spec, run_root)
        LOGGER.info("collect_smoke_experiment_load_start")
        experiment.load()
        LOGGER.info("collect_smoke_experiment_load_done")
        spec_snapshot = experiment.save_spec_snapshot()
        paths, _ = experiment.collect_examples(
            records,
            dataset_name="cerebrium_smoke",
            retain_bundles=False,
        )
        LOGGER.info("collect_smoke_collect_done paths=%s", len(paths))
        experiment.flush_writes()
        LOGGER.info("collect_smoke_flush_done")
    captures = []
    for path in paths:
        stat = path.stat()
        captures.append(
            {
                "path": str(path.relative_to(run_root)),
                "bytes": stat.st_size,
            }
        )
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "records": len(records),
        "capture_count": len(paths),
        "captures": captures,
        "spec_snapshot": str(spec_snapshot.relative_to(run_root)),
        "nvidia_smi_after_collect": _nvidia_smi(),
    }


def calibration_run() -> dict[str, object]:
    from kv_prepend_experiment.collection import InstrumentedQwen3MoeExperiment
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.logging_utils import configure_logging

    spec_path = ROOT / "spec_calibration_hf_3tasks.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None
    spec.model.torch_dtype = "auto"
    spec.collection.writer_queue_size = 1

    run_root = ROOT / "cerebrium_calibration_output"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=run_root / "artifacts" / "logs" / "calibration_run.log", level="INFO")
    LOGGER.info("calibration_run_setup spec=%s", spec_path.name)

    records = _load_nanobeir_records(["scifact", "fiqa2018", "quoraretrieval"])
    start = time.perf_counter()
    LOGGER.info("calibration_run_records_loaded records=%s", len(records))
    with _resource_heartbeat("calibration_run", run_root=run_root):
        experiment = InstrumentedQwen3MoeExperiment(spec, run_root)
        LOGGER.info("calibration_run_experiment_load_start")
        experiment.load()
        LOGGER.info("calibration_run_experiment_load_done")
        spec_snapshot = experiment.save_spec_snapshot()
        paths, _ = experiment.collect_examples(
            records,
            dataset_name="nanobeir_3tasks",
            retain_bundles=False,
        )
        LOGGER.info("calibration_run_collect_done paths=%s", len(paths))
        experiment.flush_writes()
        LOGGER.info("calibration_run_flush_done")
    captures = []
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        captures.append(
            {
                "path": str(path.relative_to(run_root)),
                "bytes": stat.st_size,
            }
        )
    return {
        "seconds": round(time.perf_counter() - start, 3),
        "records": len(records),
        "capture_count": len(paths),
        "capture_bytes": total_bytes,
        "captures": captures,
        "spec_snapshot": str(spec_snapshot.relative_to(run_root)),
        "nvidia_smi_after_collect": _nvidia_smi(),
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

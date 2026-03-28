import gc
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERSISTENT_STORAGE_ROOT = Path("/persistent-storage")
APP_STORAGE_ROOT = PERSISTENT_STORAGE_ROOT / "kv-prepend-l40s-smoke"

LOGGER = logging.getLogger(__name__)


def _app_storage_root() -> Path:
    root = APP_STORAGE_ROOT if PERSISTENT_STORAGE_ROOT.exists() else (ROOT / "cerebrium_storage")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _new_persistent_run_root(run_kind: str, run_id: str | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = run_id or f"{timestamp}_{uuid.uuid4().hex[:8]}"
    run_root = _app_storage_root() / run_kind / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _latest_persistent_run_root(run_kind: str) -> Path:
    parent = _app_storage_root() / run_kind
    if not parent.exists():
        raise FileNotFoundError(f"no persisted runs found under {parent}")
    candidates = sorted(path for path in parent.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"no persisted runs found under {parent}")
    return candidates[-1]


def _persistent_run_root(run_kind: str, run_id: str) -> Path:
    path = _app_storage_root() / run_kind / run_id
    if not path.exists():
        raise FileNotFoundError(f"persisted run not found: {path}")
    return path


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


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


def _best_effort_runtime_cleanup(label: str) -> None:
    payload: dict[str, object] = {"label": label}
    children = list(mp.active_children())
    payload["active_children"] = len(children)
    terminated = 0
    for child in children:
        if child.is_alive():
            try:
                child.terminate()
                terminated += 1
            except Exception:
                LOGGER.exception("cerebrium_cleanup_child_terminate_failed label=%s child=%s", label, child.pid)
    for child in children:
        try:
            child.join(timeout=1.0)
        except Exception:
            LOGGER.exception("cerebrium_cleanup_child_join_failed label=%s child=%s", label, child.pid)
    payload["terminated_children"] = terminated

    try:
        gc.collect()
    except Exception:
        LOGGER.exception("cerebrium_cleanup_gc_failed label=%s", label)

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
                payload["cuda_after_cleanup"] = _nvidia_smi()
        except Exception:
            LOGGER.exception("cerebrium_cleanup_cuda_failed label=%s", label)

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
            payload["malloc_trim"] = True
    except Exception:
        payload["malloc_trim"] = False

    LOGGER.info("cerebrium_cleanup %s", json.dumps(payload, sort_keys=True))


def _new_analysis_root(calibration_run_root: Path, analysis_run_id: str | None = None) -> Path:
    run_id = analysis_run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    analysis_root = calibration_run_root / "analysis_runs" / run_id
    analysis_root.mkdir(parents=True, exist_ok=True)
    return analysis_root


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


@contextmanager
def _status_heartbeat(
    status_path: Path,
    state: dict[str, object],
    *,
    run_root: Path | None = None,
    interval_s: float = 10.0,
) -> object:
    stop_event = threading.Event()

    def snapshot() -> dict[str, object]:
        payload = dict(state)
        payload["updated_at"] = _iso_now()
        payload["resource_snapshot"] = _resource_snapshot(run_root)
        return payload

    def emit() -> None:
        _write_json_atomic(status_path, snapshot())

    def loop() -> None:
        while not stop_event.wait(interval_s):
            emit()

    emit()
    worker = threading.Thread(target=loop, name=f"{status_path.stem}-heartbeat", daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        emit()


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
    try:
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
    finally:
        _best_effort_runtime_cleanup("model_load_smoke")


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


def collect_smoke(smoke_run_id=""):
    from kv_prepend_experiment.collection import InstrumentedQwen3MoeExperiment
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.logging_utils import configure_logging

    spec_path = ROOT / "spec_main_hf_teacher_forcing_l40s_3tasks.json"
    records_path = ROOT / "smoke_records.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None

    run_root = _new_persistent_run_root("smoke_runs", smoke_run_id or None)
    status_path = run_root / "status.json"
    configure_logging(log_path=run_root / "artifacts" / "logs" / "collect_smoke.log", level="INFO")
    LOGGER.info("collect_smoke_setup spec=%s records=%s", spec_path.name, records_path.name)

    records = json.loads(records_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    LOGGER.info("collect_smoke_records_loaded records=%s", len(records))
    state: dict[str, object] = {
        "run_id": run_root.name,
        "stage": "setup",
        "records": len(records),
        "run_root": str(run_root),
        "status_path": str(status_path),
        "started_at": _iso_now(),
    }

    def update_state(**kwargs: object) -> None:
        state.update(kwargs)
        _write_json_atomic(status_path, dict(state, updated_at=_iso_now(), resource_snapshot=_resource_snapshot(run_root)))

    try:
        with _resource_heartbeat("collect_smoke", run_root=run_root), _status_heartbeat(
            status_path, state, run_root=run_root
        ):
            experiment = InstrumentedQwen3MoeExperiment(spec, run_root)
            update_state(stage="experiment_load_start")
            LOGGER.info("collect_smoke_experiment_load_start")
            experiment.load()
            update_state(stage="experiment_load_done")
            LOGGER.info("collect_smoke_experiment_load_done")
            spec_snapshot = experiment.save_spec_snapshot()
            update_state(stage="collect_examples")
            paths, _ = experiment.collect_examples(
                records,
                dataset_name="cerebrium_smoke",
                retain_bundles=False,
            )
            LOGGER.info("collect_smoke_collect_done paths=%s", len(paths))
            update_state(stage="flush_writes", capture_count=len(paths))
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
        update_state(
            stage="completed",
            finished_at=_iso_now(),
            capture_count=len(paths),
            output_bytes=sum(capture["bytes"] for capture in captures),
        )
        return {
            "seconds": round(time.perf_counter() - start, 3),
            "records": len(records),
            "capture_count": len(paths),
            "captures": captures,
            "run_root": str(run_root),
            "status_path": str(status_path),
            "spec_snapshot": str(spec_snapshot.relative_to(run_root)),
            "nvidia_smi_after_collect": _nvidia_smi(),
        }
    except BaseException as exc:
        update_state(stage="failed", finished_at=_iso_now(), error=repr(exc))
        raise
    finally:
        _best_effort_runtime_cleanup("collect_smoke")


def calibration_run(calibration_run_id=""):
    from kv_prepend_experiment.collection import InstrumentedQwen3MoeExperiment
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.logging_utils import configure_logging

    spec_path = ROOT / "spec_calibration_hf_3tasks.json"
    spec = load_experiment_spec(spec_path)
    spec.model.preflight_max_used_memory_gib = None
    spec.model.torch_dtype = "auto"
    spec.collection.writer_queue_size = 1

    run_root = _new_persistent_run_root("calibration_runs", calibration_run_id or None)
    status_path = run_root / "status.json"
    configure_logging(log_path=run_root / "artifacts" / "logs" / "calibration_run.log", level="INFO")
    LOGGER.info("calibration_run_setup spec=%s", spec_path.name)

    records = _load_nanobeir_records(["scifact", "fiqa2018", "quoraretrieval"])
    start = time.perf_counter()
    LOGGER.info("calibration_run_records_loaded records=%s", len(records))
    state: dict[str, object] = {
        "run_id": run_root.name,
        "stage": "setup",
        "records": len(records),
        "run_root": str(run_root),
        "status_path": str(status_path),
        "started_at": _iso_now(),
    }

    def update_state(**kwargs: object) -> None:
        state.update(kwargs)
        _write_json_atomic(status_path, dict(state, updated_at=_iso_now(), resource_snapshot=_resource_snapshot(run_root)))

    try:
        with _resource_heartbeat("calibration_run", run_root=run_root), _status_heartbeat(
            status_path, state, run_root=run_root
        ):
            experiment = InstrumentedQwen3MoeExperiment(spec, run_root)
            update_state(stage="experiment_load_start")
            LOGGER.info("calibration_run_experiment_load_start")
            experiment.load()
            update_state(stage="experiment_load_done")
            LOGGER.info("calibration_run_experiment_load_done")
            spec_snapshot = experiment.save_spec_snapshot()
            update_state(stage="collect_examples")
            paths, _ = experiment.collect_examples(
                records,
                dataset_name="nanobeir_3tasks",
                retain_bundles=False,
            )
            LOGGER.info("calibration_run_collect_done paths=%s", len(paths))
            update_state(stage="flush_writes", capture_count=len(paths))
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
        update_state(
            stage="completed",
            finished_at=_iso_now(),
            capture_count=len(paths),
            output_bytes=total_bytes,
        )
        return {
            "seconds": round(time.perf_counter() - start, 3),
            "records": len(records),
            "capture_count": len(paths),
            "capture_bytes": total_bytes,
            "captures": captures,
            "run_root": str(run_root),
            "status_path": str(status_path),
            "spec_snapshot": str(spec_snapshot.relative_to(run_root)),
            "nvidia_smi_after_collect": _nvidia_smi(),
        }
    except BaseException as exc:
        update_state(stage="failed", finished_at=_iso_now(), error=repr(exc))
        raise
    finally:
        _best_effort_runtime_cleanup("calibration_run")


def analyze_latest_calibration(
    calibration_run_id="",
    analysis_run_id="",
    workers=0,
):
    from kv_prepend_experiment.analysis import analyze_capture_directory
    from kv_prepend_experiment.config import load_experiment_spec
    from kv_prepend_experiment.logging_utils import configure_logging

    spec_path = ROOT / "spec_calibration_hf_3tasks.json"
    spec = load_experiment_spec(spec_path)
    run_root = (
        _persistent_run_root("calibration_runs", calibration_run_id)
        if calibration_run_id
        else _latest_persistent_run_root("calibration_runs")
    )
    capture_dir = run_root / spec.output.captures_dir
    analysis_root = _new_analysis_root(run_root, analysis_run_id=analysis_run_id or None)
    output_path = analysis_root / "capture_analysis.json"
    status_path = analysis_root / "status.json"
    requested_workers = workers if workers > 0 else max(8, os.cpu_count() or 1)
    os.environ["KV_PREPEND_ANALYSIS_WORKERS"] = str(requested_workers)
    configure_logging(log_path=analysis_root / "analysis_run.log", level="INFO")
    state: dict[str, object] = {
        "analysis_run_id": analysis_root.name,
        "calibration_run_id": run_root.name,
        "stage": "setup",
        "capture_dir": str(capture_dir),
        "output_path": str(output_path),
        "status_path": str(status_path),
        "worker_count_requested": requested_workers,
        "completed_shards": 0,
        "total_shards": 0,
        "started_at": _iso_now(),
    }

    def progress_update(update: dict[str, object]) -> None:
        state.update(update)
        _write_json_atomic(status_path, dict(state, updated_at=_iso_now(), resource_snapshot=_resource_snapshot(run_root)))

    LOGGER.info(
        "analysis_run_setup spec=%s run_root=%s capture_dir=%s workers=%s output_path=%s status_path=%s",
        spec_path.name,
        run_root,
        capture_dir,
        os.environ.get("KV_PREPEND_ANALYSIS_WORKERS"),
        output_path,
        status_path,
    )
    start = time.perf_counter()
    try:
        try:
            with _resource_heartbeat("analysis_run", run_root=run_root), _status_heartbeat(
                status_path, state, run_root=run_root
            ):
                summary = analyze_capture_directory(capture_dir, output_path, progress_callback=progress_update)
        except BaseException as exc:
            state.update(
                {
                    "stage": "failed",
                    "finished_at": _iso_now(),
                    "error": repr(exc),
                }
            )
            _write_json_atomic(status_path, dict(state, updated_at=_iso_now(), resource_snapshot=_resource_snapshot(run_root)))
            raise
        state.update(
            {
                "stage": "completed",
                "finished_at": _iso_now(),
                "completed_shards": state.get("total_shards", 0),
                "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
                "num_bundles": int(summary.get("num_bundles", 0)),
                "num_layers": len(summary.get("layers", {})),
            }
        )
        _write_json_atomic(status_path, dict(state, updated_at=_iso_now(), resource_snapshot=_resource_snapshot(run_root)))
        return {
            "seconds": round(time.perf_counter() - start, 3),
            "run_root": str(run_root),
            "analysis_root": str(analysis_root),
            "capture_dir": str(capture_dir),
            "output_path": str(output_path),
            "status_path": str(status_path),
            "bundles": int(summary.get("num_bundles", 0)),
            "layers": len(summary.get("layers", {})),
            "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "nvidia_smi_after_analysis": _nvidia_smi(),
        }
    finally:
        _best_effort_runtime_cleanup("analysis_run")


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

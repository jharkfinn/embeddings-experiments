#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GPU_CONFIG = ROOT / "cerebrium.toml"
CPU_CONFIG = ROOT / "cerebrium_analysis.toml"
DEFAULT_STORAGE_APP = "kv-prepend-l40s-smoke"
DEFAULT_GPU_APP = "kv-prepend-l40s-smoke"
DEFAULT_CPU_APP = "kv-prepend-analysis-cpu"


def _read_cli_config() -> dict[str, str]:
    config_path = Path.home() / ".cerebrium" / "config.yaml"
    payload: dict[str, str] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip().strip('"').strip("'")
    return payload


def _latest_remote_dir(remote_prefix: str) -> str:
    result = subprocess.run(
        ["cerebrium", "ls", remote_prefix],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    entries: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME") or stripped.startswith("No files"):
            continue
        name = stripped.split()[0]
        if name.endswith("/"):
            name = name[:-1]
        entries.append(name)
    if not entries:
        raise RuntimeError(f"no remote entries found under {remote_prefix}")
    return entries[-1]


def _deploy_app(config_path: Path, app_name: str) -> None:
    subprocess.run(
        [
            "cerebrium",
            "deploy",
            "--config-file",
            str(config_path),
            "--name",
            app_name,
            "--disable-confirmation",
        ],
        cwd=ROOT,
        check=True,
    )


def _post_async(endpoint: str, token: str | None, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "function",
        choices=["collect_smoke", "calibration_run", "analyze_latest_calibration"],
    )
    parser.add_argument("--region", default=None)
    parser.add_argument("--storage-app", default=DEFAULT_STORAGE_APP)
    parser.add_argument("--gpu-app", default=DEFAULT_GPU_APP)
    parser.add_argument("--cpu-app", default=DEFAULT_CPU_APP)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--calibration-run-id", default=None)
    parser.add_argument("--analysis-run-id", default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip-deploy", action="store_true")
    args = parser.parse_args()

    cli_config = _read_cli_config()
    region = args.region or cli_config.get("defaultregion") or "us-east-1"
    project = cli_config.get("project")
    token = cli_config.get("accesstoken")
    if not project:
        raise RuntimeError("missing project in ~/.cerebrium/config.yaml")

    if args.function == "collect_smoke":
        app_name = args.gpu_app
        config_path = GPU_CONFIG
        function_payload = {"smoke_run_id": args.run_id or _new_run_id()}
        remote_paths = {
            "status_remote_path": f"{args.storage_app}/smoke_runs/{function_payload['smoke_run_id']}/status.json",
            "log_remote_path": f"{args.storage_app}/smoke_runs/{function_payload['smoke_run_id']}/artifacts/logs/collect_smoke.log",
            "run_root_remote_path": f"{args.storage_app}/smoke_runs/{function_payload['smoke_run_id']}/",
        }
    elif args.function == "calibration_run":
        app_name = args.gpu_app
        config_path = GPU_CONFIG
        function_payload = {"calibration_run_id": args.run_id or _new_run_id()}
        remote_paths = {
            "status_remote_path": f"{args.storage_app}/calibration_runs/{function_payload['calibration_run_id']}/status.json",
            "log_remote_path": f"{args.storage_app}/calibration_runs/{function_payload['calibration_run_id']}/artifacts/logs/calibration_run.log",
            "run_root_remote_path": f"{args.storage_app}/calibration_runs/{function_payload['calibration_run_id']}/",
        }
    else:
        app_name = args.cpu_app
        config_path = CPU_CONFIG
        calibration_run_id = args.calibration_run_id or _latest_remote_dir(f"{args.storage_app}/calibration_runs/")
        analysis_run_id = args.analysis_run_id or _new_run_id()
        function_payload = {
            "calibration_run_id": calibration_run_id,
            "analysis_run_id": analysis_run_id,
            "workers": args.workers,
        }
        remote_paths = {
            "status_remote_path": (
                f"{args.storage_app}/calibration_runs/{calibration_run_id}/analysis_runs/{analysis_run_id}/status.json"
            ),
            "log_remote_path": (
                f"{args.storage_app}/calibration_runs/{calibration_run_id}/analysis_runs/{analysis_run_id}/analysis_run.log"
            ),
            "output_remote_path": (
                f"{args.storage_app}/calibration_runs/{calibration_run_id}/analysis_runs/{analysis_run_id}/capture_analysis.json"
            ),
        }

    if not args.skip_deploy:
        _deploy_app(config_path, app_name)

    endpoint = (
        f"https://api.aws.{region}.cerebrium.ai/v4/"
        f"{urllib.parse.quote(project)}/{urllib.parse.quote(app_name)}/"
        f"{args.function}?async=true"
    )
    response = _post_async(endpoint, token, function_payload)
    result = {
        "function": args.function,
        "endpoint": endpoint,
        "app_name": app_name,
        "request": response,
        **function_payload,
        **remote_paths,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

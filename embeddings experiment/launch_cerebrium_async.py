#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
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
REST_BASE = "https://rest.cerebrium.ai/v2"
FINAL_RUN_STATUSES = {"success", "failure", "cancelled", "timeout"}


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


def _post_async_with_retry(
    endpoint: str,
    token: str | None,
    payload: dict[str, object],
    *,
    retries: int,
    delay_seconds: float,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _post_async(endpoint, token, payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 404 or attempt >= retries:
                raise
            time.sleep(delay_seconds * (attempt + 1))
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def _api_get_json(url: str, token: str | None) -> object:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"} if token else {},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _app_id(project: str, app_name: str) -> str:
    return f"{project}-{app_name}"


def _list_runs(project: str, app_name: str, token: str | None) -> list[dict[str, object]]:
    payload = _api_get_json(
        f"{REST_BASE}/projects/{urllib.parse.quote(project)}/apps/{urllib.parse.quote(_app_id(project, app_name))}/runs",
        token,
    )
    if not isinstance(payload, dict):
        return []
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _get_run(project: str, app_name: str, run_id: str, token: str | None) -> dict[str, object] | None:
    for item in _list_runs(project, app_name, token):
        if str(item.get("id", "")) == run_id:
            return item
    return None


def _wait_for_run(
    project: str,
    app_name: str,
    run_id: str,
    token: str | None,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        run = _get_run(project, app_name, run_id, token)
        if run is not None:
            status = str(run.get("status", ""))
            if status != last_status:
                print(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "app_name": app_name,
                            "status": status,
                            "completed_at": run.get("completedAt"),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
                last_status = status
            if run.get("completedAt") or status in FINAL_RUN_STATUSES:
                return run
        time.sleep(poll_seconds)
    raise TimeoutError(f"timed out waiting for run {run_id} on {app_name}")


def _delete_app(project: str, app_name: str) -> None:
    subprocess.run(
        ["cerebrium", "apps", "delete", _app_id(project, app_name)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _is_run_active(run: dict[str, object]) -> bool:
    status = str(run.get("status", ""))
    return not run.get("completedAt") and status not in FINAL_RUN_STATUSES


def _cleanup_idle_apps(project: str, token: str | None, app_names: list[str]) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for app_name in app_names:
        runs = _list_runs(project, app_name, token)
        active = [run for run in runs if _is_run_active(run)]
        deleted = False
        error = ""
        if not active:
            try:
                _delete_app(project, app_name)
                deleted = True
            except subprocess.CalledProcessError as exc:
                error = (exc.stdout or str(exc)).strip()
        results.append(
            {
                "app_name": app_name,
                "active_runs": [str(run.get("id", "")) for run in active],
                "deleted": deleted,
                "error": error,
            }
        )
    return {"project": project, "results": results}


def _new_run_id() -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "function",
        choices=["collect_smoke", "calibration_run", "main_run", "analyze_latest_calibration", "cleanup_idle"],
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
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--keep-app", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=43200.0)
    parser.add_argument("--submit-retries", type=int, default=8)
    parser.add_argument("--submit-retry-seconds", type=float, default=3.0)
    args = parser.parse_args()

    cli_config = _read_cli_config()
    region = args.region or cli_config.get("defaultregion") or "us-east-1"
    project = cli_config.get("project")
    token = cli_config.get("accesstoken")
    if not project:
        raise RuntimeError("missing project in ~/.cerebrium/config.yaml")

    if args.function == "cleanup_idle":
        result = _cleanup_idle_apps(project, token, [args.gpu_app, args.cpu_app])
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

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
    elif args.function == "main_run":
        app_name = args.gpu_app
        config_path = GPU_CONFIG
        function_payload = {"main_run_id": args.run_id or _new_run_id()}
        remote_paths = {
            "status_remote_path": f"{args.storage_app}/main_runs/{function_payload['main_run_id']}/status.json",
            "log_remote_path": f"{args.storage_app}/main_runs/{function_payload['main_run_id']}/artifacts/logs/main_run.log",
            "run_root_remote_path": f"{args.storage_app}/main_runs/{function_payload['main_run_id']}/",
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
    response = _post_async_with_retry(
        endpoint,
        token,
        function_payload,
        retries=args.submit_retries,
        delay_seconds=args.submit_retry_seconds,
    )
    run_id = str(response.get("run_id", ""))
    result = {
        "function": args.function,
        "endpoint": endpoint,
        "app_name": app_name,
        "request": response,
        **function_payload,
        **remote_paths,
    }
    if not args.detach and run_id:
        final_run = _wait_for_run(
            project,
            app_name,
            run_id,
            token,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        result["final_run"] = final_run
        if not args.keep_app:
            _delete_app(project, app_name)
            result["app_deleted_after_run"] = True
        else:
            result["app_deleted_after_run"] = False
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_CONFIG="${ROOT_DIR}/cerebrium.toml"
CPU_CONFIG="${ROOT_DIR}/cerebrium_analysis.toml"
BACKUP_CONFIG="$(mktemp "${ROOT_DIR}/.cerebrium.toml.backup.XXXXXX")"

cleanup() {
  if [[ -f "${BACKUP_CONFIG}" ]]; then
    mv -f "${BACKUP_CONFIG}" "${GPU_CONFIG}"
  fi
}

cp "${GPU_CONFIG}" "${BACKUP_CONFIG}"
trap cleanup EXIT

cp "${CPU_CONFIG}" "${GPU_CONFIG}"
cd "${ROOT_DIR}"
cerebrium run cerebrium_main.py::analyze_latest_calibration "$@"

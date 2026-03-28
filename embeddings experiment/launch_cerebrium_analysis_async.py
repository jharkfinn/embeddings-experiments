#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    command = [
        sys.executable,
        str(ROOT / "launch_cerebrium_async.py"),
        "analyze_latest_calibration",
        *sys.argv[1:],
    ]
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import os
from pathlib import Path


def bootstrap_workspace_env() -> dict[str, Path]:
    workspace_root = Path("/workspace")
    if not workspace_root.exists():
        return {}

    workspace_home = workspace_root / "home"
    cache_root = workspace_root / ".cache"
    home_related = {
        "HOME": workspace_home,
        "XDG_CACHE_HOME": cache_root / "xdg",
        "HF_HOME": cache_root / "huggingface",
        "HF_HUB_CACHE": cache_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "CUDA_CACHE_PATH": cache_root / "nv",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for path in home_related.values():
        path.mkdir(parents=True, exist_ok=True)
    for env_name, path in home_related.items():
        os.environ[env_name] = str(path)
    return home_related

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_thunder.txt

python - <<'PY'
import torch
import transformers
import datasets
import accelerate
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import fbgemm_gpu.experimental.gen_ai

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
print("cuda", torch.cuda.is_available())
print("flex", bool(flex_attention and create_block_mask))
print("genai", True)
PY

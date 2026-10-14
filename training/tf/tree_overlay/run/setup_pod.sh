#!/usr/bin/env bash
# build the training environment on a fresh runpod instance
#
# Run ON THE POD, once, after the sources are there:
#   bash setup_pod.sh /workspace/fx2t
#
# Leaves you with a `uv` virtual environment inside the tree, from HIS lockfile,
# so the torch / flash-linear-attention / triton versions are the ones the code
# was written against.  The pins matter:
#   * pysrc/optimizer.py does `from torch.optim import AdamW, Muon`; Muon
#     landed in torch 2.12, and his pyproject pins torch>=2.12,<2.13.  A pod
#     image with torch 2.8 fails at import.
#   * pysrc/model.py imports flash-linear-attention (fla) for the Kimi Linear
#     kernels (chunk_kda, FusedRMSNormGated); fla pulls its own triton.
#   * flash-attn comes from a prebuilt cu130/torch2.12/cp311 wheel URL in his
#     [tool.uv.sources]; do not let uv try to build it from source.
#   * Python must be 3.11 (his requires-python is >=3.11,<3.12).
set -euo pipefail

TREE="${1:-/workspace/fx2t}"
cd "$TREE"

echo "=== python / cuda inventory"
nvidia-smi || true
python3 --version || true

echo "=== install uv (astral) if missing"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

echo "=== python 3.11 for this tree"
uv python install 3.11
uv venv --python 3.11

echo "=== resolve from his lockfile (uv.lock is shipped with the sources)"
# --frozen: use uv.lock as-is, do not re-resolve.  If this fails because the
# prebuilt flash-attn wheel URL is gone, fall back to `uv sync` (re-resolves)
# and record which versions you ended up with in the run log; a changed fla
# or triton version is a plausible cause of a numerically different run.
uv sync --frozen || {
  echo "!! --frozen failed, re-resolving; RECORD THE VERSIONS" >&2
  uv sync
}

echo "=== versions actually installed"
uv run python - <<'PY'
import importlib.metadata as md
for name in ("torch", "flash-linear-attention", "triton", "flash-attn",
             "numpy", "pytorch-optimizer", "jaxtyping", "tqdm"):
    try:
        print(name.ljust(26), md.version(name))
    except Exception as e:
        print(name.ljust(26), "MISSING", e)
import torch
print("torch.cuda".ljust(26), torch.version.cuda, "| devices", torch.cuda.device_count())
print("device".ljust(26), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
try:
    from torch.optim import Muon  # noqa: F401
    print("torch.optim.Muon".ljust(26), "present")
except Exception as e:
    print("torch.optim.Muon".ljust(26), "MISSING ->", e)
PY

echo "=== smoke: our package imports"
uv run python -m tft.param_count

echo
echo "setup done in $TREE"
echo "next: bash run/smoke.sh   (tiny window end to end, ~20 min)"

#!/usr/bin/env bash
# Read the epoch ("Epoche") from the optimizer state file of a checkpoint
# instead of estimating it; the trainer stores {"epoch": ...} in the .opt.
#   bash /root/epoche_opt.sh /root/ckpt/xyz.tch.opt   ->  EPOCH=<number>
# Uses the python of the trainer tree /root/fx6m_8m.
set -u
F=${1:?opt file}
[ -s "$F" ] || { echo "EPOCH=MISSING"; exit 2; }
/root/fx6m_8m/.venv/bin/python - "$F" <<'PY'
import sys
import torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("EPOCH=%s" % d.get("epoch"))
PY

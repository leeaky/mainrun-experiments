#!/usr/bin/env bash
# RunPod baseline-verification script for the mainrun exercise.
# Run this in the pod shell (the pod IS the devcontainer image).
# Assumes a persistent network volume mounted at /workspace.
set -euo pipefail

REPO_URL="https://github.com/MaincodeHQ/mainrun.git"
WORK=/workspace/mainrun

echo "=== 1. GPU / CUDA check ==="
nvidia-smi || { echo "!! nvidia-smi failed - pod has no GPU"; exit 1; }
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda compiled:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch.cuda.is_available() is False - stop here"
print("device:", torch.cuda.get_device_name(0))
PY

echo "=== 2. Devcontainer gate check ==="
test -f /root/.mainrun && echo "/root/.mainrun present - utils.py gate will pass"

echo "=== 3. Clone repo onto persistent volume (if absent) ==="
if [ ! -d "$WORK/.git" ]; then
  git clone "$REPO_URL" "$WORK"
else
  echo "repo already present at $WORK - leaving as-is (dataset cache in mainrun/data persists)"
fi
cd "$WORK"

echo "=== 4. Git identity for checkpoint.mjs auto-commit ==="
git config --global user.email "phi.bedrock@gmail.com"
git config --global user.name "mainrun runpod"
git config --global --add safe.directory "$WORK"

echo "=== 5. Baseline run (task train) - timed ==="
/usr/bin/time -v task train 2>&1 | tee /workspace/baseline-run.log

echo "=== 6. Result ==="
grep -E '"event": "validation_step"' mainrun/logs/mainrun.log | tail -1
grep -E 'Elapsed \(wall clock\)' /workspace/baseline-run.log || true
echo "Done. Compare val loss to the local baseline of 1.7519."

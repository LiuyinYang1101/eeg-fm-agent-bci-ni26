#!/usr/bin/env bash
# setup_local.sh — build the local conda env `neural_interface_challenge` (NeurIPS 2026 EEG/EMG challenge kit).
# The kit wants python 3.12 + benchopt>=1.10; the Codabench worker pins neuralset/neuralfetch/neuralbench 0.3.1
# and exca 0.5.29 (third_party/2026-competition/requirements.txt). We pin the same so that local windows are
# byte-identical to what the scoring container feeds predict(). torch comes from the cu128 wheel index (RTX 5090).
# Idempotent: re-running only installs what is missing.
set -euo pipefail
ENV=neural_interface_challenge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list | awk '{print $1}' | grep -qx "$ENV" || conda create -y -n "$ENV" python=3.12 pip
conda activate "$ENV"
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$(dirname "$0")/requirements_kit.txt"
python - <<'PY'
import torch, benchopt, neuralbench, neuralset, moabb, mne, braindecode, sklearn
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
for m in (benchopt, neuralbench, neuralset, moabb, mne, braindecode, sklearn):
    print(f"{m.__name__:12s} {getattr(m, '__version__', '?')}")
PY
echo "[setup_local] OK env=$ENV"

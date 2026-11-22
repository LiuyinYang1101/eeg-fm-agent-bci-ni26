#!/usr/bin/env bash
# EEGNet control on the Scherer2015 3-class cross-session proxy (local 5090; submit via gpu-submit).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
W=~/workspace/datasets/neural_interface_challenge/t2_bci/windows/scherer2015_cue3s4
O=~/workspace/outputs/neural_interface_challenge/t2/baselines
export NUMPY_MADVISE_HUGEPAGE=0
for p in xsession within; do
  $PY scripts/t2_baselines.py --cache "$W" --code-map scherer3 --protocol "$p" --n-calib 1 --n-folds 9 \
      --models eegnet --seeds 0 1 2 --out "$O/scherer3_${p}_eegnet.json"
done

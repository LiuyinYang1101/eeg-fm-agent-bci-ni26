#!/usr/bin/env bash
# Does the 1-run cross-session screen (xsession_all) rank models like the 9-fold sealed mirror? (Scherer, 6 seeds)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
export NUMPY_MADVISE_HUGEPAGE=0
$PY scripts/t2_bench.py --cache ~/workspace/datasets/neural_interface_challenge/t2_bci/windows/scherer2015_cue3s4 \
    --code-map scherer3 --protocol xsession_all --n-calib 1 --models eegnet shallow deep4 eegnex ifnet fbcnet eegtcnet atcnet conformer \
    --recipes r1 --seeds 0 1 2 3 4 5 --out-dir ~/workspace/outputs/neural_interface_challenge/t2/bench/scherer3

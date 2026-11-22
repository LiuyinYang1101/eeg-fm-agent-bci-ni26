#!/usr/bin/env bash
# Night 2026-10-05, block S (local 5090 via gpu-submit): Scherer2015 3-command cross-day proxy (HAND/SUB/WORD).
#   S1 model zoo   x 6 seeds, sealed mirror (session 1 = calibration, session 2 scored, 9 user groups)
#   S2 calibration value: xsubject (target users' calibration REMOVED) for eegnet/shallow/deep4 x 6 seeds
#   S3 recipe sensitivity for eegnet: r0 (first-day recipe), r1_mix, r1_lowlr x 6 seeds
#   S4 control: shuffled labels must score chance (eegnet, 6 seeds)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
W=~/workspace/datasets/neural_interface_challenge/t2_bci/windows/scherer2015_cue3s4
B=~/workspace/outputs/neural_interface_challenge/t2/bench/scherer3
export NUMPY_MADVISE_HUGEPAGE=0
C=(--cache "$W" --code-map scherer3 --n-calib 1 --n-folds 9 --seeds 0 1 2 3 4 5 --out-dir "$B")
$PY scripts/t2_bench.py "${C[@]}" --protocol xsession --models eegnet shallow deep4 eegnex ifnet fbcnet eegtcnet atcnet conformer --recipes r1
$PY scripts/t2_bench.py "${C[@]}" --protocol xsubject --models eegnet shallow deep4 --recipes r1
$PY scripts/t2_bench.py "${C[@]}" --protocol xsession --models eegnet --recipes r0 r1_mix r1_lowlr
$PY scripts/t2_bench.py "${C[@]}" --protocol xsession --models eegnet --recipes r1 --shuffle-labels --tag _shuf
echo "block S done $(date -Is)"

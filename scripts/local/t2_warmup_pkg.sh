#!/usr/bin/env bash
# End-to-end warm-up submission (build items 5-6): <method> r1 × 3 seeds on the Codabench warm-up TRAINING pool
# (Dreyer users 1-60 + 82-87, duplicate recording excluded), trained on CPU (the GPU queue is full tonight),
# packaged flat, then contract-tested incl. a local replay of the Codabench warm-up evaluation.
#   bash scripts/local/t2_warmup_pkg.sh <name> [method=eegnet_r1]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
export CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=12
NAME=${1:?name}; METHOD=${2:-eegnet_r1}
[ -s "outputs/t2/submissions/$NAME.manifest.json" ] || \
  $PY scripts/t2_final.py --protocol dreyer_kit --method "$METHOD" --train-on kit --seeds 0 1 2 --name "$NAME" --device cpu
$PY scripts/t2_contract_test.py --name "$NAME" --warmup-replay

#!/usr/bin/env bash
# Contract-test already-packaged submissions (CPU, incl. warm-up replay).  bash scripts/local/t2_contract_only.sh <name>...
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
for n in "$@"; do ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/t2_contract_test.py --name "$n" --warmup-replay; done

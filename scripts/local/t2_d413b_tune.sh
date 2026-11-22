#!/usr/bin/env bash
# T2-D413b PHASE B (pre-registered 2026-11-09 11:23): one gpu-submit job that runs a batch file of generator configs on ONE protocol, each =
# t2p_meanflow.py (train, or --from-ckpt resample) + scripts/local/t2_d413b_tune_eval.py (TRAINING-WINDOW fidelity only -> tune.jsonl).
# Batch file lines: "<tag> | <t2p_meanflow args>" ('#' comments). Outputs in outputs/t2_d413b/ (tune.log, tune.jsonl, <f>_<proto>_<tag>.*).
#   bash scripts/local/t2_d413b_tune.sh <protocol> <batch file> [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
P=$1; B=$2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_meanflow.py scripts/local/t2_d413b_tune_eval.py scripts/local/t2_d413_gate.py "$B")" ] || { echo "commit the D413b code + batch file first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs/t2_d413b
mkdir -p "$O"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d413b_tune.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD) $P $B \$(date +%T)\" >> '$O/tune.log'"
  grep -v '^[[:space:]]*#' "$B" | grep -v '^[[:space:]]*$' | while IFS='|' read -r tag args; do
    tag=$(echo $tag)
    echo "echo \"### \$(date +%T) $P $tag: $args\" >> '$O/tune.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$EXP' --protocol $P --out '$O' --tag $tag $args >> '$O/tune.log' 2>&1 && \\"
    echo "  $SC $PY -u scripts/local/t2_d413b_tune_eval.py --export '$EXP' --gen-dir '$O' --protocol $P --tag $tag --note '$args' >> '$O/tune.log' 2>&1 || rc=\$?"
  done
  echo "echo \"### done \$(date +%T) rc=\$rc\" >> '$O/tune.log'"
  echo "exit \$rc"
} > "$job"
if [ "${3:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d413b-tune" bash "$job"

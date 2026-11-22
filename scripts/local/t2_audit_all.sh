#!/usr/bin/env bash
# Data-quality audit of every Track-2 window cache (CPU; run under mem-guard). Reports -> outputs/t2/audit/<cache>/
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
W=~/workspace/datasets/neural_interface_challenge/t2_bci/windows
A=~/workspace/outputs/neural_interface_challenge/t2/audit
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
$PY scripts/t2_audit_data.py --cache "$W/scherer2015_cue3s4" --code-map scherer3 --n-calib 1 --out "$A/scherer2015_cue3s4"
$PY scripts/t2_audit_data.py --cache "$W/dreyer2023" --n-calib 2 --parts A:1-60,B:61-81,C:82-87 --out "$A/dreyer2023"
$PY scripts/t2_audit_data.py --cache "$W/dreyer2023_aux" --n-calib 2 --parts A:1-60,B:61-81,C:82-87 --out "$A/dreyer2023_aux"
echo "audit done $(date -Is)"

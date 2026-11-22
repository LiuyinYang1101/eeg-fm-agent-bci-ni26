#!/usr/bin/env bash
# T2-D433 part (b): MantisV2 frozen embeddings member, CPU only (dump -> embed in eeg311 -> fit).   mem-guard bash scripts/local/t2_d433b_run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
D=$HOME/workspace/outputs/neural_interface_challenge/t2_d433
S=tracks/t2_bci/scripts/t2_d433b_mantis.py
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
$HOME/miniconda3/envs/neural_interface_challenge/bin/python -u $S dump --export "$E" --dir "$D"
$HOME/miniconda3/envs/eeg311/bin/python -u $S embed --dir "$D"
$HOME/miniconda3/envs/neural_interface_challenge/bin/python -u $S fit --export "$E" --dir "$D"
rm -f "$D/white.npy"

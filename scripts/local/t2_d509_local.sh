#!/usr/bin/env bash
# T2-D509 (the in-stream true-label fine-tune ceiling; registered 2026-11-15 before any number; zero grant, the local 5090 via gpu-submit):
# ShallowNet, M7's pooled recipe, K = 9 panel (lc9, 10 folds), seeds 790-792, --instream-ft 30,60,100 (is0 / is30 / is60 / is100 files).
#   bash scripts/local/t2_d509_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d509_local.sh)" ] || { echo "commit the D509 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
D=$HOME/workspace/outputs/neural_interface_challenge/t2_d509
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
LINE="--protocols lc9 --models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds 790,791,792 --instream-ft 30,60,100"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d509_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### \$(date +%T) d509: $LINE\" >> '$D/arm.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $LINE >> '$D/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "echo \"### done d509 \$(date +%T): \$(ls '$D/runs' | grep -c 'is100') is100 files (expect 30)\" >> '$D/arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d509" bash "$job"

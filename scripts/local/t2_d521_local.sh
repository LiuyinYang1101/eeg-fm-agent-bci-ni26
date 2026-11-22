#!/usr/bin/env bash
# T2-D521 arm R5: the RAW end-to-end net (src/ni26/rawnet.py; --aligns none --keep-mean = one global gain, no de-mean, no alignment), pooled,
# K = 9 lc9, r1_pulx_vp2_e180_vr, seeds d, d + 1000 ... d + 4000 for d in 810-812 (15 seeds, 150 nets).
#   mem-guard bash scripts/local/t2_d521_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d521_local.sh)" ] || { echo "commit the D521 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d521
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d521_job.XXXX.sh")
D=$R/R
line="--protocols lc9 --models rawnet --aligns none --keep-mean --recipe r1_pulx_vp2_e180_vr --seeds 810,1810,2810,3810,4810,811,1811,2811,3811,4811,812,1812,2812,3812,4812"
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$D/runs'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### \$(date +%T) d521 R5: $line\" >> '$R/arm.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "echo \"### done d521 R5 \$(date +%T): \$(ls '$D/runs' | wc -l) files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/arm.log'"
} > "$job"
mkdir -p "$R"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d521" bash "$job"

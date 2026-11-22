#!/usr/bin/env bash
# T2-D554b (registered 2026-11-17 before its number; the manager's OK): specnet_sf (specnet + a learned 41 -> 32 spatial projection before the
# STFT), the speech recipe r1_spec_e180_mx4, EA, lc9, pooled members, seeds 822 / 823 / 824 -> outputs/t2_d554b/runs; the local 5090 via
# gpu-submit (one job, 3 shards; ≈ 15-30 min). Read: scripts/local/t2_d554_read.py --model specnet_sf --runs outputs/t2_d554b/runs --kill 0.626
#   bash scripts/local/t2_d554b_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d554b_local.sh)" ] || { echo "commit first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
D=$O/t2_d554b
[ ! -e "$D/runs" ] || { echo "$D/runs exists: never overwrite a read's inputs"; exit 3; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
line="--protocols lc9 --models specnet_sf --aligns ea --recipe r1_spec_e180_mx4 --seeds 822,823,824"
job=$(mktemp "$O/t2_d554b_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### \$(date +%T) d554b: $line\" >> '$D/arm.log'"
  for i in 0 1 2; do echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"; done
  echo "wait"
  echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d554b \$(date +%T): \$n files (expect 30), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$D/arm.log'; [ \$n -eq 30 ]"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d554b" bash "$job"

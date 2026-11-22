#!/usr/bin/env bash
# T2-D538 (W19-F; registered 2026-11-16 19:36:59 before any number; zero grant, the local 5090 via gpu-submit): window-mean side nets for the
# FBCNet family at the D536 draws 816-818 (a NOMINATION read; the manager's condition 1: confirmation on fresh draws 819-821 follows a pass).
# = t2_d536_side_local.sh's side line for FBCNet: --protocols lc9 --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr --side dcmean,
# seeds d + k * 1000, k 0..4, d 816-818, NI26_SIDE_DUMP=1 (D539 can re-fit their heads), into outputs/t2_d538/side_fbc (150 files expected).
# The shipped FBCNet line is r1_pulxx_vp2_e180_vr_rc75; the side head is wired for the plain CE branch only (src/ni26/train.py: rc refused), so
# the side nets use the same line without rc75 (stated in D538).
#   bash scripts/local/t2_d538_side_fbc_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d538_side_fbc_local.sh)" ] || { echo "commit the D538 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d538
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$O/t2_d538_side_job.XXXX.sh")
S5=816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  D=$R/side_fbc
  line="--protocols lc9 --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr --seeds $S5 --side dcmean"
  echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d538 side_fbc: $line\" >> '$R/side_arm.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d538 side_fbc \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/side_arm.log'; [ \$n -eq 150 ] || rc=6"
  echo "echo \"### done d538 \$(date +%T) rc=\$rc\" >> '$R/side_arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d538-side-fbc" bash "$job"

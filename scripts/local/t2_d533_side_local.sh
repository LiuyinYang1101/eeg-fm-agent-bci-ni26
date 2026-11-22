#!/usr/bin/env bash
# T2-D533 (the side nets' z clamp, stage 1; registered 2026-11-16 before this job ran; zero grant, the local 5090 via gpu-submit): D532's fresh S5
# side job re-run ONCE with NI26_SIDE_DUMP=1 (a3a860d2: each net's last-epoch pre-head logits, head W / b and test z in its npz), the same line
# exactly (t2_d532_side_local.sh: --protocols lc9 --aligns ea --recipe r1_pulx_vp2_e180_vr --side dcmean, ShallowNet and IFNet, seeds d + k * 1000,
# k 0..4, d 813-815), into outputs/t2_d533 (never D532's folders). No DC bag here.
#   bash scripts/local/t2_d533_side_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d533_side_local.sh)" ] || { echo "commit the D533 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d533
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$O/t2_d533_side_job.XXXX.sh")
S5=813,1813,2813,3813,4813,814,1814,2814,3814,4814,815,1815,2815,3815,4815
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for spec in "side_sh:shallow" "side_ifn:ifnet"; do
    IFS=: read -r arm models <<< "$spec"
    D=$R/$arm
    line="--protocols lc9 --models $models --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S5 --side dcmean"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d533 $arm: $line\" >> '$R/side_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d533 $arm \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/side_arm.log'; [ \$n -eq 150 ] || rc=6"
  done
  echo "echo \"### done d533 side \$(date +%T) rc=\$rc\" >> '$R/side_arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d533-side" bash "$job"

#!/usr/bin/env bash
# T2-D511 (the diagnosed successor of D510 / D510b on fresh seeds; registered 2026-11-15 before any number; zero grant, the local 5090 via
# gpu-submit): ShallowNet, IFNet and FBCNet students (D498's pooled lines), K = 9 (lc9), seeds 800-802, every test stream, the ship schedule
# 60 / 100 / 150 with true / conf (causal) / confR (prefix re-decoded) labels, 3 repeats of confR. Teacher files: scripts/local/t2_d511_teacher.py.
#   bash scripts/local/t2_d511_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d511_local.sh scripts/local/t2_d511_teacher.py)" ] || { echo "commit the D511 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d511
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
COMMON="--protocols lc9 --seeds 800,801,802 --instream-all --instream-sched 60,100,150 --instream-sel true,conf,confR --instream-reps 3 --instream-rep-sel confR --instream-pl $R/teacher/{protocol}__s{seed}.npz"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d511_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "sh:shallow:ea:r1_pulx_vp2_e180_vr" "ifn:ifnet:ea:r1_pulx_vp2_e180_vr" "fbc:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75"; do
    IFS=: read -r fam model al rec <<< "$spec"
    D=$R/$fam
    line="--models $model --aligns $al --recipe $rec $COMMON"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d511 $fam: $line\" >> '$R/arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "echo \"### done d511 $fam \$(date +%T): \$(ls '$D/runs' | grep -c isSpr__) isSpr files (expect 30), \$(grep -c Traceback '$D'/shard_*.log | paste -sd+ | bc) tracebacks\" >> '$R/arm.log'"
  done
  echo "echo \"### done d511 all \$(date +%T)\" >> '$R/arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d511" bash "$job"

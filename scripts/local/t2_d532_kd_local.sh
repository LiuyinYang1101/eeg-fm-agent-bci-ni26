#!/usr/bin/env bash
# T2-D532 (the W18 fresh-draw read; registered 2026-11-16 before any number; zero grant, the local 5090 via gpu-submit): D530's kd ShallowNet
# students at the FRESH draws 813-815 = scripts/local/t2_d530_local.sh's 'students' stage EXACTLY with three changes: seeds 813-815; outputs in
# outputs/t2_d532 (cnn_shkd, route_shkd, p1_shkd); the P1 teacher labels = the fresh teacher (outputs/t2_d532/teacher_p1, t2_d532_p1_local.sh).
# The nested kd teachers are D530's (outputs/t2_d530/kdteacher__lc9_<f>__s0.npz: seed-free per fold, read only).
#   bash scripts/local/t2_d532_kd_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt scripts/local/t2_d530_local.sh scripts/local/t2_d532_kd_local.sh)" ] || { echo "commit the D532 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
B="tracks/t2_bci/scripts/t2p_bench.py --export '$EXP'"
SHLINE="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //')"
RT=$O/t2_d530; R=$O/t2_d532; FOLDS=0,1,2,3,4,5,6,7,8,9; SEEDS=813,814,815; stage=students
SLINE="$(echo "$SHLINE" | sed -E "s/--protocols [a-z0-9,]+ /--protocols lc9 /; s/--seeds [0-9,]+ /--seeds $SEEDS /; s/--recipe [a-z0-9_]+ /--recipe r1_pulx_vp2_e180_vr_kd /") --kd-teacher $RT/kdteacher__{fold}__s0.npz"
P1LINE="--models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr_kd --kd-teacher $RT/kdteacher__{fold}__s0.npz --protocols lc9 --seeds $SEEDS --instream-all --instream-sched 60,100,150 --instream-sel true,confR --instream-reps 3 --instream-rep-sel confR --instream-pl $R/teacher_p1/{protocol}__s{seed}.npz"
run3() {   # $1 = out dir, $2 = bench args: list the expected files, 3 shards, count the missing (as t2_d530_local.sh)
  local d=$1 line=$2
  echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
  echo "echo \"### \$(date +%T) d532 kd $stage: $line\" >> '$R/kd_arm.log'"
  echo "$SC $PY -u $B --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
  for i in 0 1 2; do echo "$SC $PY -u $B --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"; done
  echo "wait"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing, \$(cat '$d'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/kd_arm.log'; [ \$n -eq 0 ] || rc=6"
}
job=$(mktemp "$O/t2_d532_kd_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "for f in ${FOLDS//,/ }; do [ -f '$RT'/kdteacher__lc9_\${f}__s0.npz ] || { echo \"no kd teacher lc9_\$f\" >> '$R/kd_arm.log'; rc=3; }; done"
  echo "n=\$(ls '$R/teacher_p1' 2>/dev/null | wc -l); [ \$n -eq 30 ] || { echo \"fresh P1 teacher has \$n files (need 30)\" >> '$R/kd_arm.log'; rc=3; }"
  echo "if [ \$rc -eq 0 ]; then"
  run3 "$R/cnn_shkd" "$SLINE"
  echo "for f in ${FOLDS//,/ }; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$R/cnn_shkd/runs' --protocols lc9_\$f --out '$R/route_shkd' --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_kdtv8c --r 0.5 --tag 2 --seeds $SEEDS >> '$R/kd_arm.log' 2>&1 || rc=7; done"
  echo "echo \"### d532 kd routed \$(ls '$R/route_shkd' 2>/dev/null | wc -l) files (expect 30)\" >> '$R/kd_arm.log'"
  run3 "$R/p1_shkd" "$P1LINE"
  echo "echo \"### d532 kd p1: \$(ls '$R/p1_shkd/runs' | grep -c isSpr__) isSpr files (expect 30)\" >> '$R/kd_arm.log'"
  echo "fi"
  echo "echo \"### done d532 kd \$(date +%T) rc=\$rc\" >> '$R/kd_arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d532-kd" bash "$job"

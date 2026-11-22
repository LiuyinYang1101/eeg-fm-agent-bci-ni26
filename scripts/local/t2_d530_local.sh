#!/usr/bin/env bash
# T2-D530 (registered 2026-11-16 before any number; zero grant, the local 5090 via gpu-submit): ShallowNet on NESTED out-of-fold
# cross-user soft targets, in W17 on K = 9.
#   teachers : per lc9 fold f and training user j (protocol lc9_<f>oof), M7's pooled FBCNet (r1_pulxx_vp2_e180_vr_rc75 / none) and
#              ShallowNet (r1_pulx_vp2_e180_vr / ea), seed 0, trained without the fold's stranger and without j, predicting j's training
#              windows (180 fits), then t2_d530_teacher.py -> outputs/t2_d530/kdteacher__lc9_<f>__s0.npz (T2-D490's form)
#   students : D512's ShallowNet line exactly (t2d429_sh_ctl_s348 on lc9, seeds 810-812, copies + tv8c, hard-label copies) with recipe
#              _kd (kd 0.5, T 2, pooled net), routed per fold as D512 (r 0.5 tag 2); and its P1 in-stream files (D512's P1 line with the
#              _kd recipe; teacher labels = D512's outputs/t2_d512/teacher_p1)
#   bash scripts/local/t2_d530_local.sh smoke|teachers|students [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?smoke|teachers|students}
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt scripts/local/t2_d530_local.sh)" ] || { echo "commit the D530 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
B="tracks/t2_bci/scripts/t2p_bench.py --export '$EXP'"
SHLINE="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //')"
case $stage in
  smoke)    R=$O/t2_d530_smoke; FOLDS=0; SEEDS=810; X="--epochs 1"; XS="--folds lc9_0"; PROTO_T=lc9_0oof ;;
  teachers|students) R=$O/t2_d530; FOLDS=0,1,2,3,4,5,6,7,8,9; SEEDS=810,811,812; X=""; XS=""
            PROTO_T=lc9_0oof,lc9_1oof,lc9_2oof,lc9_3oof,lc9_4oof,lc9_5oof,lc9_6oof,lc9_7oof,lc9_8oof,lc9_9oof ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
SLINE="$(echo "$SHLINE" | sed -E "s/--protocols [a-z0-9,]+ /--protocols lc9 /; s/--seeds [0-9,]+ /--seeds $SEEDS /; s/--recipe [a-z0-9_]+ /--recipe r1_pulx_vp2_e180_vr_kd /") --kd-teacher $R/kdteacher__{fold}__s0.npz $X $XS"
P1LINE="--models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr_kd --kd-teacher $R/kdteacher__{fold}__s0.npz --protocols lc9 --seeds $SEEDS --instream-all --instream-sched 60,100,150 --instream-sel true,confR --instream-reps 3 --instream-rep-sel confR --instream-pl $O/t2_d512/teacher_p1/{protocol}__s{seed}.npz $X $XS"
run3() {   # $1 = out dir, $2 = bench args: list the expected files, 3 shards, count the missing
  local d=$1 line=$2
  echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
  echo "echo \"### \$(date +%T) d530 $stage: $line\" >> '$R/arm.log'"
  echo "$SC $PY -u $B --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
  for i in 0 1 2; do echo "$SC $PY -u $B --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"; done
  echo "wait"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing, \$(cat '$d'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/arm.log'; [ \$n -eq 0 ] || rc=6"
}
job=$(mktemp "$O/t2_d530_${stage}_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  if [ "$stage" != students ]; then
    run3 "$R/teachers_fbc" "--protocols $PROTO_T --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds 0 $X"
    run3 "$R/teachers_sh" "--protocols $PROTO_T --models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds 0 $X"
    echo "mkdir -p '$R/teachers_all'; ln -sf '$R'/teachers_fbc/runs/*.npz '$R'/teachers_sh/runs/*.npz '$R/teachers_all/'"
    echo "[ \$rc -eq 0 ] && { $SC $PY -u tracks/t2_bci/scripts/t2_d530_teacher.py --oof '$R/teachers_all' --out '$R' --folds $FOLDS >> '$R/arm.log' 2>&1 || rc=8; }"
  fi
  if [ "$stage" != teachers ]; then
    echo "for f in ${FOLDS//,/ }; do [ -f '$R'/kdteacher__lc9_\${f}__s0.npz ] || { echo \"no teacher lc9_\$f\" >> '$R/arm.log'; rc=3; }; done"
    echo "if [ \$rc -eq 0 ]; then"
    run3 "$R/cnn_shkd" "$SLINE"
    echo "for f in ${FOLDS//,/ }; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$R/cnn_shkd/runs' --protocols lc9_\$f --out '$R/route_shkd' --model shallow --align ea --recipe r1_pulx_vp2_e180_vr_kdtv8c --r 0.5 --tag 2 --seeds $SEEDS >> '$R/arm.log' 2>&1 || rc=7; done"
    echo "echo \"### d530 routed \$(ls '$R/route_shkd' 2>/dev/null | wc -l) files\" >> '$R/arm.log'"
    run3 "$R/p1_shkd" "$P1LINE"
    echo "echo \"### d530 p1: \$(ls '$R/p1_shkd/runs' | grep -c isSpr__) isSpr files\" >> '$R/arm.log'"
    echo "fi"
  fi
  echo "echo \"### done d530 $stage \$(date +%T) rc=\$rc\" >> '$R/arm.log'; exit \$rc"
} > "$job"
if [ "${2:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d530-$stage" bash "$job"

#!/usr/bin/env bash
# T2-D532 (the W18 fresh-draw read; registered 2026-11-16 before any number): T2-D512 part P1 stage 2 EXACTLY (t2_d512_p1_local.sh) at the FRESH
# CNN draws 813-815: first the teacher (scripts/local/t2_d532_teacher.py = t2_d512_teacher.py on t2_d532, CPU), then the in-stream students.
# The shipped form (D512 / W15): slots sh,ifn, refits 60,100,150, confR:
#   bash scripts/local/t2_d532_p1_local.sh sh,ifn 60,100,150 confR [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
SLOTS=${1:?slots}; CKS=${2:?refits}; LABEL=${3:?conf|confR}
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d512_teacher.py scripts/local/t2_d532_teacher.py scripts/local/t2_d532_p1_local.sh)" ] || { echo "commit the D532 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d532
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
COMMON="--protocols lc9 --seeds 813,814,815 --instream-all --instream-sched $CKS --instream-sel true,$LABEL --instream-reps 3 --instream-rep-sel $LABEL --instream-pl $R/teacher_p1/{protocol}__s{seed}.npz"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d532_p1_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  # T2-D532: the fresh teacher first (CPU; the routed fresh members must exist: 30 per family)
  echo "for fam in fbc sh ifn; do n=\$(ls '$R'/route_\$fam 2>/dev/null | wc -l); [ \$n -eq 30 ] || { echo \"### d532 p1: route_\$fam has \$n files (need 30)\" >> '$R/p1_arm.log'; exit 4; }; done"
  echo "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $SC $PY -u scripts/local/t2_d532_teacher.py >> '$R/p1_arm.log' 2>&1 || exit 9"
  echo "echo \"### d532 teacher: \$(ls '$R/teacher_p1' | wc -l) files (expect 30)\" >> '$R/p1_arm.log'"
  for fam in ${SLOTS//,/ }; do
    case $fam in
      sh) m="--models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr" ;;
      ifn) m="--models ifnet --aligns ea --recipe r1_pulx_vp2_e180_vr" ;;
      fbc) m="--models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75" ;;
      *) echo "unknown slot $fam"; exit 2 ;;
    esac
    D=$R/p1_$fam
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d532 p1 $fam: $m $COMMON\" >> '$R/p1_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $m $COMMON >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "echo \"### done d532 p1 $fam \$(date +%T): \$(ls '$D/runs' | grep -c isS__) isS files (expect 30), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/p1_arm.log'"
  done
  echo "echo \"### done d532 p1 all \$(date +%T)\" >> '$R/p1_arm.log'"
} > "$job"
if [ "${4:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d532-p1" bash "$job"

#!/usr/bin/env bash
# T2-D558 (the D555 / D557 second reads on FRESH draws 831-833; registered 2026-11-17 before any number; zero grant, the local 5090 via
# gpu-submit, ONLY when the manager sequences it; ONE job, stages in order). W18's form re-trained at the fresh draws = D540's chain
# (scripts/local/t2_d540_chain_local.sh) with the draw constants moved 816-818 -> 831-833 and the output dir t2_d536 -> t2_d558:
#   (1) CNN: the three families' M7 lines (copies + tv8c) on lc9 + routing (t2_d536_cnn_local.sh's job text, constants replaced);
#   (2) P1: the teacher (scripts/local/t2_d558_teacher.py, CPU) then the in-stream students sh,ifn 60,100,150 confR (t2_d536_p1_local.sh);
#   (3) side nets S5, NI26_SIDE_DUMP=1, + the DC bag at FRESH LightGBM seeds 22,23,24 (t2_d536_side_local.sh's job text);
#   (4) D543's S10 half: ShallowNet / IFNet side nets at seeds d + k * 1000, k 5..9, dumps on -> outputs/t2_d558/s10/side_{sh,ifn}.
# No FBCNet side nets (D538 is closed). The FM side = the D558 re-runs at seeds 837-842 (plain and augmented) + MI s832 (existing files).
#   bash scripts/local/t2_d558_chain_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES="scripts/local/t2_d536_cnn_local.sh scripts/local/t2_d536_p1_local.sh scripts/local/t2_d536_side_local.sh scripts/local/t2_d512_teacher.py scripts/local/t2_d558_teacher.py scripts/local/t2_d515_read.py scripts/local/t2_d558_chain_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms $FILES)" ] || { echo "commit the D558 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d558
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
job=$(mktemp "$O/t2_d558_chain_job.XXXX.sh")
sub() {   # a D536 launcher's generated job text (its --dry output) with the draw / dir constants moved; each replacement must have happened
  local t; t=$(bash "$@" --dry)
  t=${t//t2_d536_teacher.py/t2_d558_teacher.py}
  t=${t//t2_d536/t2_d558}
  t=${t//816,817,818/831,832,833}
  t=${t//816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818/831,1831,2831,3831,4831,832,1832,2832,3832,4832,833,1833,2833,3833,4833}
  t=${t//--seeds 7,8,9/--seeds 22,23,24}
  t=${t//pred_s7-8-9_dc/pred_s22-23-24_dc}
  t=${t//read_dc789/read_dc222324}
  t=${t// d536 / d558 }
  ! grep -qE "t2_d536|816|817|818|7-8-9|--seeds 7,8,9" <<< "$t" || { echo "a D536 constant survived in $1" >&2; exit 4; }
  printf '%s\n' "$t"        # its own 'exit $rc' ends only the ( ... ) subshell, so the stage's rc reaches the chain
}
S59=""; for d in 831 832 833; do for k in 5 6 7 8 9; do S59+="$((d + k * 1000)),"; done; done; S59=${S59%,}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$R'; echo \"### \$(date +%T) d558 chain start (git \$(git rev-parse --short HEAD))\" >> '$R/chain.log'"
  echo "( $(sub scripts/local/t2_d536_cnn_local.sh) ) || { echo \"### d558 cnn FAILED \$(date +%T)\" >> '$R/chain.log'; exit 5; }"
  echo "( $(sub scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR) ) || { echo \"### d558 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d558 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "( $(sub scripts/local/t2_d536_side_local.sh) ) || { echo \"### d558 side/DC FAILED \$(date +%T)\" >> '$R/chain.log'; exit 7; }"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for fam in sh ifn; do
    m=$([ $fam = sh ] && echo shallow || echo ifnet)
    D=$R/s10/side_$fam
    line="--protocols lc9 --models $m --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S59 --side dcmean"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d558 s10 $fam: $line\" >> '$R/s10_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d558 s10 $fam \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/s10_arm.log'; [ \$n -eq 150 ] || exit 8"
  done
  echo "echo \"### done d558 chain \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d558-chain" bash "$job"

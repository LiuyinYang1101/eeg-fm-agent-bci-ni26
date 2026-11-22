#!/usr/bin/env bash
# T2-D553 (the FM side-head refit read on FRESH draws 822-824; registered 2026-11-17 before any number; zero grant, the local 5090 via
# gpu-submit, ONLY when the manager sequences it; ONE job, stages in order). W18's form re-trained at the fresh draws = D540's chain
# (scripts/local/t2_d540_chain_local.sh) with the draw constants moved 816-818 -> 822-824 and the output dir t2_d536 -> t2_d553:
#   (1) CNN: the three families' M7 lines (copies + tv8c) on lc9 + routing (t2_d536_cnn_local.sh's job text, constants replaced);
#   (2) P1: the teacher (scripts/local/t2_d553_teacher.py, CPU) then the in-stream students sh,ifn 60,100,150 confR (t2_d536_p1_local.sh);
#   (3) side nets S5, NI26_SIDE_DUMP=1, + the DC bag at FRESH LightGBM seeds 13,14,15 (t2_d536_side_local.sh's job text);
#   (4) D543's S10 half: ShallowNet / IFNet side nets at seeds d + k * 1000, k 5..9, dumps on -> outputs/t2_d553/s10/side_{sh,ifn}.
# No FBCNet side nets (D538 is closed). The FM side = the D553 sofia re-runs (with head dumps) + MI s832 (existing files).
#   bash scripts/local/t2_d553_chain_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES="scripts/local/t2_d536_cnn_local.sh scripts/local/t2_d536_p1_local.sh scripts/local/t2_d536_side_local.sh scripts/local/t2_d512_teacher.py scripts/local/t2_d553_teacher.py scripts/local/t2_d515_read.py scripts/local/t2_d553_chain_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms $FILES)" ] || { echo "commit the D553 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d553
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
job=$(mktemp "$O/t2_d553_chain_job.XXXX.sh")
sub() {   # a D536 launcher's generated job text (its --dry output) with the draw / dir constants moved; each replacement must have happened
  local t; t=$(bash "$@" --dry)
  t=${t//t2_d536_teacher.py/t2_d553_teacher.py}
  t=${t//t2_d536/t2_d553}
  t=${t//816,817,818/822,823,824}
  t=${t//816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818/822,1822,2822,3822,4822,823,1823,2823,3823,4823,824,1824,2824,3824,4824}
  t=${t//--seeds 7,8,9/--seeds 13,14,15}
  t=${t//pred_s7-8-9_dc/pred_s13-14-15_dc}
  t=${t//read_dc789/read_dc131415}
  t=${t// d536 / d553 }
  ! grep -qE "t2_d536|816|817|818|7-8-9|--seeds 7,8,9" <<< "$t" || { echo "a D536 constant survived in $1" >&2; exit 4; }
  printf '%s\n' "$t"        # its own 'exit $rc' ends only the ( ... ) subshell, so the stage's rc reaches the chain
}
S59=""; for d in 822 823 824; do for k in 5 6 7 8 9; do S59+="$((d + k * 1000)),"; done; done; S59=${S59%,}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$R'; echo \"### \$(date +%T) d553 chain start (git \$(git rev-parse --short HEAD))\" >> '$R/chain.log'"
  echo "( $(sub scripts/local/t2_d536_cnn_local.sh) ) || { echo \"### d553 cnn FAILED \$(date +%T)\" >> '$R/chain.log'; exit 5; }"
  echo "( $(sub scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR) ) || { echo \"### d553 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d553 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "( $(sub scripts/local/t2_d536_side_local.sh) ) || { echo \"### d553 side/DC FAILED \$(date +%T)\" >> '$R/chain.log'; exit 7; }"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for fam in sh ifn; do
    m=$([ $fam = sh ] && echo shallow || echo ifnet)
    D=$R/s10/side_$fam
    line="--protocols lc9 --models $m --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S59 --side dcmean"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d553 s10 $fam: $line\" >> '$R/s10_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d553 s10 $fam \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/s10_arm.log'; [ \$n -eq 150 ] || exit 8"
  done
  echo "echo \"### done d553 chain \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d553-chain" bash "$job"

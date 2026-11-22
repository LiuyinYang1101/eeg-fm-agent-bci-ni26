#!/usr/bin/env bash
# T2-D615 (THE FRESH 5th DRAW SET E for D603's once-only read, the manager + Track3 R8 2026-11-19; registered before any job; the local 5090, ONE job):
# = t2_d578_chain_local.sh (set D) with the constants moved: draws 860-862 (CNN; disjoint from every earlier set and FM seed), root t2_d615,
# the DC LightGBM seeds 28-30; stages (1) CNN + routing, (2) P1 teacher + students, (3) side nets S5 + the DC bag, (4) D543's S10 half.
# The FM side = set D's sofia members (W20+D557's form, seeds 843-848), unchanged: E is an independent CNN-draw set.
#   bash scripts/local/t2_d615_chain_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES="scripts/local/t2_d536_cnn_local.sh scripts/local/t2_d536_p1_local.sh scripts/local/t2_d536_side_local.sh scripts/local/t2_d512_teacher.py scripts/local/t2_d615_teacher.py scripts/local/t2_d515_read.py scripts/local/t2_d615_chain_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms $FILES)" ] || { echo "commit the D578 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d615
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
job=$(mktemp "$O/t2_d615_chain_job.XXXX.sh")
sub() {   # a D536 launcher's generated job text (its --dry output) with the draw / dir constants moved; each replacement must have happened
  local t; t=$(bash "$@" --dry)
  t=${t//t2_d536_teacher.py/t2_d615_teacher.py}
  t=${t//t2_d536/t2_d615}
  t=${t//816,817,818/860,861,862}
  t=${t//816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818/860,1860,2860,3860,4860,861,1861,2861,3861,4861,862,1862,2862,3862,4862}
  t=${t//--seeds 7,8,9/--seeds 28,29,30}
  t=${t//pred_s7-8-9_dc/pred_s28-29-30_dc}
  t=${t//read_dc789/read_dc282930}
  t=${t// d536 / d615 }
  ! grep -qE "t2_d536|816|817|818|7-8-9|--seeds 7,8,9" <<< "$t" || { echo "a D536 constant survived in $1" >&2; exit 4; }
  printf '%s\n' "$t"        # its own 'exit $rc' ends only the ( ... ) subshell, so the stage's rc reaches the chain
}
S59=""; for d in 860 861 862; do for k in 5 6 7 8 9; do S59+="$((d + k * 1000)),"; done; done; S59=${S59%,}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$R'; echo \"### \$(date +%T) d615 chain start (git \$(git rev-parse --short HEAD))\" >> '$R/chain.log'"
  echo "( $(sub scripts/local/t2_d536_cnn_local.sh) ) || { echo \"### d615 cnn FAILED \$(date +%T)\" >> '$R/chain.log'; exit 5; }"
  echo "( $(sub scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR) ) || { echo \"### d615 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d615 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "( $(sub scripts/local/t2_d536_side_local.sh) ) || { echo \"### d615 side/DC FAILED \$(date +%T)\" >> '$R/chain.log'; exit 7; }"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for fam in sh ifn; do
    m=$([ $fam = sh ] && echo shallow || echo ifnet)
    D=$R/s10/side_$fam
    line="--protocols lc9 --models $m --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S59 --side dcmean"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d615 s10 $fam: $line\" >> '$R/s10_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d615 s10 $fam \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/s10_arm.log'; [ \$n -eq 150 ] || exit 8"
  done
  echo "echo \"### done d615 chain \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d615-chain" bash "$job"

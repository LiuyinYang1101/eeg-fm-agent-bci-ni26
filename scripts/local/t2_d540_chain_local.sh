#!/usr/bin/env bash
# T2-D540 (the W19 CONFIRM chain on FRESH draws 819-821; the manager's condition 1 of D538 / D539, 2026-11-16 ≈ 19:5x; zero grant, the local
# 5090 via gpu-submit; ONE job, stages in order). W18's form re-trained at the same fresh seeds (LESSONS A43) = D536's chain EXACTLY, with the
# draw constants moved 816-818 -> 819-821 and the output dir t2_d536 -> t2_d540:
#   (1) CNN: the three families' M7 lines (copies + tv8c) on lc9 + routing (t2_d536_cnn_local.sh's job text, constants replaced);
#   (2) P1: the teacher (scripts/local/t2_d540_teacher.py, CPU) then the in-stream students sh,ifn 60,100,150 confR (t2_d536_p1_local.sh's job text);
#   (3) side nets, NI26_SIDE_DUMP=1: ShallowNet / IFNet S5 (t2_d536_side_local.sh's line) AND FBCNet S5 (D538's line: fbcnet_bboth none
#       r1_pulxx_vp2_e180_vr --side dcmean), seeds d + k * 1000, k 0..4;
#   (4) the DC bag at FRESH LightGBM seeds 10,11,12 (t2_d515_read.py --features features_dc.npz --seeds 10,11,12, CPU) copied with its sha256.
# The FM side is REUSED (D532's W18 form: SD six 826-831 with heads, MI at 826), as in D536. Each stage refuses to start if the previous one is short.
#   bash scripts/local/t2_d540_chain_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
FILES="scripts/local/t2_d536_cnn_local.sh scripts/local/t2_d536_p1_local.sh scripts/local/t2_d536_side_local.sh scripts/local/t2_d512_teacher.py scripts/local/t2_d540_teacher.py scripts/local/t2_d515_read.py scripts/local/t2_d540_chain_local.sh"
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms $FILES)" ] || { echo "commit the D540 code first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d540
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
job=$(mktemp "$O/t2_d540_chain_job.XXXX.sh")
sub() {   # a D536 launcher's generated job text (its --dry output) with the draw / dir constants moved; each replacement must have happened
  local t; t=$(mem_dry "$@")
  t=${t//t2_d536_teacher.py/t2_d540_teacher.py}
  t=${t//t2_d536/t2_d540}
  t=${t//816,817,818/819,820,821}
  t=${t//816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818/819,1819,2819,3819,4819,820,1820,2820,3820,4820,821,1821,2821,3821,4821}
  t=${t//--seeds 7,8,9/--seeds 10,11,12}
  t=${t//pred_s7-8-9_dc/pred_s10-11-12_dc}
  t=${t//read_dc789/read_dc101112}
  t=${t// d536 / d540 }
  ! grep -qE "t2_d536|816|817|818|7-8-9|--seeds 7,8,9" <<< "$t" || { echo "a D536 constant survived in $1" >&2; exit 4; }
  printf '%s\n' "$t"        # its own 'exit $rc' ends only the ( ... ) subshell, so the stage's rc reaches the chain
}
mem_dry() { bash "$@" --dry; }
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$R'; echo \"### \$(date +%T) d540 chain start (git \$(git rev-parse --short HEAD))\" >> '$R/chain.log'"
  echo "( $(sub scripts/local/t2_d536_cnn_local.sh) ) || { echo \"### d540 cnn FAILED \$(date +%T)\" >> '$R/chain.log'; exit 5; }"
  echo "( $(sub scripts/local/t2_d536_p1_local.sh sh,ifn 60,100,150 confR) ) || { echo \"### d540 p1 FAILED \$(date +%T)\" >> '$R/chain.log'; exit 6; }"
  echo "n=\$(ls '$R'/p1_sh/runs '$R'/p1_ifn/runs 2>/dev/null | grep -c isS__); [ \$n -eq 60 ] || { echo \"### d540 p1 short: \$n isS files (need 60)\" >> '$R/chain.log'; exit 6; }"
  echo "( $(sub scripts/local/t2_d536_side_local.sh) ) || { echo \"### d540 side/DC FAILED \$(date +%T)\" >> '$R/chain.log'; exit 7; }"
  # FBCNet side nets (D538's line at the fresh draws)
  D=$R/side_fbc
  line="--protocols lc9 --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e180_vr --seeds 819,1819,2819,3819,4819,820,1820,2820,3820,4820,821,1821,2821,3821,4821 --side dcmean"
  PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
  EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
  SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
  echo "export NI26_SIDE_DUMP=1; mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d540 side_fbc: $line\" >> '$R/side_arm.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d540 side_fbc \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/side_arm.log'; [ \$n -eq 150 ] || exit 8"
  echo "echo \"### done d540 chain \$(date +%T)\" >> '$R/chain.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d540-chain" bash "$job"

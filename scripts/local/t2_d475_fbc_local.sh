#!/usr/bin/env bash
# T2-D475 arm B on the local 5090 (zero grant cost; its paired control, T2-D419's fbc_e180_s324, was made on this GPU): M7's FBCNet line
# (copies + tv8c) on common-average-referenced input, seeds 324-335, kit + seen1, then the shipped routing (t2p_fpcalib blend, r 1/3, tag 3,
# xtag __car) into outputs/t2_d475/route_fbc_car, the path scripts/local/t2_d475_read.py reads. ONE gpu-submit job; each python call in a
# MemoryMax=20G scope (t2_d419_local.sh's pattern). The arm line is tracks/t2_bci/sofia/arms/t2d475_fbccar.txt without its NSHARDS prefix.
#   bash scripts/local/t2_d475_fbc_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d475_fbccar.txt scripts/local/t2_d475_fbc_local.sh scripts/local/t2_d475_read.py)" ] || { echo "commit the D475 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
d=$O/sofia_s1/ni26-t2p-d475-fbccar-5090
R=$O/t2_d475/route_fbc_car
line=$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d475_fbccar.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //')
[[ "$line" == *--car* ]] || { echo "arm line lost --car"; exit 2; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d475_fbc_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "mkdir -p '$d/runs' '$R'; git rev-parse HEAD > '$d/git_head.txt'"
  echo "echo \"### \$(date +%T) d475 fbccar: $line\" >> '$d/arm.log'"
  echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
  echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "for p in kit seen1; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R' --model fbcnet_bboth --align none --recipe r1_pulxx_vp2_e180_vr_rc75tv8c --xtag __car --seeds $(seq -s, 324 335) >> '$d/arm.log' 2>&1 || rc=7; done"
  echo "echo \"### done fbccar \$(date +%T) rc=\$rc; routed \$(ls '$R' | wc -l) files\" >> '$d/arm.log'"
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
[ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d475" bash "$job"

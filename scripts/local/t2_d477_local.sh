#!/usr/bin/env bash
# T2-D477 (seat G10 item 2: spectral coverage of the two EA families) on the local 5090 (zero grant; the paired controls, T2-D429's
# sh_ctl / ifn_ctl, were made on this GPU). ONE gpu-submit job: arms t2d477_{ifnb3,ifnb4,sht61}_s348 one after another, each in
# outputs/sofia_s1/ni26-t2p-d477-<arm>-5090, then the shipped routing (t2p_fpcalib blend r 0.5, tag 2, as T2-D429's read) into
# outputs/t2_d477/route_<arm>. The generated job script is written into outputs/t2_d477/ (not /tmp).
# Read: mem-guard python scripts/local/t2_d477_read.py
#   bash scripts/local/t2_d477_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d477_*.txt scripts/local/t2_d477_local.sh scripts/local/t2_d477_read.py)" ] || { echo "commit the D477 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d477
declare -A MODEL=([ifnb3]=ifnet_b3 [ifnb4]=ifnet_b4 [sht61]=shallow_t61)
ARMS=${D477_ARMS:-"sht61 ifnb3 ifnb4"}
mkdir -p "$R"; job=$R/job.sh
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for arm in $ARMS; do
    d=$S/ni26-t2p-d477-$arm-5090
    line=$(grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d477_${arm}_s348.txt" | grep -v '^[[:space:]]*$')
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    echo "echo \"### \$(date +%T) d477 $arm: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "for p in kit seen1; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/route_$arm' --model ${MODEL[$arm]} --align ea --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds $(seq -s, 348 359) >> '$d/arm.log' 2>&1 || rc=7; done"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc; routed \$(ls '$R/route_$arm' | wc -l) files\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; exit 0; fi
for arm in $ARMS; do d=$S/ni26-t2p-d477-$arm-5090; [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }; done
gpu-submit -L "neural_interface_challenge:t2-d477" bash "$job"

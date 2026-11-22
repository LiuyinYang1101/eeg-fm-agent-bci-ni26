#!/usr/bin/env bash
# T2-D478 (seat G10 item 1: stream-balanced batches + the per-stream prior-blind loss, recipe tag sb16) on the local 5090 (zero grant; the paired
# controls T2-D419 fbc_e180 / T2-D429 sh_ctl, ifn_ctl were made on this GPU). ONE gpu-submit job: arms t2d478_{fbc_s324,sh_s348,ifn_s348} one
# after another, each in outputs/sofia_s1/ni26-t2p-d478-<arm>-5090, then the shipped routing (FBCNet r 1/3 tag 3; ShallowNet / IFNet r 0.5 tag 2)
# into outputs/t2_d478/route_<arm>. The generated job script is written into outputs/t2_d478/. Read: mem-guard python scripts/local/t2_d478_read.py
#   bash scripts/local/t2_d478_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d478_*.txt scripts/local/t2_d478_local.sh scripts/local/t2_d478_read.py)" ] || { echo "commit the D478 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; S=$O/sofia_s1; R=$O/t2_d478
# arm -> model : align : routed recipe : blend extra args : seeds
declare -A SPEC=([fbc_s324]="fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75_sb16tv8c::324"
                 [sh_s348]="shallow:ea:r1_pulx_vp2_e180_vr_sb16tv8c:--r 0.5 --tag 2:348"
                 [ifn_s348]="ifnet:ea:r1_pulx_vp2_e180_vr_sb16tv8c:--r 0.5 --tag 2:348")
ARMS=${D478_ARMS:-"fbc_s324 sh_s348 ifn_s348"}
mkdir -p "$R"; job=$R/job.sh
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for arm in $ARMS; do
    IFS=: read -r model al rrec bx s0 <<< "${SPEC[$arm]}"
    d=$S/ni26-t2p-d478-${arm%_s*}-5090
    line=$(grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d478_$arm.txt" | grep -v '^[[:space:]]*$')
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    echo "echo \"### \$(date +%T) d478 $arm: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "for p in kit seen1; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/route_${arm%_s*}' --model $model --align $al --recipe $rrec $bx --seeds $(seq -s, $s0 $((s0 + 11))) >> '$d/arm.log' 2>&1 || rc=7; done"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc; routed \$(ls '$R/route_${arm%_s*}' | wc -l) files\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; exit 0; fi
for arm in $ARMS; do d=$S/ni26-t2p-d478-${arm%_s*}-5090; [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }; done
gpu-submit -L "neural_interface_challenge:t2-d478" bash "$job"

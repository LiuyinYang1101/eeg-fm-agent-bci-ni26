#!/usr/bin/env bash
# T2-D179: a second paired official-kit warm-up draw of the kit-split twins: T2-W3's command (the (c) template) and T2-D167's (+ the T2-D161
# copies) at seeds 5-9 / FM 5, each built then repacked (robustness + worker-guard kit replay).
#   bash tracks/t2_bci/sofia/launch_d179.sh old|new
set -euo pipefail
cd "$(dirname "$0")/../../.."
which=${1:?old|new}
R=$S1_ROOT/runs/ni26
extra=(); [ "$which" = new ] && extra=(--recipe r1_pulxx --user-blend2 0.5 --recipe2-pu r1_pulx)
name=d179_kit_$which
out=$(S1_PURPOSE="T2-D179 build $which: kit-split twin at seeds 5-9 / FM 5 (second paired official-kit warm-up draw of the D159 + D160 template vs the (c) template)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name=ni26-t2p-pkg-$name \
  -- t2p_$name --model fbcnet_bboth --align none --epochs 60 --seeds 5,6,7,8,9 --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 60 --seeds2 5,6,7,8,9 --weight2 0.2 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds 5 \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds 5 --train kit \
     --fm-ckpt $R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth "${extra[@]}" 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D179 repack $which (robustness + worker-guard kit replay of build $job)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-$name \
  -- $R/ni26-t2p-pkg-$name-$job/t2p_$name.zip t2p_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

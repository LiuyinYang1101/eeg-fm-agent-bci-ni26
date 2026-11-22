#!/usr/bin/env bash
# T2-D211: the 15-seed CNN families under the 18.1 ms bar (the panel's T2 item 3, user-approved 2026-10-23), at w4321 (--weight2 0.3; FBCNet 0.4,
# ST-L 0.2, MOR 0.1). Otherwise as T2-D205: T2-D196's commands
# (`launch_d196.sh`: the T2-D162 all-user / T2-D167 kit-split lines) with --recipe <the winning vp2 recipe> and --epochs / --epochs2 <its pooled length>
# (both CNN families' pooled nets; every per-user copy unaugmented at r1_pulxx's / r1_pulx's lr and epochs, as in T2-D195..D204), walltime 4 h,
# then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d205.sh <name> <seeds> <fm seed> <all|kit> <recipe> <pooled epochs>
#   e.g. d140 0,...,4 0 all r1_pulxx_vp2_e180 180 | line 0,...,9 0 all ... | draw1 20,...,29 20 all ... | kit 0,...,4 0 kit ...
# ROW=d276 (T2-D276's nv15 draw 1) or d284 (T2-D284 = PLAN §6b item 10's build A: L_N 15-seed draws 5-6) relabels the purpose, job and ZIP
# names; unset = T2-D211's names exactly.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}; rec=${5:?recipe}; ep=${6:?pooled epochs}
R=$S1_ROOT/runs/ni26
r=${ROW:-d211}; [ "$r" = d211 ] || [ "$r" = d276 ] || [ "$r" = d284 ] || exit 2; RT=T2-${r^^}
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="$RT build $name: 15-seed CNN families at w4321 ($rec, $ep pooled epochs), seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=05:00:00 --job-name=ni26-t2p-pkg-$r-$name \
  -- t2p_${r}_$name --model fbcnet_bboth --align none --epochs "$ep" --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 "$ep" --seeds2 "$S" --weight2 0.3 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train $tr \
     --fm-ckpt $ck \
     --recipe "$rec" --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="$RT repack $name (robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-$r-$name \
  -- $R/ni26-t2p-pkg-$r-$name-$job/t2p_${r}_$name.zip t2p_${r}_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

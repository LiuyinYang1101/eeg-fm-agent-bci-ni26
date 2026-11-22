#!/usr/bin/env bash
# T2-D207: T2-D205's builds with T2-D206's adopted family weights w5320 (FBCNet 0.5 / ShallowNet 0.3 / ST-L 0.2 / MOR 0: no MOR member;
# --weight2 0.3, --fm-weight 0.2, no --fm2). Candidates for a swap the user decides. Otherwise as T2-D205: T2-D196's commands
# (`launch_d196.sh`: the T2-D162 all-user / T2-D167 kit-split lines) with --recipe <the winning vp2 recipe> and --epochs / --epochs2 <its pooled length>
# (both CNN families' pooled nets; every per-user copy unaugmented at r1_pulxx's / r1_pulx's lr and epochs, as in T2-D195..D204), walltime 4 h,
# then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d205.sh <name> <seeds> <fm seed> <all|kit> <recipe> <pooled epochs>
#   e.g. d140 0,...,4 0 all r1_pulxx_vp2_e180 180 | line 0,...,9 0 all ... | draw1 20,...,29 20 all ... | kit 0,...,4 0 kit ...
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}; rec=${5:?recipe}; ep=${6:?pooled epochs}
R=$S1_ROOT/runs/ni26
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="T2-D207 build $name: the vp2-180 line with T2-D206 weights w5320 (no MOR; $rec, $ep pooled epochs), seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=04:00:00 --job-name=ni26-t2p-pkg-d207-$name \
  -- t2p_d207_$name --model fbcnet_bboth --align none --epochs "$ep" --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 "$ep" --seeds2 "$S" --weight2 0.3 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --train $tr \
     --fm-ckpt $ck \
     --recipe "$rec" --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D207 repack $name (robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-d207-$name \
  -- $R/ni26-t2p-pkg-d207-$name-$job/t2p_d207_$name.zip t2p_d207_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

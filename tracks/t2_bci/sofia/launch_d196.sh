#!/usr/bin/env bash
# T2-D196: rebuild the approved T2 ZIPs with T2-D195's virtual participants (vp2, 10 mm electrode jitter): the T2-D162 (all-user) / T2-D167
# (kit-split) commands with --recipe r1_pulxx_vp2 (both families' pooled nets augmented; FBCNet copies at r1_pulxx's lr / epochs, ShallowNet copies
# at r1_pulx's via --recipe2-pu, both unaugmented, as in T2-D195); then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d196.sh <name> <seeds> <fm seed> <all|kit>
#   e.g. line 0,...,9 0 all | draw1 20,...,29 20 all | d140 0,...,4 0 all | kit 0,...,4 0 kit
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}
R=$S1_ROOT/runs/ni26
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="T2-D196 build $name: the D159 + D160 template + T2-D195 virtual participants (vp2 10 mm), seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=02:00:00 --job-name=ni26-t2p-pkg-d196-$name \
  -- t2p_d196_$name --model fbcnet_bboth --align none --epochs 60 --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 60 --seeds2 "$S" --weight2 0.2 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train $tr \
     --fm-ckpt $ck \
     --recipe r1_pulxx_vp2 --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D196 repack $name (robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-d196-$name \
  -- $R/ni26-t2p-pkg-d196-$name-$job/t2p_d196_$name.zip t2p_d196_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

#!/usr/bin/env bash
# T2-D217: upload 5's re-tuned vp2 line (T2-D215 virtual references adopted) at w4321 (--weight2 0.3; FBCNet 0.4, ST-L 0.2, MOR 0.1),
# then a reconfig repack whose --fm-from donor (arg 7) gives it the probe pair's FM files (the parts check). Otherwise as T2-D205 / D211: T2-D196's commands
# (`launch_d196.sh`: the T2-D162 all-user / T2-D167 kit-split lines) with --recipe <the winning vp2 recipe> and --epochs / --epochs2 <its pooled length>
# (both CNN families' pooled nets; every per-user copy unaugmented at r1_pulxx's / r1_pulx's lr and epochs, as in T2-D195..D204), walltime 4 h,
# then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d217.sh <name> <seeds> <fm seed> <all|kit> <recipe> <pooled epochs> <donor zip on sofia>
#   e.g. d140 0,...,4 0 all r1_pulxx_vp2_e180 180 | line 0,...,9 0 all ... | draw1 20,...,29 20 all ... | kit 0,...,4 0 kit ...
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}; rec=${5:?recipe}; ep=${6:?pooled epochs}; donor=${7:?donor zip on sofia}
R=$S1_ROOT/runs/ni26
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="T2-D217 build $name: the vp2 line with virtual references at w4321 (upload 5 candidate) ($rec, $ep pooled epochs), seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=05:00:00 --job-name=ni26-t2p-pkg-d217-$name \
  -- t2p_d217_$name --model fbcnet_bboth --align none --epochs "$ep" --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 "$ep" --seeds2 "$S" --weight2 0.3 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train $tr \
     --fm-ckpt $ck \
     --recipe "$rec" --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D217 repack $name (FM files from the donor + parts check + robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-d217-$name \
  -- $R/ni26-t2p-pkg-d217-$name-$job/t2p_d217_$name.zip t2p_d217_${name}_r --allow-differ 0 --fm-from "$donor" 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

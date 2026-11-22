#!/usr/bin/env bash
# T2-D232: upload 5's re-tuned vp2 line rebuilt with the re-coloured FBCNet (T2-D230 / D230r adopted; the dose from T2-D231r) at w4321
# (--weight2 0.3; FBCNet 0.4, ST-L 0.2, MOR 0.1). = T2-D217's launch_d217.sh with --recipe <FBCNet's pooled recipe> and --recipe2 <ShallowNet's>
# (T2-D230's re-colouring is FBCNet only: ShallowNet keeps the vr line), then the reconfig repack whose --fm-from donor (arg 8) gives it the probe
# pair's FM files (the parts check). Walltime 5 h, then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d232.sh <name> <seeds> <fm seed> <all|kit> <recipe> <recipe2> <pooled epochs> <donor zip on sofia>
#   e.g. u5rc 0,...,9 0 all r1_pulxx_vp2_e180_vr_rc5 r1_pulxx_vp2_e180_vr 180 $R/ni26-t2p-repack-d210-u2-1533013/t2p_d210_u2.zip
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}; rec=${5:?recipe}; rec2=${6:?recipe2}; ep=${7:?pooled epochs}; donor=${8:?donor zip on sofia}
R=$S1_ROOT/runs/ni26
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="T2-D232 build $name: upload 5's vp2 + vr line with the re-coloured FBCNet at w4321 (FBCNet $rec, ShallowNet $rec2, $ep pooled epochs), seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=05:00:00 --job-name=ni26-t2p-pkg-d232-$name \
  -- t2p_d232_$name --model fbcnet_bboth --align none --epochs "$ep" --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 "$ep" --seeds2 "$S" --weight2 0.3 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train $tr \
     --fm-ckpt $ck \
     --recipe "$rec" --recipe2 "$rec2" --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D232 repack $name (FM files from the donor + parts check + robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-d232-$name \
  -- $R/ni26-t2p-pkg-d232-$name-$job/t2p_d232_$name.zip t2p_d232_${name}_r --allow-differ 0 --fm-from "$donor" 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

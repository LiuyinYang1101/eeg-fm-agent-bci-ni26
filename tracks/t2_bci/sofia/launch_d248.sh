#!/usr/bin/env bash
# T2-D248: upload 5's candidate rebuilt with the adopted ST-L fine-tune (T2-D247 / D247r: re-colouring 0.75 + layer-wise lr decay 0.75) at w4321
# (--weight2 0.3; FBCNet 0.4, ST-L 0.2, MOR 0.1). = T2-D232's launch_d232.sh (the re-coloured FBCNet line) with --fm-llrd / --fm-rc on the ST-L
# member, then the PLAIN repack (no --fm-from donor: the donor route needs every FM member config to match, and the new ST-L's has llrd / rc),
# so every part of this ZIP is this build's own. Walltime 5 h, then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d248.sh <name> <seeds> <fm seed> <all|kit> <recipe> <recipe2> <pooled epochs> <fm llrd> <fm rc>
#   e.g. u5st 0,1,2,3,4,5,6,7,8,9 0 all r1_pulxx_vp2_e180_vr_rc75 r1_pulxx_vp2_e180_vr 180 0.75 0.75
# ROW=d249 (T2-D249's draws, launch_d249_draws.sh), d279 (T2-D279's kit-split warm-up twins), d276 (T2-D276's 15-seed line + draw 1) or d284 (T2-D284 = PLAN §6b
# item 10's build A: L_V 15-seed draws 2-7) relabels the purpose, job and ZIP names;
# unset = T2-D248's names exactly.
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}; tr=${4:?all|kit}; rec=${5:?recipe}; rec2=${6:?recipe2}; ep=${7:?pooled epochs}; ll=${8:?fm llrd}; rc=${9:?fm rc}
R=$S1_ROOT/runs/ni26
r=${ROW:-d248}; [ "$r" = d248 ] || [ "$r" = d249 ] || [ "$r" = d279 ] || [ "$r" = d276 ] || [ "$r" = d284 ] || exit 2; RT=T2-${r^^}
case $tr in all) ck=$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth;; kit) ck=$R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth;; *) exit 2;; esac
out=$(S1_PURPOSE="$RT build $name: the re-tuned vp2 line (vp2 + vr + re-coloured FBCNet; 5f538614's) at w4321 with the adopted ST-L (fm llrd $ll, fm rc $rc; T2-D247r), FBCNet $rec, ShallowNet $rec2, $ep pooled epochs, seeds $S / FM $b, --train $tr, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=05:00:00 --job-name=ni26-t2p-pkg-$r-$name \
  -- t2p_${r}_$name --model fbcnet_bboth --align none --epochs "$ep" --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 "$ep" --seeds2 "$S" --weight2 0.3 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" --fm-llrd "$ll" --fm-rc "$rc" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train $tr \
     --fm-ckpt $ck \
     --recipe "$rec" --recipe2 "$rec2" --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="$RT repack $name (robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-$r-$name \
  -- $R/ni26-t2p-pkg-$r-$name-$job/t2p_${r}_$name.zip t2p_${r}_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

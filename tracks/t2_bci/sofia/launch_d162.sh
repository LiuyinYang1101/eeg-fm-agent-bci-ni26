#!/usr/bin/env bash
# T2-D162: rebuild the last-five line on the T2-D159 + D160 template (T2-D161 flags): `f403527d`'s / T2-D150's command (FBCNet B3 x10 +
# D124c user blend, ShallowNet-EA x10, ST-L x1 (mixup, all-user CPT), role-LoRA MOR x1, prior, --train all) with the FBCNet copies at
# r1_pulxx (r 1/3) and the ShallowNet copies at r1_pulx (r 1/2); then t2p_repack (robustness + worker-guard kit replay) after it.
#   bash tracks/t2_bci/sofia/launch_d162.sh <name> <seeds> <fm seed>     # e.g. line 0,1,...,9 0 | draw1 20,...,29 20
set -euo pipefail
cd "$(dirname "$0")/../../.."
name=${1:?name}; S=${2:?seeds}; b=${3:?fm seed}
R=$S1_ROOT/runs/ni26
out=$(S1_PURPOSE="T2-D162 build $name: f403527d's line on the D159 + D160 template (FBCNet r1_pulxx copies r 1/3 + ShallowNet r1_pulx copies r 1/2), seeds $S / FM $b, D140 bars" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name=ni26-t2p-pkg-d162-$name \
  -- t2p_d162_$name --model fbcnet_bboth --align none --epochs 60 --seeds "$S" --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 60 --seeds2 "$S" --weight2 0.2 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds "$b" \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds "$b" --train all \
     --fm-ckpt $R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth \
     --recipe r1_pulxx --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D162 repack $name (robustness + worker-guard kit replay of build $job; D140 bars)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-d162-$name \
  -- $R/ni26-t2p-pkg-d162-$name-$job/t2p_d162_$name.zip t2p_d162_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

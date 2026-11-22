#!/usr/bin/env bash
# T2-D167: the kit-split twin of the D159 + D160 template = T2-W3's build (`27f9794e`, t2p_d140c_kit_g3: --train kit, seeds 0-4 / FM 0, the
# kit-split ST-L CPT d89b) + the T2-D161 flags (FBCNet r1_pulxx copies r 1/3, ShallowNet r1_pulx copies r 1/2); then t2p_repack after it.
#   bash tracks/t2_bci/sofia/launch_d167.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=$S1_ROOT/runs/ni26
name=d167_kit
out=$(S1_PURPOSE="T2-D167 build: kit-split twin of the D159 + D160 template (T2-W3's command + the D161 flags), official-kit warm-up replay vs T2-W3's 0.5606" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:30:00 --job-name=ni26-t2p-pkg-$name \
  -- t2p_$name --model fbcnet_bboth --align none --epochs 60 --seeds 0,1,2,3,4 --tta prior:1:3 \
     --model2 shallow --align2 ea --n02 0 --epochs2 60 --seeds2 0,1,2,3,4 --weight2 0.2 \
     --fm st_large --fm-align sz --fm-calib gain --fm-epochs 30 --fm-weight 0.2 --fm-mixup 0.4 --fm-seeds 0 \
     --user-blend 0.3333 --fm2 mor_p1v2 --fm2-weight 0.1 --fm2-seeds 0 --train kit \
     --fm-ckpt $R/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth \
     --recipe r1_pulxx --user-blend2 0.5 --recipe2-pu r1_pulx 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
S1_PURPOSE="T2-D167 repack (robustness + worker-guard kit replay of build $job)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --dependency=afterok:$job --time=01:00:00 --job-name=ni26-t2p-repack-$name \
  -- $R/ni26-t2p-pkg-$name-$job/t2p_$name.zip t2p_${name}_r --allow-differ 0 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"

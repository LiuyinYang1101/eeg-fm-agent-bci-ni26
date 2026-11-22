#!/usr/bin/env bash
# T2-D541 (MI-SD: the MI-expert FM member WITH the window-mean side head; PREPARED 2026-11-16, launched only after the manager's OK; inside GO and
# the s1 caps, every job <= 3 GPU-h). Parts:
#   nom      : MI-SD s826 fold halves (the nomination read on 816-818 vs D532's MI s826)                       2 jobs x 2:30
#   confirm  : MI s832 + MI-SD s832 fold halves (fresh FM seed for the 819-821 stack, LESSONS A43)               4 jobs x 2:30
#   donor d1|d2 : the all-user MI-SD donor for the build (D529's midonor line + --fm-side dcmean), seed 610 / 620   1 job x 1:15
#                 build PREPARATION only (the FM fine-tune cut-off is Nov 18 12:00); used only if the W19 stack keeps D541 and the owner's line
#   mem-guard bash tracks/t2_bci/sofia/launch_d541.sh dry|GO nom|confirm|donor [d1|d2]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?nom|confirm|donor}; draw=${3:-d1}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == nom || "$part" == confirm || "$part" == donor ]] && [[ "$draw" == d1 || "$draw" == d2 ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d541.sh tracks/t2_bci/sofia/arms/t2d541_*.txt)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D541 MI-SD (the MI-expert FM member with the window-mean side head), K = 9; feeds the T2 W19 stack (owner direction 2026-11-16)"
if [ "$part" == donor ]; then
  s=610; [ "$draw" == d2 ] && s=620
  KN="{\"$s\":{\"tilt\":0.15}}"
  env "${dry[@]}" S1_PURPOSE="$P; build PREPARATION donor $draw seed $s (all users, MI root, --fm-side dcmean), used only if the stack keeps D541 and the owner's line" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:15:00 --job-name="ni26-t2p-d541-misddonor-$draw-s$s" -- \
    "t2p_d541_misddonor_${draw}_s$s" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
    --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds "$s" \
    --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-preft-d529-mi-root/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor --fm-side dcmean \
    2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
  exit 0
fi
if [ "$part" == nom ]; then arms=(tracks/t2_bci/sofia/arms/t2d541_misd_s826_p{0..4}.txt); else arms=(tracks/t2_bci/sofia/arms/t2d541_{mi,misd}_s832_p{0..4}.txt); fi
for f in "${arms[@]}"; do
  a=$(basename "$f" .txt); a=${a#t2d541_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=01:15:00 \
    --job-name="ni26-t2p-d541-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

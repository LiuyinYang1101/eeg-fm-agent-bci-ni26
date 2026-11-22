#!/usr/bin/env bash
# T2-D524 / D531 build PREPARATION (not an adoption, never a candidate by itself; the manager 05:0x): FM donors WITH the window-mean side head
# for upload B, one per draw = D512's tilt donors exactly (launch_d512_tiltdonor.sh: same seeds 610-615 / 620-625, d111-all CPT, sz, gain,
# 100 epochs, mixup 0.4, llrd 0.75, rc 0.75, tilt 0.15; 3 plain + 3 CAR) + t2p_final --fm-side dcmean (the head in fm_<i>_side.pt).
# Used only if D524 passes its full read AND the W18 fresh read, and the owner types the W18 line; started now because the FM fine-tune
# cut-off is Nov 18 12:00.
#   mem-guard bash tracks/t2_bci/sofia/launch_d531_sddonor.sh dry|GO d1|d2
#   mem-guard bash tracks/t2_bci/sofia/launch_d531_sddonor.sh smoke     (1 plain + 1 CAR seed, 1 epoch, 45-min wall; checks the files)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO|smoke}; draw=${2:-d1}
[[ "$mode" == dry || "$mode" == GO || "$mode" == smoke ]] && [[ "$draw" == d1 || "$draw" == d2 ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d531_sddonor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
b=610; [ "$draw" == d2 ] && b=620
S="$b,$((b+1)),$((b+2)),$((b+3)),$((b+4)),$((b+5))"
KN="{\"$b\":{\"tilt\":0.15},\"$((b+1))\":{\"tilt\":0.15},\"$((b+2))\":{\"tilt\":0.15},\"$((b+3))\":{\"car\":1,\"tilt\":0.15},\"$((b+4))\":{\"car\":1,\"tilt\":0.15},\"$((b+5))\":{\"car\":1,\"tilt\":0.15}}"
name="t2p_d531_sddonor_$draw"; ep=100; wall=06:00:00; part=(); extra=()
if [ "$mode" == smoke ]; then
  S="$b,$((b+3))"; KN="{\"$b\":{\"tilt\":0.15},\"$((b+3))\":{\"car\":1,\"tilt\":0.15}}"; name="t2p_d531_sddonor_smoke"; ep=1; wall=01:00:00
  wall=00:45:00   # zen4_h200 (backfills a short wall; zen5_vis is one GPU per user across projects)
fi
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" != dry ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D524/D531 build PREPARATION (not an adoption): side-head FM donor $draw ($mode; 3 plain + 3 CAR ST-L, seeds $S, all users, --fm-side dcmean) for upload B, used only if D524 passes its full and fresh reads and the owner's W18 line" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm "${part[@]}" --time=$wall --job-name="ni26-t2p-d531-sddonor-$draw-$mode" -- \
  "$name" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs $ep --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds "$S" \
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor --fm-side dcmean "${extra[@]}" \
  2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

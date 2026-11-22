#!/usr/bin/env bash
# T2-D529 build PREPARATION (not an adoption, never a candidate by itself; the manager 09:0x): the all-user MI-expert FM members for upload B, one
# build per draw, HEAD-LESS (every MI member the reads score is trained without --fm-side): D512's plain tilt-donor line exactly (sz, gain, 100
# epochs, mixup 0.4, llrd 0.75, rc 0.75, tilt 0.15) from D529's stage-1 checkpoint (ni26-t2p-preft-d529-mi-root, sha 74b9e61f8b0e) instead of the
# d111-all CPT, at the first two PLAIN donor seeds (d1 610, 611; d2 620, 621): 610 / 620 replace the first plain member; 611 / 621 serve only
# if D529's one-vs-two rule keeps a second MI member. Used only if D532 keeps D529 and the owner types the W18 line.
#   mem-guard bash tracks/t2_bci/sofia/launch_d529_midonor.sh dry|GO d1|d2      (two single-member jobs per draw)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; draw=${2:-d1}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$draw" == d1 || "$draw" == d2 ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/scripts/t2p_bench.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d529_midonor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
b=610; [ "$draw" == d2 ] && b=620
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
for s in $b $((b+1)); do                # one member per job (about 42 min each at 25 s / epoch): a 1:15 wall backfills sooner than one 3:00 job
  KN="{\"$s\":{\"tilt\":0.15}}"
  env "${dry[@]}" S1_PURPOSE="T2-D529 build PREPARATION (not an adoption): head-less MI-expert FM donor $draw seed $s (plain ST-L, all users, from D529's stage-1 checkpoint) for upload B, used only if D532 keeps D529 and the owner's W18 line" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:15:00 --job-name="ni26-t2p-d529-midonor-$draw-s$s" -- \
    "t2p_d529_midonor_${draw}_s$s" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
    --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds "$s" \
    --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-preft-d529-mi-root/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor \
    2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200
done

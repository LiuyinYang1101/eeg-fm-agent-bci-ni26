#!/usr/bin/env bash
# T2-D512 build PREPARATION (not an adoption, never a candidate by itself): FM donors for part P2 (the ST-L spectral tilt 0.15) of the final
# stack, one per draw: 3 plain + 3 CAR ST-L members, the shipped FM recipe (T2-D475e's donor line: d111-all CPT, sz, gain, 100 epochs, mixup 0.4,
# llrd 0.75, rc 0.75) + tilt 0.15 per seed, trained on ALL users (--fm-donor). Used only if D512's leave-one-out keeps P2 (then t2p_repack
# --fm-from replaces W14-lite's six FM members); started before that read so the final build is not delayed by ≈ 3 h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d512_tiltdonor.sh dry|GO d1|d2
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; draw=${2:?d1|d2}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$draw" == d1 || "$draw" == d2 ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py src/ni26 tracks/t2_bci/sofia/t2p_final.slurm tracks/t2_bci/sofia/launch_d512_tiltdonor.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
b=610; [ "$draw" == d2 ] && b=620
S="$b,$((b+1)),$((b+2)),$((b+3)),$((b+4)),$((b+5))"
KN="{\"$b\":{\"tilt\":0.15},\"$((b+1))\":{\"tilt\":0.15},\"$((b+2))\":{\"tilt\":0.15},\"$((b+3))\":{\"car\":1,\"tilt\":0.15},\"$((b+4))\":{\"car\":1,\"tilt\":0.15},\"$((b+5))\":{\"car\":1,\"tilt\":0.15}}"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D512 build PREPARATION (not an adoption): tilt-0.15 FM donor $draw (3 plain + 3 CAR ST-L, seeds $S, all users) for the final stack's part P2, used only if D512 keeps P2" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=06:00:00 --job-name="ni26-t2p-d512-tiltdonor-$draw" -- \
  "t2p_d512_tiltdonor_$draw" --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds "$S" \
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt "$R/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth" --fm-seed-knobs "$KN" --fm-donor \
  2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

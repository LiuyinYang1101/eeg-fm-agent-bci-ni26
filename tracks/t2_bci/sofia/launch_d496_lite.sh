#!/usr/bin/env bash
# T2-D496 (W14-lite): repack the GATED M7-W14 with the approved W14 template unchanged, keeping FM members 0-2 (plain) + 5-7 (CAR) = 3 + 3
# (t2p_repack --fm-keep, checked against kref = the base CNNs + only the kept members) and the 3 + 3 fm start vector by T2-D459's rule
# (tracks/t2_bci/submission/configs/t2_d496_fm_start_vector.json). t2p_repack.slurm on zen5_vis: every W14 gate + the kit replay; ms / window
# on the H200 is a separate timing job. Config + member list only (R3); the user's W14 line named 5 + 3, so the upload choice is theirs.
#   mem-guard bash tracks/t2_bci/sofia/launch_d496_lite.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; zip=ni26-t2p-repack-t2p_ms_M7_w14-1615940/t2p_ms_M7_w14.zip; name=t2p_ms_M7_w14lite33
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d496_lite.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D496 W14-lite build (M7-W14 minus plain FM members 3-4 = 3 + 3, ~42 ms; config + member list only, R3): the user's choice for the Nov 15 upload" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=zen5_vis --time=04:00:00 --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" --template tracks/t2_bci/submission/t2p_submission_w14.py --fm-keep 0,1,2,5,6,7 \
  --set "tta.per_family.fm=@tracks/t2_bci/submission/configs/t2_d496_fm_start_vector.json#fm" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

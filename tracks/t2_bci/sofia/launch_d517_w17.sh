#!/usr/bin/env bash
# T2-D517 (template W17 = W16 + the window-mean side nets; PREPARED 2026-11-15): build a W17 package from a W15 final-stack zip (the step-B output)
# in two repacks (t2p_repack's --set-ref needs the base's own template, byte for byte):
#   step E = the W17 template alone on the W15 zip (no cfg dcfam / side = W15 bitwise; t2p_repack's clean equality proves it);
#   step F = cfg dcfam (configs/t2_d516_dcfam.json, share 0.075) + cfg side (configs/t2_d517_side.json, r 1/3 'thirds', cnn2 + cnn3) + their files
#            (runs/ni26/t2_d516_dcfam/dcfam.npz sha256 a5c9d8cba8ed595d…; runs/ni26/t2_d517_side/side.pt sha256 17e3a192161d49ee…,
#            tracks/t2_bci/scripts/t2p_side_final.py) via --add-file, --set-ref.
# REFUSES TO RUN unless W17_APPROVED carries the user's own W16 + W17 line typed in T2's shell (date + the words).
#   W17_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_d517_w17.sh dry|GO E|F <base zip under runs/ni26> <new name>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; step=${2:?E|F}; zip=${3:?base zip under runs/ni26}; name=${4:?new name}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$step" == E || "$step" == F ]] || exit 2
[[ "${W17_APPROVED:-}" == *"template change approved"* && "${W17_APPROVED:-}" == *"W17"* ]] || { echo "W17 is PREPARED, NOT APPROVED: set W17_APPROVED to the user's own W16 + W17 line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d517_w17.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "$step" == E ]; then
  ex=(--template tracks/t2_bci/submission/t2p_submission_w17.py)
else
  ex=(--template tracks/t2_bci/submission/t2p_submission_w17.py --set-ref
      --set "dcfam=@tracks/t2_bci/submission/configs/t2_d516_dcfam.json#dcfam" --set "side=@tracks/t2_bci/submission/configs/t2_d517_side.json#side"
      --add-file "$R/t2_d516_dcfam/dcfam.npz" --add-file "$R/t2_d517_side/side.pt")
fi
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D517 W17 step $step ($zip -> $name; approved: ${W17_APPROVED:0:60}): feeds T2's final packages" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=${T2_PART:-zen5_vis} --time=${T2_TIME:-04:00:00} --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" "${ex[@]}" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

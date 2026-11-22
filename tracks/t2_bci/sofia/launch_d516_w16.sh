#!/usr/bin/env bash
# T2-D516 (template W16 = W15 + the window-mean family; PREPARED 2026-11-15): build a W16 package from a GATED W15 final-stack zip in two repacks,
# as W15 was built (t2p_repack's --set-ref needs the base's own template, byte for byte):
#   step C = the W16 template alone on the W15 zip (no cfg dcfam = W15 bitwise; t2p_repack's clean equality proves it);
#   step D = cfg dcfam (tracks/t2_bci/submission/configs/t2_d516_dcfam.json: share 0.075) + the family's tree arrays
#            (runs/ni26/t2_d516_dcfam/dcfam.npz, sha256 a5c9d8cba8ed595d…, scripts/local/t2_d516_final.py) via --add-file, --set-ref.
# REFUSES TO RUN unless W16_APPROVED carries the user's own W16 line typed in T2's shell (date + the words).
#   W16_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_d516_w16.sh dry|GO C|D <base zip under runs/ni26> <new name>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; step=${2:?C|D}; zip=${3:?base zip under runs/ni26}; name=${4:?new name}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$step" == C || "$step" == D ]] || exit 2
[[ "${W16_APPROVED:-}" == *"template change approved"* ]] || { echo "W16 is PREPARED, NOT APPROVED: set W16_APPROVED to the user's own line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d516_w16.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "$step" == C ]; then
  ex=(--template tracks/t2_bci/submission/t2p_submission_w16.py)
else
  ex=(--template tracks/t2_bci/submission/t2p_submission_w16.py --set-ref --set "dcfam=@tracks/t2_bci/submission/configs/t2_d516_dcfam.json#dcfam"
      --add-file "$R/t2_d516_dcfam/dcfam.npz")
fi
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
env "${dry[@]}" S1_PURPOSE="T2-D516 W16 step $step ($zip -> $name; approved: ${W16_APPROVED:0:60}): feeds T2's final packages" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=${T2_PART:-zen5_vis} --time=${T2_TIME:-04:00:00} --job-name="ni26-t2p-repack-$name" -- \
  "$R/$zip" "$name" "${ex[@]}" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

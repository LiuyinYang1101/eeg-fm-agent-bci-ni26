#!/usr/bin/env bash
# T2-D557 build PREPARATION, ONLY after D557-2 AND the W20 interaction read both pass (the owner's W20 line: "D557's augmented donors only if both
# D557 reads pass"): the merged donor (t2p_merge_chain.slurm, zen5_vis) -> then the dry donor check (t2p_donor_check.slurm vs W18 b2 with the
# W19 template) afterok. The W20+D557 build itself is a separate step (W19-style chain on the H200, then the W20 repacks).
#   mem-guard bash tracks/t2_bci/sofia/launch_d557_merge.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_merge_donors.py tracks/t2_bci/sofia/t2p_merge_chain.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/launch_d557_merge.sh tests/test_t2p_merge_donors.py)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
BASE=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip        # the sofia paths (the job checks they exist; this script runs locally)
A=$R/ni26-t2p-d557-donor-a-1629736/t2p_d557_donor_a.zip; B=$R/ni26-t2p-d557-donor-b-1629737/t2p_d557_donor_b.zip
C=$R/ni26-t2p-d557-donor-c-1629738/t2p_d557_donor_c.zip
K2=$R/ni26-t2p-d557-cont-622-1629739/t2p_d557_cont_622.zip; K4=$R/ni26-t2p-d557-cont-624-1629740/t2p_d557_cont_624.zip
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
JM=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-d557-merge,NAME=t2p_w20d557_donor" S1_PURPOSE="T2-D557 build PREPARATION: W19 b2's donor with the five augmented headed members (after D557-2 + the W20 interaction read passed)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_merge_chain.slurm --partition=zen5_vis --account=vsc --time=01:00:00 --job-name=ni26-t2p-d557-merge -- "$BASE" "$A" "$B" "$C" "$K2" "$K4")
[ "$mode" == dry ] && { echo "dry: the merge validated; the donor check needs its id"; exit 0; }
JD=$(sub S1_PURPOSE="T2-D557: the dry donor check of the merged augmented donor (vs W18 b2, the W19 template) before any build slot" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm --partition=zen5_vis --account=vsc --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-d557-donorcheck -- \
  $R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip $R/ni26-t2p-d557-merge/t2p_w20d557_donor.zip t2p_submission_w19.py)
echo "D557 merge $JM -> donor check $JD; donor runs/ni26/ni26-t2p-d557-merge/t2p_w20d557_donor.zip"

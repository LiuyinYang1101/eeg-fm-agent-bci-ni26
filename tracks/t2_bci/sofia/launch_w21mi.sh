#!/usr/bin/env bash
# T2-D567 -> W21 = W20+D557 with slot 0 (the head-less MI member) replaced by D567's augmented all-user MI donor (W20's template
# t2p_submission_w20.py UNCHANGED, donors only; NOT the closed D563 template t2p_submission_w21.py).
#   merge: ONLY after all three registered D567 reads pass (DECISIONS T2-D567; D567_READS_PASSED=1): the merged donor = W20+D557's donor
#          (runs/ni26/ni26-t2p-d557-merge/t2p_w20d557_donor.zip) with slot 0 <- the D567 donor's fm_0 (head-less swap, no --keep-side;
#          t2p_merge.slurm, zen5_vis) -> the dry donor check afterok (vs W18 b2, the W19 template, as D557's).
#   build: ALSO needs the owner's NEW typed line (W21_APPROVED, naming W21 and D567) and the merge job's id (MERGE_JOB): W20+D557's chain
#          exactly (launch_w20d557.sh): (1) the W19-style build (H200) -> (2) the W20 repacks R5a / R5b + K1 -> (3) the post-build check ->
#          (4) K2; (5) the H200 timing afterok (2).
#   D567_READS_PASSED=1 mem-guard bash tracks/t2_bci/sofia/launch_w21mi.sh merge dry|GO
#   W21_APPROVED="..." D567_READS_PASSED=1 MERGE_JOB=<id> mem-guard bash tracks/t2_bci/sofia/launch_w21mi.sh build dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
part=${1:?merge|build}; mode=${2:?dry|GO}
[[ "$part" == merge || "$part" == build ]] && [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ "${D567_READS_PASSED:-}" = 1 ] || { echo "all three D567 reads must have passed (record them in DECISIONS first)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_merge.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w21mi.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
BASE=$R/ni26-t2p-d557-merge/t2p_w20d557_donor.zip                      # W20+D557's merged donor (sha 84136714...)
MI=$R/ni26-t2p-d567-midonor-s620-1631892/t2p_d567_midonor_s620.zip     # D567's all-user donor (sha d0be8487...)
W18=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip; SIDE=$R/t2_d549_side10/side.pt
V=(--partition=zen5_vis --account=vsc)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
if [ "$part" == merge ]; then
  JM=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-d567-merge,NAME=t2p_w21mi_donor" S1_PURPOSE="T2-D567 build PREPARATION: W20+D557's donor with slot 0 <- D567's augmented MI donor (after all three D567 reads passed)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_merge.slurm "${V[@]}" --time=00:45:00 --job-name=ni26-t2p-d567-merge -- \
    --base "$BASE" --swap "$MI" --slots 0 --swap-index 0)
  [ "$mode" == dry ] && { echo "dry: the merge validated; the donor check needs its id"; exit 0; }
  JD=$(sub S1_PURPOSE="T2-D567: the dry donor check of the W21 merged donor (vs W18 b2, the W19 template) before any build slot" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm "${V[@]}" --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-d567-donorcheck -- \
    "$W18" "$R/ni26-t2p-d567-merge/t2p_w21mi_donor.zip" t2p_submission_w19.py)
  echo "D567 merge $JM -> donor check $JD; donor runs/ni26/ni26-t2p-d567-merge/t2p_w21mi_donor.zip"
  exit 0
fi
[[ "${W21_APPROVED:-}" == *"W21"* && "${W21_APPROVED:-}" == *"D567"* && "${W21_APPROVED:-}" == *"approved"* ]] || { echo "set W21_APPROVED to the owner's typed W21 line (it names D567)"; exit 3; }
[ -n "${MERGE_JOB:-}" ] || { echo "set MERGE_JOB to the COMPLETED merge job's id (and its donor check must have passed)"; exit 3; }
DONOR=$R/ni26-t2p-d567-merge/t2p_w21mi_donor.zip
N1=t2p_fin2_w19w21mi; N2=t2p_fin2_w21mi
P="T2-D567 W21 = W20+D557 + D567's MI donor in slot 0 (W20's template unchanged, donors only; the owner's W21 line) for gating"
J1=$(sub S1_PURPOSE="$P: (1) the W19-style build with the W21 merged donor (R1-R4 + K1, H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm \
  --time=05:00:00 --job-name="ni26-t2p-w19-$N1" -- "$W18" "$DONOR" "$SIDE" "$N1")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z1="$R/ni26-t2p-w19-$N1-$J1/$N1.zip"; Z1_REL="ni26-t2p-w19-$N1-$J1/$N1.zip"
J2=$(sub S1_PURPOSE="$P: (2) the W20 repacks R5a / R5b + K1 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm "${V[@]}" --time=05:00:00 \
  --dependency="afterok:$J1" --job-name="ni26-t2p-w20-$N2" -- "$Z1" "$N2")
Z2_REL="ni26-t2p-w20-$N2-$J2/$N2.zip"
J3=$(sub S1_PURPOSE="$P: (3) the post-build check vs (1) (gates I / H / HF / HP / B / S; zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" \
  --time=08:00:00 --dependency="afterok:$J2" --job-name="ni26-t2p-pkgcheck-$N2" -- w20 "$Z1_REL" "$Z2_REL" "$N2")
J4=$(sub S1_PURPOSE="$P: (4) condition K2 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=03:00:00 \
  --dependency="afterok:$J3" --job-name="ni26-t2p-w20kit-$N2" -- "$Z2_REL" "ni26-t2p-pkgcheck-$N2-$J3/$N2.pkgcheck.pred.npz" "$N2")
J5=$(sub S1_PURPOSE="$P: (5) the H200 timing (3 reps), the D517 line <= 3,150 s for the -lite rule" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-time-$N2" -- "$Z2_REL" "$N2" 3)
echo "W21 (D567): (1) $J1 -> (2) $J2 -> (3) check $J3 -> (4) K2 $J4; (5) timing $J5; the zip runs/ni26/$Z2_REL"

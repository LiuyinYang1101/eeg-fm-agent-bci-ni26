#!/usr/bin/env bash
# T2-D652 (W30) ONE-HOP PREBUILD (the manager's amendment 2026-11-21 17:31 + GO 17:4x). W30 = W25-ALL6 with its
# FM members swapped for the W30 donors' (the MI slot = D642's MI donor, W25-ALL6's own), in ONE repack: the tested chain's output, because
# W25-ALL6 vs W21 differ in fm_members ONLY and every hop after R1 is a template / config step (the FM-dependent refits run in-stream).
#   (M) t2p_merge_chain6.slurm (= launch_w30.sh's merge, on zen5_vis): W19 b2's donor with ALL SIX members <- MI / A / B / C / cont-622 / cont-624
#       (afterok on the five W30 donor jobs: it starts the moment they end)
#   (C) the dry donor check vs W25-ALL6 with W21lp's template (zen5_vis)
#   (1) t2p_w30_onehop_build.slurm: ONE repack of W25-ALL6 --fm-from the merged donor + K1 (zen5_vis)
#   (2) t2p_pkgcheck.slurm w30: the FM-swap package check vs W25-ALL6 (config diff = fm_members; non-FM files byte-equal; finite + rerun-equal;
#       the paired ms / window; NEW predictions) (zen5_vis)
#   (3) K2 (zen5_vis); (4) the H200 timing after (1). The structural zipcheck (t2p_w25_zipcheck.py --base W25-ALL6 --cand W30 --donor <merged>
#       --keep-slots 0) runs locally on the fetched files. PREBUILD ONLY: promotion needs the owner's typed W30 line (a separate step).
#   mem-guard bash tracks/t2_bci/sofia/launch_w30_onehop.sh dry|PREBUILD <A> <B> <C> <K2> <K4>   (the W30 donor job ids)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILD}; [[ "$mode" == dry || "$mode" == PREBUILD ]] || { echo "dry|PREBUILD only (promotion = the owner's line, a separate step)"; exit 2; }
IA=${2:?A id}; IB=${3:?B id}; IC=${4:?C id}; IK2=${5:?K2 id}; IK4=${6:?K4 id}
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_merge_chain6.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/t2p_w30_onehop_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w30_onehop.sh)" ] || { echo "commit first"; exit 2; }
W25_SHA=2b03eb037d3ac4962ec5d54e146364c1a5fc6bacb57fcf976b55614b9714ff1a
W25_REL=ni26-t2p-w21lp-t2p_fin2_w25all6_PREBUILD-1644829/t2p_fin2_w25all6_PREBUILD.zip   # the promoted W25-ALL6 bytes (verified on sofia 2026-11-21)
R=$S1_ROOT/runs/ni26
W25=$R/$W25_REL
W19B2DONOR=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip
DMI=$R/ni26-t2p-d642-donor-mi-1644013/t2p_d642_donor_mi.zip   # the MI slot = D642's MI donor (W25-ALL6's slot 0)
DA=$R/ni26-t2p-w30-donor-a-$IA/t2p_w30_donor_a.zip; DB=$R/ni26-t2p-w30-donor-b-$IB/t2p_w30_donor_b.zip
DC=$R/ni26-t2p-w30-donor-c-$IC/t2p_w30_donor_c.zip
DK2=$R/ni26-t2p-w30-cont-622-$IK2/t2p_w30_cont_622.zip; DK4=$R/ni26-t2p-w30-cont-624-$IK4/t2p_w30_cont_624.zip
N=t2p_w30oh_PREBUILD
P="T2-D652 W30 ONE-HOP PREBUILD (the manager's amendment 2026-11-21 17:31; the owner's extra upload): NOT approved, never uploaded without the owner's typed W30 line; cut-off 21:45"
V=(--partition=zen5_vis --account=vsc); VH=(--partition=zen4_h200)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == PREBUILD ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee -a "${LAUNCH_LOG:-/dev/null}" | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
MRUN=ni26-t2p-w30oh-merge; MD=$R/$MRUN/t2p_w30oh_donor.zip
JM=$(sub S1_EXPORT="RUN_NAME=$MRUN,NAME=t2p_w30oh_donor" S1_PURPOSE="$P: (M) W19 b2's donor with ALL SIX members <- MI + the W30 donors" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_merge_chain6.slurm "${V[@]}" --time=01:00:00 --dependency="afterok:$IA:$IB:$IC:$IK2:$IK4" --job-name=$MRUN -- \
  "$W19B2DONOR" "$DMI" "$DA" "$DB" "$DC" "$DK2" "$DK4")
[ "$mode" == dry ] && { echo "dry: (M) validated; the rest need its id"; exit 0; }
[ -n "$JM" ] || { echo "(M) not submitted"; exit 2; }
JC=$(sub S1_PURPOSE="$P: (C) the dry donor check vs W25-ALL6 (W21lp template)" s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm "${V[@]}" \
  --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-w30oh-donorcheck -- "$W25" "$MD" t2p_submission_w21lp.py)
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$W25_SHA,W30_PREBUILD=1" S1_PURPOSE="$P: (1) ONE repack of W25-ALL6 --fm-from the merged W30 donor + K1" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_w30_onehop_build.slurm "${V[@]}" --time=02:00:00 --dependency=afterok:$JC --job-name="ni26-t2p-w30oh-$N" -- "$W25" "$MD" "$N")
Z_REL="ni26-t2p-w30oh-$N-$J1/$N.zip"
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm "${VH[@]}" \
  --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$W25_SHA" S1_PURPOSE="$P: (2) the FM-swap package check vs W25-ALL6 + the paired timing" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=02:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- w30 "$W25_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=02:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-w30ohkit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
echo "W30 ONE-HOP PREBUILD (NOT APPROVED): (M) $JM -> (C) $JC -> (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL; the merged donor $MD"

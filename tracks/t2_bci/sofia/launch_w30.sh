#!/usr/bin/env bash
# T2-D652 (W30) BUILD CHAIN = launch_w25.sh's chain (W21's six-swap merge on W19 b2's donor -> (C) -> W19-style build -> W20 repacks -> W21 cnn_lp
# repacks -> the post-build check -> K2 -> the H200 timing) with the W30 donors in place of the D642 donors. PREBUILD (no owner line): the names
# carry _PREBUILD, S1_PURPOSE says 'not approved'; it may start as soon as the donors exist (speculative, W25's pattern). GO needs the owner's typed
# line (W30_APPROVED containing "T2 W30 approved" and "D652") AND decide_d652.json "included": true. Rule (the owner's 16:31:32 inclusion rule): the member sign gate + two set reads: mean dS > 0, mean OLD sealed > 0, read 2 OLD sealed > 0 and dS > 0 (D652; fixed 2026-11-21).
#   mem-guard bash tracks/t2_bci/sofia/launch_w30.sh dry|PREBUILD|GO <A> <B> <C> <K2> <K4>   (the W30 donor job ids)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILD|GO}; [[ "$mode" =~ ^(dry|PREBUILD|GO)$ ]] || exit 2
IA=${2:?A id}; IB=${3:?B id}; IC=${4:?C id}; IK2=${5:?K2 id}; IK4=${6:?K4 id}
if [ "$mode" == GO ]; then
  [[ "${W30_APPROVED:-}" == *"T2 W30 approved"* && "${W30_APPROVED:-}" == *"D652"* ]] || { echo "set W30_APPROVED to the owner's typed W30 line (\"T2 W30 approved ... D652 ...\")"; exit 3; }
  grep -q '"included": true' "$HOME/workspace/outputs/neural_interface_challenge/t2_d642/setread/bw21/decide_d652.json" 2>/dev/null \
    || { echo "GO refuses: decide_d652.json does not say included"; exit 3; }   # 2026-11-21: the decide writes "included" (was a stale "adopt" grep)
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_merge_chain6.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_w21lp_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w30.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
W18=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip
SIDE10=$R/t2_d549_side10/side.pt
W19B2DONOR=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip
DMI=$R/ni26-t2p-d642-donor-mi-1644013/t2p_d642_donor_mi.zip   # the MI slot = D642's MI donor (W25-ALL6's slot 0; W30 has no V for MI)
DA=$R/ni26-t2p-w30-donor-a-$IA/t2p_w30_donor_a.zip; DB=$R/ni26-t2p-w30-donor-b-$IB/t2p_w30_donor_b.zip
DC=$R/ni26-t2p-w30-donor-c-$IC/t2p_w30_donor_c.zip
DK2=$R/ni26-t2p-w30-cont-622-$IK2/t2p_w30_cont_622.zip; DK4=$R/ni26-t2p-w30-cont-624-$IK4/t2p_w30_cont_624.zip
TAG=w30; [ "$mode" != GO ] && TAG=w30_PREBUILD
N1=t2p_fin2_w19$TAG; N2=t2p_fin2_w20$TAG; N3=t2p_fin2_$TAG
P="T2 W30 ($TAG; D652) for gating; $([ "$mode" == GO ] && echo "approved: ${W30_APPROVED:0:60}" || echo "PREBUILD, NOT approved (the owner's line + decide_d652.json pending)")"
H=(--partition=zen4_h200); V=(--partition=zen4_h200 --gpus-per-node=1 --cpus-per-task=24)   # the manager's option (ii): (M) + (C) on the H200 (zen5_vis queued to 04:33)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" != dry ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
MRUN=ni26-t2p-w30-merge-$TAG; MD=$R/$MRUN/t2p_${TAG}_donor.zip
JM=$(sub S1_EXPORT="RUN_NAME=$MRUN,NAME=t2p_${TAG}_donor" S1_PURPOSE="$P: (M) W19 b2's donor with ALL SIX members <- the W30 donors" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_merge_chain6.slurm "${V[@]}" --time=01:00:00 --job-name=$MRUN -- "$W19B2DONOR" "$DMI" "$DA" "$DB" "$DC" "$DK2" "$DK4")
[ "$mode" == dry ] && { echo "dry: (M) validated; the rest need its id"; exit 0; }
[ -n "$JM" ] || { echo "(M) not submitted"; exit 2; }
JC=$(sub S1_PURPOSE="$P: (C) the dry donor check (vs W18 b2, the W19 template)" s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm "${V[@]}" \
  --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-w30-donorcheck-$TAG -- "$W18" "$MD" t2p_submission_w19.py)
J1=$(sub S1_PURPOSE="$P: (1) the W19-style build (R1-R4 + K1)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm "${H[@]}" --time=05:00:00 \
  --dependency=afterok:$JC --job-name="ni26-t2p-w19-$N1" -- "$W18" "$MD" "$SIDE10" "$N1")
Z1="$R/ni26-t2p-w19-$N1-$J1/$N1.zip"
J2=$(sub S1_PURPOSE="$P: (2) the W20 repacks R5a / R5b + K1" s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm "${H[@]}" --time=05:00:00 \
  --dependency=afterok:$J1 --job-name="ni26-t2p-w20-$N2" -- "$Z1" "$N2")
Z2="$R/ni26-t2p-w20-$N2-$J2/$N2.zip"; Z2_REL="ni26-t2p-w20-$N2-$J2/$N2.zip"
J3=$(sub S1_EXPORT="BASE_SHA_EXPECT=sibling" S1_PURPOSE="$P: (3) the W21 repacks R6a (inert) / R6b (cnn_lp) + K1" s1 submit ni26 tracks/t2_bci/sofia/t2p_w21lp_build.slurm \
  "${H[@]}" --time=05:00:00 --dependency=afterok:$J2 --job-name="ni26-t2p-w21lp-$N3" -- "$Z2" "$N3")
Z3_REL="ni26-t2p-w21lp-$N3-$J3/$N3.zip"
J4=$(sub S1_PURPOSE="$P: (4) the post-build check w21lp (2) vs (3) (gates IL / H / HF / HP / FM / B / S)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm \
  "${H[@]}" --time=08:00:00 --dependency=afterok:$J3 --job-name="ni26-t2p-pkgcheck-$N3" -- w21lp "$Z2_REL" "$Z3_REL" "$N3")
J5=$(sub S1_PURPOSE="$P: (5) condition K2" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${H[@]}" --time=03:00:00 \
  --dependency=afterok:$J4 --job-name="ni26-t2p-w30kit-$N3" -- "$Z3_REL" "ni26-t2p-pkgcheck-$N3-$J4/$N3.pkgcheck.pred.npz" "$N3")
J6=$(sub S1_PURPOSE="$P: (6) the H200 timing (3 reps): the D517 line <= 3,300 s (owner)" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency=afterok:$J3 --job-name="ni26-t2p-time-$TAG" -- "$Z3_REL" "$TAG" 3)
echo "W30 $TAG: (M) $JM -> (C) $JC -> (1) $J1 -> (2) $J2 -> (3) $J3 -> (4) check $J4 -> (5) K2 $J5; (6) timing $J6; the final zip runs/ni26/$Z3_REL"

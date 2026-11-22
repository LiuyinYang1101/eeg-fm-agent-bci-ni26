#!/usr/bin/env bash
# T2-D642 x9 -> W25-ALL6 build chain (the manager 2026-11-20 ≈ 01:2x: a SPECULATIVE PREBUILD, launched after Track3's review of this diff, NOT
# waiting for D642's --combine; discarded if the combine fails = W22's pattern). W25-ALL6 = W21 (W20+D557 + D569 cnn_lp) with ALL SIX FM members
# replaced by the user-balanced D642 donors (t2p_final --fm-donor --user-balance, the ship CPTs: MI 620 = the MI root; 621 / 623 / 625 and the
# continuations 622 / 624 = d111-all), through EXACTLY W22's FM-part chain (launch_w22.sh) with a six-swap merge:
#   (M) t2p_merge_chain6.slurm on W19 b2's donor (slot 0 <- MI, 1 <- A, 3 <- B, 5 <- C, 2 <- cont-622, 4 <- cont-624; --keep-side; the merge
#       refuses a CPT-sha change) -> (C) the dry donor check (vs W18 b2, the W19 template) -> (1) the W19-style build (R1-R4 + K1) -> (2) the W20
#       repacks R5a / R5b + K1 -> (3) the W21 repacks R6a (inert) / R6b (cnn_lp) + K1 -> (4) the post-build check w21lp (2) vs (3) -> (5) K2 ->
#       (6) the H200 timing (3 reps; the D517 line <= 3,300 s). Every GPU step on the H200; (M) / (C) on zen5_vis.
#   PREBUILD (no owner line yet): every name carries _PREBUILD, S1_PURPOSE says 'not approved'; PROMOTION only via the owner's typed W25 line
#   + D642's --combine --set all6 PASS + Track3's fresh review + the W21-vs-W25 structural check (a HARD gate: t2p_w25_zipcheck.py on the
#   fetched zip). GO needs W25_APPROVED = the owner's typed line ("T2 W25 approved ... D642 ...") AND set_combined_all6.json passes true.
#   Base W21 only (W23 = W21 + the session-1 route file, a config step on the final zip, only if the owner types item 166: not wired here).
#   mem-guard bash tracks/t2_bci/sofia/launch_w25.sh dry|PREBUILD|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|PREBUILD|GO}; [[ "$mode" =~ ^(dry|PREBUILD|GO)$ ]] || exit 2
if [ "$mode" == GO ]; then
  [[ "${W25_APPROVED:-}" == *"T2 W25 approved"* && "${W25_APPROVED:-}" == *"D642"* ]] || { echo "set W25_APPROVED to the owner's typed W25 line (\"T2 W25 approved ... D642 ...\")"; exit 3; }
  grep -q '"passes": true' "$HOME/workspace/outputs/neural_interface_challenge/t2_d642/setread/bw21/set_combined_all6.json" 2>/dev/null \
    || { echo "GO refuses: D642's --combine --set all6 has not PASSED (setread/bw21/set_combined_all6.json)"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_merge_chain6.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_w21lp_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w25.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
W18=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip
SIDE10=$R/t2_d549_side10/side.pt
W19B2DONOR=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip
DMI=$R/ni26-t2p-d642-donor-mi-1644013/t2p_d642_donor_mi.zip
DA=$R/ni26-t2p-d642-donor-a-1644014/t2p_d642_donor_a.zip; DB=$R/ni26-t2p-d642-donor-b-1644015/t2p_d642_donor_b.zip
DC=$R/ni26-t2p-d642-donor-c-1644016/t2p_d642_donor_c.zip
DK2=$R/ni26-t2p-d642-cont-622-1644017/t2p_d642_cont_622.zip; DK4=$R/ni26-t2p-d642-cont-624-1644018/t2p_d642_cont_624.zip
TAG=w25all6; [ "$mode" != GO ] && TAG=w25all6_PREBUILD
N1=t2p_fin2_w19$TAG; N2=t2p_fin2_w20$TAG; N3=t2p_fin2_$TAG
P="T2 W25-ALL6 ($TAG; D642 x9) for gating; $([ "$mode" == GO ] && echo "approved: ${W25_APPROVED:0:60}" || echo "PREBUILD, NOT approved (the owner's line + the combine pending)")"
H=(--partition=zen4_h200); V=(--partition=zen4_h200 --gpus-per-node=1 --cpus-per-task=24)   # the manager's option (ii): (M) + (C) on the H200 (zen5_vis queued to 04:33)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" != dry ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
MRUN=ni26-t2p-w25-merge-$TAG; MD=$R/$MRUN/t2p_${TAG}_donor.zip
JM=$(sub S1_EXPORT="RUN_NAME=$MRUN,NAME=t2p_${TAG}_donor" S1_PURPOSE="$P: (M) W19 b2's donor with ALL SIX members <- the D642 balanced donors" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_merge_chain6.slurm "${V[@]}" --time=01:00:00 --job-name=$MRUN -- "$W19B2DONOR" "$DMI" "$DA" "$DB" "$DC" "$DK2" "$DK4")
[ "$mode" == dry ] && { echo "dry: (M) validated; the rest need its id"; exit 0; }
[ -n "$JM" ] || { echo "(M) not submitted"; exit 2; }
JC=$(sub S1_PURPOSE="$P: (C) the dry donor check (vs W18 b2, the W19 template)" s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm "${V[@]}" \
  --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-w25-donorcheck-$TAG -- "$W18" "$MD" t2p_submission_w19.py)
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
  --dependency=afterok:$J4 --job-name="ni26-t2p-w25kit-$N3" -- "$Z3_REL" "ni26-t2p-pkgcheck-$N3-$J4/$N3.pkgcheck.pred.npz" "$N3")
J6=$(sub S1_PURPOSE="$P: (6) the H200 timing (3 reps): the D517 line <= 3,300 s (owner)" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency=afterok:$J3 --job-name="ni26-t2p-time-$TAG" -- "$Z3_REL" "$TAG" 3)
echo "W25-ALL6 $TAG: (M) $JM -> (C) $JC -> (1) $J1 -> (2) $J2 -> (3) $J3 -> (4) check $J4 -> (5) K2 $J5; (6) timing $J6; the final zip runs/ni26/$Z3_REL"

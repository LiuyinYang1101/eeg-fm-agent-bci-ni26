#!/usr/bin/env bash
# T2 W22 build chain (PREPARED on the manager's night plan 2026-11-18, item 5; nothing runs before the owner's typed W22 line in T2's shell).
# W22 = W21 (W20+D557 + D569 cnn_lp) with the parts that passed C + B + D swapped in:
#   ONE FM part at most (the speculative all-user donors exist per part, never for a combination of two FM flags):
#     D583 / D582 / D588 = the five headed donors (A = 621, B = 623, C = 625, K2 = cont-622, K4 = cont-624: t2p_merge_chain.slurm on W19 b2's
#                          donor, exactly as the D557 merge built W20+D557's donor);
#     D596              = the same five-donor merge (launch_d596_donor.sh), but OUTSIDE T2-D592's line: GO needs an owner line NAMING D596;
#     D586              = donor 621 only (t2p_merge.slurm: slot 1 of W20+D557's donor <- the D586 donor, --keep-side);
#   D580 = the side.pt replacement (t2p_w22_side.slurm: --side-replace with the smx4 side.pt, seeds 51700-51709, uploaded to sofia first).
# Chain with an FM part: (M) merge -> (C) the dry donor check (t2p_donor_check.slurm vs W18 b2, the W19 template) -> (1) the W19-style build
# (t2p_w19_build.slurm: W18 b2 + the merged donor + W19's S10 side.pt) -> (2) the W20 repacks (t2p_w20_build.slurm) -> (3) the W21 repacks
# (t2p_w21lp_build.slurm, BASE_SHA_EXPECT=sibling: R6a inert + R6b cnn_lp + K1) -> (4) the post-build check w21lp (2) vs (3) -> (5) K2 on (3);
# [+ D580: (S) t2p_w22_side.slurm on (3)] ; (6) the H200 timing of the final zip (3 reps; the D517 line <= 3,300 s). Every GPU step on the H200.
# Without an FM part (D580 alone): (S) on W21's zip (sha 94281e72…) -> (6) timing.
# OPEN (to finish before a D580 W22 is gated): K2 and the side-golden check for the (S) zip (t2p_w22_side.slurm runs the repack's own parts
# gate + K1 only).
#   PARTS=D583[,D580] D583_READS_PASSED=1 [D580_READS_PASSED=1] DONOR_A=.. DONOR_B=.. DONOR_C=.. DONOR_K2=.. DONOR_K4=.. [SIDE_PT=.. SIDE_SHA=..] \
#   W22_APPROVED="<the owner's line>" mem-guard bash tracks/t2_bci/sofia/launch_w22.sh dry|GO <tag>
#   (D586: DONOR_621=..; dry runs validate the first step without the owner's line; GO refuses without it)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; TAG=${2:?tag (e.g. d583 or d583d580)}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$TAG" =~ ^[a-z0-9]+$ ]] || exit 2
IFS=, read -ra PS <<< "${PARTS:?PARTS=D583[,D580] ...}"
FMP=""; SIDEP=0; CFGP=0; TPLP=0
for p in "${PS[@]}"; do
  case $p in
    D582|D583|D586|D588|D596) [ -z "$FMP" ] || { echo "two FM parts ($FMP, $p): no donor carries both flags; refused"; exit 2; }; FMP=$p ;;
    D580) SIDEP=1 ;;
    D591) CFGP=1 ;;    # T2-D592: the FM share 0.45 (a config step on the W21-form zip)
    D589) TPLP=1 ;;    # T2-D592: the P1-copy mixup (a template step)
    *) echo "unknown part $p (D580 / D582 / D583 / D586 / D588 / D589 / D591 / D596)"; exit 2 ;;
  esac
  v="${p}_READS_PASSED"; [ "${!v:-}" = 1 ] || { echo "$p: set ${p}_READS_PASSED=1 only after its C + B + D reads passed (recorded in DECISIONS first)"; exit 3; }
done
if [ "$mode" == GO ] && { [ $CFGP = 1 ] || [ $TPLP = 1 ]; }; then   # T2-D592: their build steps + the final-zip K2 / parts gate are not wired yet
  echo "D589 / D591 in W22: their build step and the FINAL-zip K2 + parts gate are not wired yet: GO refused until they are"; exit 3
fi
if [ "$mode" == GO ] && [ $SIDEP = 1 ]; then   # Track3 MEDIUM on 6364a657: the uploaded zip would be the side-replaced one, gated by K1 + timing only
  echo "D580 in W22: K2, the side golden and a parts gate on the FINAL (side-replaced) zip are not wired yet: GO refused until they are"; exit 3
fi
OL=$HOME/workspace/outputs/neural_interface_challenge
declare -A VLOG=([D580]=$OL/t2_d580_v2.log [D582]=$OL/t2_d582_v2.log [D583]=$OL/t2_d583_v2.log [D586]=$OL/t2_d586_v2.log [D588]=$OL/t2_d588_v2.log [D596]=$OL/t2_d596_v2.log \
                 [D589]=$OL/t2_d589_d591.log [D591]=$OL/t2_d589_d591.log)
passes() {   # part: its LAST verdict line says DISPLACES (Track3 LOW 2: a re-run's newer verdict wins); D589 / D591 share one log (T2-D592)
  case $1 in
    D589|D591) grep -E "^### $1 (pooled|does not)" "${VLOG[$1]}" 2>/dev/null | tail -1 | grep -q "DISPLACES W21" ;;
    *) grep "^### pooled C + B + D" "${VLOG[$1]}" 2>/dev/null | tail -1 | grep -q ": $1 DISPLACES W21" ;;
  esac; }
cbmean() {   # part: the pooled C + B mean ten of its LAST 'POOL CB' line
  grep "^### POOL CB " "${VLOG[$1]}" 2>/dev/null | tail -1 | grep -oE "mean [+-][0-9.]+" | head -1 | awk '{print $2}'; }
for p in "${PS[@]}"; do   # T2-D590: every part must carry its own pooled C + B + D displacement verdict in its read log (mechanical, not typed)
  if ! passes "$p"; then
    [ "$mode" == dry ] && { echo "WARNING (dry run): $p has no 'DISPLACES W21' verdict in ${VLOG[$p]} (GO refuses)"; continue; }   # Track3 LOW 3
    echo "$p: no 'DISPLACES W21' verdict in ${VLOG[$p]}"; exit 3
  fi
done
if [ -n "$FMP" ]; then   # Track3 LOW 1 / T2-D590 (c): ONE FM recipe ships = the argmax of pooled C + B among the FM parts that displace W21
  best=""; bestv=""
  CANDS="D582 D583 D586 D588"   # T2-D592's line: the argmax among ITS four recipes; D596 competes only when the owner's line names D596
  if [[ "${W22_APPROVED:-}" == *"D596"* ]]; then CANDS="$CANDS D596"
  elif passes D596; then echo "### D596 displaces W21 but the owner's line does not name it: outside the argmax (T2-D592's four recipes only)"; fi
  for q in $CANDS; do
    passes "$q" || continue
    v=$(cbmean "$q"); [ -n "$v" ] || { echo "$q displaces W21 but has no POOL CB line"; exit 3; }
    echo "### FM passer $q: pooled C + B mean $v"
    if [ -z "$best" ] || awk -v a="$v" -v b="$bestv" 'BEGIN { exit !(a > b) }'; then best=$q; bestv=$v; fi
  done
  if [ -n "$best" ] && [ "$best" != "$FMP" ]; then
    [ "$mode" == dry ] && echo "WARNING (dry run): $FMP is not the best FM passer ($best, pooled C + B $bestv; GO refuses)" || { echo "$FMP is not the best FM passer ($best, pooled C + B $bestv): T2-D590 (c)"; exit 3; }
  fi
fi
if [ "$mode" == GO ]; then
  [[ "${W22_APPROVED:-}" == *"T2 W22 approved"* ]] || { echo "set W22_APPROVED to the owner's typed W22 line (\"T2 W22 approved ...\")"; exit 3; }
  if [[ "$W22_APPROVED" != *"every part that clears T2-D590"* ]]; then   # the conditional line covers the passers; otherwise each part is named
    for p in "${PS[@]}"; do [[ "$W22_APPROVED" == *"$p"* ]] || { echo "the owner's W22 line does not name $p"; exit 3; }; done
  fi
  if [ "$FMP" = D596 ] && [[ "$W22_APPROVED" != *"D596"* ]]; then   # T2-D592's conditional line lists D582 / D583 / D586 / D588 only
    echo "D596 is outside T2-D592's line: the owner's W22 line must name D596"; exit 3
  fi
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_merge_chain.slurm tracks/t2_bci/sofia/t2p_merge.slurm tracks/t2_bci/sofia/t2p_donor_check.slurm tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_w21lp_build.slurm tracks/t2_bci/sofia/t2p_w22_side.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w22.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
W18=$R/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip
SIDE10=$R/t2_d549_side10/side.pt
W19B2DONOR=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip
W20D557DONOR=$R/ni26-t2p-d557-merge/t2p_w20d557_donor.zip
W21_REL=ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip; W21_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc
N1=t2p_fin2_w19$TAG; N2=t2p_fin2_w20$TAG; N3=t2p_fin2_w22$TAG; NS=t2p_fin2_w22${TAG}s
P="T2 W22 ($TAG: ${PARTS}) for gating (approved: ${W22_APPROVED:0:60})"
H=(--partition=zen4_h200); V=(--partition=zen5_vis --account=vsc)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
FINAL_REL=""; JF=""
if [ -n "$FMP" ]; then
  MRUN=ni26-t2p-w22-merge-$TAG; MD=$R/$MRUN/t2p_w22${TAG}_donor.zip
  if [ "$FMP" = D586 ]; then
    : "${DONOR_621:?DONOR_621 = the D586 donor zip on sofia}"
    JM=$(sub S1_EXPORT="RUN_NAME=$MRUN,NAME=t2p_w22${TAG}_donor" S1_PURPOSE="$P: (M) W20+D557's donor, slot 1 <- the D586 donor 621" s1 submit ni26 \
      tracks/t2_bci/sofia/t2p_merge.slurm "${V[@]}" --time=01:00:00 --job-name=$MRUN -- --base "$W20D557DONOR" --swap "$DONOR_621" --slots 1 --swap-index 0 --keep-side)
  else
    for v in DONOR_A DONOR_B DONOR_C DONOR_K2 DONOR_K4; do [ -n "${!v:-}" ] || { echo "$FMP: set $v (the speculative donor zips on sofia)"; exit 2; }; done
    JM=$(sub S1_EXPORT="RUN_NAME=$MRUN,NAME=t2p_w22${TAG}_donor" S1_PURPOSE="$P: (M) W19 b2's donor with the five $FMP headed members" s1 submit ni26 \
      tracks/t2_bci/sofia/t2p_merge_chain.slurm "${V[@]}" --time=01:00:00 --job-name=$MRUN -- "$W19B2DONOR" "$DONOR_A" "$DONOR_B" "$DONOR_C" "$DONOR_K2" "$DONOR_K4")
  fi
  [ "$mode" == dry ] && { echo "dry: (M) validated; the rest need its id"; exit 0; }
  [ -n "$JM" ] || { echo "(M) not submitted"; exit 2; }
  JC=$(sub S1_PURPOSE="$P: (C) the dry donor check (vs W18 b2, the W19 template)" s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm "${V[@]}" \
    --time=00:30:00 --dependency=afterok:$JM --job-name=ni26-t2p-w22-donorcheck-$TAG -- "$W18" "$MD" t2p_submission_w19.py)
  J1=$(sub S1_PURPOSE="$P: (1) the W19-style build (R1-R4 + K1)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm "${H[@]}" --time=05:00:00 \
    --dependency=afterok:$JC --job-name="ni26-t2p-w19-$N1" -- "$W18" "$MD" "$SIDE10" "$N1")
  Z1="$R/ni26-t2p-w19-$N1-$J1/$N1.zip"; Z1_REL="ni26-t2p-w19-$N1-$J1/$N1.zip"
  J2=$(sub S1_PURPOSE="$P: (2) the W20 repacks R5a / R5b + K1" s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm "${H[@]}" --time=05:00:00 \
    --dependency=afterok:$J1 --job-name="ni26-t2p-w20-$N2" -- "$Z1" "$N2")
  Z2="$R/ni26-t2p-w20-$N2-$J2/$N2.zip"; Z2_REL="ni26-t2p-w20-$N2-$J2/$N2.zip"
  J3=$(sub S1_EXPORT="BASE_SHA_EXPECT=sibling" S1_PURPOSE="$P: (3) the W21 repacks R6a (inert) / R6b (cnn_lp) + K1" s1 submit ni26 tracks/t2_bci/sofia/t2p_w21lp_build.slurm \
    "${H[@]}" --time=05:00:00 --dependency=afterok:$J2 --job-name="ni26-t2p-w21lp-$N3" -- "$Z2" "$N3")
  Z3_REL="ni26-t2p-w21lp-$N3-$J3/$N3.zip"
  J4=$(sub S1_PURPOSE="$P: (4) the post-build check w21lp (2) vs (3) (gates IL / H / HF / HP / FM / B / S)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm \
    "${H[@]}" --time=08:00:00 --dependency=afterok:$J3 --job-name="ni26-t2p-pkgcheck-$N3" -- w21lp "$Z2_REL" "$Z3_REL" "$N3")
  J5=$(sub S1_PURPOSE="$P: (5) condition K2" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${H[@]}" --time=03:00:00 \
    --dependency=afterok:$J4 --job-name="ni26-t2p-w22kit-$N3" -- "$Z3_REL" "ni26-t2p-pkgcheck-$N3-$J4/$N3.pkgcheck.pred.npz" "$N3")
  FINAL_REL=$Z3_REL; JF=$J3
  echo "W22 FM part $FMP: (M) $JM -> (C) $JC -> (1) $J1 -> (2) $J2 -> (3) $J3 -> (4) check $J4 -> (5) K2 $J5"
fi
if [ $SIDEP = 1 ]; then
  : "${SIDE_PT:?SIDE_PT = the D580 side.pt on sofia (put it there first)}"; : "${SIDE_SHA:?SIDE_SHA = its sha256}"
  if [ -n "$FINAL_REL" ]; then BEXP=sibling; BZ="$R/$FINAL_REL"; dep=(--dependency=afterok:$JF); else BEXP=$W21_SHA; BZ="$R/$W21_REL"; dep=(); fi
  JS=$(sub S1_EXPORT="BASE_SHA_EXPECT=$BEXP,SIDE_SHA_EXPECT=$SIDE_SHA" S1_PURPOSE="$P: (S) the D580 side.pt replacement + K1" s1 submit ni26 \
    tracks/t2_bci/sofia/t2p_w22_side.slurm "${H[@]}" --time=03:00:00 "${dep[@]}" --job-name="ni26-t2p-w22side-$NS" -- "$BZ" "$SIDE_PT" "$NS")
  [ "$mode" == dry ] && { echo "dry: (S) validated"; exit 0; }
  FINAL_REL="ni26-t2p-w22side-$NS-$JS/$NS.zip"; JF=$JS
  echo "W22 D580: (S) $JS (K2 + the side-golden check of this zip: OPEN, see the header)"
fi
J6=$(sub S1_PURPOSE="$P: (6) the H200 timing (3 reps): the D517 line <= 3,300 s (owner)" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency=afterok:$JF --job-name="ni26-t2p-time-w22$TAG" -- "$FINAL_REL" "w22$TAG" 3)
echo "W22 $TAG: the final zip runs/ni26/$FINAL_REL; timing $J6"

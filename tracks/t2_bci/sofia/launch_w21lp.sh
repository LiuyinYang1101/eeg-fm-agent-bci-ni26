#!/usr/bin/env bash
# T2-D569 -> W21 (the owner's typed W21 line in T2's shell, 2026-11-18: "T2 W21 approved: W21 = W20+D557 plus every part that passes all its
# registered reads: D569 ... and / or D568 ...; built and gated speculatively, uploaded only if its parts pass ...; every W20+D557 gate + an H200
# timing <= 3,300 s at the D517 line + a fresh review; W20+D557 the fallback"). THIS launcher = the D569 part only: W21 = W20+D557 (sha 096a428d…,
# the reviewed upload candidate) + cfg cnn_lp {alpha 0.5, k 10} on t2p_submission_w21lp.py (donors unchanged). REFUSES without W21_APPROVED (the
# owner's line, naming D569) and D569_READS_PASSED=1 (A / B / C recorded in DECISIONS first). Chain (Track3 F3 / F4 adopted):
#   (1) t2p_w21lp_build.slurm (zen5_vis): BASE sha == 096a428d… checked; R6a inert template (clean_max_dp == 0 asserted), R6b cnn_lp, K1 with
#       the D569 cnn_lp gate -> (2) the post-build check t2p_w21lp_pkgcheck.py (gates IL / H / HF / HP / FM / B / S; OLD sha checked, NEW vs its
#       .sha256) -> (3) K2 (the kit on all public streams vs (2)'s predictions; sha checked); (4) the H200 timing afterok (1) (sha checked).
#   W21_APPROVED="..." D569_READS_PASSED=1 mem-guard bash tracks/t2_bci/sofia/launch_w21lp.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[[ "${W21_APPROVED:-}" == *"T2 W21 approved"* && "${W21_APPROVED:-}" == *"D569"* ]] || { echo "set W21_APPROVED to the owner's typed W21 line (it names D569)"; exit 3; }
[ "${D569_READS_PASSED:-}" = 1 ] || { echo "D569's three registered reads must have passed (record them in DECISIONS first)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_w21lp_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_w21lp.sh)" ] || { echo "commit first"; exit 2; }
SHA=096a428dd5a30fcbeb169a0e2dfed6301f90dce0e21dabb62f984699642d824b
LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-w20-t2p_fin2_w20d557-1631447/t2p_fin2_w20d557.zip
h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of W20+D557: $h"; [ "$h" = "$SHA" ] || { echo "!! the local W20+D557 copy's sha256 != $SHA"; exit 4; }
R=$S1_ROOT/runs/ni26
BASE_REL=ni26-t2p-w20-t2p_fin2_w20d557-1631447/t2p_fin2_w20d557.zip; BASE=$R/$BASE_REL
N=t2p_fin2_w21lp
P="T2-D569 W21 = W20+D557 + cnn_lp (the owner's W21 line; D569 passed A / B / C on the real teacher) for gating"
V=(--partition=zen4_h200)                         # zen5_vis = 1 GPU per user across projects (T3 holds it: --test-only said Nov 21); time binds, so the H200
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$SHA" S1_PURPOSE="$P: (1) the W21 repacks R6a (inert) / R6b (cnn_lp) + K1 (H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w21lp_build.slurm \
  "${V[@]}" --time=05:00:00 --job-name="ni26-t2p-w21lp-$N" -- "$BASE" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-w21lp-$N-$J1/$N.zip"
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$SHA" S1_PURPOSE="$P: (2) the post-build check vs W20+D557 (gates IL / H / HF / HP / FM / B / S; H200)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=08:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- w21lp "$BASE_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2 (H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=03:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-w21kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s (owner) / 3,150 s (-lite rule)" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)
echo "W21 (D569): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL"

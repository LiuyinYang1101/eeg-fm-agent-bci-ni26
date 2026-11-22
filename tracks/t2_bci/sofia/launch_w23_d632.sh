#!/usr/bin/env bash
# T2-D634 -> W23 = W21 + D632 P1 (the copy router's route space, i.e. scaler, LDA and user centroids, fitted on each user's session-1 windows;
# copies, r and template unchanged; a config-only repack of W21). The owner chose "Yes, build it as an exception" at the manager's prompt
# (manager shell, 2026-11-19 13:44); it counts only when USER_LINES item 166 is TYPED in T2's shell.
#   GO       REFUSES without W23_APPROVED = the owner's typed line (item 166: "T2 W23 approved, small-gain exception: W23 = W21 + D632 ...").
#   PREBUILD the owner's exception choice relayed by the manager (NOT item 166): PREBUILD_CHOICE_AT="2026-11-19 13:44" required, W23_APPROVED
#            must be unset; N, the zip and every job name carry _PREBUILD; S1_PURPOSE says 'not approved until typed'; the build job writes
#            README_PREBUILD.txt. A PREBUILD zip becomes the upload candidate ONLY via promote_w23_d632_prebuild.sh (sha-pinned, item 166).
# Chain (= launch_w22_d603.sh's, W21 the base). PARTITIONS (the manager, 2026-11-19 ≈ 14:1x: --test-only put the H200 start at Nov 21):
#   (1)-(3) on zen5_vis (account vsc, free; RTX 5000 Ada 32 GB; W20's check + K2 passed there; 1 GPU per user across projects, so the three
#   run one after another through their dependencies); (4) the timing stays on zen4_h200 (the D517 line), submitted at launch afterok (1).
#   Walltimes 2 h / 2 h / 2 h / 1 h. Hard stop (the manager): any gate or the H200 timing not passed by Nov 21 06:00 -> W23 is out, W21 ships.
#   (1) t2p_w23_build.slurm: W21 sha checked on sofia; ONE repack with W21's own template, the four --set route-space edits from
#       tracks/t2_bci/sofia/w23_route_s1.json, --set-ref; K1 (the official kit; W21's D553 / D560 / D569 lines) ->
#   (2) the package check t2p_w23_pkgcheck.py vs W21 on every public stream (config / files, finite, rerun, OLD / NEW routes == the lab's,
#       GOLDEN: OLD replaying NEW's routes == NEW; NEW's .pred.npz) -> (3) K2 (the kit on all public streams vs (2)'s predictions);
#   (4) the H200 timing afterok (1) (the D517 line <= 3,300 s).
#   W23_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_w23_d632.sh dry|GO
#   PREBUILD_CHOICE_AT="2026-11-19 13:44" mem-guard bash tracks/t2_bci/sofia/launch_w23_d632.sh PREBUILDdry|PREBUILD
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO|PREBUILDdry|PREBUILD}; [[ "$mode" == dry || "$mode" == GO || "$mode" == PREBUILDdry || "$mode" == PREBUILD ]] || exit 2
pb=0; [[ "$mode" == PREBUILD* ]] && pb=1
if [ $pb = 1 ]; then   # the ONLY difference from GO / dry: no W23_APPROVED test; the choice is recorded instead (never the owner's line)
  [[ "${PREBUILD_CHOICE_AT:-}" =~ ^2026-11-[0-9]{2}\ [0-9]{2}:[0-9x]{2}$ ]] || { echo "set PREBUILD_CHOICE_AT='YYYY-MM-DD HH:MM' (the owner's exception choice at the manager's prompt)"; exit 3; }
  [ -z "${W23_APPROVED:-}" ] || { echo "W23_APPROVED is set: an approved build is GO, not PREBUILD"; exit 3; }
  [ "$mode" == PREBUILD ] && mode=GO || mode=dry
fi
if [ "$mode" == GO ] && [ $pb = 0 ]; then
  [[ "${W23_APPROVED:-}" == *"T2 W23 approved"* && "${W23_APPROVED:-}" == *"D632"* ]] || { echo "set W23_APPROVED to the owner's typed line (item 166, naming D632)"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/w23_route_s1.json tracks/t2_bci/sofia/t2p_w23_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_w23_d632.sh)" ] || { echo "commit first"; exit 2; }
W21_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc
W21_REL=ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip
LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/$W21_REL
h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of W21: $h"; [ "$h" = "$W21_SHA" ] || { echo "!! the local W21 copy's sha256 != $W21_SHA"; exit 4; }
echo "### the route-space JSON: $(sha256sum tracks/t2_bci/sofia/w23_route_s1.json | cut -d' ' -f1) (git $(git rev-parse --short HEAD))"
R=$S1_ROOT/runs/ni26
BASE=$R/$W21_REL
N=t2p_fin2_w23d632
P="T2-D634 W23 = W21 + D632 P1 (session-1 route space; the owner's typed line, item 166) for gating"
PBX=""
if [ $pb = 1 ]; then
  N=${N}_PREBUILD
  P="T2-D634 W23 PREBUILD = W21 + D632 P1 (session-1 route space): owner-chosen exception (manager prompt, $PREBUILD_CHOICE_AT), not approved until typed"
  PBX=",W23_PREBUILD=1"
fi
V=(--partition=zen5_vis --account=vsc)              # (1)-(3): free, 1 GPU per user (they chain by afterok)
VH=(--partition=zen4_h200)                          # (4): the H200 timing (the D517 line)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
echo "### BASE $W21_REL (sha $W21_SHA) -> $N"
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$W21_SHA$PBX" S1_PURPOSE="$P: (1) the config-only repack (--set-ref) + K1 (zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_w23_build.slurm "${V[@]}" --time=02:00:00 --job-name="ni26-t2p-w23d632-$N" -- "$BASE" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-w23d632-$N-$J1/$N.zip"
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  "${VH[@]}" --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)      # first in the H200 queue
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$W21_SHA" S1_PURPOSE="$P: (2) the package check vs W21 (config / files / routes / golden replay; zen5_vis)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=02:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- w23 "$W21_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2 (zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=02:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-w23kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
echo "W23 (D632 on W21$([ $pb = 1 ] && echo ', PREBUILD: NOT APPROVED')): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL"

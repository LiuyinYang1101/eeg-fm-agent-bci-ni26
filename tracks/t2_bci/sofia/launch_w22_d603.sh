#!/usr/bin/env bash
# T2-D603 -> a W22 candidate = BASE + cfg p1_prior_informed on template t2p_submission_w22d603.py (PREPARED as code only, the manager 2026-11-19:
# Track3's informed restart REPLACES D595 as the owner's optional small-gain exception; pooled ten +0.0006, returning +0.0011 with P04 AND P10
# positive on every set, sealed +0.0008; below T2-D590's bar). REFUSES without W22_APPROVED = the owner's typed line in T2's shell NAMING D603.
# BASE (= launch_w22_d595.sh's, the manager's composition decision 2026-11-18):
#   - default = W21 (sha 94281e72…; its local copy is checked too): only D603 is approved.
#   - D603 AND an FM part passing: launch_w22.sh builds and gates the FM chain FIRST; D603 is then the LAST step, on THAT chain's
#     final zip: BASE_REL=<its path under runs/ni26> BASE_SHA=<its 64-hex sha256, from the gated chain> BASE_TAG=<the chain's tag, e.g. d588>.
#     The post-build check's OLD is that BASE (pre-refit equality and 'every other family unchanged' against BASE, not W21).
# Chain (W21's pattern):
#   (1) t2p_w22_d603_build.slurm (H200): BASE sha == BASE_SHA checked on sofia; R6a the W22 template without the flag (clean_max_dp == 0 = BASE
#       bitwise), R6b --set p1_prior_informed=true --set-ref, K1 (the official kit; the D553 / D560 / D569 line counts + the D603 restart lines) ->
#   (2) the post-build check t2p_w22_d603_pkgcheck.py vs BASE (config, finite, == BASE before the first refit, reproducible, the P1 correctors
#       restart exactly at the fired refits on their own running mean, the parts unchanged, the decode == the replayed corrector, others == BASE;
#       NEW's .pred.npz) -> (3) K2 (the kit on all public streams vs (2)'s predictions); (4) the H200 timing afterok (1) (the D517 line <= 3,300 s).
#   [BASE_REL=.. BASE_SHA=.. BASE_TAG=..] W22_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_w22_d603.sh dry|GO
# PREBUILD (the owner chose 'Pre-build now' at the manager's prompt, 2026-11-19 ~09:5x; NOT item 164; Track3's / the manager's checklist):
#   PREBUILD_CHOICE_AT="<YYYY-MM-DD HH:MM of the owner's choice, as the manager relayed it>" bash .../launch_w22_d603.sh PREBUILDdry|PREBUILD
#   = GO with ONLY the W22_APPROVED test skipped (git-clean, W21 / BASE sha, every chain gate identical); N, the zip and every job name carry
#   _PREBUILD; S1_PURPOSE says 'not approved'; the build job writes README_PREBUILD.txt (W22_PREBUILD=1). A PREBUILD zip becomes the upload
#   candidate ONLY through promote_w22_d603_prebuild.sh (sha-pinned, requires W22_APPROVED naming D603), never by renaming it by hand.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO|PREBUILDdry|PREBUILD}; [[ "$mode" == dry || "$mode" == GO || "$mode" == PREBUILDdry || "$mode" == PREBUILD ]] || exit 2
pb=0; [[ "$mode" == PREBUILD* ]] && pb=1
if [ $pb = 1 ]; then   # the ONLY difference from GO / dry: no W22_APPROVED test; the choice is recorded instead (never the owner's line)
  [[ "${PREBUILD_CHOICE_AT:-}" =~ ^2026-11-[0-9]{2}\ [0-9]{2}:[0-9x]{2}$ ]] || { echo "set PREBUILD_CHOICE_AT='YYYY-MM-DD HH:MM' (the owner's 'Pre-build now' at the manager's prompt)"; exit 3; }
  [ -z "${W22_APPROVED:-}" ] || { echo "W22_APPROVED is set: an approved build is GO, not PREBUILD"; exit 3; }
  [ "$mode" == PREBUILD ] && mode=GO || mode=dry
fi
if [ "$mode" == GO ] && [ $pb = 0 ]; then
  [[ "${W22_APPROVED:-}" == *"T2 W22"* && "${W22_APPROVED:-}" == *"D603"* ]] || { echo "set W22_APPROVED to the owner's typed line naming D603"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_w22_d603_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w22_d603.sh)" ] || { echo "commit first"; exit 2; }
W21_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc
W21_REL=ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip
BASE_REL=${BASE_REL:-$W21_REL}; BASE_TAG=${BASE_TAG:-}
if [ "$BASE_REL" = "$W21_REL" ]; then
  [ -z "${BASE_SHA:-}" ] || [ "$BASE_SHA" = "$W21_SHA" ] || { echo "BASE_SHA $BASE_SHA != W21's sha for W21's path"; exit 2; }
  [ -z "$BASE_TAG" ] || { echo "BASE_TAG is for an FM-chain base only"; exit 2; }
  SHA=$W21_SHA; BNAME=W21
  LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/$W21_REL
  h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of W21: $h"; [ "$h" = "$SHA" ] || { echo "!! the local W21 copy's sha256 != $SHA"; exit 4; }
else   # an FM chain's final zip (launch_w22.sh): its sha is given explicitly (never 'sibling': D603 runs only after that chain is gated)
  [[ "${BASE_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "BASE_SHA = the 64-hex sha256 of $BASE_REL (from the gated FM chain)"; exit 2; }
  [[ "$BASE_TAG" =~ ^d5[0-9][0-9][a-z0-9]*$ ]] || { echo "BASE_TAG = the FM chain's launch_w22.sh tag (e.g. d588)"; exit 2; }
  [[ "$BASE_REL" =~ ^ni26-t2p-w21lp-t2p_fin2_w22${BASE_TAG}-[0-9]+/t2p_fin2_w22${BASE_TAG}\.zip$ ]] || { echo "BASE_REL is not launch_w22.sh's (3) zip for tag $BASE_TAG"; exit 2; }
  SHA=$BASE_SHA; BNAME="the ${BASE_TAG^^} chain's W22"   # that chain's own GO already required its owner line (launch_w22.sh)
fi
R=$S1_ROOT/runs/ni26
BASE=$R/$BASE_REL
N=t2p_fin2_w22${BASE_TAG}d603
P="T2-D603 W22 = $BNAME + p1_prior_informed (the owner's line naming D603) for gating"
PBX=""
if [ $pb = 1 ]; then
  N=${N}_PREBUILD
  P="T2-D603 W22 PREBUILD = $BNAME + p1_prior_informed: owner-chosen PREBUILD for readiness (manager prompt, $PREBUILD_CHOICE_AT); not approved"
  PBX=",W22_PREBUILD=1"
fi
V=(--partition=zen4_h200)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
echo "### BASE $BASE_REL (sha $SHA) -> $N"
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$SHA$PBX" S1_PURPOSE="$P: (1) R6a (the W22 template inert on BASE) / R6b (p1_prior_informed) + K1 (H200)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_w22_d603_build.slurm "${V[@]}" --time=05:00:00 --job-name="ni26-t2p-w22d603-$N" -- "$BASE" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-w22d603-$N-$J1/$N.zip"
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$SHA" S1_PURPOSE="$P: (2) the post-build check vs BASE (config / pre-refit / rerun / restarts / parts / golden / others; H200)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=08:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- w22_d603 "$BASE_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2 (H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=03:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-w22kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)
echo "W22 (D603 on ${BASE_TAG:-W21}$([ $pb = 1 ] && echo ', PREBUILD: NOT APPROVED')): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL"

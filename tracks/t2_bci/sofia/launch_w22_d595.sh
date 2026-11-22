#!/usr/bin/env bash
# T2-D595 -> a W22 candidate = BASE + cfg p1_prior_reset on template t2p_submission_w22.py (PREPARED as code only, the manager 2026-11-18: D595
# did NOT clear T2-D590's displacement bar (pooled ten +0.0008) but is sign-consistent and zero-runtime; the owner decides). REFUSES without
# W22_APPROVED = the owner's typed line in T2's shell that NAMES D595 (T2-D592's line does not cover it).
# BASE (the manager's composition decision 2026-11-18, after 855efefb):
#   - default = W21 (sha 94281e72…; its local copy is checked too): only D595 is approved.
#   - D595 AND an FM part (D588 / D596) passing: launch_w22.sh builds and gates the FM chain FIRST; D595 is then the LAST step, on THAT chain's
#     final zip: BASE_REL=<its path under runs/ni26> BASE_SHA=<its 64-hex sha256, from the gated chain> BASE_TAG=<the chain's tag, e.g. d588>.
#     The post-build check's OLD is that BASE (pre-refit equality and 'every other family unchanged' against BASE, not W21).
# Chain (W21's pattern):
#   (1) t2p_w22_d595_build.slurm (H200): BASE sha == BASE_SHA checked on sofia; R6a the W22 template without the flag (clean_max_dp == 0 = BASE
#       bitwise), R6b --set p1_prior_reset=true --set-ref, K1 (the official kit; the D553 / D560 / D569 line counts + the D595 reset lines) ->
#   (2) the post-build check t2p_w22_d595_pkgcheck.py vs BASE (config, finite, == BASE before the first refit, reproducible, the P1 correctors
#       reset exactly at the fired refits, the parts unchanged, the reset decode == the restarted corrector, every other family == BASE;
#       NEW's .pred.npz) -> (3) K2 (the kit on all public streams vs (2)'s predictions); (4) the H200 timing afterok (1) (the D517 line <= 3,300 s).
#   [BASE_REL=.. BASE_SHA=.. BASE_TAG=..] W22_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_w22_d595.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
if [ "$mode" == GO ]; then
  [[ "${W22_APPROVED:-}" == *"T2 W22"* && "${W22_APPROVED:-}" == *"D595"* ]] || { echo "set W22_APPROVED to the owner's typed line naming D595"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_w22_d595_build.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w22_d595.sh)" ] || { echo "commit first"; exit 2; }
W21_SHA=94281e72198c40f1d238fcae22fe6eecb6c987a73fc6c83890037bf7864973fc
W21_REL=ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip
BASE_REL=${BASE_REL:-$W21_REL}; BASE_TAG=${BASE_TAG:-}
if [ "$BASE_REL" = "$W21_REL" ]; then
  [ -z "${BASE_SHA:-}" ] || [ "$BASE_SHA" = "$W21_SHA" ] || { echo "BASE_SHA $BASE_SHA != W21's sha for W21's path"; exit 2; }
  [ -z "$BASE_TAG" ] || { echo "BASE_TAG is for an FM-chain base only"; exit 2; }
  SHA=$W21_SHA; BNAME=W21
  LOC=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/$W21_REL
  h=$(sha256sum "$LOC" | cut -d' ' -f1); echo "### the local copy of W21: $h"; [ "$h" = "$SHA" ] || { echo "!! the local W21 copy's sha256 != $SHA"; exit 4; }
else   # an FM chain's final zip (launch_w22.sh): its sha is given explicitly (never 'sibling': D595 runs only after that chain is gated)
  [[ "${BASE_SHA:-}" =~ ^[0-9a-f]{64}$ ]] || { echo "BASE_SHA = the 64-hex sha256 of $BASE_REL (from the gated FM chain)"; exit 2; }
  [[ "$BASE_TAG" =~ ^d5[0-9][0-9][a-z0-9]*$ ]] || { echo "BASE_TAG = the FM chain's launch_w22.sh tag (e.g. d588)"; exit 2; }
  [[ "$BASE_REL" =~ ^ni26-t2p-w21lp-t2p_fin2_w22${BASE_TAG}-[0-9]+/t2p_fin2_w22${BASE_TAG}\.zip$ ]] || { echo "BASE_REL is not launch_w22.sh's (3) zip for tag $BASE_TAG"; exit 2; }
  SHA=$BASE_SHA; BNAME="the ${BASE_TAG^^} chain's W22"   # that chain's own GO already required its owner line (launch_w22.sh)
fi
R=$S1_ROOT/runs/ni26
BASE=$R/$BASE_REL
N=t2p_fin2_w22${BASE_TAG}d595
P="T2-D595 W22 = $BNAME + p1_prior_reset (the owner's line naming D595) for gating"
V=(--partition=zen4_h200)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
echo "### BASE $BASE_REL (sha $SHA) -> $N"
J1=$(sub S1_EXPORT="BASE_SHA_EXPECT=$SHA" S1_PURPOSE="$P: (1) R6a (the W22 template inert on BASE) / R6b (p1_prior_reset) + K1 (H200)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_w22_d595_build.slurm "${V[@]}" --time=05:00:00 --job-name="ni26-t2p-w22d595-$N" -- "$BASE" "$N")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
Z_REL="ni26-t2p-w22d595-$N-$J1/$N.zip"
J2=$(sub S1_EXPORT="OLD_SHA_EXPECT=$SHA" S1_PURPOSE="$P: (2) the post-build check vs BASE (config / pre-refit / rerun / resets / parts / golden / others; H200)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" --time=08:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-pkgcheck-$N" -- w22_d595 "$BASE_REL" "$Z_REL" "$N")
J3=$(sub S1_PURPOSE="$P: (3) condition K2 (H200)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm "${V[@]}" --time=03:00:00 \
  --dependency="afterok:$J2" --job-name="ni26-t2p-w22kit-$N" -- "$Z_REL" "ni26-t2p-pkgcheck-$N-$J2/$N.pkgcheck.pred.npz" "$N")
J4=$(sub S1_PURPOSE="$P: (4) the H200 timing (3 reps): the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm \
  --time=01:00:00 --dependency="afterok:$J1" --job-name="ni26-t2p-time-$N" -- "$Z_REL" "$N" 3)
echo "W22 (D595 on ${BASE_TAG:-W21}): (1) $J1 -> (2) check $J2 -> (3) K2 $J3; (4) timing $J4; the zip runs/ni26/$Z_REL"

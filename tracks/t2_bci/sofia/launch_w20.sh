#!/usr/bin/env bash
# T2-D560 (the W20 build: W19 b2 + cfg fm_proto, on zen5_vis). The FULL build REFUSES TO RUN unless W20_APPROVED carries the owner's own W20
# line (typed in T2's shell; it names W20 and fm_proto). The SMOKE (R5a on 2 streams, no package kept) needs no line: it tests memory / env.
#   mem-guard bash tracks/t2_bci/sofia/launch_w20.sh dry|GO smoke
#   W20_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_w20.sh dry|GO full
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:?smoke|full}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == smoke || "$part" == full ]] || exit 2
if [ "$part" == full ]; then
  [[ "${W20_APPROVED:-}" == *"template change approved"* && "${W20_APPROVED:-}" == *"W20"* && "${W20_APPROVED:-}" == *"fm_proto"* ]] || \
    { echo "set W20_APPROVED to the owner's own W20 line (typed in T2's shell; it names W20 and fm_proto)"; exit 3; }
fi
[ -z "$(git status --porcelain -- tracks/t2_bci/submission tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_w20.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
BASE="$R/ni26-t2p-w19-t2p_fin2_w19-1629649/t2p_fin2_w19.zip"      # W19 b2 (sha256 92b8482b..., K1 clean)
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_PURPOSE="T2-D560 W20 build SMOKE on zen5_vis (free): R5a template-only repack of W19 b2 on 2 streams (memory / env), no package kept" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm --partition=zen5_vis --account=vsc --time=00:45:00 --job-name=ni26-t2p-w20-smoke \
    -- "$BASE" t2p_fin2_w20 smoke 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-160
  exit 0
fi
# FULL: build (zen5_vis) -> post-build check w20 (zen5_vis, afterok) -> K2 (zen5_vis, afterok the check); H200 timing (afterok the build) = the
# only job in the zen4_h200 queue. zen5_vis allows 1 GPU per user: the three zen5_vis jobs run one after another by their dependencies.
NAME=t2p_fin2_w20
P="T2-D560 W20 (W19 b2 + fm_proto) for gating (approved: ${W20_APPROVED:0:60})"
V=(--partition=zen5_vis --account=vsc)
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
JB=$(sub S1_PURPOSE="$P: R5a / R5b repacks + kit replay K1 (zen5_vis, free)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm "${V[@]}" \
  --time=05:00:00 --job-name="ni26-t2p-w20-$NAME" -- "$BASE" "$NAME")
[ "$mode" == dry ] && { echo "dry: the build validated; check / K2 / timing need its id"; exit 0; }
[ -n "$JB" ] || { echo "build not submitted"; exit 2; }
NEW_REL="ni26-t2p-w20-$NAME-$JB/$NAME.zip"; BASE_REL=${BASE#$R/}
JC=$(sub S1_PURPOSE="$P: post-build check vs W19 b2 (gates I / H / HF / HP / B / S; zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm "${V[@]}" \
  --time=08:00:00 --dependency="afterok:$JB" --job-name="ni26-t2p-pkgcheck-$NAME" -- w20 "$BASE_REL" "$NEW_REL" "$NAME")
JK=$(sub S1_PURPOSE="$P: condition K2 (the kit harness on all public streams vs the check; zen5_vis)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm \
  "${V[@]}" --time=03:00:00 --dependency="afterok:$JC" --job-name="ni26-t2p-w20kit-$NAME" -- "$NEW_REL" "ni26-t2p-pkgcheck-$NAME-$JC/$NAME.pkgcheck.pred.npz" "$NAME")
JT=$(sub S1_PURPOSE="$P: H200 timing (3 reps) + kit replay, the D517 line <= 3,300 s" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:00:00 \
  --dependency="afterok:$JB" --job-name="ni26-t2p-time-$NAME" -- "$NEW_REL" "$NAME" 3)
echo "W20: build $JB -> check $JC -> K2 $JK (zen5_vis); timing $JT (H200, afterok the build); new zip runs/ni26/$NEW_REL"

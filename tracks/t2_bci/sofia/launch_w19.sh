#!/usr/bin/env bash
# T2-D549 (the W19 build for gating). REFUSES TO RUN unless W19_APPROVED carries the owner's own W19 line (typed in T2's shell; date + the words).
# Per draw N = 2 (primary, W18 b2) | 1 (the fallback, W18 b1, only after b2's gates):
#   build  = t2p_w19_build.slurm (R1 --fm-from the W19 merged donor N, R2 --side-from side.pt S10, R3 --set side.refit --set-ref,
#            R4 --set fm_side_refit --set-ref (T2-D553, named in the owner's line), K1 kit replay)
#   timing = t2p_time_zip.slurm on the new zip (3 reps + kit replay), afterok the build
#   check  = t2p_pkgcheck.slurm w19 (NEW vs W18 build N: config, finite / moves / re-run, gates I, H, HF, B, S, 0 fallbacks), afterok the build
#   K2     = t2p_w19_kitfull.slurm (the kit harness on all public streams vs the check's predictions), afterok the check
#   W19_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_w19.sh dry|GO 2|1
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; n=${2:?2|1}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$n" == 1 || "$n" == 2 ]] || exit 2
[[ "${W19_APPROVED:-}" == *"template change approved"* && "${W19_APPROVED:-}" == *"W19"* && "${W19_APPROVED:-}" == *"D553"* ]] || { echo "set W19_APPROVED to the owner's own W19 line (typed in T2's shell; it names D553)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission src/ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_w19.sh scripts/local/t2_d539_read.py)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
SIDE=$R/t2_d549_side10/side.pt
if [ "$n" == 2 ]; then
  BASE_REL=ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip; DONOR=$R/ni26-t2p-d546b-merge-b2/t2p_w19b2_donor.zip
else
  BASE_REL=ni26-t2p-w18-t2p_fin1_w18-1622522/t2p_fin1_w18.zip; DONOR=$R/ni26-t2p-d546b-merge-b1/t2p_w19b1_donor.zip
fi
NAME=t2p_fin${n}_w19
P="T2-D549 W19 build $n for gating (approved: ${W19_APPROVED:0:60})"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
JB=$(sub S1_PURPOSE="$P: R1 / R2 / R3 / R4 repacks + kit replay K1" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm --time=05:00:00 \
  --job-name="ni26-t2p-w19-$NAME" -- "$R/$BASE_REL" "$DONOR" "$SIDE" "$NAME")
[ "$mode" == dry ] && { echo "dry: the build validated; timing / check / K2 need its id"; exit 0; }
[ -n "$JB" ] || { echo "build not submitted"; exit 2; }
NEW_REL="ni26-t2p-w19-$NAME-$JB/$NAME.zip"
JT=$(sub S1_PURPOSE="$P: H200 timing (3 reps) + kit replay" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:00:00 \
  --dependency="afterok:$JB" --job-name="ni26-t2p-time-$NAME" -- "$NEW_REL" "$NAME" 3)
JC=$(sub S1_PURPOSE="$P: post-build check vs W18 build $n (gates I / H / HF / B / S)" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm --time=02:00:00 \
  --dependency="afterok:$JB" --job-name="ni26-t2p-pkgcheck-$NAME" -- w19 "$BASE_REL" "$NEW_REL" "$NAME")
JK=$(sub S1_PURPOSE="$P: condition K2 (the kit harness on all public streams vs the check)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm \
  --time=01:00:00 --dependency="afterok:$JC" --job-name="ni26-t2p-w19kit-$NAME" -- "$NEW_REL" "ni26-t2p-pkgcheck-$NAME-$JC/$NAME.pkgcheck.pred.npz" "$NAME")
echo "W19 build $n: build $JB -> timing $JT, check $JC -> K2 $JK; new zip runs/ni26/$NEW_REL"

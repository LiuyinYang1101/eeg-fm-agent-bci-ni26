#!/usr/bin/env bash
# T2-D534 (W18 builds for gating, APPROVED by the user's line typed in T2's shell 2026-11-16 ≈ 10:25): per draw N = 1 | 2
#   build  = t2p_w18_build.slurm (merge SD donor dN + head-less MI dN at slot 0; repack W17 build N with template W18; kit replay),
#            afterok the SD donor's job (it lands ≈ 12:40 / 13:00);
#   timing = t2p_time_zip.slurm on the new zip (3 reps + kit replay), afterok the build;
#   check  = t2p_pkgcheck.slurm w18 (NEW vs W17 build N), afterok the build.
# REFUSES TO RUN unless W18_APPROVED carries the user's own W18 line (date + the words).
#   W18_APPROVED="..." mem-guard bash tracks/t2_bci/sofia/launch_d534_w18.sh dry|GO 1|2
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; n=${2:?1|2}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$n" == 1 || "$n" == 2 ]] || exit 2
[[ "${W18_APPROVED:-}" == *"template change approved"* && "${W18_APPROVED:-}" == *"W18"* ]] || { echo "set W18_APPROVED to the user's own W18 line (typed in T2's shell)"; exit 3; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_w18_build.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/t2p_pkgcheck.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d534_w18.sh)" ] || { echo "commit first"; exit 2; }
R=$S1_ROOT/runs/ni26
if [ "$n" == 1 ]; then
  BASE_REL=ni26-t2p-repack-t2p_fin1_w17-1619616/t2p_fin1_w17.zip; SDJ=1620636
  SD=$R/ni26-t2p-d531-sddonor-d1-GO-1620636/t2p_d531_sddonor_d1.zip; MI=$R/ni26-t2p-d529-midonor-d1-s610-1620733/t2p_d529_midonor_d1_s610.zip
else
  BASE_REL=ni26-t2p-repack-t2p_fin2_w17-1619617/t2p_fin2_w17.zip; SDJ=1620637
  SD=$R/ni26-t2p-d531-sddonor-d2-GO-1620637/t2p_d531_sddonor_d2.zip; MI=$R/ni26-t2p-d529-midonor-d2-s620-1620735/t2p_d529_midonor_d2_s620.zip
fi
NAME=t2p_fin${n}_w18
P="T2-D534 W18 build $n for gating (approved: ${W18_APPROVED:0:60})"
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
JB=$(sub S1_PURPOSE="$P: merge + repack + kit replay" s1 submit ni26 tracks/t2_bci/sofia/t2p_w18_build.slurm --time=02:30:00 \
  --dependency="afterok:$SDJ" --job-name="ni26-t2p-w18-$NAME" -- "$R/$BASE_REL" "$SD" "$MI" "$NAME")
[ "$mode" == dry ] && { echo "dry: the build validated; timing / check need its id"; exit 0; }
[ -n "$JB" ] || { echo "build not submitted"; exit 2; }
NEW_REL="ni26-t2p-w18-$NAME-$JB/$NAME.zip"
JT=$(sub S1_PURPOSE="$P: H200 timing (3 reps) + kit replay" s1 submit ni26 tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:00:00 \
  --dependency="afterok:$JB" --job-name="ni26-t2p-time-$NAME" -- "$NEW_REL" "$NAME" 3)
JC=$(sub S1_PURPOSE="$P: post-build check vs W17 build $n" s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm --time=00:45:00 \
  --dependency="afterok:$JB" --job-name="ni26-t2p-pkgcheck-$NAME" -- w18 "$BASE_REL" "$NEW_REL" "$NAME")
echo "W18 build $n: build $JB (afterok $SDJ) -> timing $JT, check $JC; new zip runs/ni26/$NEW_REL"

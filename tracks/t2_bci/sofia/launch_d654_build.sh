#!/usr/bin/env bash
# T2-D654 ship path (only after --decide654 INCLUDES D654 and --final picks it; the owner's typed line still gates any upload): the W25-ALL6 PREBUILD
# + the W31v template with the PICKED arm's fm_views (c gate -> "cnn", p gate -> "prob"; f2 / f4 = q 1 with the cnn gate; an m-gated pick is NOT
# shippable by this template) -> (1) t2p_w31v_build.slurm (patch + structural check + golden + predictions) -> (2) K2 (the kit harness vs those
# predictions) and (3) the official H200 timing (3 reps; the owner's D654 bar: mean <= 3,480 s, every rep <= 3,550 s on the D517 line).
#   mem-guard bash tracks/t2_bci/sofia/launch_d654_build.sh dry|GO <arm>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; ARM=${2:?the picked arm}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_w31v_build.slurm tracks/t2_bci/sofia/t2p_w19_kitfull.slurm tracks/t2_bci/sofia/t2p_time_zip.slurm tracks/t2_bci/sofia/launch_d654_build.sh)" ] || { echo "commit first"; exit 2; }
case $ARM in
  f2) V='{"K": 2, "q": 1.0, "gate": "cnn"}' ;; f4) V='{"K": 4, "q": 1.0, "gate": "cnn"}' ;;
  g[24]q[0-9][0-9]c) V="{\"K\": ${ARM:1:1}, \"q\": 0.${ARM:3:2}, \"gate\": \"cnn\"}" ;;
  p[24]q[0-9][0-9]c) V="{\"K\": ${ARM:1:1}, \"q\": 0.${ARM:3:2}, \"gate\": \"prob\"}" ;;
  *) echo "REFUSED: arm $ARM is not shippable by the W31v template (an m gate or unknown)"; exit 3 ;;
esac
BASE=ni26-t2p-w21lp-t2p_fin2_w25all6_PREBUILD-1644829/t2p_fin2_w25all6_PREBUILD.zip
N=t2p_fin2_w31v_$ARM
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
sub() { env "${dry[@]}" "$@" 2>&1 | tee /dev/stderr | { grep -oE 'SUBMITTED [0-9]+' || true; } | awk '{print $2}'; }
P="T2-D654 W31v ship path ($ARM, fm_views $V) for gating; NOT approved (the owner's typed line pending)"
RUN=ni26-t2p-w31v-build-$ARM
J1=$(sub S1_EXPORT="RUN_NAME=$RUN" S1_PURPOSE="$P: (1) patch + structural check + golden + predictions" s1 submit ni26 tracks/t2_bci/sofia/t2p_w31v_build.slurm \
  --time=01:30:00 --job-name=$RUN -- "$BASE" "$N" "$V")
[ "$mode" == dry ] && { echo "dry: (1) validated; the rest need its id"; exit 0; }
[ -n "$J1" ] || { echo "(1) not submitted"; exit 2; }
J2=$(sub S1_PURPOSE="$P: (2) K2 (the kit harness vs (1)'s predictions)" s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm --time=01:30:00 \
  --dependency=afterok:$J1 --job-name=ni26-t2p-w31vkit-$ARM -- "$RUN/$N.zip" "$RUN/$N.pred.npz" "$N")
J3=$(sub S1_PURPOSE="$P: (3) the H200 timing (3 reps; the owner's D654 bar mean <= 3,480 / rep <= 3,550 on the D517 line)" s1 submit ni26 \
  tracks/t2_bci/sofia/t2p_time_zip.slurm --time=01:30:00 --dependency=afterok:$J1 --job-name=ni26-t2p-time-w31v-$ARM -- "$RUN/$N.zip" "w31v_$ARM" 3)
echo "D654 W31v $ARM: (1) $J1 -> (2) K2 $J2; (3) timing $J3; the zip runs/ni26/$RUN/$N.zip"

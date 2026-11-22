#!/usr/bin/env bash
# T2-D484 (the manager's advance 2, 2026-11-14; weights only, R3 standing approval; registered before any number): the CAR-view CPT.
# Three CPTs = the shipped ST-L CPT recipe exactly (st_large, 5,000 steps, batch 64 x accum 8, lr 6.58e-5, wd 1e-5, mask 0.75, PROTEUS +
# Dreyer2023 wide + Scherer2015) + t2p_cpt --car (stream.car_windows on every pool's draw):
#   kit   (PROTEUS pool kit_tv = d89b's)  -> the kit read      seen1 (pool seen1 = d111-with's) -> the seen1 read
#   all   (pool all = d111-all's)         -> the build's CPT (used only if the read passes)
# Then T2-D475's CAR ST-L line from the protocol's CAR CPT, seeds 472-485, 4 parts per protocol (afterok on its CPT).
#   mem-guard bash tracks/t2_bci/sofia/launch_d484.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_cpt.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d484.sh tracks/t2_bci/sofia/arms/t2d484_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D484 the CAR-view CPT (the ST-L's CPT on CAR input; CAR members from it vs T2-D475's CAR members from the plain CPT): feeds W14's CAR members"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
declare -A POOL=([kit]=kit_tv [seen1]=seen1 [all]=all) J
for pr in kit seen1 all; do
  J[$pr]=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-cpt-d484-car-$pr" S1_PURPOSE="$P; CPT $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_cpt.slurm --time=03:00:00 \
    --job-name="ni26-t2p-cpt-d484-car-$pr" -- --fm st_large --extra export_dreyer2023_wide_v1 export_scherer2015_v1 --steps 5000 --batch 64 --accum 8 \
    --lr 6.58e-05 --wd 1e-05 --mask-ratio 0.75 --eval-every 500 --proteus-pool "${POOL[$pr]}" --car)
done
for pr in kit seen1; do
  dep=(); [ -n "${J[$pr]:-}" ] && dep=(--dependency="afterok:${J[$pr]}")
  for j in 0 1 2 3; do
    sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $pr part $j" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 "${dep[@]}" \
      --job-name="ni26-t2p-d484-stlcarcpt-$pr-$j" -- "tracks/t2_bci/sofia/arms/t2d484_stlcarcpt_${pr}_$j.txt" >/dev/null
  done
done

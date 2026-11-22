#!/usr/bin/env bash
# T2-D492 (the manager's (b), the user's "make efforts on improvements"; GPU inside GO, R3; registered before any number): in-domain CPT
# of REVE-base and LaBraM-base (t2p_cpt_fm.py: masked-patch reconstruction; ST-L's CPT pools: PROTEUS + Dreyer2023 wide + Scherer2015;
# 5,000 steps, batch 64 x accum 8), one CPT per protocol pool (kit -> kit_tv = d89b's, seen1 -> seen1 = d111-with's, all -> the build's,
# used only on a pass), then T2-D452's member line from the protocol's CPT on D452's seeds 424-435 (3 parts each, afterok on its CPT).
#   mem-guard bash tracks/t2_bci/sofia/launch_d492.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts/t2p_cpt_fm.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/sofia/t2p_cpt_fm.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d492.sh tracks/t2_bci/sofia/arms/t2d492_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D492 in-domain CPT of REVE / LaBraM (masked-patch) + D452's members from it: feeds a CPT'd different-FM family under W14"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
declare -A POOL=([kit]=kit_tv [seen1]=seen1 [all]=all) J
for fm in reve labram; do
  for pr in kit seen1 all; do
    J[$fm-$pr]=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-cpt-d492-$fm-$pr" S1_PURPOSE="$P; CPT $fm $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_cpt_fm.slurm \
      --time=03:00:00 --job-name="ni26-t2p-cpt-d492-$fm-$pr" -- --fm $fm --extra export_dreyer2023_wide_v1 export_scherer2015_v1 --steps 5000 \
      --batch 64 --accum 8 --eval-every 500 --proteus-pool "${POOL[$pr]}")
  done
  for pr in kit seen1; do
    dep=(); [ -n "${J[$fm-$pr]:-}" ] && dep=(--dependency="afterok:${J[$fm-$pr]}")
    for j in 0 1 2; do
      sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $fm $pr part $j" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=04:00:00 "${dep[@]}" \
        --job-name="ni26-t2p-d492-$fm-$pr-$j" -- "tracks/t2_bci/sofia/arms/t2d492_${fm}_${pr}_$j.txt" >/dev/null
    done
  done
done

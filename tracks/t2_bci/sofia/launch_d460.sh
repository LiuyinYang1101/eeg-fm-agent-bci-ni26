#!/usr/bin/env bash
# T2-D460: pretrained ST-EEGFormer base / small as added or replacing FM members in M7's shape (the manager's seed, 2026-11-12).
# 4 CPT jobs (CPT v1's recipe per protocol) -> 8 member jobs (with / without CPT), one GPU each.
#   mem-guard bash tracks/t2_bci/sofia/launch_d460.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_cpt.slurm tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d460.sh tracks/t2_bci/sofia/arms/t2d460_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
P="T2-D460 (pretrained ST-base / ST-small as FM members in M7's shape): feeds T2's third sealed upload"
sub() { local out; out=$(env "${dry[@]}" "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150 >&2; sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p' <<<"$out"; }
declare -A FM=([b]=st_base [s]=st_small) J
for z in b s; do
  for pr in kit seen1; do
    pool=(); [ "$pr" == seen1 ] && pool=(--proteus-pool seen1)
    J[$z$pr]=$(sub S1_EXPORT="RUN_NAME=ni26-t2p-cpt-d460-$z-$pr" S1_PURPOSE="$P; CPT ${FM[$z]} $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_cpt.slurm --time=05:00:00 \
      --job-name="ni26-t2p-cpt-d460-$z-$pr" -- --fm "${FM[$z]}" --extra export_dreyer2023_wide_v1 export_scherer2015_v1 --steps 5000 --batch 64 --accum 8 --eval-every 500 "${pool[@]}")
    for c in c n; do
      dep=(); [ "$c" == c ] && [ -n "${J[$z$pr]:-}" ] && dep=(--dependency="afterok:${J[$z$pr]}")
      sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members ${FM[$z]} $c $pr" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=05:00:00 "${dep[@]}" \
        --job-name="ni26-t2p-d460-$z$c-$pr" -- "tracks/t2_bci/sofia/arms/t2d460_${z}${c}_${pr}.txt" >/dev/null
    done
  done
done

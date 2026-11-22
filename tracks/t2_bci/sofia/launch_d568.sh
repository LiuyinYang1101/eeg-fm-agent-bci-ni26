#!/usr/bin/env bash
# T2-D568 (D557's augmentation at a stronger dose, ONE arm, C first; registered 2026-11-18 11:34:11 before any job; inside GO): D558's 16
# augmented headed-line jobs (seeds 837-841) with --fm-aug 0.3:45:0.15 --fm-mixup 1.0 (was 0.2:30:0.1 / 0.8), nothing else changed; side +
# feature dumps. The code path = D558's k jobs (smoke-tested lines; only two numbers change), so no new smoke. 16 jobs.
#   mem-guard bash tracks/t2_bci/sofia/launch_d568.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d568.sh tracks/t2_bci/sofia/arms/t2d568_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d568_aug_*.txt | wc -l)" -eq 16 ] || { echo "need 16 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D568 one-arm stronger augmentation dose (gain 0.3, +-45, ch-drop 0.15, mixup 1.0) on the five headed FM lines at C's seeds 837-841, read on C vs D557's dose (stack bar); W21 part only on C + B"
for f in tracks/t2_bci/sofia/arms/t2d568_aug_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d568_}
  w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    --job-name="ni26-t2p-d568-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

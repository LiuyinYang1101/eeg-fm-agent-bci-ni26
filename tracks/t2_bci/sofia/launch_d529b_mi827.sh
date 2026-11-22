#!/usr/bin/env bash
# T2-D529 rule (b) (registered 2026-11-16 before the s821 members landed; PREPARATION): the MI-expert ST-L member at the FRESH seed 827 (the second MI
# member on fresh draws, swapped for the plain s827 member), stage 2 from D529's stage-1 checkpoint, fold halves. Same line as launch_d532_mi.sh's s826.
#   mem-guard bash tracks/t2_bci/sofia/launch_d529b_mi827.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d529b_mi827.sh tracks/t2_bci/sofia/arms/t2d529b_mi_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D529 rule (b): the second MI-expert ST-L member at fresh seed 827 (swap for the plain s827 member) in the W18 fresh-draw read; decides one vs two MI members in upload B"
for f in tracks/t2_bci/sofia/arms/t2d529b_mi_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d529b_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 \
    --job-name="ni26-t2p-d529b-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

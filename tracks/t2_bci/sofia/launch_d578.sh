#!/usr/bin/env bash
# T2-D578 (the fresh confirmation set D, draws 834-836; registered 2026-11-18 before any job; inside GO): W20+D557's six members at NEW seeds with
# the side + feature dumps = D558's head-less MI arms (seed 842 -> 848) + D558's five augmented headed lines (837-841 -> 843-847). The code and
# arms = D558's (smoke-tested lines; only the seeds differ), so no new smoke. 21 jobs.
#   mem-guard bash tracks/t2_bci/sofia/launch_d578.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d578.sh tracks/t2_bci/sofia/arms/t2d578_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d578_*.txt | wc -l)" -eq 21 ] || { echo "need 21 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D578 fresh confirmation set D (W20+D557's members at seeds 843-848, draws 834-836): read once for W21's LP gain + pre-declared W22 parts"
for f in tracks/t2_bci/sofia/arms/t2d578_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d578_}
  w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    --job-name="ni26-t2p-d578-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

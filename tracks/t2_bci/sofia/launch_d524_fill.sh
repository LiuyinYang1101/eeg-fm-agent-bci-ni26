#!/usr/bin/env bash
# T2-D524 fold fill (2026-11-16): 1620616 / 17 / 18 hit their 2:00 walls at 4 / 5 folds (about 1,605 s per fold), so the
# missing fold of each runs alone with a 0:45 wall. Job names equal the originals', so t2_d524_read.py's glob finds them.
#   mem-guard bash tracks/t2_bci/sofia/launch_d524_fill.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d524_fill.sh tracks/t2_bci/sofia/arms/t2d524f_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D524 window means into the FM members, the missing fold of a job that hit its wall; feeds T2 upload B (needs its own owner line)"
for f in tracks/t2_bci/sofia/arms/t2d524f_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d524f_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; fold fill $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:45:00 \
    --job-name="ni26-t2p-d524-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

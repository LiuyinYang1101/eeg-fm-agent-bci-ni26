#!/usr/bin/env bash
# T2-D554 kill-read (registered 2026-11-17 before any number; inside GO): the speech-recipe family (specnet) on lc9, seeds 822 / 823 / 824, pooled
# members only, one H200 job per seed (3 shards), walls 1:30.
#   mem-guard bash tracks/t2_bci/sofia/launch_d554.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d554.sh tracks/t2_bci/sofia/arms/t2d554_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D554 the owner's speech-recipe bet: specnet (log-spectrogram 2-D CNN + SpecAugment) kill-read on lc9; feeds a possible W20 family (owner's template question only on a pass)"
for f in tracks/t2_bci/sofia/arms/t2d554_spec_s82{2,3,4}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d554_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=3" S1_PURPOSE="$P; $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=01:30:00 \
    --job-name="ni26-t2p-d554-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

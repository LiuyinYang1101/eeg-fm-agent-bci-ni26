#!/usr/bin/env bash
# T2-D521 (the raw end-to-end net; registered 2026-11-15 before any number; inside GO and the s1 caps): RawNet on the raw window (keep-mean) and on the de-meaned window, K = 9 (lc9), 15 seeds, fold halves (4 jobs x 1 H200).
#   mem-guard bash tracks/t2_bci/sofia/launch_d521.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d521.sh tracks/t2_bci/sofia/arms/t2d521_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D521 raw end-to-end net (learned compression + 0.1-1 s learned temporal bank + level branch; raw and de-meaned arms) in the side slot of FINAL+DC, K = 9 strangers; feeds T2 upload B (needs its own owner line)"
for f in tracks/t2_bci/sofia/arms/t2d521_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d521_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:00:00 \
    --job-name="ni26-t2p-d521-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

#!/usr/bin/env bash
# T2-D520 (the grid head as an ADDED FM member; registered 2026-11-15 before any number; inside GO and the s1 caps): G = grid64d ST-L (seeds 830-832) and C7 = a 7th mean ST-L control (833-835), tilt 0.15, K = 9 (lc9), each split in two fold halves (12 jobs x 1 H200).
#   mem-guard bash tracks/t2_bci/sofia/launch_d520.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d520.sh tracks/t2_bci/sofia/arms/t2d520_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D520 grid (all-token, position-preserving) head as an ADDED FM member vs a 7th plain ST-L control, K = 9 strangers in FINAL+DC+side; feeds T2 upload B (FM members rebuilt on a pass)"
for f in tracks/t2_bci/sofia/arms/t2d520_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d520_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=05:00:00 \
    --job-name="ni26-t2p-d520-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

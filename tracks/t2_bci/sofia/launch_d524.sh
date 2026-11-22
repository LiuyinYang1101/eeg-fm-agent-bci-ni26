#!/usr/bin/env bash
# T2-D524 (the window means INTO the FM members; registered 2026-11-15 before any number; inside GO and the s1 caps): D512's six tilt ST-L (seeds 820-825) with the dcmean side head (sd) and retrained without it (ctl), K = 9 (lc9), fold halves (24 jobs x 1 H200).
#   mem-guard bash tracks/t2_bci/sofia/launch_d524.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d524.sh tracks/t2_bci/sofia/arms/t2d524_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D524 window means into the FM members (dcmean side head on the six tilt ST-L vs a retrained control), K = 9 strangers in the W17 stack; feeds T2 upload B (needs its own owner line)"
for f in tracks/t2_bci/sofia/arms/t2d524_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d524_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=05:00:00 \
    --job-name="ni26-t2p-d524-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

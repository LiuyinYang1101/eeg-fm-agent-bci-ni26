#!/usr/bin/env bash
# T2-D531 (W18 fresh-draw FM members, PREPARATION; registered 2026-11-16 before any number; inside GO and the s1 caps): D524's six lines at FRESH seeds 826-831 (plain 826-828, CAR 829-831), with the side head (sd) and without (ctl), K = 9 (lc9), fold halves (24 jobs x 1 H200, walls 2:30 = the measured 1:15 + margin).
#   mem-guard bash tracks/t2_bci/sofia/launch_d531.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d531.sh tracks/t2_bci/sofia/arms/t2d531_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D531 W18 fresh-draw FM members (D524 lines at fresh seeds 826-831, side head and control), K = 9; feeds the W18 stack read (D524 / D525 / D529 / D530 parts) for T2 upload B"
for f in tracks/t2_bci/sofia/arms/t2d531_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d531_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 \
    --job-name="ni26-t2p-d531-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

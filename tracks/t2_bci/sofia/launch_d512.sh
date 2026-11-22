#!/usr/bin/env bash
# T2-D512 (the final-week stack row; registered 2026-11-15 before any number; inside GO and the s1 caps): the FM members on K = 9 (lc9) with
# fresh seeds: 3 plain (820-822) + 3 CAR (823-825) ST-L, each with and without the spectral tilt 0.15 (part P2), each split in two fold halves
# (24 jobs x 1 H200; D498's line ran 10 folds in one job; halves halve the wall time). No smoke: D498's exact line plus D442's tilt flag, both
# already run on sofia; each job < 16 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d512.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d512.sh tracks/t2_bci/sofia/arms/t2d512_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D512 final-week stack row (fresh-seed W14-lite base + parts, leave-one-part-out): FM members, control and tilt (P2); feeds the Nov 20 final stack"
for f in tracks/t2_bci/sofia/arms/t2d512_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d512_}
  env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=05:00:00 \
    --job-name="ni26-t2p-d512-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

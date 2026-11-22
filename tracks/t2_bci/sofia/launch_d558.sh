#!/usr/bin/env bash
# T2-D558 (the sprint's second FM-seed set for D555 / D557; registered 2026-11-17 before any number; inside GO): every W19 FM line re-run at
# NEW seeds (SD 837 plain / 838, 839 CAR; D546's base + cb 840 plain / 841 CAR; MI 842) with the side + feature dumps (the D555-2 REF / ARM
# and D557-2 REF), plus the five headed lines with the D557 augmentation at the same new seeds (D557-2 ARM). The code and arms = D555's /
# D557's (both smoke-tested), only the seeds differ, so no new smoke. 37 jobs.
#   mem-guard bash tracks/t2_bci/sofia/launch_d558.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d558.sh tracks/t2_bci/sofia/arms/t2d558_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D558 second FM-seed set (seeds 837-842) for the D555 / D557 replication reads on fresh draws 831-833 (W20 parts need both reads); FM fine-tunes before the Nov 18 12:00 cut-off"
for f in tracks/t2_bci/sofia/arms/t2d558_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d558_}
  w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    --job-name="ni26-t2p-d558-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

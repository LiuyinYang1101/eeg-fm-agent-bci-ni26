#!/usr/bin/env bash
# T2-D582 (vp 2 on top of D557's dose, the five headed FM lines; registered 2026-11-18 before any job; inside GO): C (16, nice 0) + B (32 = REF
# re-run with dumps + ARM) and D (16) PRE-LAUNCHED at nice 100 under the row's rule (no B / D output opened before a C pass; on a C fail every
# B / D job cancelled and the dirs deleted unread). The arms = D557 / D558 / D578's lines + --fm-vp 2 (the REF unchanged); no new smoke.
#   mem-guard bash tracks/t2_bci/sofia/launch_d582.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d582.sh tracks/t2_bci/sofia/arms/t2d582_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d582_*.txt | wc -l)" -eq 64 ] || { echo "need 64 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
for part in c bref barm d; do
  nice=100; [ $part = c ] && nice=0
  P="T2-D582 vp 2 on D557's dose ($part; $([ $part = c ] && echo "the C stage, read first" || echo "pre-launched at nice 100: read only after a C pass, cancelled + deleted unread on a C fail"))"
  for f in tracks/t2_bci/sofia/arms/t2d582_${part}_*.txt; do
    a=$(basename "$f" .txt); a=${a#t2d582_}
    w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
    env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
      --nice=$nice --job-name="ni26-t2p-d582-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
    [ "$mode" == dry ] && break
  done
  [ "$mode" == dry ] && break
done

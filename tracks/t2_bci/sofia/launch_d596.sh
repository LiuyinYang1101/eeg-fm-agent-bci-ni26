#!/usr/bin/env bash
# T2-D596 (the WORD-margin loss in the FM fine-tune: L += 0.3 x mean softplus(0.5 - (z_y - z_c)) over the WORD / MI / SUB confuser pairs, ramped
# over the first 10 % of the steps; registered 2026-11-18 before any code or number): the five headed FM lines on D557's dose.
#   smoke: ONE 1 + 1-epoch job on lc9_0 (t2d596smoke.txt: the margin term, its same-class share and the logits-identity line logged)
#   c:     the 16 C jobs (t2d558_aug_* + --fm-wmargin 0.3), nice 20 (the manager: D583 / D586 / D588's reads stay ahead)
#   bd:    the 16 B ARM (t2d557_* + the flag) + 16 D (t2d578_aug_* + the flag) PRE-LAUNCHED at nice 100 (T2-D590's rules: a C kill cancels and
#          deletes them unread; B's REF = D582's bref)
# Walltimes: single view, D583's (which fitted D558's lines on the H200 inside 02:30 / 02:00) + margin: SD 03:00, base + cb 02:30.
#   mem-guard bash tracks/t2_bci/sofia/launch_d596.sh smoke|c|bd dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
part=${1:?smoke|c|bd}; mode=${2:?dry|GO}
[[ "$part" == smoke || "$part" == c || "$part" == bd ]] && [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d596.sh tracks/t2_bci/sofia/arms/t2d596*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d596_*.txt | wc -l)" -eq 48 ] || { echo "need 48 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="T2-D596 SMOKE: 1 + 1 epochs on lc9_0 with --fm-wmargin 0.3 (the margin term vs CE, its same-class share) before the C stage" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:40:00 --job-name=ni26-t2p-d596-smoke -- tracks/t2_bci/sofia/arms/t2d596smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
pts=c; [ "$part" == bd ] && pts="barm d"
for pt in $pts; do
  nice=100; [ $pt = c ] && nice=20
  P="T2-D596 WORD-margin loss (0.3, m 0.5) on D557's dose ($pt; $([ $pt = c ] && echo "the C stage, read first under T2-D590" || echo "pre-launched at nice 100: read only after a C continue, cancelled + deleted unread on a C kill"))"
  for f in tracks/t2_bci/sofia/arms/t2d596_${pt}_*.txt; do
    a=$(basename "$f" .txt); a=${a#t2d596_}
    w=02:30:00; [[ "$a" == *stl* ]] && w=03:00:00
    env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
      --nice=$nice --job-name="ni26-t2p-d596-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
    [ "$mode" == dry ] && break
  done
  [ "$mode" == dry ] && break
done

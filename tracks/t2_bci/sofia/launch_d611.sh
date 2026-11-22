#!/usr/bin/env bash
# T2-D611 (ONE 8-30 Hz band-pass specialist FM member: the headed CAR SD slot 831; registered 35e9432b before any job; inside GO):
#   smoke: ONE 1-epoch job on lc9_0 (t2d611smoke.txt: the --fm-band path on the GPU with the side / feature dumps)
#   c:     the 2 C jobs (D558's aug line at 839 + --fm-band 8:30, fold halves a / b), nice 10
#   bd:    B ARM (2: D582's bref line at 831 + the flag; the B REF = D582's bref jobs, no new REF) + D (2: D578's line at 845 + the flag),
#          PRE-LAUNCHED at nice 100 (no B / D output opened before a C pass; on a C KILL / CB FAIL cancelled and the dirs deleted unread)
# Walltime: D558's aug 5-fold SD jobs ran in ~8,000 s on the H200 (D588's measurement); one view here -> 03:30:00 (~1.6x, margin).
#   mem-guard bash tracks/t2_bci/sofia/launch_d611.sh smoke|c|bd dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
part=${1:?smoke|c|bd}; mode=${2:?dry|GO}
[[ "$part" == smoke || "$part" == c || "$part" == bd ]] && [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d611.sh tracks/t2_bci/sofia/arms/t2d611*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d611_*.txt | wc -l)" -eq 6 ] || { echo "need 6 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="T2-D611 SMOKE: 1 epoch on lc9_0 with --fm-band 8:30 (the band-pass path on the GPU, the dumps) before the C stage" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:40:00 --job-name=ni26-t2p-d611-smoke -- tracks/t2_bci/sofia/arms/t2d611smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
parts=(c); [ "$part" == bd ] && parts=(barm d)
for pt in "${parts[@]}"; do
  nice=100; [ $pt = c ] && nice=10
  P="T2-D611 one 8-30 Hz band-pass specialist FM member (slot 831, D557's dose + --fm-band 8:30) ($pt; $([ $pt = c ] && echo "the C stage, read first under T2-D590" || echo "pre-launched at nice 100: read only after a C pass, cancelled + deleted unread on a C KILL / CB FAIL"))"
  for f in tracks/t2_bci/sofia/arms/t2d611_${pt}_*.txt; do
    a=$(basename "$f" .txt); a=${a#t2d611_}
    env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:30:00 \
      --nice=$nice --job-name="ni26-t2p-d611-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
    [ "$mode" == dry ] && break
  done
  [ "$mode" == dry ] && break
done

#!/usr/bin/env bash
# T2-D583 (within-class segment recombination on top of D557's dose, the five headed FM lines; registered 2026-11-18 before any number):
#   smoke: ONE 1 + 1-epoch job on lc9_0 (t2d583smoke.txt; the recombination count per epoch in its log) -> read before C
#   c:     the 16 C jobs (D558's aug arms at seeds 837-841 + --fm-recomb 4:0.5), nice 0; B / D only on a C pass (not pre-launched)
#   bd:    ON A C PASS ONLY (the night plan 2026-11-18): the 16 B ARM jobs (t2d583_barm_*, nice 0; B's REF = D582's bref) + the 16 D jobs
#          (t2d583_d_*, nice 100; cancelled + deleted unread on a B fail). Guard: t2_d583_reads.log must hold the C PASS line.
#   mem-guard bash tracks/t2_bci/sofia/launch_d583.sh smoke|c|bd dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
part=${1:?smoke|c}; mode=${2:?dry|GO}
[[ "$part" == smoke || "$part" == c || "$part" == bd ]] && [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d583.sh tracks/t2_bci/sofia/arms/t2d583*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
if [ "$part" == bd ]; then
  [ "$mode" == dry ] || grep -q "^### POOL C CONTINUE" "$HOME/workspace/outputs/neural_interface_challenge/t2_d583_v2.log" || { echo "bd needs D583's C CONTINUE line (T2-D590)"; exit 2; }
  [ "$(ls tracks/t2_bci/sofia/arms/t2d583_barm_*.txt | wc -l)" -eq 16 ] && [ "$(ls tracks/t2_bci/sofia/arms/t2d583_d_*.txt | wc -l)" -eq 16 ] || { echo "need 16 + 16 B / D arms"; exit 2; }
  for sub in barm d; do
    nice=0; [ $sub = d ] && nice=100
    P="T2-D583 segment recombination, $([ $sub = barm ] && echo "B ARM (read after a C pass; REF = D582's bref)" || echo "D (nice 100; read once after a B pass, cancelled + deleted unread on a B fail)")"
    for f in tracks/t2_bci/sofia/arms/t2d583_${sub}_*.txt; do
      a=$(basename "$f" .txt); a=${a#t2d583_}
      w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
      env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
        --nice=$nice --job-name="ni26-t2p-d583-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
      [ "$mode" == dry ] && break
    done
  done
  exit 0
fi
if [ "$part" == smoke ]; then
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="T2-D583 SMOKE: 1 + 1 epochs on lc9_0 with --fm-recomb 4:0.5 (the recombination count logged) before the C stage" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:40:00 --job-name=ni26-t2p-d583-smoke -- tracks/t2_bci/sofia/arms/t2d583smoke.txt 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  exit 0
fi
[ "$(ls tracks/t2_bci/sofia/arms/t2d583_c_*.txt | wc -l)" -eq 16 ] || { echo "need 16 C arms"; exit 2; }
P="T2-D583 within-class segment recombination on D557's dose (C, seeds 837-841), read on C at the stack bar; B / D only on a C pass"
for f in tracks/t2_bci/sofia/arms/t2d583_c_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d583_}
  w=02:00:00; [[ "$a" == *stl* ]] && w=02:30:00
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$w \
    --job-name="ni26-t2p-d583-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done

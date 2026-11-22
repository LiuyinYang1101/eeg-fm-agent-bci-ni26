#!/usr/bin/env bash
# T2-D586 (the shared-phase Fourier surrogate on the plain SD line only; registered 2026-11-18 before any code or number): the 2 C jobs
# (t2d586_c_stlsd_s837_a / b = D558's k twin of seed 837 + --fm-phase 0.25:0.7854), read as a one-member replacement on set C; B / D only on
# a C pass (not pre-launched).
#   bd (ON A C PASS ONLY, the night plan 2026-11-18): the 2 B ARM jobs (t2d586_barm_*, nice 0; B's REF = D582's bref) + the 2 D jobs (t2d586_d_*,
#   nice 100; cancelled + deleted unread on a B fail). Guard: t2_d586_reads.log must hold the C PASS line.
#   mem-guard bash tracks/t2_bci/sofia/launch_d586.sh dry|GO [c|bd]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; part=${2:-c}
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$part" == c || "$part" == bd ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d586.sh tracks/t2_bci/sofia/arms/t2d586*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d586_c_*.txt | wc -l)" -eq 2 ] && [ "$(ls tracks/t2_bci/sofia/arms/t2d586_barm_*.txt tracks/t2_bci/sofia/arms/t2d586_d_*.txt | wc -l)" -eq 4 ] || { echo "need 2 C + 2 B + 2 D arms"; exit 2; }
[ "$part" == c ] || [ "$mode" == dry ] || grep -q "^### POOL C CONTINUE" "$HOME/workspace/outputs/neural_interface_challenge/t2_d586_v2.log" || { echo "bd needs D586's C CONTINUE line (T2-D590)"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="T2-D586 shared-phase Fourier surrogate (p 0.25, +-pi/4) on the plain SD line 837 only (C), read as a one-member replacement at the stack bar; B / D only on a C pass"
subs=c; [ "$part" == bd ] && subs="barm d"
for sub in $subs; do
for f in tracks/t2_bci/sofia/arms/t2d586_${sub}_*.txt; do
  a=$(basename "$f" .txt); a=${a#t2d586_}
  nice=0; [ $sub = d ] && nice=100
  [ $sub = c ] || P="T2-D586 shared-phase surrogate, $([ $sub = barm ] && echo "B ARM (read after a C pass; REF = D582's bref)" || echo "D (nice 100; read once after a B pass, cancelled + deleted unread on a B fail)")"
  env "${dry[@]}" S1_EXPORT="NSHARDS=1,NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; members $a" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=02:30:00 \
    --nice=$nice --job-name="ni26-t2p-d586-$a" -- "$f" 2>&1 | grep -E "SUBMITTED|DRY RUN|refus|rror" | cut -c1-150
  [ "$mode" == dry ] && break
done
done

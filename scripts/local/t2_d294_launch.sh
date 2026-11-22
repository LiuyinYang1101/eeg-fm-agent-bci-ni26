#!/usr/bin/env bash
# T2-D294 launch (backlog 69, FM -> CNN distillation): GPU only on the user's reply to the final improvement list (PLAN §6b item 11.3).
# One `s1 sync ni26` (git archive HEAD: commit the D294 code first), then the teacher job (RUN_NAME fixed: the student arms read its dumps at
# /sofia/.../runs/ni26/ni26-t2p-d294-teacher/runs/kdteacher__<protocol>__s0.npz) and three student jobs after it (afterok).
# Read: `s1 fetch ni26 ni26-t2p-d294-<arm>-<id>` x 3 (+ the teacher's small files), then scripts/local/t2_d294_read.sh.
#   bash scripts/local/t2_d294_launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d294_*.txt)" ] || { echo "commit the D294 code first"; exit 2; }
s1 sync ni26
out=$(S1_EXPORT=RUN_NAME=ni26-t2p-d294-teacher,NSHARDS=1 S1_PURPOSE="T2-D294 distillation teachers (adopted ST-L, seed 0, kit + seen1) + dumps; feeds the iteration builds (PLAN 6b item 11)" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=03:00:00 --job-name=ni26-t2p-d294-teacher -- tracks/t2_bci/sofia/arms/t2d294_teacher.txt 2>&1)
echo "$out" | grep -E "SUBMITTED|refus|error|ERROR" || { echo "$out"; exit 2; }
job=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$job" ] || exit 2
for arm in fbc_kit fbc_seen1 sh; do
  S1_EXPORT=NSHARDS=3 S1_PURPOSE="T2-D294 distillation students: arms t2d294_$arm (kd / kdx) after teacher $job; feeds the iteration builds (PLAN 6b item 11)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --dependency=afterok:$job --time=03:00:00 --job-name=ni26-t2p-d294-${arm//_/-} \
    -- tracks/t2_bci/sofia/arms/t2d294_$arm.txt 2>&1 | grep -E "SUBMITTED|refus|error|ERROR"
done

#!/usr/bin/env bash
# T2-D493: ST-L 5-soup vs the ensemble of the same five nets (the manager's 'buy the budget back' bet). smoke = one 1-epoch soup task
# (kit, seed 730) that must write main + soup5 + soup5m1..m4; GO (needs S1_SMOKE=<the COMPLETED smoke id>) = 12 one-task H200 jobs
# (kit / seen1 x seeds 730-735; each task = 5 x 100-epoch fine-tunes ≈ 2.2 h), each afterok on the smoke.
#   mem-guard bash tracks/t2_bci/sofia/launch_d493.sh smoke|dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?smoke|dry|GO}; [[ "$mode" == smoke || "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d493.sh tracks/t2_bci/sofia/arms/t2d493_*.txt)" ] || { echo "commit first"; exit 2; }
P="T2-D493 (ST-L 5-soup vs the ensemble of the same 5 nets: frees ~17 ms/window for different-in-kind members if it holds): feeds T2's upload C / framework bet"
sub() { local out; out=$(env "$@" 2>&1); echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-160; }
[ "$mode" != dry ] && s1 sync ni26 >/dev/null 2>&1
if [ "$mode" == smoke ]; then
  sub S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; smoke (1 epoch)" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=00:45:00 \
    --job-name="ni26-t2p-d493-smoke" -- tracks/t2_bci/sofia/arms/t2d493_smoke.txt
  exit 0
fi
[ -n "${S1_SMOKE:-}" ] || { echo "set S1_SMOKE to the COMPLETED smoke job id"; exit 3; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
for pr in kit seen1; do
  for sd in 730 731 732 733 734 735; do
    sub "${dry[@]}" S1_SMOKE="$S1_SMOKE" S1_EXPORT="NSHARDS=1" S1_PURPOSE="$P; $pr seed $sd" s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm \
      --time=04:00:00 --dependency="afterok:$S1_SMOKE" --job-name="ni26-t2p-d493-$pr-s$sd" -- tracks/t2_bci/sofia/arms/t2d493_${pr}_s$sd.txt
  done
done

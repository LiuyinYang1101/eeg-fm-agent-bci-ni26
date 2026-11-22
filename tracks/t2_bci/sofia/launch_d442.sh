#!/usr/bin/env bash
# T2-D442 (night plan 2026-11-10; R3 standing approval, GO 1,200): the paired FRESH-seed read of the spectral tilt on M7's five ST-L members.
# 12 one-GPU jobs (they backfill): M7's two ST-L member recipes (plain 100 epochs; lp10 at 30) x control / tilt 0.15 x kit / seen1, seeds 400-411;
# the 100-epoch legs split in two halves of 6 seeds. The controls are also T2-D443's fresh current-recipe members.
#   bash tracks/t2_bci/sofia/launch_d442.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d442.sh tracks/t2_bci/sofia/arms/t2d442_*.txt scripts/local/t2_d442_read.py)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null
for r in e100 e100tl lp10 lp10tl; do
  case $r in e100*) halves="a b"; t=04:00:00;; *) halves="x"; t=03:00:00;; esac
  for h in $halves; do
    for p in kit seen1; do
      a=tracks/t2_bci/sofia/arms/t2d442_${r}_${h}_$p.txt; jn=ni26-t2p-d442-$r-$h-$p
      out=$(env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="T2-D442 (slot-5 menu, GO 1,200): ST-L member $r $h $p, fresh seeds 400-411; paired read of the tilt on M7's ST-L slot = the M8 menu row" \
        s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=$t --job-name="$jn" -- "$a" 2>&1)
      echo "$jn: $(echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|error|ERROR" | cut -c1-160 | tr '\n' ' ')"
    done
  done
done

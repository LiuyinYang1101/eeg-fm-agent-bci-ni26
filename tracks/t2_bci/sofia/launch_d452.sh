#!/usr/bin/env bash
# T2-D452 (R3; GO 1,200): REVE-base and LaBraM-base as T2 members under the CURRENT fine-tune recipe (they were last read on Oct 13, before
# rc / LLRD / mixup / 100 epochs). 6 one-GPU jobs, fresh seeds 424-435.   bash tracks/t2_bci/sofia/launch_d452.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; [[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_fm.py tracks/t2_bci/sofia/t2p_multi.slurm tracks/t2_bci/sofia/launch_d452.sh tracks/t2_bci/sofia/arms/t2d452_*.txt scripts/local/t2_d452_read.py)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26 >/dev/null 2>&1
for r in reve_none reve_sz labram_sz; do for p in kit seen1; do
  jn=ni26-t2p-d452-${r//_/-}-$p
  out=$(env "${dry[@]}" S1_EXPORT="NSHARDS=1" S1_PURPOSE="T2-D452: $r $p, a different pretrained member under the current fine-tune recipe, seeds 424-435; feeds T2's third sealed upload (a sixth FM member)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm --time=06:00:00 --job-name="$jn" -- tracks/t2_bci/sofia/arms/t2d452_${r}_$p.txt 2>&1)
  echo "$jn: $(echo "$out" | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-150 | tr '\n' ' ')"
done; done

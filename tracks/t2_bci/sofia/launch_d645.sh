#!/usr/bin/env bash
# T2-D645 (FM-DRO, Track3's registration; TRIGGERED by D642's 6 / 6): the DRO member screen on s1_stlsd: arm R = D642's B
# line + --fm-dro 0.003 (arms/t2d645_s1_stlsd_R.txt), paired with D642's B (seed 901); ONE array
# (6 tasks = fold pairs, 1 H200 each, 2:30); reserved 15 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d645.sh dry|GO <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d645.sh tracks/t2_bci/sofia/arms/t2d645_*.txt)" ] || { echo "commit first"; exit 2; }
ARM=R
[ "$(ls tracks/t2_bci/sofia/arms/t2d645_s*_$ARM.txt | wc -l)" -eq 1 ] || { echo "need 1 R arm file"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-19: target first place, all resources; T2-D645 FM-DRO (Track3, triggered by D642), member screen R vs D642 B"
for f in tracks/t2_bci/sofia/arms/t2d645_s*_$ARM.txt; do
  a=$(basename "$f" .txt); a=${a#t2d645_}
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d645-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

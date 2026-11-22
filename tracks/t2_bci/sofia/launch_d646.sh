#!/usr/bin/env bash
# T2-D646 (the third independent ALL6 read, a veto check on W25-ALL6; registered before any number): arms X = D642's B lines and Y = D642's C
# lines at FRESH seeds 1000 + slot, all six slots (arms/t2d646_<slot>_<X|Y>.txt; only --seeds changed); one Slurm ARRAY per (slot, arm) = 6 fold
# pairs, 1 H200 each, D642's walls (3:00 MI / cont, 2:30 others); S1_SMOKE / S1_PURPOSE as launch_d642a.sh. Reserved <= 12 x 6 x 3 h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d646.sh dry|GO <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d646.sh tracks/t2_bci/sofia/arms/t2d646_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d646_s*_[XY].txt | wc -l)" -eq 12 ] || { echo "need 12 D646 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-19: target first place, all resources; T2-D646 the third independent ALL6 read (a veto check on W25-ALL6 / USER_LINES item 167)"
for f in tracks/t2_bci/sofia/arms/t2d646_s*_[XY].txt; do
  a=$(basename "$f" .txt); a=${a#t2d646_}
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d646-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

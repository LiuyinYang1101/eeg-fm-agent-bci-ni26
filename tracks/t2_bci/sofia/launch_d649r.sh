#!/usr/bin/env bash
# T2-D649 (W29) the SPECULATIVE REPLICATE members (the manager 2026-11-20 14:3x GO): C + the engagement-picked G at 1300 + slot (pinned seeds),
# all six slots, ss8 + rt (arms/t2d649r_<slot>_<C|G..>.txt = the screen's arm files with the seed 1200 + slot -> 1300 + slot), launched AT THE PICK
# (≈ 15:45) without waiting for the screen read; cancelled + deleted UNREAD if the screen kills. 12 arrays x 6 tasks, reserved <= 216 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d649r.sh dry|GO <completed smoke job id> <G01|G03|G1 = the picked arm>   [SKIP=<regex>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; G=${3:?the picked arm G01|G03|G1}; [[ "$mode" =~ ^(dry|GO)$ && "$G" =~ ^(G01|G03|G1)$ ]] || exit 2
pk=$HOME/workspace/outputs/neural_interface_challenge/t2_d649/lambda_pick.json
want=$("$HOME/miniconda3/envs/neural_interface_challenge/bin/python" -c "import json; v = json.load(open('$pk'))['lambda_max']; print({0.1: 'G01', 0.3: 'G03', 1.0: 'G1'}.get(v, 'NONE'))" 2>/dev/null || echo NOPICK)
[ "$mode" == dry ] || [ "$want" == "$G" ] || { echo "REFUSED: the engagement pick says $want, not $G"; exit 3; }
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d649r.sh tracks/t2_bci/sofia/arms/t2d649r_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D649 (W29 GRL) the speculative REPLICATE members C + $G at 1300 + slot (the manager GO); cancelled unread if the screen kills"
for f in tracks/t2_bci/sofia/arms/t2d649r_s*_{C,$G}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d649r_}
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); [ "$n" -eq 6 ] || { echo "!! $f: $n lines (need 6)"; exit 2; }
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "d649r-$a" =~ ${SKIP} ]] && { echo "skip d649r-$a (SKIP=$SKIP)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d649r-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! d649r $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

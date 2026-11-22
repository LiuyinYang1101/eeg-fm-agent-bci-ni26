#!/usr/bin/env bash
# T2-D649 (W29) the rtl arm (the manager 2026-11-20 15:2x: W29 x W30 rule, W29's own dRL): G_picked on rtl at 1400 + slot = D651's C line + --fm-grl,
# all six slots (arms/t2d649rl_<slot>_<G..>.txt), paired with D651's C (no new C fit), launched AT THE PICK
# (≈ 15:45) without waiting for the screen read; cancelled + deleted UNREAD if the screen kills. 6 arrays x 4 tasks, reserved <= 72 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d649rl.sh dry|GO <completed smoke job id> <G01|G03|G1 = the picked arm>   [SKIP=<regex>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; G=${3:?the picked arm G01|G03|G1}; [[ "$mode" =~ ^(dry|GO)$ && "$G" =~ ^(G01|G03|G1)$ ]] || exit 2
pk=$HOME/workspace/outputs/neural_interface_challenge/t2_d649/lambda_pick.json
want=$("$HOME/miniconda3/envs/neural_interface_challenge/bin/python" -c "import json; v = json.load(open('$pk'))['lambda_max']; print({0.1: 'G01', 0.3: 'G03', 1.0: 'G1'}.get(v, 'NONE'))" 2>/dev/null || echo NOPICK)
[ "$mode" == dry ] || [ "$want" == "$G" ] || { echo "REFUSED: the engagement pick says $want, not $G"; exit 3; }
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d649rl.sh tracks/t2_bci/sofia/arms/t2d649rl_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D649 (W29 GRL) the rtl arm $G at 1400 + slot vs D651 C (the manager GO, W29 x W30 rule); cancelled unread if the screen kills"
for f in tracks/t2_bci/sofia/arms/t2d649rl_s*_$G.txt; do
  a=$(basename "$f" .txt); a=${a#t2d649rl_}
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); [ "$n" -eq 4 ] || { echo "!! $f: $n lines (need 4)"; exit 2; }
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "d649rl-$a" =~ ${SKIP} ]] && { echo "skip d649rl-$a (SKIP=$SKIP)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-3 --time=$w --job-name="ni26-t2p-d649rl-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! d649rl $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

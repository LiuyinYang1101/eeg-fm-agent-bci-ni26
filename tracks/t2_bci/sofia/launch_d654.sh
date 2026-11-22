#!/usr/bin/env bash
# T2-D654 (FM view-TTA, Track3's pins adopted by the manager 15:33:36) the VALUE RUN: the C lines (D642's C) at FRESH seeds, all six slots, with
# S1_EXPORT NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1,NI26_VIEW_DUMP=4 (the fixed views v1-v3 dumped; the fits unchanged = identity vs views paired in a fit):
#   set 1600: ship (arms/t2d654_<slot>_C1600.txt, 6 fold-pair tasks) + rtl (arms/t2d654rl_<slot>_C1600.txt, 4 tasks) -> 12 arrays, <= 180 GPU-h;
#   set 1700: ship only (arms/t2d654_<slot>_C1700.txt) -> 6 arrays, <= 108 GPU-h.
# Job names (Track3's viewmix --pre): ni26-t2p-d654a-<slot>_C (1600 ship), ni26-t2p-d654rl-<slot>_C (1600 rtl), ni26-t2p-d654b-<slot>_C (1700 ship).
#   mem-guard bash tracks/t2_bci/sofia/launch_d654.sh dry|GO <completed smoke job id> 1600|1700   [SKIP=<regex on <pre>-<slot>_C>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; SET=${3:?1600|1700}; [[ "$mode" =~ ^(dry|GO)$ && "$SET" =~ ^(1600|1700)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d654.sh tracks/t2_bci/sofia/arms/t2d654*.txt)" ] || { echo "commit first"; exit 2; }
FS=(tracks/t2_bci/sofia/arms/t2d654_s*_C$SET.txt); [ "$SET" == 1600 ] && FS+=(tracks/t2_bci/sofia/arms/t2d654rl_s*_C1600.txt)
[ "${#FS[@]}" -eq $([ "$SET" == 1600 ] && echo 12 || echo 6) ] || { echo "need the D654 arm files of set $SET"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D654 (FM view-TTA) the value run, C lines at $SET + slot with NI26_VIEW_DUMP=4 (the manager GO)"
for f in "${FS[@]}"; do
  a=$(basename "$f" .txt); k=${a%%_*}; a=${a#"$k"_}; a=${a%"$SET"}          # a = <slot>_C (Track3's viewmix glob ni26-t2p-<pre>-<slot>_C-*)
  pre=d654a; [ "$SET" == 1700 ] && pre=d654b; [ "$k" == t2d654rl ] && pre=d654rl   # d654a = 1600 ship, d654rl = 1600 rtl, d654b = 1700 ship
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); arr="0-$((n - 1))"
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "$pre-$a" =~ ${SKIP} ]] && { echo "skip $pre-$a (SKIP=$SKIP)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1,NI26_VIEW_DUMP=4" S1_PURPOSE="$P; read $pre-$a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=$arr --time=$w --job-name="ni26-t2p-$pre-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $pre $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

#!/usr/bin/env bash
# T2-D652 (W30) the SECOND-read V members, SPECULATIVE (the manager 2026-11-20 14:3x GO): V = D648's W line (D642's C at 1100 + slot) + --fm-vp 2,
# the five headed slots, ss8 + rt (arms/t2d652b_<slot>_V.txt), paired with D648's EXISTING W members for W30's second full-chain read; cancelled +
# deleted UNREAD if gate 1 (D651 pooled C - B < 0) or W30's member gate fails. 5 arrays x 6 tasks, reserved <= 90 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d652b.sh dry|GO <completed smoke job id>   [SKIP=<regex>] [PURPOSE=<text>] [XARGS="--dependency=after:<id> --nice=<n>"]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d652b.sh tracks/t2_bci/sofia/arms/t2d652b_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d652b_s*_V.txt | wc -l)" -eq 5 ] || { echo "need 5 D652b arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D652 (W30 vp 2) the speculative second-read V members at 1100 + slot vs D648's W (the manager GO); cancelled unread on a gate fail"
[ -n "${PURPOSE:-}" ] && P=$PURPOSE   # 2026-11-21: the reopened information read names itself (T2-D652)
for f in tracks/t2_bci/sofia/arms/t2d652b_s*_V.txt; do
  a=$(basename "$f" .txt); a=${a#t2d652b_}
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); [ "$n" -eq 6 ] || { echo "!! $f: $n lines (need 6)"; exit 2; }
  w=02:30:00; [[ "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "d652b-$a" =~ ${SKIP} ]] && { echo "skip d652b-$a (SKIP=$SKIP)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w ${XARGS:-} --job-name="ni26-t2p-d652b-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! d652b $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

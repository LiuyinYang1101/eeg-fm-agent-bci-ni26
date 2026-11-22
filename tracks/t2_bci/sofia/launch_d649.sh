#!/usr/bin/env bash
# T2-D649 (W29, the FM subject-adversarial GRL) MEMBER SCREEN, launched IN PARALLEL with the lambda smoke (the manager 2026-11-20 ≈ 14:2x: buy back
# the smoke's 2.5 h with compute): per slot (all six) arm C (D642's C line) and arms G01 / G03 / G1 (C + --fm-grl 0.1 / 0.3 / 1) at the FRESH seeds
# 1200 + slot, ss8 + rt (arms/t2d649_<slot>_<C|G01|G03|G1>.txt; 6 fold-pair tasks each). ONLY the G arm at the lambda max the smoke picks by
# ENGAGEMENT (rule 3, scripts/local/t2_d649_lambda_pick.py) is ever read; the other G arms are s1-cancelled + deleted UNREAD by the automation
# (outputs/t2_d649_reads.sh) before any screen number is opened. 24 arrays x 6 tasks, D642's walls, SIDE / FEAT dumps; reserved <= 432 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d649.sh dry|GO <completed smoke job id>     [SKIP=<regex on d649-<slot>_<arm>>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d649.sh tracks/t2_bci/sofia/arms/t2d649_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d649_s*_{C,G01,G03,G1}.txt | wc -l)" -eq 24 ] || { echo "need 24 D649 screen arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D649 (W29 GRL) member screen G vs C at 1200 + slot, launched beside the lambda smoke (the manager); only the engagement-picked lambda is read"
for f in tracks/t2_bci/sofia/arms/t2d649_s*_{C,G01,G03,G1}.txt; do
  a=$(basename "$f" .txt); a=${a#t2d649_}
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); [ "$n" -eq 6 ] || { echo "!! $f: $n lines (need 6)"; exit 2; }
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "d649-$a" =~ ${SKIP} ]] && { echo "skip d649-$a (SKIP=$SKIP: already submitted)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d649-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true   # a reply without these lines must reach the retry, not exit
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! d649 $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

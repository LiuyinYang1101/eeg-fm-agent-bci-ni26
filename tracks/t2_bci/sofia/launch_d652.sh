#!/usr/bin/env bash
# T2-D652 (W30 = W25's C lines + vp 2, D582's recipe; the manager's GO 2026-11-20 with Track3's pins; registered before any number): arm V ONLY
# (the manager: halve the compute), the five headed slots (the MI slot excluded, as D582 / D567):
#   (a) ship composition: arms/t2d652_<slot>_V.txt = D642's Cp line (950 + slot, ss8 + rt) + --fm-vp 2, paired with the EXISTING C' (d642a Cp);
#       one ARRAY of 6 fold-pair tasks per slot (ni26-t2p-d652-<slot>_V);
#   (b) rtl: arms/t2d652rl_<slot>_V.txt = D651's C line (1400 + slot, rtl) + --fm-vp 2, paired with D651's C; one ARRAY of 4 rtl fold-pair tasks
#       per slot (ni26-t2p-d652rl-<slot>_V).
# 1 H200 per task, D642's walls (3:00 MI / cont, 2:30 others), SIDE / FEAT dumps as D642. Reserved <= 5 x 6 x 3 + 5 x 4 x 3 = 150 GPU-h.
# Gate 1 (pinned): D651's pooled C - B < 0 -> W30 stops UNREAD (its dirs deleted unread).
#   mem-guard bash tracks/t2_bci/sofia/launch_d652.sh dry|GO <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d652.sh tracks/t2_bci/sofia/arms/t2d652_*.txt tracks/t2_bci/sofia/arms/t2d652rl_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d652_s*_V.txt | wc -l)" -eq 5 ] && [ "$(ls tracks/t2_bci/sofia/arms/t2d652rl_s*_V.txt | wc -l)" -eq 5 ] || { echo "need 5 + 5 D652 arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 keep trying; T2-D652 (W30 = W25's C lines + vp 2; the manager GO + Track3's pins): member gate V vs C' (strangers) and V vs D651's C (rtl)"
[ -n "${PURPOSE:-}" ] && P=$PURPOSE   # 2026-11-21: the reopened information read names itself (T2-D652)
for f in tracks/t2_bci/sofia/arms/t2d652_s*_V.txt tracks/t2_bci/sofia/arms/t2d652rl_s*_V.txt; do
  a=$(basename "$f" .txt); k=${a%%_*}; a=${a#"$k"_}
  n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); arr="0-$((n - 1))"
  w=02:30:00; [[ "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  [[ -n "${SKIP:-}" && "${k#t2}-$a" =~ ${SKIP} ]] && { echo "skip ${k#t2}-$a (SKIP=$SKIP: already submitted)"; continue; }
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read ${k#t2}-$a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=$arr --time=$w --job-name="ni26-t2p-${k#t2}-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true   # a reply without these lines must reach the retry, not exit (set -e + pipefail)
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $k $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

#!/usr/bin/env bash
# T2-D644 (W27, astra; the manager 2026-11-19): the member screen of the ST channel-embedding lr release on fm_1 / fm_3: arm E = D642's B
# lines + --fm-chemb-lr 0.0001 (arms/t2d644_<slot>_E.txt), paired with D642's B at the same seeds 901 / 903; one ARRAY per slot
# (6 tasks = fold pairs, 1 H200 each, 2:30); reserved 2 x 6 x 2.5 h = 30 GPU-h.
#   [ARM=E|U] mem-guard bash tracks/t2_bci/sofia/launch_d644.sh dry|GO <completed smoke job id>   (U: the combination rule (b), both screens passed)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d644.sh tracks/t2_bci/sofia/arms/t2d644_*.txt)" ] || { echo "commit first"; exit 2; }
ARM=${ARM:-E}; [[ "$ARM" =~ ^(E|U)$ ]] || exit 2       # U = the combination rule's (b) arm C + chemb: launched ONLY if both D642 and D644 pass
[ "$(ls tracks/t2_bci/sofia/arms/t2d644_s*_$ARM.txt | wc -l)" -eq 2 ] || { echo "need 2 $ARM arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-19: target first place, all resources; T2-D644 = W27 (the ST channel-embedding lr release on fm_1 / fm_3), member screen E vs D642 B"
for f in tracks/t2_bci/sofia/arms/t2d644_s*_$ARM.txt; do
  a=$(basename "$f" .txt); a=${a#t2d644_}
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d644-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

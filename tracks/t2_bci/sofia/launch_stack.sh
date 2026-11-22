#!/usr/bin/env bash
# THE STACK READ's member fits (the owner's inclusion rule 16:31:32, rule 3; Track3's spec): at the FRESH seeds 1800 + slot, ship composition (ss8 +
# rt), S1_EXPORT NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1,NI26_VIEW_DUMP=4: C (D642's C, six slots: the base), G (C + --fm-grl 0.1, six), V (C + --fm-vp 2,
# s1-s5; s0 = C's), J (C + --fm-grl 0.1 + --fm-vp 2, s1-s5; s0 = G's). Speculative: a part's arms are cancelled + deleted unread when its member sign
# gate fails (G / J on W29's, V / J on W30's). One array of 6 fold-pair tasks per (slot, arm), D642's walls; a submit refused by the pool cap is
# retried every 10 min up to RETRY_H hours (default 3).
#   mem-guard bash tracks/t2_bci/sofia/launch_stack.sh dry|GO <completed smoke job id> <arms, e.g. C,G or V,J>   [SKIP=<regex on stk-<slot>_<arm>>]
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; ARMS=${3:?arms C,G,V,J}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_stack.sh tracks/t2_bci/sofia/arms/tstk_*.txt)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-20 16:31:32 'any small gain': THE STACK READ's member fits at 1800 + slot (VIEW_DUMP=4); cancelled unread per part on a member-gate fail"
tries=$(( ${RETRY_H:-3} * 6 ))
for arm in ${ARMS//,/ }; do
  for f in tracks/t2_bci/sofia/arms/tstk_s*_$arm.txt; do
    a=$(basename "$f" .txt); a=${a#tstk_}
    n=$(grep -v '^[[:space:]]*#' "$f" | grep -c .); [ "$n" -eq 6 ] || { echo "!! $f: $n lines"; exit 2; }
    w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
    [[ -n "${SKIP:-}" && "stk-$a" =~ ${SKIP} ]] && { echo "skip stk-$a"; continue; }
    for try in $(seq 1 $tries); do
      o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1,NI26_VIEW_DUMP=4" S1_PURPOSE="$P; read stk-$a (array, fold pairs)" s1 submit ni26 \
          tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-stk-$a" -- "$f" 2>&1 || true)
      echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170 || true
      echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
      echo "!! stk $a: refused (try $try / $tries): $(echo "$o" | tail -1 | cut -c1-160)"; [ "$try" -eq "$tries" ] && exit 4; sleep 600
    done
    [ "$mode" == dry ] && exit 0
    sleep 8
  done
done

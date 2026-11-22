#!/usr/bin/env bash
# T2-D642 (W25) member READ, re-split (the manager 2026-11-19 ≈ 19:25: the 18 8-GPU jobs sat PENDING (Priority) and were s1-cancelled):
# one Slurm ARRAY per (slot, arm) = 6 tasks (fold pairs), 1 H200 each, short walls (3:00 for the MI / continuation slots, 2:30 for the
# others); the same arm lines (B / C / T paired by fold and seed), the same S1_SMOKE / S1_PURPOSE; 18 submits, spaced with a retry on a
# refusal (the per-user Slurm RPC rate limit, SOFIA_OPS §6). Reserved ≈ 18 x 6 x 2.5-3 h ≈ 300 GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d642a.sh dry|GO <completed smoke job id>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; SMOKE=${2:?the completed smoke job id}; [[ "$mode" =~ ^(dry|GO)$ ]] || exit 2
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/t2p_multi_array.slurm tracks/t2_bci/sofia/launch_d642a.sh tracks/t2_bci/sofia/arms/t2d642a_*.txt)" ] || { echo "commit first"; exit 2; }
[ "$(ls tracks/t2_bci/sofia/arms/t2d642a_s*_[BCT].txt | wc -l)" -eq 18 ] || { echo "need 18 array arm files"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh
P="owner 2026-11-19: target first place, all resources; T2-D642 user-balanced FM (W25), member screen + speculative donors"
for f in tracks/t2_bci/sofia/arms/t2d642a_s*_[BCT].txt; do
  a=$(basename "$f" .txt); a=${a#t2d642a_}
  w=02:30:00; [[ "$a" == s0_* || "$a" == s2_* || "$a" == s4_* ]] && w=03:00:00
  for try in 1 2 3; do
    o=$(env "${dry[@]}" S1_SMOKE="$SMOKE" S1_EXPORT="NI26_SIDE_DUMP=1,NI26_FEAT_DUMP=1" S1_PURPOSE="$P; read $a (array, fold pairs)" s1 submit ni26 \
        tracks/t2_bci/sofia/t2p_multi_array.slurm --array=0-5 --time=$w --job-name="ni26-t2p-d642a-$a" -- "$f" 2>&1 || true)
    echo "$o" | grep -E "SUBMITTED|DRY RUN|start at" | cut -c1-170
    echo "$o" | grep -q -E "SUBMITTED|DRY RUN" && break
    echo "!! $a: refused (try $try): $(echo "$o" | tail -1 | cut -c1-160)"; sleep 60
  done
  [ "$mode" == dry ] && break
  sleep 8
done

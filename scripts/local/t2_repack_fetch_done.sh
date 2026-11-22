#!/usr/bin/env bash
# Fetch only the FINISHED T2 repack dirs (S1_DONE present on sofia) + their slurm logs, via t2_fetch_repacks.sh (sha256 check, no unpacked
# copies kept). Unfinished ones are listed and skipped (run again later). Used for T2-D288's SK10 repacks.
#   bash scripts/local/t2_repack_fetch_done.sh ni26-t2p-repack-d288-vp2r15sk-1541403 ...
set -uo pipefail
cd "$(dirname "$0")/../.."
S=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1
df -h "/media/USER/New Volume2" | tail -1
done_=()
for d in "$@"; do
  if ls "$S/$d"/t2p_*.zip >/dev/null 2>&1 && [ -f "$S/$d/S1_DONE" ]; then echo "$d: already fetched"; continue; fi
  s1 fetch ni26 "$d/S1_DONE" >/dev/null 2>&1 || { echo "$d: not done yet"; continue; }
  echo "$d: $(tr '\n' ' ' < "$S/$d/S1_DONE")"
  [ -f "$S/logs/$d.out" ] || s1 fetch ni26 "logs/$d.out" >/dev/null 2>&1 || echo "logs/$d.out: fetch failed"
  done_+=("$d")
done
[ ${#done_[@]} -gt 0 ] && bash scripts/local/t2_fetch_repacks.sh "${done_[@]}"
df -h "/media/USER/New Volume2" | tail -1

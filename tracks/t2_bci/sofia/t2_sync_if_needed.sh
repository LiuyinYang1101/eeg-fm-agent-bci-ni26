#!/usr/bin/env bash
# T2 (the manager's file-cap note, 2026-11-14 01:13): `s1 sync ni26` ONLY when the code a T2 job runs changed since T2's last sync.
# Every sync of this repo ships HEAD, so a later sync by any session still holds the code of T2's last one: if nothing under
# tracks/t2_bci (except the local launchers) or src/ni26 changed between T2's last synced commit and HEAD, the newest snapshot already has it -> no new snapshot.
#   bash tracks/t2_bci/sofia/t2_sync_if_needed.sh        (called by the T2 launchers on GO)
set -euo pipefail
cd "$(dirname "$0")/../../.."
rec=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/.t2_last_sync
head=$(git rev-parse HEAD)
if [ -s "$rec" ] && git cat-file -e "$(cat "$rec")^{commit}" 2>/dev/null && git diff --quiet "$(cat "$rec")" "$head" -- tracks/t2_bci src/ni26 \
     ':(exclude)tracks/t2_bci/sofia/launch_*.sh' ':(exclude)tracks/t2_bci/sofia/t2_sync_if_needed.sh'; then
  echo "t2 sync: no T2 code change since $(cut -c1-12 "$rec") -> no new snapshot" >&2
  exit 0
fi
s1 sync ni26 >/dev/null 2>&1 || { echo "t2 sync: s1 sync failed" >&2; exit 2; }
echo "$head" > "$rec"
echo "t2 sync: synced $(echo "$head" | cut -c1-12)" >&2

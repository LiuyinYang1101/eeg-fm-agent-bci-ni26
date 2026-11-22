#!/usr/bin/env bash
# T2 build verification, step 1 (repo copy of T2-D276's scratchpad fetch; used from T2-D284 on): for each build "<name>:<pkg job>:<repack job>"
# of row <ROW>, fetch the build log, the package dir's small files and, once the repack is done, the repack dir (t2_fetch_repacks.sh: sha256 +
# CRC checks, unpacked copy removed). No waiting: a build whose repack has no S1_DONE yet is reported and skipped (run again later).
#   mem-guard bash scripts/local/t2_build_fetch.sh d284 vp2r15d2:1539888:1539889 ...
set -uo pipefail
cd "$(dirname "$0")/../.."
S=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1
row=${1:?row}; shift
df -h "/media/USER/New Volume2" | tail -1
reps=()
for spec in "$@"; do
  IFS=: read -r n pj rj <<< "$spec"
  P=ni26-t2p-pkg-$row-$n-$pj; R=ni26-t2p-repack-$row-$n-$rj
  [ -f "$S/$R/S1_DONE" ] || s1 fetch ni26 "$R/S1_DONE" >/dev/null 2>&1 || { echo "$n: repack $rj not done yet"; continue; }
  s1 fetch ni26 "$P/S1_DONE" >/dev/null 2>&1 || { echo "$n: build $pj has no S1_DONE"; continue; }
  echo "$n: build $(tr '\n' ' ' < "$S/$P/S1_DONE")| repack $(tr '\n' ' ' < "$S/$R/S1_DONE")"
  for x in final.json guard_refused.txt guard_selftest.err kit_sim.json kit_sim.log kit_warmup.json kit_warmup.log; do
    f=t2p_${row}_$n.$x; [ -f "$S/$P/$f" ] || s1 fetch ni26 "$P/$f" >/dev/null 2>&1 || echo "$P/$f: fetch failed"
  done
  [ -f "$S/logs/$P.out" ] && [ "$(grep -c 'zip_sha256' "$S/logs/$P.out")" -gt 0 ] || s1 fetch ni26 "logs/$P.out" >/dev/null 2>&1 || echo "logs/$P.out: fetch failed"
  reps+=("$R")
done
[ ${#reps[@]} -gt 0 ] && bash scripts/local/t2_fetch_repacks.sh "${reps[@]}"
for R in "${reps[@]}"; do sha256sum "$S/$R"/t2p_*.zip; done
df -h "/media/USER/New Volume2" | tail -1

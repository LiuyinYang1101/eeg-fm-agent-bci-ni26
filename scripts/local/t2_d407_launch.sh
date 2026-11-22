#!/usr/bin/env bash
# T2-D407 launch (pre-registered 2026-11-07 10:08): the in-house FM project's C + P1 as T2's MOR member. ONE launch group per call = one `s1 sync ni26`
# (commit first), then (a) if any NAME=PATH is given, the free zen5_vis contract job t2p_d407_ckpts.slurm (RUN_NAME ni26-t2p-d407-ckpts: a
# symlink + fm_contract record per checkpoint), and (b) the member fine-tunes t2d407_<x>_{kit,seen1} (t2p_multi, one H200 each, ≈ 0.45 GPU-h
# per pair as D386), the checkpoint members afterok on (a); p1v2 (the pinned control) needs no checkpoint.
#   bash scripts/local/t2_d407_launch.sh [DRY] "C150k=/sofia/.../x.pt CP1a=/sofia/.../y.pt" "p1v2 C150k CP1a"
#   STRIP="CP1a+CP1b" bash ... = write head-less copies of those checkpoints (the P1 supervised heads dropped; T2-D407 amendment 12:5x)
# Read: `s1 fetch ni26 ni26-t2p-d407-<x>-<kit|seen1>-<id>` each, then `mem-guard bash scripts/local/t2_d407_read.sh` (X="..." for a subset).
set -euo pipefail
cd "$(dirname "$0")/../.."
dry=(); [ "${1:-}" == DRY ] && { dry=(S1_DRYRUN=1); shift; }
ck="${1:-}"; ft="${2:-}"
[ -n "$ft" ] || { echo "usage: $0 [DRY] \"NAME=PATH ...\" \"members\""; exit 2; }
[ -z "$(git status --porcelain -- tracks/t2_bci/sofia/t2p_d407_ckpts.slurm tracks/t2_bci/sofia/arms/t2d407_*.txt scripts/local/t2_d407_read.sh)" ] || { echo "commit the D407 code first"; exit 2; }
names=(); exp="RUN_NAME=ni26-t2p-d407-ckpts"
for kv in $ck; do
  k=${kv%%=*}; v=${kv#*=}
  [[ $k =~ ^(C150k|CP1a|CP1b)$ ]] && [[ $v == /sofia/* ]] || { echo "bad checkpoint '$kv'"; exit 2; }
  names+=("$k"); exp+=",SRC_$k=$v"
done
for x in $ft; do
  [[ $x =~ ^(p1v2|C150k|CP1a|CP1b)$ ]] || { echo "bad member '$x'"; exit 2; }
done
[ ${#dry[@]} -gt 0 ] || s1 sync ni26
dep=()
if [ ${#names[@]} -gt 0 ]; then
  nm=$(IFS=+; echo "${names[*]}")
  out=$(env "${dry[@]}" S1_EXPORT="$exp,NAMES=$nm${STRIP:+,STRIP=$STRIP}" S1_PURPOSE="T2-D407 (0 GPU-h, zen5_vis): stage + contract-check the in-house FM project's $nm for the bench; feeds the C + P1 MOR-member read (T2's hold / row 101)" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_d407_ckpts.slurm --time=00:30:00 --job-name=ni26-t2p-d407-ckpts 2>&1 | grep -E "SUBMITTED|DRYRUN|test-only|refus|error|ERROR" || true)
  echo "$out"
  j=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p')
  [ ${#dry[@]} -gt 0 ] || [ -n "$j" ] || { echo "contract job not submitted"; exit 2; }
  [ -n "$j" ] && dep=(--dependency=afterok:$j)
fi
for x in $ft; do
  d=(); [ "$x" != p1v2 ] && d=("${dep[@]}")
  for p in kit seen1; do
    env "${dry[@]}" S1_EXPORT=NSHARDS=1 S1_PURPOSE="T2-D407 member fine-tune $x $p (D386's MOR line, fresh seeds 36-47, ≈ 0.23 GPU-h); feeds the C + P1 MOR-member read (T2's hold / row 101)" \
      s1 submit ni26 tracks/t2_bci/sofia/t2p_multi.slurm "${d[@]}" --time=01:30:00 --job-name=ni26-t2p-d407-$x-$p -- tracks/t2_bci/sofia/arms/t2d407_${x}_$p.txt 2>&1 \
      | grep -E "SUBMITTED|DRYRUN|test-only|refus|error|ERROR" || true
  done
done

#!/usr/bin/env bash
# T2-D437 (bagging without removing a user: Dirichlet user-share bags uw1 / uw4, run-level bootstrap, vs same-size seed bags) on the local 5090, ONE gpu-submit job
#   t2d437_sh_{seeds,uw1,uw4,runs}_s372 -> outputs/sofia_s1/ni26-t2p-d437-sh-{seeds,uw1,uw4,runs}-5090; seeds 372-383, kit + seen1.
# Read: mem-guard python scripts/local/t2_d437_read.py
#   bash scripts/local/t2_d437_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d437_*.txt scripts/local/t2_d437_read.py tests/test_t2p_bag_rows.py)" ] || { echo "commit the D437 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
ARMS=${D437_ARMS:-"t2d437_sh_seeds_s372 t2d437_sh_uw1_s372 t2d437_sh_uw4_s372 t2d437_sh_runs_s372"}
dir_of() { case "$1" in t2d437_*_s372) local k=${1#t2d437_}; k=${k%_s372}; echo "$S/ni26-t2p-d437-${k//_/-}-5090";; *) echo "unknown arm $1" >&2; exit 2;; esac; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d437_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in $ARMS; do
    d=$(dir_of "$arm")
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d437 $arm: $line\" >> '$d/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
for arm in $ARMS; do
  d=$(dir_of "$arm")
  [ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
done
gpu-submit -L "neural_interface_challenge:t2-d437" bash "$job"

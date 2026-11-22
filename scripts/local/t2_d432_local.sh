#!/usr/bin/env bash
# T2-D432 (stock braindecode ShallowFBCSPNet + EEGNet with causal EA: the commodity floor) on the local 5090 (zero grant cost), ONE gpu-submit job, two arms one after the other, each in its own dir
#   t2d432_{sh,eegnet}_stock_s360 -> outputs/sofia_s1/ni26-t2p-d432-{sh,ifn}-{ctl,eaf}-5090; seeds 360-371, kit + seen1, stock recipe r1, no copies.
# Read: mem-guard python scripts/local/t2_d432_read.py
#   bash scripts/local/t2_d432_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d432_*.txt scripts/local/t2_d432_read.py)" ] || { echo "commit the D432 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
ARMS=${D432_ARMS:-"t2d432_sh_stock_s360 t2d432_eegnet_stock_s360"}
dir_of() { case "$1" in t2d432_*_s360) local k=${1#t2d432_}; k=${k%_s360}; echo "$S/ni26-t2p-d432-${k//_/-}-5090";; *) echo "unknown arm $1" >&2; exit 2;; esac; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d432_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in $ARMS; do
    d=$(dir_of "$arm")
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d432 $arm: $line\" >> '$d/arm.log'"
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
gpu-submit -L "neural_interface_challenge:t2-d432" bash "$job"

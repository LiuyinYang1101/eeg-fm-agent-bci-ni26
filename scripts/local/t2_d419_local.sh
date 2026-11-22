#!/usr/bin/env bash
# T2-D419 (the fit audit follow-up: a shorter pooled FBCNet schedule) on the local 5090 (zero grant cost), ONE gpu-submit job: the two FBCNet arms one after
# the other, each in its own dir: t2d419_fbc_e90_s324 (90 pooled epochs) and t2d419_fbc_e180_s324 (the paired control = M7's FBCNet as shipped, 180 epochs)
# seeds 324-335, kit + seen1, copies + tv8c. Each python call in a MemoryMax=20G scope (t2_d367_local.sh's pattern).
# Outputs: outputs/sofia_s1/ni26-t2p-d419-<arm>-5090/{runs/, arm.log, expected.txt}. Read: mem-guard bash scripts/local/t2_d419_read.sh
# T2-D419b (2026-11-10): D419_ARMS overrides the arm list, e.g. D419_ARMS=fbc_e60_s324 (the control = D419's e180 dir, not re-run).
#   bash scripts/local/t2_d419_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d419_*.txt scripts/local/t2_d419_read.sh)" ] || { echo "commit the D419 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
ARMS=${D419_ARMS:-"fbc_e90_s324 fbc_e180_s324"}
job=$(mktemp "${TMPDIR:-/tmp}/t2_d419_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in $ARMS; do
    d="$S/ni26-t2p-d419-$arm-5090"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d419_$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d419 $arm: $line\" >> '$d/arm.log'"
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
  d="$S/ni26-t2p-d419-$arm-5090"
  [ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
done
gpu-submit -L "neural_interface_challenge:t2-d419" bash "$job"

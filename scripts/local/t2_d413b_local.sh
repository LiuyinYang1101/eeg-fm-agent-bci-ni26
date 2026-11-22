#!/usr/bin/env bash
# T2-D413b STEP 1 (= T2-D413 Step 1 on the tuned bank; only after the D413b final gate PASSED) on the local 5090 (zero grant cost), ONE gpu-submit job: the two FBCNet arms one after
# the other, each in its own dir: t2d413b_fbc_tl15mf_s300 (MeanFlow syn 0.5) and t2d413b_fbc_tl15_s300 (the paired control = M6's
# tilt line as shipped), seeds 300-311, kit + seen1, copies + tv8c. Each python call in a MemoryMax=20G scope (t2_d367_local.sh's pattern).
# Outputs: outputs/sofia_s1/ni26-t2p-d413b-<arm>-5090/{runs/, arm.log, expected.txt}. Read: mem-guard bash scripts/local/t2_d413b_read.sh
#   bash scripts/local/t2_d413b_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d413b_*.txt tests/test_syn.py scripts/local/t2_d413b_read.sh)" ] || { echo "commit the D413b code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
$PY -c "import json, sys; sys.exit(0 if json.load(open('outputs/t2_d413b/gate_final.json'))['gate_pass'] is True else 1)" \
  || { echo "T2-D413b: the final gate did not PASS (outputs/t2_d413b/gate_final.json): no member run"; exit 3; }
for p in kit seen1; do [ -s "outputs/t2_d413b/bank_${p}_final.npz" ] || { echo "missing outputs/t2_d413b/bank_${p}_final.npz"; exit 3; }; done
S=$(pwd)/outputs/sofia_s1
job=$(mktemp "${TMPDIR:-/tmp}/t2_d413b_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in fbc_tl15mf_s300 fbc_tl15_s300; do
    d="$S/ni26-t2p-d413b-$arm-5090"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d413b_$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d413b $arm: $line\" >> '$d/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
for arm in fbc_tl15mf_s300 fbc_tl15_s300; do
  d="$S/ni26-t2p-d413b-$arm-5090"
  [ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
done
gpu-submit -L "neural_interface_challenge:t2-d413b-step1" bash "$job"

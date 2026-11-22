#!/usr/bin/env bash
# T2-D367 (outlier B6-3) on the local 5090 (zero grant cost), ONE gpu-submit job: (1) t2p_cnn_ssl.py per protocol (kit, seen1) on the export
# + the staged session crops (sofia 1568601, sha-checked) -> ssl_<protocol>.pt; (2) the t2d367_fbc_ssl arms (the shipped FBCNet recipe from that
# init, seeds 192-203, copies + tv8c). Each python call in a MemoryMax=20G scope (the pattern of t2_d358_local.sh).
# Outputs: outputs/sofia_s1/ni26-t2p-d367-fbc-ssl-5090/{ssl_kit.pt, ssl_seen1.pt, runs/, arm.log}.
#   bash scripts/local/t2_d367_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_cnn_ssl.py tracks/t2_bci/sofia/arms/t2d367_*.txt tests/test_cnn_init.py)" ] || { echo "commit the D367 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
CROPS=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-stage-psess-1568601/proteus_sessions_v1
S=$(pwd)/outputs/sofia_s1
d="$S/ni26-t2p-d367-fbc-ssl-5090"
[ -f "$CROPS/X.npy" ] || { echo "no staged crops at $CROPS"; exit 2; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d367_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "mkdir -p '$d/runs'"
  for p in kit seen1; do
    echo "echo \"### \$(date +%T) d367 ssl $p\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_cnn_ssl.py --export '$EXP' --crops '$CROPS' --protocol $p --out '$d/ssl_$p.pt' >> '$d/arm.log' 2>&1 || rc=\$?"
  done
  echo "[ \$rc -eq 0 ] || { echo \"### done \$(date +%T) rc=\$rc (ssl failed)\" >> '$d/arm.log'; exit \$rc; }"
  grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d367_fbc_ssl.txt | grep -v '^[[:space:]]*$' | while read -r line; do
    echo "echo \"### \$(date +%T) d367 bench: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
  done
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done \$(date +%T) rc=\$rc\" >> '$d/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
[ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d367" bash "$job"

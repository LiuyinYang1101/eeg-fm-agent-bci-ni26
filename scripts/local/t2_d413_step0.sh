#!/usr/bin/env bash
# T2-D413 STEP 0 on the local 5090 (zero grant cost; inside GO 430), ONE gpu-submit job: the MeanFlow generators of kit and seen1
# (tracks/t2_bci/scripts/t2p_meanflow.py, generator seed 0, 12k steps each) and then the pre-registered gate (scripts/local/t2_d413_gate.py).
# Each python call in a MemoryMax=20G scope (t2_d412_local.sh's pattern). Outputs: outputs/t2_d413/{gen,own,interp,bank,info}_<proto>_g0.*,
# gate_g0.json, step0.log (the gate's verdict line at its end).
#   bash scripts/local/t2_d413_step0.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_meanflow.py tracks/t2_bci/scripts/t2p_bench.py scripts/local/t2_d413_gate.py tests/test_meanflow.py tests/test_syn.py)" ] || { echo "commit the D413 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs/t2_d413
[ ! -e "$O/gate_g0.json" ] || { echo "$O/gate_g0.json exists: Step 0 already ran"; exit 2; }
mkdir -p "$O"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d413_step0.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
cat > "$job" <<EOF
set -uo pipefail; cd '$(pwd)'; rc=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### git \$(git rev-parse --short=12 HEAD) \$(date +%T)" >> '$O/step0.log'
git rev-parse HEAD > '$O/git_head.txt'
for p in kit seen1; do
  $SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$EXP' --protocol \$p --out '$O' --seed 0 >> '$O/step0.log' 2>&1 || rc=\$?
done
[ \$rc -eq 0 ] && { $SC $PY -u scripts/local/t2_d413_gate.py --export '$EXP' --gen-dir '$O' --seed 0 >> '$O/step0.log' 2>&1 || rc=\$?; }
echo "### done \$(date +%T) rc=\$rc" >> '$O/step0.log'
exit \$rc
EOF
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d413-step0" bash "$job"

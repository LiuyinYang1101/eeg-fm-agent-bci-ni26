#!/usr/bin/env bash
# T2-D413c GATE (a labelled deviation from T2-D413b's "final gate ONCE", typed by the user in T2's shell 2026-11-09: "Run D413c (Recommended)"),
# ONCE: the wfm30k checkpoints D413b already trained (kit gen_kit_wfm30k.pt, seen1 gen_seen1_wfm30k.pt) resampled with calib 3 (filter-bank
# spatial re-colouring + spectral gains; the config that fixed covariance on kit) -> tag final3, bank 8 x n_train; seen1 training-window metrics
# reported; scripts/local/t2_d413_gate.py --tag final3 on BOTH protocols (D413's four checks and bars). No retraining.
#   bash scripts/local/t2_d413c_gate.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_meanflow.py tracks/t2_bci/scripts/t2p_mf_calib.py scripts/local/t2_d413_gate.py scripts/local/t2_d413b_tune_eval.py scripts/local/t2_d413c_gate.sh)" ] || { echo "commit the D413c code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs/t2_d413b
for p in kit seen1; do [ -s "$O/gen_${p}_wfm30k.pt" ] || { echo "missing $O/gen_${p}_wfm30k.pt"; exit 2; }; done
[ ! -e "$O/gate_final3.json" ] || { echo "$O/gate_final3.json exists: the D413c gate runs ONCE"; exit 2; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d413c_gate.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
MF="$SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$EXP' --out '$O'"
cat > "$job" <<EOF
set -uo pipefail; cd '$(pwd)'; rc=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L='$O/final3.log'
echo "### git \$(git rev-parse --short=12 HEAD) D413c: wfm30k eu100 calib 3 \$(date +%T)" >> \$L
for p in kit seen1; do
  [ \$rc -eq 0 ] && { $MF --protocol \$p --tag final3 --from-ckpt '$O/gen_'\$p'_wfm30k.pt' --sample eu100 --calib 3 --bank-mult 8 >> \$L 2>&1 || rc=\$?; }
done
[ \$rc -eq 0 ] && { $SC $PY -u scripts/local/t2_d413b_tune_eval.py --export '$EXP' --gen-dir '$O' --protocol seen1 --tag final3 --note 'D413c seen1 (reported)' >> \$L 2>&1 || rc=\$?; }
[ \$rc -eq 0 ] && { $SC $PY -u scripts/local/t2_d413_gate.py --export '$EXP' --gen-dir '$O' --tag final3 >> \$L 2>&1 || rc=\$?; }
echo "### done \$(date +%T) rc=\$rc" >> \$L
exit \$rc
EOF
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d413c-gate" bash "$job"

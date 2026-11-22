#!/usr/bin/env bash
# T2-D413b FINAL GATE (pre-registered 2026-11-09 11:23), ONCE, for the config picked on kit training-window fidelity (lowest own-user AUC with
# (b) and (c) passing, outputs/t2_d413b/tune.jsonl). ONE gpu-submit job:
#   kit:   resample the picked kit checkpoint (gen_kit_<base>.pt) with the picked sampler + calibration, bank 8 x n_train -> tag "final";
#   seen1: train the SAME config on seen1 (tag <base>), then resample it the same way -> tag "final"; its training-window metrics reported;
#   gate:  scripts/local/t2_d413_gate.py --tag final on BOTH protocols (D413's four checks and bars; TSTR on the kit val >= 0.45 on both).
#   bash scripts/local/t2_d413b_final.sh <base tag> "<training args>" <sampler> <calib> [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
BASE=$1; TRAIN=$2; SAMPLE=$3; CAL=$4
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_meanflow.py tracks/t2_bci/scripts/t2p_mf_calib.py scripts/local/t2_d413_gate.py scripts/local/t2_d413b_tune_eval.py scripts/local/t2_d413b_final.sh)" ] || { echo "commit the D413b code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs/t2_d413b
[ -s "$O/gen_kit_$BASE.pt" ] || { echo "missing $O/gen_kit_$BASE.pt"; exit 2; }
[ ! -e "$O/gate_final.json" ] || { echo "$O/gate_final.json exists: the final gate runs ONCE"; exit 2; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d413b_final.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
MF="$SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$EXP' --out '$O'"
cat > "$job" <<EOF
set -uo pipefail; cd '$(pwd)'; rc=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L='$O/final.log'
echo "### git \$(git rev-parse --short=12 HEAD) base $BASE | $TRAIN | $SAMPLE calib $CAL \$(date +%T)" >> \$L
$MF --protocol kit --tag final --from-ckpt '$O/gen_kit_$BASE.pt' --sample $SAMPLE --calib $CAL --bank-mult 8 >> \$L 2>&1 || rc=\$?
[ \$rc -eq 0 ] && { $MF --protocol seen1 --tag $BASE $TRAIN --sample $SAMPLE --bank-mult 1 >> \$L 2>&1 || rc=\$?; }
[ \$rc -eq 0 ] && { $MF --protocol seen1 --tag final --from-ckpt '$O/gen_seen1_$BASE.pt' --sample $SAMPLE --calib $CAL --bank-mult 8 >> \$L 2>&1 || rc=\$?; }
[ \$rc -eq 0 ] && { $SC $PY -u scripts/local/t2_d413b_tune_eval.py --export '$EXP' --gen-dir '$O' --protocol seen1 --tag final --note 'final seen1 (reported)' >> \$L 2>&1 || rc=\$?; }
[ \$rc -eq 0 ] && { $SC $PY -u scripts/local/t2_d413_gate.py --export '$EXP' --gen-dir '$O' --tag final >> \$L 2>&1 || rc=\$?; }
echo "### done \$(date +%T) rc=\$rc" >> \$L
exit \$rc
EOF
if [ "${5:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d413b-final" bash "$job"

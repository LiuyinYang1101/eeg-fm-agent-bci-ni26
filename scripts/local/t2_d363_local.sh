#!/usr/bin/env bash
# T2-D363 (outlier bet B4: CBraMod's readout / fit check) on the local 5090, ONE gpu-submit job (≈ 5 min): step 1 = fm:cbramodx on
# T2-D341's line, kit seeds 0-1; the fit check (diagnostic only) = cbramod and cbramodx, kit seed 0, no mixup / rc, LLRD 1.0,
# lr enc 1e-3. Each bench call in a MemoryMax=20G scope (the pattern of t2_d358_local.sh).
# Outputs: outputs/sofia_s1/ni26-t2p-d363-cbramodx-5090/{step1,fit}.
#   bash scripts/local/t2_d363_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ tracks/t2_bci/scripts/ tracks/t1_image/scripts/t1_fm.py)" ] || { echo "commit the D363 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
d=$(pwd)/outputs/sofia_s1/ni26-t2p-d363-cbramodx-5090
BASE="--protocols kit --aligns sz --fm-calib gain"
declare -A ARM=(
  [step1]="$BASE --seeds 0,1 --models fm:cbramodx:mean --fm-mixup 0.4 --fm-rc 0.75 --fm-llrd 0.75"
  [fit]="$BASE --seeds 0 --models fm:cbramod:mean,fm:cbramodx:mean --fm-lr-enc 1e-3"
)
job=$(mktemp "${TMPDIR:-/tmp}/t2_d363_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for a in step1 fit; do
    echo "mkdir -p '$d/$a'"
    echo "echo \"### \$(date +%T) d363 $a: ${ARM[$a]}\" >> '$d/arm.log'"
    echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/$a' ${ARM[$a]} >> '$d/arm.log' 2>&1 || rc=\$?"
  done
  echo "echo \"### done \$(date +%T) rc=\$rc\" >> '$d/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d363" bash "$job"

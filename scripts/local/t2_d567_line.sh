#!/usr/bin/env bash
# T2-D567 local backup (the manager 2026-11-18 ≈ 06:5x): ONE D567 augmented MI line on the local 5090, all 10 folds in one gpu job, the D567 arms EXACTLY (their args; only the
# sofia --fm-ckpt path mapped to the local copy) + NI26_FM_GRADCKPT=1 (activation checkpointing: the same computation; batch 64 fits 32 GB)
# + the side / feature dumps. Output: outputs/t2_d567/mi_s<seed>/runs (the reader's ami form with NI26_D567_LOCAL=<seed>); complete = 10 files.
#   mem-guard bash scripts/local/t2_d567_line.sh mi842|mi832 [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
LINE=${1:?mi842|mi832}
case $LINE in
  mi842) ARMS="aug_mi_s842_p0 aug_mi_s842_p1 aug_mi_s842_p2 aug_mi_s842_p3 aug_mi_s842_p4"; D=mi_s842 ;;
  mi832) ARMS="aug_mi_s832_p0 aug_mi_s832_p1 aug_mi_s832_p2 aug_mi_s832_p3 aug_mi_s832_p4"; D=mi_s832 ;;
  *) exit 2 ;;
esac
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts src/ni26 tracks/t2_bci/sofia/arms/t2d567_*.txt scripts/local/t2_d567_line.sh)" ] || { echo "commit first"; exit 2; }
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d567/$D
[ ! -e "$R/runs" ] || [ -z "$(ls -A "$R/runs" 2>/dev/null)" ] || { echo "$R/runs is not empty"; exit 3; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
SOF=$S1_ROOT/runs/ni26
job=$(mktemp "$O/t2_d567_line.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R/runs'; git rev-parse HEAD > '$R/git_head.txt'"
  echo "export NI26_FM_GRADCKPT=1 NI26_SIDE_DUMP=1 NI26_FEAT_DUMP=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for a in $ARMS; do
    line=$(grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d567_$a.txt" | grep -v '^[[:space:]]*$' | sed -e 's/^NSHARDS=[0-9]* //' -e "s#$SOF/#$O/sofia_s1/#g")
    echo "echo \"### \$(date +%T) d567 $LINE $a: $line\" >> '$R/arm.log'"
    echo "$PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$R/runs' $line >> '$R/arm.log' 2>&1 || rc=\$?"
  done
  echo "n=\$(ls '$R/runs' | grep -c '\.npz\$'); echo \"### done d567 $LINE \$(date +%T): \$n files (expect 10), rc \$rc, \$(grep -c Traceback '$R/arm.log') tracebacks, \$(grep -c 'CUDA OOM' '$R/arm.log') OOM\" >> '$R/arm.log'"
} > "$job"
if [ "${2:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d567-$LINE" bash "$job"

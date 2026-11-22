#!/usr/bin/env bash
# T2-D434: fine-tuned MantisV2 members (arms sp, sp0, ch; seeds 0-5; kit + seen1) on the local 5090, ONE gpu-submit job. eeg311 python (the `mantis`
# package lives there); the data dump (project env) must exist: outputs/t2_d434/ft_data.npz.   bash scripts/local/t2_d434_local.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=$HOME/workspace/outputs/neural_interface_challenge/t2_d434
[ -f "$D/ft_data.npz" ] || { echo "dump first"; exit 2; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2_d434_mantis_ft.py scripts/local/t2_d434_read.py)" ] || { echo "commit the D434 code first"; exit 2; }
for a in sp sp0 ch; do [ ! -e "$D/runs_$a" ] || { echo "$D/runs_$a exists"; exit 2; }; done
job="$D/job.sh"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\" > '$D/train.log'"
  for a in sp sp0 ch; do
    echo "$HOME/miniconda3/envs/eeg311/bin/python -u tracks/t2_bci/scripts/t2_d434_mantis_ft.py train --dir '$D' --arm $a >> '$D/train.log' 2>&1 || rc=\$?"
  done
  echo "echo \"### done d434 rc=\$rc\" >> '$D/train.log'; exit \$rc"
} > "$job"
gpu-submit -L "neural_interface_challenge:t2-d434" bash "$job"

#!/usr/bin/env bash
# T2-D491b (seat G10 item 5; registered 2026-11-14 before any number; zero grant, the local 5090 via gpu-submit): the TSMNet-form SPD member
# (src/ni26/tsmnet.py, tracks/t2_bci/scripts/t2_d491_tsmnet.py), kit + seen1, seeds 360-371, three shards in parallel, then the batch-1
# ms / window line (CPU + GPU). Read: mem-guard python scripts/local/t2_d491_read.py
#   bash scripts/local/t2_d491_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/tsmnet.py tracks/t2_bci/scripts/t2_d491_tsmnet.py scripts/local/t2_d491_local.sh scripts/local/t2_d491b_local.sh scripts/local/t2_d491_read.py)" ] || { echo "commit the D491b code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d491
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d491b_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R/runs'; git rev-parse HEAD > '$R/git_head.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  echo "echo \"### \$(date +%T) d491b (nore) start\" >> '$R/run.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2_d491_tsmnet.py --nore --out '$R/runs' --shard $i --nshards 3 >> '$R/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "n=\$(ls '$R/runs' | grep -c '__tsmnetnore__ea__s[0-9]*\.npz\$'); [ \$n -eq 24 ] || rc=7"
  echo "echo \"### done d491b \$(date +%T) rc=\$rc; \$n files (24 expected)\" >> '$R/run.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d491b" bash "$job"

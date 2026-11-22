#!/usr/bin/env bash
# T2-D519 arms (pooled side nets with the dcmean side head and a temporal input VIEW, t2p_bench --view; K = 9 lc9, r1_pulx_vp2_e180_vr):
#   V5 = the shipped config under views ma6 / ema3 / dt37 at the draws' seeds 810-812 (+ D517's plain seed-d net + D518's S5 seed d + 1000).
#   mem-guard bash scripts/local/t2_d519_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d519_local.sh)" ] || { echo "commit the D519 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d519
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d519_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "sh:shallow" "ifn:ifnet"; do
    IFS=: read -r fam model <<< "$spec"
    D=$R/V_$fam
    echo "mkdir -p '$D/runs'"
    for view in ma6 ema3 dt37; do
      line="--protocols lc9 --models $model --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds 810,811,812 --side dcmean --view $view"
      echo "echo \"### \$(date +%T) d519 $fam $view: $line\" >> '$R/arm.log'"
      for i in 0 1 2; do
        echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_${view}_$i.log' 2>&1 &"
      done
      echo "wait"
    done
    echo "echo \"### done d519 $fam \$(date +%T): \$(ls '$D/runs' | wc -l) files (expect 90), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/arm.log'"
  done
  echo "echo \"### done d519 all \$(date +%T)\" >> '$R/arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d519" bash "$job"

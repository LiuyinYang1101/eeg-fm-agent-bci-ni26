#!/usr/bin/env bash
# T2-D517 arms (a) / (c) / (d): side inputs at the logits (ni26.train side head; t2p_bench --side), POOLED ShallowNet and IFNet only (no copies /
# views), K = 9 (lc9), seeds 810-812 (paired with D512's plain pooled nets of the same seeds, the is0 files in outputs/t2_d512/p1_<fam>).
#   bash scripts/local/t2_d517_side_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d517_side_local.sh)" ] || { echo "commit the D517 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d517
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d517_side_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for kind in dcmean dc8 lvwin lvstream; do
    for spec in "sh:shallow" "ifn:ifnet"; do
      IFS=: read -r fam model <<< "$spec"
      D=$R/side_${kind}_$fam
      line="--protocols lc9 --models $model --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds 810,811,812 --side $kind"
      echo "mkdir -p '$D/runs'; echo \"### \$(date +%T) d517 side $kind $fam: $line\" >> '$R/side_arm.log'"
      for i in 0 1 2; do
        echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
      done
      echo "wait"
      echo "echo \"### done d517 side $kind $fam \$(date +%T): \$(ls '$D/runs' | wc -l) files (expect 30), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/side_arm.log'"
    done
  done
  echo "echo \"### done d517 side all \$(date +%T)\" >> '$R/side_arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d517-side" bash "$job"

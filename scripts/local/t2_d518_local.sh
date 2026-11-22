#!/usr/bin/env bash
# T2-D518 arms (pooled side nets with the dcmean side head, K = 9 lc9, r1_pulx_vp2_e180_vr, as D517's side arm):
#   C5 = the 4 config variants per family at the draws' seeds 810-812 (the shipped config's seed-d net is D517's side_dcmean file);
#   S5 = the shipped config at seeds d + 1000 ... d + 4000 for d in 810-812 (with D517's seed-d net: 5 seeds per draw).
#   mem-guard bash scripts/local/t2_d518_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d518_local.sh)" ] || { echo "commit the D518 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d518
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d518_job.XXXX.sh")
S5=1810,2810,3810,4810,1811,2811,3811,4811,1812,2812,3812,4812
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "C5_sh:shallow_t13,shallow_t41,shallow_p50,shallow_f60:810,811,812" "C5_ifn:ifnet_k127,ifnet_k31,ifnet_s96,ifnet_st4:810,811,812" \
              "S5_sh:shallow:$S5" "S5_ifn:ifnet:$S5"; do
    IFS=: read -r arm models seeds <<< "$spec"
    D=$R/$arm
    line="--protocols lc9 --models $models --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $seeds --side dcmean"
    echo "mkdir -p '$D/runs'; echo \"### \$(date +%T) d518 $arm: $line\" >> '$R/arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "echo \"### done d518 $arm \$(date +%T): \$(ls '$D/runs' | wc -l) files (expect 120), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/arm.log'"
  done
  echo "echo \"### done d518 all \$(date +%T)\" >> '$R/arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d518" bash "$job"

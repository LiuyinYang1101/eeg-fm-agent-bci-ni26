#!/usr/bin/env bash
# Track1's T2 experiment XDAY (lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md, section XDAY; registered before any number; zero grant, the
# local 5090 via gpu-submit): the cross-day returning panel on Scherer2015. FBCNet (D419's line) + ShallowNet (D429's line), seeds 780-785.
#   stage xd: protocol xd with copies + tv8c (reads 1 and 3);  stage xh: protocols xh0,xh1,xh2, pooled + tv8 only (read 2).
#   bash scripts/local/t1seat_t2_xday_local.sh xd|xs|xh|xl [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:?xd|xs|xh|xl}
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t1seat_t2_xday_bench.py scripts/local/t1seat_t2_xday_local.sh)" ] || { echo "commit the XDAY code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
O=$HOME/workspace/outputs/neural_interface_challenge/t1seat_t2
EXP=$O/export_combo_v1
[ -f "$EXP/X.npy" ] || { echo "run t1seat_t2_xday_export.py first"; exit 2; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
case $stage in
  xd) protos=xd; extra="--peruser --tta-vp 8 --tta-copies" ;;
  xs) protos=xs; extra="--peruser --tta-vp 8 --tta-copies" ;;
  xl) protos=xl0,xl1,xl2; extra="--peruser --tta-vp 8 --tta-copies" ;;
  xh) protos=xh0,xh1,xh2; extra="--tta-vp 8" ;;
  *) echo "stage xd|xs|xh|xl"; exit 2 ;;
esac
job=$(mktemp "${TMPDIR:-/tmp}/t1seat_xday.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$O/runs'; git rev-parse HEAD > '$O/git_head_$stage.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75" "shallow:ea:r1_pulx_vp2_e180_vr"; do
    IFS=: read -r model al rec <<< "$spec"
    line="--protocols $protos --models $model --aligns $al --recipe $rec --seeds 780,781,782,783,784,785 $extra"
    echo "echo \"### \$(date +%T) xday $stage $model: $line\" >> '$O/arm_$stage.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u scripts/local/t1seat_t2_xday_bench.py --export '$EXP' --out '$O/runs' --shard $i --nshards 3 $line >> '$O/shard_${stage}_${model}_$i.log' 2>&1 &"
    done
    echo "wait"
  done
  echo "echo \"### done xday $stage \$(date +%T) rc=\$rc; \$(ls '$O/runs' | grep -c npz) files\" >> '$O/arm_$stage.log'; exit \$rc"
} > "$job"
if [ "${2:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t1seat-xday-$stage" bash "$job"

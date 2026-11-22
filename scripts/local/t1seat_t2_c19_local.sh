#!/usr/bin/env bash
# Track1's T2 experiment C19 (SEAT_T1_on_T2.md, section C19; registered before any number; zero grant, the local 5090): pooled FBCNet (D419's
# line) + ShallowNet (D429's line), tv8, seeds 790-795, kit split.  R41 = kit on export_proteus_v1 (41 ch); A = kit on export_c19_v1;
# B = kx; B' = kxh (both on export_c19_v1).      bash scripts/local/t1seat_t2_c19_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t1seat_t2_xday_bench.py scripts/local/t1seat_t2_c19_local.sh)" ] || { echo "commit the C19 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
O=$HOME/workspace/outputs/neural_interface_challenge/t1seat_t2
E41=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
E19=$O/export_c19_v1
[ -f "$E19/X.npy" ] || { echo "run t1seat_t2_c19_export.py first"; exit 2; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "${TMPDIR:-/tmp}/t1seat_c19.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$O/c19/r41' '$O/c19/c19'; git rev-parse HEAD > '$O/c19/git_head.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75" "shallow:ea:r1_pulx_vp2_e180_vr"; do
    IFS=: read -r model al rec <<< "$spec"
    for arm in "r41:$E41:kit" "c19:$E19:kit,kx,kxh"; do
      IFS=: read -r out exp protos <<< "$arm"
      line="--protocols $protos --models $model --aligns $al --recipe $rec --seeds 790,791,792,793,794,795 --tta-vp 8"
      echo "echo \"### \$(date +%T) c19 $out $model: $line\" >> '$O/c19/arm.log'"
      for i in 0 1 2; do
        echo "$SC $PY -u scripts/local/t1seat_t2_xday_bench.py --export '$exp' --out '$O/c19/$out' --shard $i --nshards 3 $line >> '$O/c19/shard_${out}_${model}_$i.log' 2>&1 &"
      done
      echo "wait"
    done
  done
  echo "echo \"### done c19 \$(date +%T); \$(ls '$O/c19/r41' '$O/c19/c19' | grep -c npz) files\" >> '$O/c19/arm.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t1seat-c19" bash "$job"

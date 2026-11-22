#!/usr/bin/env bash
# T2-D536 (the W19 fresh-draw read, a copy of D532's launcher at seeds 816-818; registered 2026-11-16 before any number; zero grant, the local 5090 via gpu-submit):
#   (1) the S5 side nets at the FRESH draws, with NI26_SIDE_DUMP=1 (D533: the clamp is recomputed at read time; outputs unchanged): D517 / D518's dcmean side line EXACTLY (t2_d518_local.sh's S5 arm: --protocols lc9 --aligns ea
#       --recipe r1_pulx_vp2_e180_vr --side dcmean) for ShallowNet and IFNet at seeds d + k * 1000, k 0..4, d in 816-818 (k 0 = D517's seed-d net);
#   (2) then, CPU only, the DC tree family's bag at FRESH LightGBM seeds 7,8,9: D516's run EXACTLY (scripts/local/t2_d515_read.py --features
#       features_dc.npz, the in-fold top-300 selection) with --seeds 7,8,9 -> outputs/t2_d515/pred_s7-8-9_dc.npz (its own name: it cannot clobber
#       pred_s1-2-3_dc.npz / _dm.npz), copied to outputs/t2_d536/pred_s7-8-9_dc.npz with its sha256 beside it.
#   bash scripts/local/t2_d536_side_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d515_read.py scripts/local/t2_d536_side_local.sh)" ] || { echo "commit the D536 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d536
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$O/t2_d536_side_job.XXXX.sh")
S5=816,1816,2816,3816,4816,817,1817,2817,3817,4817,818,1818,2818,3818,4818
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  for spec in "side_sh:shallow" "side_ifn:ifnet"; do
    IFS=: read -r arm models <<< "$spec"
    D=$R/$arm
    line="--protocols lc9 --models $models --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S5 --side dcmean"
    echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) d536 $arm: $line\" >> '$R/side_arm.log'"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(ls '$D/runs' | wc -l); echo \"### done d536 $arm \$(date +%T): \$n files (expect 150), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$R/side_arm.log'; [ \$n -eq 150 ] || rc=6"
  done
  echo "echo \"### \$(date +%T) d536 DC bag: t2_d515_read.py --features features_dc.npz --seeds 7,8,9\" >> '$R/side_arm.log'"
  echo "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $SC $PY -u scripts/local/t2_d515_read.py --features features_dc.npz --seeds 7,8,9 > '$R/read_dc789.log' 2>&1 || rc=8"
  echo "cp -p '$O/t2_d515/pred_s7-8-9_dc.npz' '$R/pred_s7-8-9_dc.npz' && sha256sum '$R/pred_s7-8-9_dc.npz' > '$R/pred_s7-8-9_dc.npz.sha256' || rc=8"
  echo "echo \"### done d536 side + DC bag \$(date +%T) rc=\$rc; \$(cat '$R/pred_s7-8-9_dc.npz.sha256' 2>/dev/null | cut -c1-16)\" >> '$R/side_arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d536-side" bash "$job"

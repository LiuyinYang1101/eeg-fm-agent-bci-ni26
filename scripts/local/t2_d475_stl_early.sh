#!/usr/bin/env bash
# T2-D475 arm A EARLY LOOK on the local 5090 (the manager's option 2, 2026-11-13): the kit ST-L CAR line on its first five seeds
# (472-476), while the paired sofia jobs 1614071 / 1614072 keep their queue place and stay the registered verdict. DESCRIPTIVE only:
# its controls (T2-D455 ctl kit) were trained on an H200, so the pairing is weaker than D419's same-GPU control.
# Needs the kit CPT checkpoint fetched locally (s1 fetch ni26 ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth).
# The generated job script is written into the run dir (not /tmp). Read: mem-guard python scripts/local/t2_d475_read.py early
#   bash scripts/local/t2_d475_stl_early.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d475_stlcar_kit.txt scripts/local/t2_d475_stl_early.sh scripts/local/t2_d475_read.py)" ] || { echo "commit the D475 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
CK=$O/sofia_s1/ni26-t2p-cpt-stlarge-d89b/checkpoint-cpt.pth
[ -f "$CK" ] || { echo "fetch $CK first"; exit 2; }
d=$O/sofia_s1/ni26-t2p-d475-stlcar-kitearly-5090
line=$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d475_stlcar_kit.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' \
  | sed "s#--fm-ckpt [^ ]*#--fm-ckpt $CK#; s#--seeds [0-9,]*#--seeds 472,473,474,475,476#")
[[ "$line" == *--car* && "$line" == *"$CK"* && "$line" == *"--seeds 472,473,474,475,476 "* ]] || { echo "arm line edit failed: $line"; exit 2; }
# batch 64 OOMs on the 32 GB 5090 (job 607, 4 OOMs a seed): exact gradient accumulation over 4 chunks (t2p_fm micro path; stem tag mb4)
line="$line --fm-micro ${D475_MICRO:-4}"
mkdir -p "$d/runs"
job=$d/job.sh
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "git rev-parse HEAD > '$d/git_head.txt'"
  echo "echo \"### \$(date +%T) d475 stlcar kit early: $line\" >> '$d/arm.log'"
  echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
  echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done stlcar-kitearly \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; exit 0; fi
[ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d475-early" bash "$job"

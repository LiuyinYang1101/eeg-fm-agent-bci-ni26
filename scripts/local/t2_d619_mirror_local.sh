#!/usr/bin/env bash
# T2-D619 (DIAGNOSTIC; registered 2026-11-19 before any code or number; the local 5090 via gpu-submit, ONE job): FBCNet's set-E line
# (scripts/local/t2_d536_cnn_local.sh's fbc spec at draws 860-862: fbcnet_bboth, align none, r1_pulxx_vp2_e180_vr_rc75, --tta-vp 8) on lc9
# WITHOUT --peruser / --tta-copies, plus --test-mirror: every pooled net also predicts its TEST windows left-right mirrored (twins mr, tv8mr).
# -> outputs/t2_d619/runs; read with scripts/local/t2_d619_mirror_read.py.
#   bash scripts/local/t2_d619_mirror_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt scripts/local/t2_d619_mirror_local.sh)" ] || { echo "commit the D619 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d619
d=$R/cnn_fbc
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
line="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' \
  | sed -E 's/--protocols [a-z0-9,]+ /--protocols lc9 /; s/--seeds [0-9,]+ /--seeds 860,861,862 /; s/ --peruser//; s/ --tta-copies//') --test-mirror"
grep -q -- "--recipe r1_pulxx_vp2_e180_vr_rc75 " <<< "$line" && grep -q -- "--tta-vp 8" <<< "$line" && ! grep -q -- "--peruser" <<< "$line" || { echo "unexpected line: $line"; exit 4; }
job=$(mktemp "$O/t2_d619_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### \$(date +%T) d619 fbc: $line\" >> '$R/arm.log'"
  echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"fbc: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$R/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done d619 \$(date +%T) rc=\$rc\" >> '$R/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d619-mirror" bash "$job"

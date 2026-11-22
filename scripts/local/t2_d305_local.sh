#!/usr/bin/env bash
# T2-D305 (row 77) on the local 5090 (zero grant): the MIRepNet screen, ONE gpu-submit job, each bench call in a MemoryMax=20G scope.
# Output: outputs/sofia_s1/ni26-t2p-d305-mirep-5090/runs (where t2_d305_read.sh looks). 48 fits of a 5 M model, est. 0.5-1.5 GPU-h (cap 5).
#   bash scripts/local/t2_d305_local.sh            (then `mem-guard bash scripts/local/t2_d305_read.sh`)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py src/ni26/fm/mirepnet.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d305_mirep.txt)" ] || { echo "commit the D305 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
d=$(pwd)/outputs/sofia_s1/ni26-t2p-d305-mirep-5090
ls -d "$(pwd)"/outputs/sofia_s1/ni26-t2p-d305-mirep-[0-9]*/ >/dev/null 2>&1 && { echo "a d305 dir already exists"; exit 2; }
mkdir -p "$d/runs"
line=$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d305_mirep.txt | grep -v '^[[:space:]]*$')
job=$(mktemp "${TMPDIR:-/tmp}/t2_d305_local.XXXX.sh")
cat > "$job" <<JOB
set -uo pipefail; cd '$(pwd)'; rc=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### \$(date +%T) start"
systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5
systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?
n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\$f ] || echo miss; done | wc -l); echo "\$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing"; [ \$n -eq 0 ] || rc=6
echo "### done \$(date +%T) rc=\$rc"; exit \$rc
JOB
gpu-submit bash "$job"

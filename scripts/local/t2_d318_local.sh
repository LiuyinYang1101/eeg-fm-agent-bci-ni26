#!/usr/bin/env bash
# T2-D318 on the local 5090 (zero grant): one gpu-submit job per arm, each python call in a MemoryMax=20G scope.
#   bash scripts/local/t2_d318_local.sh ha   H-a: HEEGNet + its EEGNet control, kit + seen1, seeds 0-11 (t2_d318_heeg.py)
#   bash scripts/local/t2_d318_local.sh hb   H-b: FBCNet (bboth) with the Lorentz MLR head vs its linear twin, rc75 recipe, pooled
# Outputs: outputs/sofia_s1/ni26-t2p-d318-{ha,hb}-5090/runs (where t2_d318_read.sh looks).
set -euo pipefail
cd "$(dirname "$0")/../.."
arm="${1:-}"; [[ "$arm" == ha || "$arm" == hb ]] || { sed -n '2,5p' "$0"; exit 2; }
[ -z "$(git status --porcelain -- src/ni26/heegnet.py src/ni26/train.py tracks/t2_bci/scripts/t2_d318_heeg.py tracks/t2_bci/scripts/t2p_bench.py)" ] \
  || { echo "commit the D318 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
d=$(pwd)/outputs/sofia_s1/ni26-t2p-d318-$arm-5090
mkdir -p "$d/runs"
S=0,1,2,3,4,5,6,7,8,9,10,11
if [ "$arm" == ha ]; then
  cmd="tracks/t2_bci/scripts/t2_d318_heeg.py --export '$EXP' --out '$d/runs' --models heeg,eegnetctl --protocols kit,seen1 --seeds $S"
else
  cmd="tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --protocols kit,seen1 --models fbcnet_bboth,fbcnet_bboth_lmlr --aligns none --recipe r1_pulxx_vp2_e180_vr_rc75 --seeds $S"
fi
job=$(mktemp "${TMPDIR:-/tmp}/t2_d318_$arm.XXXX.sh")
cat > "$job" <<JOB
set -uo pipefail; cd '$(pwd)'; rc=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### \$(date +%T) start $arm (git \$(git rev-parse --short HEAD))"
systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u $cmd --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5
systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u $cmd >> '$d/arm.log' 2>&1 || rc=\$?
n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\$f ] || echo miss; done | wc -l); echo "\$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing"; [ \$n -eq 0 ] || rc=6
echo "### done \$(date +%T) rc=\$rc"; exit \$rc
JOB
gpu-submit -L "ni26:t2-d318-$arm" bash "$job"

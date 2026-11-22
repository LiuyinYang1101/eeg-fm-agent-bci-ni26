#!/usr/bin/env bash
# T2-D166 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26 ni26-t2p-d166-morheads-1523492`:
# the D124c blend (r 1/3, hard route) of the MOR member's per-user heads -> t2_d166/route_mor; c_morh vs cA_m (the adopted package with
# the SAME job's pooled MOR) on outputs/t2_d166/spec.json, bar d141
#   mem-guard bash scripts/local/t2_d166_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/sofia_s1/ni26-t2p-d166-morheads-1523492/runs
echo "### $(date +%T) blends"
for p in kit seen1; do
  $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" --runs "$R" --protocols $p --out "$O/t2_d166/route_mor" \
    --model fm-mor_p1v2-mean --align wz --recipe fmE12b64le0.001lh0.001wd0.05ll1ema0dp0li16roled0.1 --seeds 0,1,2,3,4,5,6,7,8,9,10,11
done
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d166/spec.json" --out "$O/t2_d166/read.json" > "$O/t2_d166/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

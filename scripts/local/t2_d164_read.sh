#!/usr/bin/env bash
# T2-D164 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
# ni26-t2p-d164-sh-r1_pulx-<jobid> and ni26-t2p-d164-sh-r1_pulxx-<jobid> (ShallowNet, fresh seeds 24-35):
# ShallowNet blends at r 1/2; c_shxx vs cA (FBCNet = T2-D159's r1_pulxx blend, seeds 48-59) on outputs/t2_d164/spec.json, bar d141
#   mem-guard bash scripts/local/t2_d164_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
echo "### $(date +%T) blends"
for r in r1_pulx r1_pulxx; do
  d=( "$S"/ni26-t2p-d164-sh-$r-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $r: ${d[*]}"; exit 2; }
  for p in kit seen1; do
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d164/route_sh_${r}_r2" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds 24,25,26,27,28,29,30,31,32,33,34,35
  done
done
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d164/spec.json" --out "$O/t2_d164/read.json" > "$O/t2_d164/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

#!/usr/bin/env bash
# T2-D165 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md: the soft top-2 inverse-distance route
# (t2p_fpcalib blend --route inv2) of T2-D159's FBCNet r1_pulxx copies (r 1/3, seeds 48-59) and T2-D164's ShallowNet r1_pulx copies
# (r 1/2, seeds 24-35); c_soft vs cA (the adopted hard route on the same files) on outputs/t2_d165/spec.json, bar d141
#   mem-guard bash scripts/local/t2_d165_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" --route inv2 "$@"; }
echo "### $(date +%T) soft blends"
for p in kit seen1; do
  d=( "$S"/ni26-t2p-d159-fbc-r1_pulxx-$p-*/ ); [ ${#d[@]} -eq 1 ] || exit 2
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d165/route_fbc_soft" --model fbcnet_bboth --recipe r1_pulxx --tag 3s --seeds 48,49,50,51,52,53,54,55,56,57,58,59
done
d=( "$S"/ni26-t2p-d164-sh-r1_pulx-[0-9]*/ ); [ ${#d[@]} -eq 1 ] || exit 2
for p in kit seen1; do
  blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d165/route_sh_soft" --model shallow --align ea --recipe r1_pulx --r 0.5 --tag 2s --seeds 24,25,26,27,28,29,30,31,32,33,34,35
done
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d165/spec.json" --out "$O/t2_d165/read.json" > "$O/t2_d165/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

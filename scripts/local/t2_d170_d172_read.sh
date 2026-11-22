#!/usr/bin/env bash
# T2-D170 / D171 / D172 reads (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md:
#   d170: t2p_repick of the causal prior (alpha, n0) on the adopted package's files (T2-D163's members; weights / r fixed), cross-fitted
#   d171: T2-D168's FBCNet r1_pulxx files (seeds 60-71) re-blended at r 0.4 (tag 4); c_fr04 vs cB
#   d172: T2-D169's ShallowNet r1_pulx files (seeds 36-47) re-blended at r 2/3 (tag 6); c_sr067 vs cC
#   mem-guard bash scripts/local/t2_d170_d172_read.sh d170|d171|d172
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
case "${1:?d170|d171|d172}" in
  d170) $PY -u tracks/t2_bci/scripts/t2p_repick.py --export "$E" --spec "$O/t2_d170/spec.json" --out "$O/t2_d170/read.json" --jobs 12 > "$O/t2_d170/read.txt" 2>&1
        echo "rc $?"; exit 0 ;;
  d171) for p in kit seen1; do d=( "$S"/ni26-t2p-d168-fbc-r1_pulxx-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] || exit 2
          blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d171/route_fbc_r04" --model fbcnet_bboth --recipe r1_pulxx --r 0.4 --tag 4 --seeds 60,61,62,63,64,65,66,67,68,69,70,71
        done ;;
  d172) d=( "$S"/ni26-t2p-d169-sh-r1_pulx-[0-9]*/ ); [ ${#d[@]} -eq 1 ] || exit 2
        for p in kit seen1; do
          blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d172/route_sh_r067" --model shallow --align ea --recipe r1_pulx --r 0.666666666667 --tag 6 --seeds 36,37,38,39,40,41,42,43,44,45,46,47
        done ;;
esac
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_$1/spec.json" --out "$O/t2_$1/read.json" > "$O/t2_$1/read.txt" 2>&1
echo "rc $?"

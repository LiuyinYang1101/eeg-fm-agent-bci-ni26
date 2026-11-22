#!/usr/bin/env bash
# T2-D342: the tv16c per-user blends of the local d342 runs, exactly as t2_d296r_read.sh blends tv8c (FBCNet r 1/3 tag 3, ShallowNet r 0.5 tag 2).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$(pwd)/outputs; S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
mkdir -p "$O/t2_d342"
for p in kit seen1; do
  r=r1_pulxx_vp2_e180_vr_rc75tv16c
  blend --runs "$S/ni26-t2p-d342-fbc-$p-5090/runs" --protocols $p --out "$O/t2_d342/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds $(seq -s, 192 203)
  r=r1_pulx_vp2_e180_vrtv16c
  blend --runs "$S/ni26-t2p-d342-sh-5090/runs" --protocols $p --out "$O/t2_d342/route_sh_$r" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds $(seq -s, 156 167)
done

#!/usr/bin/env bash
# T2-D153 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
# ni26-t2p-d153-fbc-r1_ema-kit-1519488, ni26-t2p-d153-fbc-r1_ema-seen1-1519489 and ni26-t2p-d153-sh-r1_ema-1519490:
# 1 the D124c blend (r 1/3, T2-D124b's cached fingerprints) of FBCNet B3 r1_ema + its per-user copies -> t2_d153/route_fbc_ema
# 2 t2p_pkgshape on t2_d153/spec.json (anchor c0 = T2-D146's sofia r1 controls; deciding arm c_ema; c_fe / c_se beside)
#   mem-guard bash scripts/local/t2_d153_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
echo "### $(date +%T) blends"
blend --runs "$S/ni26-t2p-d153-fbc-r1_ema-kit-1519488/runs" --protocols kit --out "$O/t2_d153/route_fbc_ema" --model fbcnet_bboth --recipe r1_ema
blend --runs "$S/ni26-t2p-d153-fbc-r1_ema-seen1-1519489/runs" --protocols seen1 --out "$O/t2_d153/route_fbc_ema" --model fbcnet_bboth --recipe r1_ema
ls "$O"/t2_d153/route_fbc_ema/*.npz | wc -l
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d153/spec.json" --out "$O/t2_d153/read.json" > "$O/t2_d153/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

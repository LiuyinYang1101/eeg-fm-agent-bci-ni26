#!/usr/bin/env bash
# T2-D146 / D148 / D149 reads (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md:
# 1 the D124c blends (r 1/3, fingerprint every window; cached fingerprints of T2-D124b) of the sofia per-user runs:
#   FBCNet B3 r1 (1516836 kit / 1516837 seen1) and r1_mx4 (1516838 / 1516839) -> t2_d146/route_fbc_{r1,mx4};
#   ShallowNet-EA r1 (1516841, kit + seen1) -> t2_d148/route_sh
# 2 t2p_pkgshape on t2_d146/spec_sofia.json (anchor c0 = (c) with the sofia r1 controls), t2_d148/spec.json (anchor c_sh0),
#   t2_d149/spec.json (anchor c); every spec is bar d141.
#   mem-guard bash scripts/local/t2_d146_d149_reads.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
echo "### $(date +%T) blends"
blend --runs "$S/ni26-t2p-d146-fbc-r1-kit-1516836/runs" --protocols kit --out "$O/t2_d146/route_fbc_r1" --model fbcnet_bboth --recipe r1
blend --runs "$S/ni26-t2p-d146-fbc-r1-seen1-1516837/runs" --protocols seen1 --out "$O/t2_d146/route_fbc_r1" --model fbcnet_bboth --recipe r1
blend --runs "$S/ni26-t2p-d146-fbc-r1_mx4-kit-1516838/runs" --protocols kit --out "$O/t2_d146/route_fbc_mx4" --model fbcnet_bboth --recipe r1_mx4
blend --runs "$S/ni26-t2p-d146-fbc-r1_mx4-seen1-1516839/runs" --protocols seen1 --out "$O/t2_d146/route_fbc_mx4" --model fbcnet_bboth --recipe r1_mx4
blend --runs "$S/ni26-t2p-d148-shpu-1516841/runs" --out "$O/t2_d148/route_sh" --model shallow --align ea --recipe r1 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11
ls "$O"/t2_d146/route_fbc_r1/*.npz "$O"/t2_d146/route_fbc_mx4/*.npz "$O"/t2_d148/route_sh/*.npz | wc -l
for d in t2_d146:spec_sofia t2_d148:spec t2_d149:spec; do
  dir=${d%%:*}; sp=${d##*:}
  echo "### $(date +%T) $dir pkgshape"
  $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/$dir/$sp.json" --out "$O/$dir/read.json" > "$O/$dir/read.txt" 2>&1
  echo "rc $?"
done
echo "### $(date +%T) done"

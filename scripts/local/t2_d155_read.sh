#!/usr/bin/env bash
# T2-D155 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the four
# ni26-t2p-d155-fbc-<recipe>-<protocol>-<jobid> dirs (recipes r1 = the fresh-seed control, r1_pulx; kit + seen1; seeds 24-35):
# 1 the D124c blend (r 1/3, T2-D124b's cached fingerprints) of each recipe's FBCNet B3 + its per-user copies -> t2_d155/route_fbc_<recipe>
# 2 t2p_pkgshape on t2_d155/spec.json (anchor c0f = fresh r1 FBCNet + D146's sh0 / stl / mor; the one arm c_pulx decides on bar d141)
#   mem-guard bash scripts/local/t2_d155_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
echo "### $(date +%T) blends"
for r in r1 r1_pulx; do
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d155-fbc-$r-$p-*/ )
    [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $r $p: ${d[*]}"; exit 2; }
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d155/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds 24,25,26,27,28,29,30,31,32,33,34,35
  done
  echo "$r: $(ls "$O"/t2_d155/route_fbc_$r/*.npz | wc -l) blended files"
done
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d155/spec.json" --out "$O/t2_d155/read.json" > "$O/t2_d155/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

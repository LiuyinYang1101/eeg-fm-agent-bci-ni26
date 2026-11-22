#!/usr/bin/env bash
# T2-D191 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the four
# ni26-t2p-d191-fbc-<warp|plain>-<kit|seen1>-<jobid> dirs: D124c blends (r 1/3) of each -> t2_d191/route_<warp|plain>; c_iaf vs c0w (bar d141)
#   mem-guard bash scripts/local/t2_d191_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
for v in warp plain; do for p in kit seen1; do
  d=( "$S"/ni26-t2p-d191-fbc-$v-$p-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $v $p: ${d[*]}"; exit 2; }
  $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" --runs "${d[0]}runs" --protocols $p \
    --out "$O/t2_d191/route_$v" --model fbcnet_bboth --recipe r1_pulxx --seeds 72,73,74,75,76,77,78,79,80,81,82,83
done; done
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d191/spec.json" --out "$O/t2_d191/read.json" > "$O/t2_d191/read.txt" 2>&1
echo "rc $?"

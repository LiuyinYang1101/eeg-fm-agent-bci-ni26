#!/usr/bin/env bash
# T2-D176 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the four
# ni26-t2p-d176-<kit|seen1>-<a|b>-<jobid> dirs: links both seed halves into t2_d176/stl_runs, then t2p_pkgshape on t2_d176/spec.json
# (c_stsoup vs cS decides, bar d141; soup_alone / single_alone are the member-alone views, never deciding)
#   mem-guard bash scripts/local/t2_d176_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
mkdir -p "$O/t2_d176/stl_runs"
for p in kit seen1; do for h in a b; do
  d=( "$S"/ni26-t2p-d176-$p-$h-[0-9]*/ ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}runs" ] || { echo "need exactly one fetched dir for $p $h: ${d[*]}"; exit 2; }
  for f in "${d[0]}"runs/*.npz; do ln -sfn "$f" "$O/t2_d176/stl_runs/$(basename "$f")"; done
done; done
echo "$(ls "$O/t2_d176/stl_runs" | wc -l) linked files (expect 48)"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d176/spec.json" --out "$O/t2_d176/read.json" > "$O/t2_d176/read.txt" 2>&1
echo "rc $?"

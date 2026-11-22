#!/usr/bin/env bash
# T2-D159 / D160 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the five dirs
# ni26-t2p-d159-fbc-<r1|r1_pulxx>-<kit|seen1>-<jobid> (FBCNet, fresh seeds 48-59) and ni26-t2p-d159-sh-r1_pulx-<jobid> (ShallowNet, seeds 12-23):
# 1 FBCNet blends (r 1/3) of r1 and r1_pulxx; ShallowNet blends of r1_pulx at r 1/3 (D159) and r 1/2 (D160)
# 2 the spec = outputs/t2_d159/spec_template.json (committed sha in the rows) with the ShallowNet job id filled in -> spec.json
# 3 t2p_pkgshape: c_both vs c0h decides D159, c_both_shr2 vs c0h decides D160 (bar d141); c_fbc_only / c_sh_only beside
#   mem-guard bash scripts/local/t2_d159_read.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( $1 ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}/runs" ] || { echo "need exactly one fetched dir: $1" >&2; return 2; }; echo "${d[0]}"; }
echo "### $(date +%T) blends"
for r in r1 r1_pulxx; do for p in kit seen1; do
  d=$(one "$S/ni26-t2p-d159-fbc-$r-$p-*") || exit 2
  blend --runs "$d/runs" --protocols $p --out "$O/t2_d159/route_fbc_${r}_r3" --model fbcnet_bboth --recipe $r --seeds 48,49,50,51,52,53,54,55,56,57,58,59
done; done
sh=$(one "$S/ni26-t2p-d159-sh-r1_pulx-*") || exit 2
for p in kit seen1; do
  blend --runs "$sh/runs" --protocols $p --out "$O/t2_d159/route_sh_pulx_r3" --model shallow --align ea --recipe r1_pulx --seeds 12,13,14,15,16,17,18,19,20,21,22,23
  blend --runs "$sh/runs" --protocols $p --out "$O/t2_d159/route_sh_pulx_r2" --model shallow --align ea --recipe r1_pulx --r 0.5 --tag 2 --seeds 12,13,14,15,16,17,18,19,20,21,22,23
done
job=${sh##*-}; sed "s/ni26-t2p-d159-sh-r1_pulx-JOB/ni26-t2p-d159-sh-r1_pulx-$job/" "$O/t2_d159/spec_template.json" > "$O/t2_d159/spec.json"
echo "### $(date +%T) pkgshape"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d159/spec.json" --out "$O/t2_d159/read.json" > "$O/t2_d159/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

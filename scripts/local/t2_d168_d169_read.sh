#!/usr/bin/env bash
# T2-D168 / D169 reads (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of
#   d168: ni26-t2p-d168-fbc-<r1_pulxx|r1_pulxxx>-<kit|seen1>-<jobid> (FBCNet, fresh seeds 60-71; blends at r 1/3)
#   d169: ni26-t2p-d169-sh-<r1_pulx|r1_pul>-<jobid> (ShallowNet, fresh seeds 36-47; blends at r 1/2)
#   mem-guard bash scripts/local/t2_d168_d169_read.sh d168|d169
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( $1 ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}/runs" ] || { echo "need exactly one fetched dir: $1" >&2; return 2; }; echo "${d[0]}/runs"; }
case "${1:?d168|d169}" in
  d168)
    for r in r1_pulxx r1_pulxxx; do for p in kit seen1; do
      runs=$(one "$S/ni26-t2p-d168-fbc-$r-$p-[0-9]*") || exit 2
      blend --runs "$runs" --protocols $p --out "$O/t2_d168/route_fbc_$r" --model fbcnet_bboth --recipe $r --seeds 60,61,62,63,64,65,66,67,68,69,70,71
    done; done ;;
  d169)
    for r in r1_pulx r1_pul; do
      runs=$(one "$S/ni26-t2p-d169-sh-$r-[0-9]*") || exit 2
      for p in kit seen1; do
        blend --runs "$runs" --protocols $p --out "$O/t2_d169/route_sh_${r}_r2" --model shallow --align ea --recipe $r --r 0.5 --tag 2 --seeds 36,37,38,39,40,41,42,43,44,45,46,47
      done
    done ;;
esac
echo "### $(date +%T) pkgshape $1"
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_$1/spec.json" --out "$O/t2_$1/read.json" > "$O/t2_$1/read.txt" 2>&1
echo "rc $?"
echo "### $(date +%T) done"

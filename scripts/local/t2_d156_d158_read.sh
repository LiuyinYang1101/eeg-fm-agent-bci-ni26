#!/usr/bin/env bash
# T2-D156 / D157 / D158 reads (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of the job dirs:
#   d156  ni26-t2p-d156-fbc-<r1|r1_pulx|r1_pulxx>-<kit|seen1>-<jobid> (fresh seeds 36-47)
#   d158  ni26-t2p-d158-sh-r1_pulx-<jobid> (ShallowNet-EA seeds 0-11, kit + seen1)
# D156: blend each recipe at r 1/3 (tag 3); arm c_pulxx vs c0g decides (bar d141), c_pulx_g beside (D155's arm on fresh seeds)
# D157: the same files blended at r 1/2 (tag 2); arm c_pulxx_r2 vs c0g decides (bar d141), c_pulx_r2 beside
# D158: the ShallowNet blend (r 1/3) of the r1_pulx copies; c_shpx vs c0 (T2-D146's anchor) decides (bar d141)
#   mem-guard bash scripts/local/t2_d156_d158_read.sh d156|d158
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
one() { local d=( $1 ); [ ${#d[@]} -eq 1 ] && [ -d "${d[0]}/runs" ] || { echo "need exactly one fetched dir: $1"; exit 2; }; echo "${d[0]}/runs"; }
case "${1:?d156|d158}" in
  d156)
    echo "### $(date +%T) blends"
    for r in r1 r1_pulx r1_pulxx; do for p in kit seen1; do
      runs=$(one "$S/ni26-t2p-d156-fbc-$r-$p-*") || exit 2
      for t in 3 2; do
        [ $r = r1 ] && [ $t = 2 ] && continue
        rr=(); [ $t = 2 ] && rr=(--r 0.5 --tag 2)                     # tag 3 = the blend's default r 1/3 exactly (as D154 / D155)
        blend --runs "$runs" --protocols $p --out "$O/t2_d156/route_fbc_${r}_r$t" --model fbcnet_bboth --recipe $r "${rr[@]}" --seeds 36,37,38,39,40,41,42,43,44,45,46,47
      done
    done; done
    echo "### $(date +%T) pkgshape"
    $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d156/spec.json" --out "$O/t2_d156/read.json" > "$O/t2_d156/read.txt" 2>&1; echo "rc $?" ;;
  d158)
    runs=$(one "$S/ni26-t2p-d158-sh-r1_pulx-*") || exit 2
    echo "### $(date +%T) blends"
    for p in kit seen1; do
      blend --runs "$runs" --protocols $p --out "$O/t2_d158/route_sh_pulx" --model shallow --align ea --recipe r1_pulx --seeds 0,1,2,3,4,5,6,7,8,9,10,11
    done
    echo "### $(date +%T) pkgshape"
    $PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d158/spec.json" --out "$O/t2_d158/read.json" > "$O/t2_d158/read.txt" 2>&1; echo "rc $?" ;;
esac
echo "### $(date +%T) done"

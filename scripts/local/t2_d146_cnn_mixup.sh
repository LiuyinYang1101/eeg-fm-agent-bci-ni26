#!/usr/bin/env bash
# T2-D146: input mixup 0.4 (recipe r1_mx4) on the CNN members of (c), on the local 5090 = the hardware of every anchor CNN file
# (t2_d124c/b3_gpu.sh, sh_kit, sh_seen1). 4 shards per arm on one GPU (t2p_bench retries an OOM). Run through the GPU queue:
#   gpu-submit bash scripts/local/t2_d146_cnn_mixup.sh
# 1 sentinel  FBCNet B3 r1 seed 12 kit --peruser + ShallowNet-EA r1 seed 0 kit on the current code (vs the anchor files)
# 2 FBCNet B3 r1_mx4, seeds 12-23, kit + seen1, --peruser (per-user FT copies = ni26.train.PERUSER, unchanged)
# 3 ShallowNet-EA r1_mx4, seeds 0-11, kit + seen1
# 4 the D124c blend of (2): t2p_fpcalib blend --recipe r1_mx4 (r 1/3, fingerprint every window)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge/t2_d146
mkdir -p "$O"
bench() {   # $1 = out dir, rest = t2p_bench args; 4 shards in parallel
  local out=$1; shift
  for i in 0 1 2 3; do
    $PY -u tracks/t2_bci/scripts/t2p_bench.py --export "$E" --out "$out" "$@" --shard $i --nshards 4 > "$out.shard$i.log" 2>&1 &
  done
  wait
  grep -h "DONE\|Traceback\|Error" "$out".shard*.log | tail -5
}
echo "### $(date +%T) sentinel"
bench "$O/sentinel_fbc" --protocols kit --models fbcnet_bboth --aligns none --seeds 12 --peruser
bench "$O/sentinel_sh" --protocols kit --models shallow --aligns ea --seeds 0
echo "### $(date +%T) FBCNet B3 r1_mx4"
bench "$O/runs_fbc" --protocols kit,seen1 --models fbcnet_bboth --aligns none --recipe r1_mx4 \
  --seeds 12,13,14,15,16,17,18,19,20,21,22,23 --peruser
echo "### $(date +%T) ShallowNet-EA r1_mx4"
bench "$O/runs_sh" --protocols kit,seen1 --models shallow --aligns ea --recipe r1_mx4 --seeds 0,1,2,3,4,5,6,7,8,9,10,11
echo "### $(date +%T) blend"
$PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --runs "$O/runs_fbc" --out "$O/route_fbc" --model fbcnet_bboth \
  --recipe r1_mx4 > "$O/route_fbc.log" 2>&1; echo "blend rc $?"
ls "$O"/runs_fbc/*.npz "$O"/runs_sh/*.npz "$O"/route_fbc/*.npz 2>/dev/null | wc -l
echo "### $(date +%T) done"

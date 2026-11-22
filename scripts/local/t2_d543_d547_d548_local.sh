#!/usr/bin/env bash
# T2-D543 / D547 / D548 (registered 2026-11-16 20:03:46 / 20:11:20 before any number; zero grant, the local 5090 via gpu-submit, queued
# behind T3's start reads; ONE job, stages in order, each sharded 3 ways as D536's side job):
#   D547: side ShallowNet S5 at 65 epochs (recipe r1_pulx_vp2_e65_vr), seeds d + k * 1000, k 0..4, d 816-818 and 819-821, dumps on
#         -> outputs/t2_d547/side_sh65
#   D543: side ShallowNet / IFNet at seeds d + k * 1000, k 5..9 (the S10 half), d 816-818 and 819-821, dumps on (D536's side line)
#         -> outputs/t2_d543/side_{sh,ifn}
#   D548: FBCNet pooled at 240 epochs (recipe r1_pulxx_vp2_e240_vr_rc75), seed 816, lc9 (diagnosis only) -> outputs/t2_d548/fbc240
#   bash scripts/local/t2_d543_d547_d548_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d543_d547_d548_local.sh)" ] || { echo "commit the code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "$O/t2_d543_d547_d548_job.XXXX.sh")
seeds() { local ks=$1 out="" d k; for d in 816 817 818 819 820 821; do for k in $ks; do out+="$((d + k * 1000)),"; done; done; echo "${out%,}"; }
S5=$(seeds "0 1 2 3 4"); S59=$(seeds "5 6 7 8 9")
stage() {   # name dir line expected
  local name=$1 D=$2 line=$3 n=$4
  echo "mkdir -p '$D/runs'; git rev-parse HEAD > '$D/git_head.txt'; echo \"### \$(date +%T) $name: $line\" >> '$D/../arm.log'"
  for i in 0 1 2; do
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$D/runs' --shard $i --nshards 3 $line >> '$D/shard_$i.log' 2>&1 &"
  done
  echo "wait"
  echo "n=\$(ls '$D/runs' | wc -l); echo \"### done $name \$(date +%T): \$n files (expect $n), \$(cat '$D'/shard_*.log | grep -c Traceback) tracebacks\" >> '$D/../arm.log'; [ \$n -eq $n ] || rc=6"
}
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$O/t2_d547' '$O/t2_d543' '$O/t2_d548'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NI26_SIDE_DUMP=1"
  stage d547_side_sh65 "$O/t2_d547/side_sh65" "--protocols lc9 --models shallow --aligns ea --recipe r1_pulx_vp2_e65_vr --seeds $S5 --side dcmean" 300
  stage d543_side_sh "$O/t2_d543/side_sh" "--protocols lc9 --models shallow --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S59 --side dcmean" 300
  stage d543_side_ifn "$O/t2_d543/side_ifn" "--protocols lc9 --models ifnet --aligns ea --recipe r1_pulx_vp2_e180_vr --seeds $S59 --side dcmean" 300
  echo "unset NI26_SIDE_DUMP"
  stage d548_fbc240 "$O/t2_d548/fbc240" "--protocols lc9 --models fbcnet_bboth --aligns none --recipe r1_pulxx_vp2_e240_vr_rc75 --seeds 816" 10
  echo "echo \"### done d543 / d547 / d548 \$(date +%T) rc=\$rc\" >> '$O/t2_d543/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d543-d547-d548" bash "$job"

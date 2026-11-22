#!/usr/bin/env bash
# T2-D499 (Track1's seat C3 (i); registered 2026-11-14 before any number; zero grant, the local 5090 via gpu-submit): the same-day returning
# panel (t2p_bench protocol sd46: session 1 runs 1-4 train, runs 5-6 test), FBCNet (D419's line) + ShallowNet (D429's line) with copies + tv8c,
# seeds 770-772, each routed by the shipped blend at r 1/3 (tag 3), 1/2 (tag 2), 2/3 (tag 6) and 1 (tag 1); r 0 = the pooled net (the main file).
#   bash scripts/local/t2_d499_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt scripts/local/t2_d499_local.sh)" ] || { echo "commit the D499 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d499
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d499_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "fbc:t2d419_fbc_e180_s324:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75tv8c" "sh:t2d429_sh_ctl_s348:shallow:ea:r1_pulx_vp2_e180_vrtv8c"; do
    IFS=: read -r fam arm model al brec <<< "$spec"
    d=$O/sofia_s1/ni26-t2p-d499-${fam}-sd46-5090
    line="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/$arm.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' | sed -E 's/--protocols [a-z0-9,]+ /--protocols sd46 /; s/--seeds [0-9,]+ /--seeds 770,771,772 /')"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
    echo "echo \"### \$(date +%T) d499 $fam: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    for rt in "0.333333333333:3" "0.5:2" "0.666666666667:6" "1.0:1"; do
      IFS=: read -r r tag <<< "$rt"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols sd46 --out '$R/route_${fam}' --model $model --align $al --recipe $brec --r $r --tag $tag --seeds 770,771,772 >> '$d/arm.log' 2>&1 || rc=7"
    done
    echo "echo \"### done d499 $fam \$(date +%T) rc=\$rc; routed \$(ls '$R/route_${fam}' 2>/dev/null | wc -l) files\" >> '$d/arm.log'"
  done
  echo "echo \"### done d499 all \$(date +%T) rc=\$rc\" >> '$R/done.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d499" bash "$job"

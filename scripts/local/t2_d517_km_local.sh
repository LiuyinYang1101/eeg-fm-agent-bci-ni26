#!/usr/bin/env bash
# T2-D517 arm (b), keep-mean (de-meaning off), CNN side (registered 2026-11-15 before any number; zero grant, the local 5090 via gpu-submit): the three CNN families' M7
# lines (copies + tv8c) on the K = 9 panel (lc9, 10 folds), FRESH seeds 810-812 (the base draws) and 840-842 (part P3: the second net of each
# family's bag), plus ShallowNet with the stream-balanced loss (part P4, recipe r1_pulx_vp2_e180_vr_sb16) on the same six seeds; each routed
# per fold as shipped (t2p_fpcalib blend: FBCNet as D419, ShallowNet / IFNet r 0.5 tag 2), exactly as T2-D498's CNN side.
#   bash scripts/local/t2_d512_cnn_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt tracks/t2_bci/sofia/arms/t2d429_ifn_ctl_s348.txt scripts/local/t2_d512_cnn_local.sh)" ] || { echo "commit the D517 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d517
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
SEEDS=810,811,812
job=$(mktemp "$O/t2_d517_km_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "sh:t2d429_sh_ctl_s348:shallow:ea:r1_pulx_vp2_e180_vr:r1_pulx_vp2_e180_vrtv8c:--r 0.5 --tag 2 --xtag __km" \
              "ifn:t2d429_ifn_ctl_s348:ifnet:ea:r1_pulx_vp2_e180_vr:r1_pulx_vp2_e180_vrtv8c:--r 0.5 --tag 2 --xtag __km"; do
    IFS=: read -r fam arm model al rec brec bx <<< "$spec"
    d=$R/cnn_$fam
    line="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/$arm.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' | sed -E "s/--protocols [a-z0-9,]+ /--protocols lc9 /; s/--seeds [0-9,]+ /--seeds $SEEDS /; s/--recipe [a-z0-9_]+ /--recipe $rec /") --keep-mean"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
    echo "echo \"### \$(date +%T) d512 $fam: $line\" >> '$R/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$fam: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$R/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "for p in lc9_0 lc9_1 lc9_2 lc9_3 lc9_4 lc9_5 lc9_6 lc9_7 lc9_8 lc9_9; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/route_$fam' --model $model --align $al --recipe $brec $bx --seeds $SEEDS >> '$R/arm.log' 2>&1 || rc=7; done"
    echo "echo \"### done d517 $fam \$(date +%T) rc=\$rc; routed \$(ls '$R/route_$fam' 2>/dev/null | wc -l) files\" >> '$R/arm.log'"
  done
  echo "echo \"### done d517 cnn all \$(date +%T) rc=\$rc\" >> '$R/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d517-km" bash "$job"

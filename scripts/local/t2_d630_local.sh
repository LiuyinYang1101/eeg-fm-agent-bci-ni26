#!/usr/bin/env bash
# T2-D630 (registered + amended 2026-11-19 before any code or number; the local 5090 via gpu-submit, ONE job): the pooled-training composition.
# The lab's CNN stage line (t2_d536_cnn_local.sh's three family specs: the arm files' lines, --peruser --tta-vp 8 --tta-copies, routed by
# t2p_fpcalib blend as shipped) on the D630 protocols ss8 (the 8 single-session strangers) + rt (rt1 / rt2 x P04 / P10), seeds 880-882, in two
# PAIRED arms: B = unweighted (the ship composition), C = + --user-balance (1 / the user's training sessions; tag __ubw). Same rows / seeds /
# batches / augmentation draws; only the loss weights differ. -> outputs/t2_d630/{B,C}/{cnn,route}_<fam>; read: scripts/local/t2_d630_read.py.
#   mem-guard bash scripts/local/t2_d630_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt tracks/t2_bci/sofia/arms/t2d429_ifn_ctl_s348.txt scripts/local/t2_d630_local.sh)" ] || { echo "commit the D630 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d630
[ ! -e "$R" ] || { echo "$R exists: never overwrite a read's inputs"; exit 3; }
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
SEEDS=880,881,882
FOLDS="ss8_P01 ss8_P02 ss8_P03 ss8_P05 ss8_P06 ss8_P07 ss8_P08 ss8_P09 rt1_P04 rt1_P10 rt2_P04 rt2_P10"
job=$(mktemp "$O/t2_d630_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'; echo \"### \$(date +%T) d630 start (git \$(git rev-parse --short HEAD))\" >> '$R/arm.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "fbc:t2d419_fbc_e180_s324:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75:r1_pulxx_vp2_e180_vr_rc75tv8c:" \
              "sh:t2d429_sh_ctl_s348:shallow:ea:r1_pulx_vp2_e180_vr:r1_pulx_vp2_e180_vrtv8c:--r 0.5 --tag 2" \
              "ifn:t2d429_ifn_ctl_s348:ifnet:ea:r1_pulx_vp2_e180_vr:r1_pulx_vp2_e180_vrtv8c:--r 0.5 --tag 2"; do
    IFS=: read -r fam arm model al rec brec bx <<< "$spec"
    base="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/$arm.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' | sed -E "s/--protocols [a-z0-9,]+ /--protocols ss8,rt /; s/--seeds [0-9,]+ /--seeds $SEEDS /; s/--recipe [a-z0-9_]+ /--recipe $rec /")"
    grep -q -- "--peruser" <<< "$base" && grep -q -- "--tta-vp 8" <<< "$base" && grep -q -- "--tta-copies" <<< "$base" && grep -q -- "--protocols ss8,rt " <<< "$base" || { echo "unexpected $fam line: $base"; exit 4; }
    for A in B C; do
      line="$base"; xt=""
      [ $A = C ] && { line="$base --user-balance"; xt="--xtag __ubw"; }
      d=$R/$A/cnn_$fam
      echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
      echo "echo \"### \$(date +%T) d630 $A $fam: $line\" >> '$R/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      for i in 0 1 2; do
        echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
      done
      echo "wait"
      echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$A $fam: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$R/arm.log'; [ \$n -eq 0 ] || rc=6"
      echo "for p in $FOLDS; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/$A/route_$fam' --model $model --align $al --recipe $brec $bx $xt --seeds $SEEDS >> '$R/arm.log' 2>&1 || rc=7; done"
      echo "echo \"### done d630 $A $fam \$(date +%T) rc=\$rc; routed \$(ls '$R/$A/route_$fam' 2>/dev/null | wc -l) files\" >> '$R/arm.log'"
    done
  done
  echo "echo \"### done d630 all \$(date +%T) rc=\$rc\" >> '$R/arm.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d630" bash "$job"

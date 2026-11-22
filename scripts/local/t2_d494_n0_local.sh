#!/usr/bin/env bash
# T2-D494 (G16 item 1, label-shift-aware alignment, the manager's relay; registered 2026-11-14 before any number; zero grant, the local 5090
# via gpu-submit): the two EA families' M7 lines (ShallowNet / IFNet = T2-D429's control lines, seeds 348-359, copies + tv8c) with the EA
# population prior n0 = 10 and n0 = 30 windows (t2p_bench --n0: the stream's running covariance starts from n0 pseudo-windows of the
# training population's covariance, so the first windows' class mix is not absorbed into the alignment), routed as D429 (r 0.5, tag 2).
#   bash scripts/local/t2_d494_n0_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt tracks/t2_bci/sofia/arms/t2d429_ifn_ctl_s348.txt scripts/local/t2_d494_n0_local.sh)" ] || { echo "commit the D494 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_d494
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d494_n0_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for n0 in 10 30; do
    for spec in "sh:t2d429_sh_ctl_s348:shallow" "ifn:t2d429_ifn_ctl_s348:ifnet"; do
      IFS=: read -r fam arm model <<< "$spec"
      d=$O/sofia_s1/ni26-t2p-d494-${fam}n0${n0}-5090
      line="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/$arm.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //') --n0 $n0"
      echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
      echo "echo \"### \$(date +%T) d494 $fam n0 $n0: $line\" >> '$d/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      for i in 0 1 2; do
        echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
      done
      echo "wait"
      echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
      echo "for p in kit seen1; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/route_${fam}_n0${n0}' --model $model --align ea --n0tag n0${n0} --recipe r1_pulx_vp2_e180_vrtv8c --r 0.5 --tag 2 --seeds \$(seq -s, 348 359) >> '$d/arm.log' 2>&1 || rc=7; done"
      echo "echo \"### done d494 $fam n0 $n0 \$(date +%T) rc=\$rc; routed \$(ls '$R/route_${fam}_n0${n0}' 2>/dev/null | wc -l) files\" >> '$d/arm.log'"
    done
  done
  echo "echo \"### done d494 all \$(date +%T) rc=\$rc\" >> '$R/done.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d494" bash "$job"

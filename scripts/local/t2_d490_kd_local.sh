#!/usr/bin/env bash
# T2-D490 (seat G10 item 4; registered 2026-11-14 before any number; zero grant, the local 5090 via gpu-submit): the three CNN
# families' M7 lines trained with 0.5 hard CE + 0.5 T^2 KL to OUT-OF-FOLD cross-user soft targets (t2_d490_teacher.py; recipes _kd = control + kd 0.5),
# paired with their controls (FBCNet T2-D419 seeds 324-335; ShallowNet / IFNet T2-D429 seeds 348-359; both made on this GPU), then routed as
# the controls were (FBCNet: t2p_fpcalib blend defaults as D419; ShallowNet / IFNet: r 0.5 tag 2 as D429).
#   bash scripts/local/t2_d490_kd_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/sofia/arms/t2d419_fbc_e180_s324.txt tracks/t2_bci/sofia/arms/t2d429_sh_ctl_s348.txt tracks/t2_bci/sofia/arms/t2d429_ifn_ctl_s348.txt tracks/t2_bci/scripts/t2_d490_teacher.py scripts/local/t2_d490_kd_local.sh)" ] || { echo "commit the D490 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
KD="--kd-teacher $O/t2_d490/kdteacher__{protocol}__s0.npz"; [ -f "$O/t2_d490/kdteacher__kit__s0.npz" ] && [ -f "$O/t2_d490/kdteacher__seen1__s0.npz" ] || { echo "build the teacher first"; exit 3; }
R=$O/t2_d490
SC="systemd-run --user --scope -p MemoryHigh=10G -p MemoryMax=12G --quiet --"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d490_kd_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$R'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for spec in "fbc:t2d419_fbc_e180_s324:fbcnet_bboth:none:r1_pulxx_vp2_e180_vr_rc75_kdtv8c:324:335:" "sh:t2d429_sh_ctl_s348:shallow:ea:r1_pulx_vp2_e180_vr_kdtv8c:348:359:--r 0.5 --tag 2" "ifn:t2d429_ifn_ctl_s348:ifnet:ea:r1_pulx_vp2_e180_vr_kdtv8c:348:359:--r 0.5 --tag 2"; do
    IFS=: read -r fam arm model al brec s0 s1 bx <<< "$spec"
    d=$O/sofia_s1/ni26-t2p-d490-${fam}kd-5090
    line="$(grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/$arm.txt | grep -v '^[[:space:]]*$' | sed 's/^NSHARDS=[0-9]* //' | sed -E 's/--recipe (r1_pulxx?_vp2_e180_vr(_rc75)?) /--recipe \1_kd /') $KD"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; : > '$d/expected.txt'"
    echo "echo \"### \$(date +%T) d490 $fam: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    for i in 0 1 2; do
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' --shard $i --nshards 3 $line >> '$d/shard_$i.log' 2>&1 &"
    done
    echo "wait"
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "for p in kit seen1; do $SC $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export '$EXP' --fp '$O/t2_d124b/fp_proteus.npy' --runs '$d/runs' --protocols \$p --out '$R/route_${fam}_kd' --model $model --align $al --recipe $brec $bx --seeds \$(seq -s, $s0 $s1) >> '$d/arm.log' 2>&1 || rc=7; done"
    echo "echo \"### done d490 $fam \$(date +%T) rc=\$rc; routed \$(ls '$R/route_${fam}_kd' 2>/dev/null | wc -l) files\" >> '$d/arm.log'"
  done
  echo "echo \"### done d490 all \$(date +%T) rc=\$rc\" >> '$R/done.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d490" bash "$job"

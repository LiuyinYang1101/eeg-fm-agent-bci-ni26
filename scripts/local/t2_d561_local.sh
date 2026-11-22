#!/usr/bin/env bash
# T2-D561 STAGE 1 (the synthetic-dose question; registered 2026-11-17 before any number; INSIGHT ONLY: no ship path by Nov 20) on the
# local 5090, ONE gpu-submit job:
#   (0) D413c's calib-3 banks re-drawn from the SAME checkpoints (outputs/t2_d413b/gen_<p>_wfm30k.pt, sampler eu100, calib 3, bank 8 x n_train;
#       = t2_d413c_gate.sh's command) into outputs/t2_d561 (the D413b dir is never written); the bank sha256 vs D413c's recorded one is
#       REPORTED (a re-draw is fine: all three arms below share this bank);
#   (1) three arms per CNN family, PAIRED on fresh seeds 340-345, kit + seen1, copies + tv8c: syn 0.2 (mf20), syn 0.5 (mf50, D413c's dose),
#       control; FBCNet tilt (its shipped align none) + ShallowNet + IFNet (the owner's "why only fbcnet"; ALIGN NONE for all three arms of
#       each: the syn path needs it, synthetic interpolated users have no EA whitening; the shipped sh / ifn lines are EA).
# Outputs: outputs/sofia_s1/ni26-t2p-d561-<arm>-5090/. Read: mem-guard bash scripts/local/t2_d561_read.sh
#   bash scripts/local/t2_d561_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/scripts/t2p_meanflow.py tracks/t2_bci/sofia/arms/t2d561_*.txt tests/test_syn.py scripts/local/t2_d561_local.sh scripts/local/t2_d561_read.sh)" ] || { echo "commit the D561 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
G=$(pwd)/outputs/t2_d413b; O=$(pwd)/outputs/t2_d561; S=$(pwd)/outputs/sofia_s1
for p in kit seen1; do [ -s "$G/gen_${p}_wfm30k.pt" ] || { echo "missing $G/gen_${p}_wfm30k.pt"; exit 3; }; done
[ ! -e "$O" ] || { echo "$O exists: never overwrite a read's inputs"; exit 3; }
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
MF="$SC $PY -u tracks/t2_bci/scripts/t2p_meanflow.py --export '$EXP' --out '$O'"
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_d561_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$O'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### \$(date +%T) d561 start (git \$(git rev-parse --short=12 HEAD))\" >> '$O/run.log'"
  echo "for p in kit seen1; do $MF --protocol \$p --tag final3 --from-ckpt '$G/gen_'\$p'_wfm30k.pt' --sample eu100 --calib 3 --bank-mult 8 >> '$O/run.log' 2>&1 || exit 4; done"
  echo "$PY -c \"import json; [print('### d561 bank', p, 'sha256', json.load(open('$O/info_'+p+'_final3.json'))['sha256']['bank'][:12], '| D413c', json.load(open('$G/info_'+p+'_final3.json'))['sha256']['bank'][:12]) for p in ('kit', 'seen1')]\" >> '$O/run.log' 2>&1"
  for arm in fbc_tl15mf20_s340 fbc_tl15mf_s340 fbc_tl15_s340 sh_mf20_s340 sh_mf50_s340 sh_ctl_s340 ifn_mf20_s340 ifn_mf50_s340 ifn_ctl_s340; do
    d="$S/ni26-t2p-d561-$arm-5090"
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d561_$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d561 $arm: $line\" >> '$d/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  done
  echo "echo \"### done d561 \$(date +%T) rc=\$rc\" >> '$O/run.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
for arm in fbc_tl15mf20_s340 fbc_tl15mf_s340 fbc_tl15_s340 sh_mf20_s340 sh_mf50_s340 sh_ctl_s340 ifn_mf20_s340 ifn_mf50_s340 ifn_ctl_s340; do
  d="$S/ni26-t2p-d561-$arm-5090"; [ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
done
gpu-submit -L "neural_interface_challenge:t2-d561-dose" bash "$job"

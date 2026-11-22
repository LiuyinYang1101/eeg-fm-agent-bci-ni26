#!/usr/bin/env bash
# T2-D430 (panel P7 I1 = M1, stranger-safe copies: FBCNet ctl + kl, ShallowNet kl, IFNet kl; the ShallowNet / IFNet controls = D429's dirs) on the local 5090 (zero grant cost), ONE gpu-submit job, arms one after the other (ShallowNet and IFNet first, FBCNet last), each in its own dir
#   t2d430_{sh_kl,ifn_kl,fbc_ctl,fbc_kl}_s348 -> outputs/sofia_s1/ni26-t2p-d430-{sh-kl,ifn-kl,fbc-ctl,fbc-kl}-5090; seeds 348-359, kit + seen1, copies + tv8c.
# Read: mem-guard bash scripts/local/t2_d430_read.sh
#   bash scripts/local/t2_d430_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d430_*.txt scripts/local/t2_d430_read.sh)" ] || { echo "commit the D430 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
ARMS=${D430_ARMS:-"t2d430_sh_kl_s348 t2d430_ifn_kl_s348 t2d430_fbc_ctl_s348 t2d430_fbc_kl_s348"}
dir_of() { case "$1" in t2d430_*_s348) local k=${1#t2d430_}; k=${k%_s348}; echo "$S/ni26-t2p-d430-${k//_/-}-5090";; *) echo "unknown arm $1" >&2; exit 2;; esac; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d430_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in $ARMS; do
    d=$(dir_of "$arm")
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d430 $arm: $line\" >> '$d/arm.log'"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
    echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
    echo "echo \"### done $arm \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  done
  echo "exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
for arm in $ARMS; do
  d=$(dir_of "$arm")
  [ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
done
gpu-submit -L "neural_interface_challenge:t2-d430" bash "$job"

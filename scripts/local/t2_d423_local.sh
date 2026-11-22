#!/usr/bin/env bash
# T2-D423 Stage B on the local 5090 (zero grant; 2026-11-10, before any Stage B number: the sofia queue estimate put the rb arm ≈ 14 h out).
# ONE gpu-submit job: the rb arm t2d423_sh_rb_s336 on a codec job's fetched re-expressed export, dir outputs/sofia_s1/ni26-t2p-d423-sh-<arm>-5090.
# The paired control is the local ni26-t2p-d424-sh-ctl-5090 (the same arm file t2d423_sh_ctl_s336, same seeds, same hardware).
# Read: D423_LOCAL=1 mem-guard bash scripts/local/t2_d423_read.sh <arm>
#   bash scripts/local/t2_d423_local.sh rbS <codec job dir name> [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=$1; CJ=$2
case "$ARM" in rbI|rbS) ;; *) echo "unknown arm $ARM" >&2; exit 2;; esac
[ -z "$(git status --porcelain -- src/ni26/stream.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d423_sh_rb_s336.txt scripts/local/t2_d423_read.sh)" ] || { echo "commit the code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
S=$(pwd)/outputs/sofia_s1
EXP="$S/$CJ/export_rb"
[ -f "$EXP/X.npy" ] && [ -f "$EXP/fidelity.json" ] || { echo "no fetched export at $EXP" >&2; exit 2; }
grep -q '"killed": false' "$EXP/fidelity.json" || { echo "$CJ did not pass Stage A" >&2; exit 2; }
d="$S/ni26-t2p-d423-sh-$ARM-5090"
job=$(mktemp "${TMPDIR:-/tmp}/t2_d423_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'; echo '$EXP' > '$d/export.txt'"
  grep -v '^[[:space:]]*#' tracks/t2_bci/sofia/arms/t2d423_sh_rb_s336.txt | grep -v '^[[:space:]]*$' | while read -r line; do
    echo "echo \"### \$(date +%T) d423 $ARM: $line\" >> '$d/arm.log'"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
    echo "$SC $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
  done
  echo "n=\$(sort -u '$d/expected.txt' | while read -r f; do [ -f '$d/runs/'\"\$f\" ] || echo miss; done | wc -l); echo \"$d: \$(sort -u '$d/expected.txt' | wc -l) expected, \$n missing\" >> '$d/arm.log'; [ \$n -eq 0 ] || rc=6"
  echo "echo \"### done $ARM \$(date +%T) rc=\$rc\" >> '$d/arm.log'"
  echo "exit \$rc"
} > "$job"
if [ "${3:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
[ ! -e "$d/runs" ] || [ -z "$(ls -A "$d/runs" 2>/dev/null)" ] || { echo "$d/runs is not empty"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d423" bash "$job"

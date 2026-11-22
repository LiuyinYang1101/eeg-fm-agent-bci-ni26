#!/usr/bin/env bash
# T2-D424 (filter-bank EA for M7's ShallowNet) on the local 5090 (zero grant cost), ONE gpu-submit job, two arms one after the other, each in its own dir:
#   t2d423_sh_ctl_s336  -> outputs/sofia_s1/ni26-t2p-d424-sh-ctl-5090   (the PAIRED control on the same hardware: M7's ShallowNet-EA arm file)
#   t2d424_sh_fbea_s336 -> outputs/sofia_s1/ni26-t2p-d424-sh-fbea-5090  (the same with --aligns fbea)
# seeds 336-347, kit + seen1, copies + tv8c. Each python call in a MemoryMax=20G scope (t2_d419_local.sh's pattern).
# D424_ARMS overrides the list (e.g. D424_ARMS=t2d424_sh_fbea_s336 when the control dir already exists).
# T2-D426 (2026-11-10): D424_ARMS=t2d426_sh_eab_s336 runs the EA-β arm (dir ni26-t2p-d426-sh-eab-5090) against the same control.
# T2-D427: D424_ARMS=t2d427_sh_eal_s336 (the 0-8 Hz EA matrix; dir ni26-t2p-d427-sh-eal-5090).
# T2-D428: D424_ARMS=t2d428_sh_eaf_s336 (forgetting EA, gamma 0.98; dir ni26-t2p-d428-sh-eaf-5090).
# Read: mem-guard bash scripts/local/t2_d424_read.sh
#   bash scripts/local/t2_d424_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/stream.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d423_sh_ctl_s336.txt tracks/t2_bci/sofia/arms/t2d424_*.txt tracks/t2_bci/sofia/arms/t2d426_*.txt tracks/t2_bci/sofia/arms/t2d427_*.txt tracks/t2_bci/sofia/arms/t2d428_*.txt scripts/local/t2_d424_read.sh scripts/local/t2_d426_read.sh scripts/local/t2_d427_read.sh scripts/local/t2_d428_read.sh)" ] || { echo "commit the D424 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
ARMS=${D424_ARMS:-"t2d423_sh_ctl_s336 t2d424_sh_fbea_s336"}
dir_of() { case "$1" in t2d423_sh_ctl_s336) echo "$S/ni26-t2p-d424-sh-ctl-5090";; t2d424_sh_fbea_s336) echo "$S/ni26-t2p-d424-sh-fbea-5090";; t2d426_sh_eab_s336) echo "$S/ni26-t2p-d426-sh-eab-5090";; t2d427_sh_eal_s336) echo "$S/ni26-t2p-d427-sh-eal-5090";; t2d428_sh_eaf_s336) echo "$S/ni26-t2p-d428-sh-eaf-5090";; *) echo "unknown arm $1" >&2; exit 2;; esac; }
job=$(mktemp "${TMPDIR:-/tmp}/t2_d424_local.XXXX.sh")
SC="systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet --"
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "echo \"### git \$(git rev-parse --short=12 HEAD)\""
  for arm in $ARMS; do
    d=$(dir_of "$arm")
    echo "mkdir -p '$d/runs'; git rev-parse HEAD > '$d/git_head.txt'"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) d424 $arm: $line\" >> '$d/arm.log'"
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
gpu-submit -L "neural_interface_challenge:t2-d424" bash "$job"

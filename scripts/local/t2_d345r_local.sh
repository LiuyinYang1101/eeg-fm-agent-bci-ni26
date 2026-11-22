#!/usr/bin/env bash
# T2-D345r (fresh IFNet-EA seeds 168-179) on the local 5090 (zero grant cost): the t2d345r_ifnet_s168 arms file in ONE gpu-submit job,
# each bench call in a MemoryMax=20G scope (the pattern of t2_d310_local.sh). Outputs: outputs/sofia_s1/ni26-t2p-d345r-ifnet-s168-5090/runs.
#   bash scripts/local/t2_d345r_local.sh --dry      (prints the commands, submits nothing)
#   bash scripts/local/t2_d345r_local.sh            (submits; then the D345r blend + read)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d345r_*.txt)" ] || { echo "commit the D345r code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
job=$(mktemp "${TMPDIR:-/tmp}/t2_d345r_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for arm in ifnet_s168; do
    d="$S/ni26-t2p-d345r-${arm//_/-}-5090"
    if [ "${1:-}" != --dry ]; then
      ls -d "$S"/ni26-t2p-d345r-${arm//_/-}-[0-9]*/ >/dev/null 2>&1 && { echo "a d345r-${arm//_/-} dir already exists under $S" >&2; exit 2; }
      mkdir -p "$d/runs"
    fi
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d345r_$arm.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) $arm: $line\""
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
  done
  echo "for d in '$S'/ni26-t2p-d345r-*-5090; do n=\$(sort -u \"\$d/expected.txt\" | while read -r f; do [ -f \"\$d/runs/\$f\" ] || echo miss; done | wc -l); echo \"\$d: \$(sort -u \"\$d/expected.txt\" | wc -l) expected, \$n missing\"; [ \$n -eq 0 ] || rc=6; done"
  echo "echo \"### done \$(date +%T) rc=\$rc\"; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit bash "$job"

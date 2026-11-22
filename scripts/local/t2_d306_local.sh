#!/usr/bin/env bash
# T2-D306 on the local 5090 (zero grant): the softclip arms (ch / glob), ONE gpu-submit job, each bench call in a MemoryMax=20G scope; outputs
# outputs/sofia_s1/ni26-t2p-d306-<mode>-5090/runs. ≈ 2 x 16 min. Then `mem-guard bash scripts/local/t2_d306_read.sh`.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/sofia/arms/t2d306_*.txt)" ] || { echo "commit the D306 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
job=$(mktemp "${TMPDIR:-/tmp}/t2_d306_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for m in ch glob; do
    d="$S/ni26-t2p-d306-$m-5090"
    ls -d "$S"/ni26-t2p-d306-$m-[0-9]*/ >/dev/null 2>&1 && { echo "a d306-$m dir already exists" >&2; exit 2; }
    mkdir -p "$d/runs"
    grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d306_$m.txt" | grep -v '^[[:space:]]*$' | while read -r line; do
      echo "echo \"### \$(date +%T) $m: $line\""
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
  done
  echo "for d in '$S'/ni26-t2p-d306-*-5090; do n=\$(sort -u \"\$d/expected.txt\" | while read -r f; do [ -f \"\$d/runs/\$f\" ] || echo miss; done | wc -l); echo \"\$d: \$(sort -u \"\$d/expected.txt\" | wc -l) expected, \$n missing\"; [ \$n -eq 0 ] || rc=6; done"
  echo "echo \"### done \$(date +%T) rc=\$rc\"; exit \$rc"
} > "$job"
gpu-submit bash "$job"

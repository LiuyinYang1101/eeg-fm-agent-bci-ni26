#!/usr/bin/env bash
# T2-D293 local fallback (lead contingency 2026-10-28 18:2x): if sofia's acc nodes stay unavailable, run the three D293 arms files on the
# local 5090 instead of jobs 1541880-82 (D294 / D295 stay on sofia). ONE gpu-submit job, the arms one after another, each line in a
# MemoryMax=20G scope; outputs land where t2_d293_read.sh looks: outputs/sofia_s1/ni26-t2p-d293-<arm>-5090/runs (the read wants exactly one
# dir per arm, so a later sofia fetch of the same arm must be set aside, or the 1541880-82 jobs scancelled when this runs).
# Each arm dir also re-runs the lambda 0 control (the line's own recipe, same seeds; lead 18:3x: argmax agreement 0.98-1.00 with the sofia
# anchor is not tight enough to pair a ~+0.003 effect), so the read is local vs local: LOCAL=1 t2_d293_read.sh.
# Sizing (local smoke + same-seed anchor today: ~80 s per seed x lambda for all four cells): 36 seed x recipe groups ~ 48 min, < 20 GB RAM
# (export 203 MB, memmapped).
#   bash scripts/local/t2_d293_local.sh --dry      (prints the commands, submits nothing)
#   bash scripts/local/t2_d293_local.sh            (submits the gpu-submit job; then `LOCAL=1 mem-guard bash scripts/local/t2_d293_read.sh`)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/sofia/arms/t2d293_*.txt)" ] || { echo "commit the D293 code first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$(pwd)/outputs/sofia_s1
job=$(mktemp "${TMPDIR:-/tmp}/t2_d293_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  for arm in fbc_kit fbc_seen1 sh; do
    d="$S/ni26-t2p-d293-${arm//_/-}-5090"
    if [ "${1:-}" != --dry ]; then
      ls -d "$S"/ni26-t2p-d293-${arm//_/-}-[0-9]*/ >/dev/null 2>&1 && { echo "a d293-${arm//_/-} dir already exists under $S" >&2; exit 2; }
      mkdir -p "$d/runs"
    fi
    arms=$(grep -v '^[[:space:]]*#' "tracks/t2_bci/sofia/arms/t2d293_$arm.txt" | grep -v '^[[:space:]]*$')
    ctl=$(grep '_ua1 ' <<< "$arms" | sed 's/_ua1 / /')     # the lambda 0 control (the line's own recipe), same seeds: the read is local vs local
    [ "$(wc -l <<< "$ctl")" -eq 1 ] && ! grep -q '_ua[0-9]' <<< "$ctl" || { echo "cannot derive the control line for $arm" >&2; exit 2; }
    printf '%s\n%s\n' "$ctl" "$arms" | while read -r line; do
      echo "echo \"### \$(date +%T) $arm: $line\""
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line --list-outputs '$d/expected.txt' >/dev/null 2>&1 || rc=5"
      echo "systemd-run --user --scope -p MemoryHigh=16G -p MemoryMax=20G --quiet -- $PY -u tracks/t2_bci/scripts/t2p_bench.py --export '$EXP' --out '$d/runs' $line >> '$d/arm.log' 2>&1 || rc=\$?"
    done
  done
  echo "for d in '$S'/ni26-t2p-d293-*-5090; do n=\$(sort -u \"\$d/expected.txt\" | while read -r f; do [ -f \"\$d/runs/\$f\" ] || echo miss; done | wc -l); echo \"\$d: \$(sort -u \"\$d/expected.txt\" | wc -l) expected, \$n missing\"; [ \$n -eq 0 ] || rc=6; done"
  echo "echo \"### done \$(date +%T) rc=\$rc\"; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit bash "$job"

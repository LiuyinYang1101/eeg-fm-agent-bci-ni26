#!/usr/bin/env bash
# T2-D482 runtime gate, same-GPU form (written 2026-11-14 before any number; the H200 queue holds the direct timing 1616363 with a start
# estimate of Nov 16): on the local 5090, through t2p_fm_stack_time (t2p_repack load() + run_stream(), every public stream, one window at a
# time, after an untimed warm-up stream), one job, one after another:
#   w13 --dups 0,3 : M7-W13 as built, and with three copies of its first three ST-L = the stand-in the H200 measured at 51.31 ms (1615577)
#   w14 --dups 0   : the built M7-W14 (five plain + three CAR ST-L) as is
# H200-basis estimate of W14 = 51.31 ms x (ms_w14 / ms_w13+3) on the 5090 (the real package against its stand-in on one GPU; the CAR step is
# the only difference in compute). Gate: <= 52 ms (the user's bar). If the direct H200 timing lands, it governs.
#   bash scripts/local/t2_d482_time_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d482_time_local.sh)" ] || { echo "commit first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
S=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1
d=$HOME/workspace/outputs/neural_interface_challenge/t2_d482/time5090
Z13=$S/ni26-t2p-repack-t2p_ms_M7_w13-1615509/t2p_ms_M7_w13.zip
Z14=$S/ni26-t2p-repack-t2p_ms_M7_w14-1615940/t2p_ms_M7_w14.zip
for z in "$Z13" "$Z14"; do [ -f "$z" ] || { echo "missing $z"; exit 2; }; done
job=$(mktemp "${TMPDIR:-/tmp}/t2_d482_time_local.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; rc=0; mkdir -p '$d'; git rev-parse HEAD > '$d/git_head.txt'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4"
  echo "$PY -u tracks/t2_bci/scripts/t2p_fm_stack_time.py --zip '$Z13' --export '$EXP' --out '$d/w13' --dups 0,3 > '$d/w13.log' 2>&1 || rc=\$?"
  echo "$PY -u tracks/t2_bci/scripts/t2p_fm_stack_time.py --zip '$Z14' --export '$EXP' --out '$d/w14' --dups 0 > '$d/w14.log' 2>&1 || rc=\$?"
  echo "echo \"### done d482 time5090 \$(date +%T) rc=\$rc\" >> '$d/done.log'; exit \$rc"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d482-time" bash "$job"

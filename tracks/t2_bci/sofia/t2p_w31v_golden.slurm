#!/usr/bin/env bash
# T2-D654 (W31v) the split-graph view template's GOLDEN + TIMING check on one H200 (the manager's first gate): from the base ZIP A (W25-ALL6) build
# B = W31v gate "off" (K 2) and C = W31v gated (K 2, q 0.1, the CNN gate) with t2p_w31v_patch.py, then t2p_w31v_golden.py A / B / C over the export.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w31v_golden.slurm --time=01:30:00 --job-name=... -- <base zip under runs/ni26> <name> [q]
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
A="$S1_ROOT/runs/ni26/${1:?base zip under runs/ni26}"; NAME=${2:?name}; Q=${3:-0.1}
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
[ -f "$A" ] || { echo "!! no base zip $A"; s1_finish 2 "$OUT" none 1; }
W="${S1_JOBTMP:-/tmp}/w31v"; mkdir -p "$W"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_patch.py" zip "$A" "$W/B_off.zip" '{"K": 2, "q": 0.0, "gate": "off"}' || s1_finish 3 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_patch.py" zip "$A" "$W/C_q.zip" "{\"K\": 2, \"q\": $Q, \"gate\": \"cnn\"}" || s1_finish 3 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_golden.py" --export "$D/export_proteus_v1" --out "$OUT/$NAME" --zip "A=$A" --zip "B=$W/B_off.zip" --zip "C=$W/C_q.zip" --reps 2; rc=$?
s1_finish $rc "$OUT" "$NAME.json" 1

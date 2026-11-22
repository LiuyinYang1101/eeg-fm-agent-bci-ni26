#!/usr/bin/env bash
# T2-D475e (the manager's ask, 2026-11-14): MEASURED ms / window of a gated package with k extra ST-L members (copies as stand-ins for
# k CAR members), through t2p_repack's load() + run_stream() on every public stream; implied sealed seconds vs 3,400 / 3,600 s.
# Timing only: no training, no new package.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_fm_stack_time.slurm --time=01:30:00 --job-name=ni26-t2p-stacktime-<name> \
#       -- <gated zip path on sofia> <name> [--dups 0,2,3,5]
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
OUT=$(s1_outdir)
ZIP="$1"; NAME="$2"; shift 2
[ -f "$ZIP" ] || { echo "!! no zip $ZIP"; s1_finish 2 "$OUT" 'none' 1; }
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name --format=csv,noheader
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_fm_stack_time.py" --zip "$ZIP" --export "$D/export_proteus_v1" --out "$OUT/$NAME" "$@"
rc=$?
s1_finish "$rc" "$OUT" "$NAME.stack_time.json" 1

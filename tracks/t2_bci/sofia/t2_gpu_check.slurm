#!/usr/bin/env bash
# a short GPU check on zen5_vis (free): runs one tests/gpu_*.py from the synced snapshot.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_gpu_check.slurm --partition=zen5_vis --account=vsc --time=00:15:00 -- tests/gpu_<x>.py
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8   # = t2p_w19_build.slurm
cd "$S1_CODE" && python -u "$1" 2>&1 | tee "$OUT/check.log"; rc=${PIPESTATUS[0]}
s1_finish "$rc" "$OUT" "check.log" 1

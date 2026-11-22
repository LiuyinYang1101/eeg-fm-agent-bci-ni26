#!/usr/bin/env bash
# T2-D654 (FM view-TTA) timing probe on one H200 (D517's timing host): t2p_d654_fmprobe.py on a packaged ZIP under runs/ni26.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_d654_probe.slurm --time=00:30:00 --job-name=... -- <zip under runs/ni26> <name>
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
ZIP="$S1_ROOT/runs/ni26/${1:?zip under runs/ni26}"; NAME=${2:?name}
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
[ -f "$ZIP" ] || { echo "!! no zip $ZIP"; s1_finish 2 "$OUT" none 1; }
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_d654_fmprobe.py" --zip "$ZIP" --export "$D/export_proteus_v1" --out "$OUT/$NAME" --n 300; rc=$?
s1_finish $rc "$OUT" "$NAME.json" 1

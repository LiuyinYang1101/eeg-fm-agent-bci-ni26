#!/usr/bin/env bash
# Track 2: balanced accuracy of packaged zips with one flat channel per stream, each channel in turn (t2p_flatba.py).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_flatba.slurm --time=00:30:00 --job-name=ni26-t2p-flatba \
#       -- <zip1,zip2 (paths on sofia)> <label1,label2>
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
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_flatba.py" --zips "$1" --labels "$2" --export "$D/export_proteus_v1" --out "$OUT/flatba.json"
s1_finish $? "$OUT" 'flatba.json' 1

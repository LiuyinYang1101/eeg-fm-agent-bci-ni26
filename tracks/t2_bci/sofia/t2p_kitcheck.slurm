#!/usr/bin/env bash
# Re-check an EXISTING T2 package zip with the official kit under the worker-image import guard (no training):
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_kitcheck.slurm --time=01:00:00 --job-name=ni26-t2p-kitcheck-<name> \
#       -- <zip path on sofia> <name>
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
ZIP="$1"; NAME="$2"
[ -f "$ZIP" ] || { echo "!! no zip $ZIP"; s1_finish 2 "$OUT" 'none' 1; }
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
cp "$ZIP" "$OUT/$NAME.zip"
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

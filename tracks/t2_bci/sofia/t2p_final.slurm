#!/usr/bin/env bash
# Track 2 PROTEUS package: t2p_final.py (fixed-epoch training, package, golden replay) then the OFFICIAL kit on the
# package, inference-only like Codabench, from a READ-ONLY copy: (a) Simulated 41 x 480 @ 120 Hz, (b) the PROTEUS
# warm-up (BCI[study=dreyer2026proteus], test P04 + P09). Args after `--` go to t2p_final.py (--out is set here).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_final.slurm --time=01:00:00 --job-name=ni26-t2p-pkg-<name> \
#       -- <name> --model shallow --align ea --epochs 40 --seeds 0,1,2 --train kit [--tta prior:1:3] [--extra ...]
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
NAME="$1"; shift
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_final.py" --export "$D/export_proteus_v1" --out "$OUT/$NAME" "$@" || s1_finish 3 "$OUT" 'none' 1
# official kit (Simulated + warm-up) on a read-only unzip of the package, under the worker-image import guard
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

#!/usr/bin/env bash
# T2-D654 ship path (W31v = the base ZIP + the split-graph view template + the picked arm's fm_views; no donors): (1) t2p_w31v_patch.py zip -> the
# candidate ZIP in this job's run dir, (2) the structural check vs the base, (3) the golden over the export (A = base vs B = the candidate: differing
# predictions, max |dp|, ms, the realised view fraction; Track3's view checks first) saving B's predictions for K2. Then launch_d654_build.sh chains K2
# (t2p_w19_kitfull.slurm) and the official 3-rep timing (t2p_time_zip.slurm).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w31v_build.slurm --time=01:30:00 --job-name=... -- <base zip under runs/ni26> <name> '<fm_views JSON>'
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
A="$S1_ROOT/runs/ni26/${1:?base zip under runs/ni26}"; NAME=${2:?name}; VIEWS=${3:?fm_views JSON}
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
[ -f "$A" ] || { echo "!! no base zip $A"; s1_finish 2 "$OUT" none 1; }
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_patch.py" zip "$A" "$OUT/$NAME.zip" "$VIEWS" || s1_finish 3 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_patch.py" check "$A" "$OUT/$NAME.zip" || s1_finish 4 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w31v_golden.py" --export "$D/export_proteus_v1" --out "$OUT/$NAME.golden" --zip "A=$A" --zip "B=$OUT/$NAME.zip" \
  --reps 1 --save-pred B; rc=$?
[ $rc -eq 0 ] && cp "$OUT/$NAME.golden.B.pred.npz" "$OUT/$NAME.pred.npz"
s1_finish $rc "$OUT" "$NAME.pred.npz" 1

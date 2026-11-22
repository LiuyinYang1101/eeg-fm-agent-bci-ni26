#!/usr/bin/env bash
# T2-D544 stage 1b: the weight interpolation 0.5 * pinned ST-L root + 0.5 * the stage-1 word expert (t2p_interp_ckpt.py), CPU only -> zen5_vis.
#   S1_EXPORT=RUN_NAME=ni26-t2p-d544-wordalpha,EXPERT=ni26-t2p-preft-d544-word-root S1_PURPOSE="..." \
#     s1 submit ni26 tracks/t2_bci/sofia/t2p_interp.slurm --partition=zen5_vis --account=vsc --time=00:30:00 --dependency=afterok:<stage 1>
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26/t2_proteus"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=16
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_interp_ckpt.py" --expert "$S1_ROOT/runs/ni26/${EXPERT:?EXPERT run name}" \
    --names-from "$D/export_proteus_v1" --alpha 0.5 --out "$OUT"
rc=$?
s1_finish "$rc" "$OUT" 'checkpoint-cpt.pth' 1

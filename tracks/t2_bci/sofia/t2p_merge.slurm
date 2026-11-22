#!/usr/bin/env bash
# T2-D546b build PREPARATION (not a package, not a candidate): one merged FM donor = BASE donor with slots replaced by SWAP members
# (t2p_merge_donors.py, CPU only -> zen5_vis). Args after `--` go to t2p_merge_donors.py; --out is set here ($OUT/$NAME.zip).
#   S1_EXPORT=RUN_NAME=<run>,NAME=<zip name> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_merge.slurm --partition=zen5_vis \
#     --account=vsc --time=00:45:00 -- --base <zip> --swap <zip> --slots 2,4 --swap-index 0,1 --keep-side
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_merge_donors.py" "$@" --out "$OUT/${NAME:?NAME}.zip"
rc=$?
[ $rc = 0 ] && sha256sum "$OUT/$NAME.zip" > "$OUT/$NAME.zip.sha256"
s1_finish "$rc" "$OUT" "$NAME.zip" 1

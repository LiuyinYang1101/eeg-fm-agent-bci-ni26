#!/usr/bin/env bash
# T2-D129: continuous-session PROTEUS crops (t2p_session_export.py) -> data/ni26/t2_proteus/cpt_pools/proteus_sessions_v1 (free CPU)
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_session_export.slurm --partition=zen5_vis --time=02:00:00 --job-name=ni26-t2p-sessexp
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
DST="$D/cpt_pools/proteus_sessions_v1"
[ -s "$DST/meta.json" ] && { echo "### exists: $DST"; cp "$DST/meta.json" "$OUT/"; s1_finish 0 "$OUT" 'meta.json' 1; }
export OMP_NUM_THREADS=8 NUMPY_MADVISE_HUGEPAGE=0
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_session_export.py" --bids "$D/benchopt/neural_compet/Dreyer2026Proteus/download/nm000290" \
    --names-from "$D/export_proteus_v1" --out "$DST" --crop-s 4 --hop-s 2
rc=$?
cp "$DST/meta.json" "$OUT/" 2>/dev/null
s1_finish "$rc" "$OUT" 'meta.json' 1

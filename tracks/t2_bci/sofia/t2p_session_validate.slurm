#!/usr/bin/env bash
# T2-D129: validate the session export's preprocessing / timing against the contract windows (t2p_session_export --validate-only)
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_session_export.py" --bids "$D/benchopt/neural_compet/Dreyer2026Proteus/download/nm000290" \
    --names-from "$D/export_proteus_v1" --out "$OUT/none" --validate-only 6 | tee "$OUT/validate.txt"
s1_finish $? "$OUT" 'validate.txt' 1

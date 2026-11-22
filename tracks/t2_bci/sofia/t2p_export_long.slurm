#!/usr/bin/env bash
# Track 2 (first-principles neurophysiology, 2026-11-13): PROTEUS long epochs through the kit's own pipeline (t2p_export.py), -3 to +8 s
# re cue (a pre-cue rest baseline for ERD / ERS maps, as the earlier Scherer2015 / Dreyer2023 analyses), written into THIS job's run dir
# so `s1 fetch` brings it home. CPU only -> zen5_vis (free). Reads the PROTEUS release already in data/ni26/t2_proteus/benchopt.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_export_long.slurm --partition=zen5_vis --account=vsc --time=03:00:00
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
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_export.py" --no-download --start -3.0 --duration 11.0 --out "$OUT/export_proteus_long_m3p8"
rc=$?
s1_finish "$rc" "$OUT" 'export_proteus_long_m3p8/meta.json' 1

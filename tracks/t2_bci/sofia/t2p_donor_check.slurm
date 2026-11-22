#!/usr/bin/env bash
# T2-D549 (the manager's guard): a DRY donor check before a build slot = t2p_repack --fm-from <donor> --donor-check-only on the base ZIP with
# the build's template (the repack's own checks, no package written). CPU only -> zen5_vis.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_donor_check.slurm --partition=zen5_vis --account=vsc --time=00:20:00 \
#     -- <base zip> <donor zip> <template under tracks/t2_bci/submission/>
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
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"   # = t2p_w19_build.slurm
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py" --zip "$1" --fm-from "$2" --out "$OUT/donorcheck" --export "$D/export_proteus_v1" \
  --template "$S1_CODE/tracks/t2_bci/submission/$3" --donor-check-only 2>&1 | tee "$OUT/donorcheck.log"
rc=${PIPESTATUS[0]}
rm -rf "$OUT/donorcheck" "$OUT/donorcheck_"*
grep -q "DONOR CHECK OK" "$OUT/donorcheck.log" || rc=3
s1_finish "$rc" "$OUT" "donorcheck.log" 1

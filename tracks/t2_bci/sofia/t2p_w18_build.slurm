#!/usr/bin/env bash
# T2-D534 (W18 builds for gating, APPROVED by the user's line 2026-11-16 ≈ 10:25): one draw's W18 build on one H200.
#   (M) t2p_merge_donors.py: the SD donor (six ST-L with window-mean side heads) with slot 0 swapped for the head-less MI-expert member
#       -> <out>/<name>_donor.zip (CRC-checked against both sources);
#   (R) t2p_repack.py --zip <W17 build> --template t2p_submission_w18.py --fm-from <merged donor> -> <out>/<name>.zip (clean pass, parts,
#       the side-head checks, recombination, robustness), then the official kit under the worker-image import guard (t2p_kit_replay.sh).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w18_build.slurm --time=02:30:00 --job-name=ni26-t2p-w18-<name> \
#       -- <W17 zip> <SD donor zip> <MI donor zip> <name>
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
BASE="$1"; SD="$2"; MI="$3"; NAME="$4"
for z in "$BASE" "$SD" "$MI"; do [ -f "$z" ] || { echo "!! no zip $z"; s1_finish 2 "$OUT" none 1; }; done
sha256sum "$BASE" "$SD" "$MI" | tee "$OUT/$NAME.inputs.sha256"
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
echo "### $(date +%T) merge"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_merge_donors.py" --base "$SD" --swap "$MI" --slots 0 --out "$OUT/${NAME}_donor.zip" \
  || { echo "!! merge failed"; s1_finish 3 "$OUT" none 1; }
echo "### $(date +%T) repack"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py" --zip "$BASE" --out "$OUT/$NAME" --export "$D/export_proteus_v1" \
  --template "$S1_CODE/tracks/t2_bci/submission/t2p_submission_w18.py" --fm-from "$OUT/${NAME}_donor.zip"
rc=$?
[ $rc = 0 ] || { echo "!! t2p_repack failed (rc $rc)"; s1_finish "$rc" "$OUT" "$NAME.repack.json" 1; }
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
rm -rf "$OUT/${NAME}_ref" "$OUT/${NAME}_dref" "$OUT/${NAME}_old" "$OUT/$NAME"      # the check trees (files cap); the ZIP and the merged donor stay
echo "### $(date +%T) kit replay"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

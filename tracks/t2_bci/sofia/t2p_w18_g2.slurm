#!/usr/bin/env bash
# T2-D534 condition (1): rehearsal G2 = the W18 build's exact shape on W17 build 1 with a FAKE donor (never a candidate, never uploaded):
#   (a) fake donor = W17 build 1's own six FM members + small random window-mean side heads (scripts/local/t2_w18_fakedonor.py);
#   (b) t2p_merge_donors.py: slot 0 swapped for W17 build 1's own member 0, head-less (5 heads + 1 head-less member, as the real builds);
#   (c) t2p_repack.py --template W18 --fm-from the merged fake donor (all repack gates incl. the in-stream-aware parts check and gate R);
#   (d) t2p_w18_pkgcheck.py NEW vs W17 build 1 (the approved config shape, finite / moves / re-run, gates I and S, gate B's inputs).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w18_g2.slurm --time=03:00:00 --job-name=ni26-t2p-w18-g2 -- <W17 build 1 zip>
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
BASE="$1"
[ -f "$BASE" ] || { echo "!! no zip $BASE"; s1_finish 2 "$OUT" none 1; }
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
S="$S1_JOBTMP/g2"; mkdir -p "$S"
echo "### $(date +%T) fake donor"
python -u "$S1_CODE/scripts/local/t2_w18_fakedonor.py" --base "$BASE" --out "$S/fakedonor.zip" || s1_finish 3 "$OUT" none 1
echo "### $(date +%T) merge"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_merge_donors.py" --base "$S/fakedonor.zip" --swap "$BASE" --slots 0 --out "$S/fakemerged.zip" || s1_finish 3 "$OUT" none 1
echo "### $(date +%T) repack (G2)"
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py" --zip "$BASE" --out "$S/rehG2" --export "$D/export_proteus_v1" \
  --template "$S1_CODE/tracks/t2_bci/submission/t2p_submission_w18.py" --fm-from "$S/fakemerged.zip" 2>&1 | tee "$OUT/rehG2.log"
rc=${PIPESTATUS[0]}
cp "$S/rehG2.repack.json" "$OUT/" 2>/dev/null
[ $rc = 0 ] || { echo "!! G2 repack failed (rc $rc)"; s1_finish "$rc" "$OUT" "rehG2.repack.json" 1; }
cp "$S/rehG2.zip" "$OUT/rehG2.zip"     # run 1622536 lost its ZIP to a check crash: keep it, so a check-only rerun needs no repack (t2p_pkgcheck.slurm w18)
echo "### $(date +%T) post-build check (G2 vs W17 build 1)"
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$BASE" "$S/old" || s1_finish 2 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w18_pkgcheck.py" --old "$S/old" --new "$S/rehG2" --export "$D/export_proteus_v1" \
  --out "$OUT/rehG2.pkgcheck.json" 2>&1 | tee "$OUT/rehG2.pkgcheck.log"
rc=${PIPESTATUS[0]}
rm -rf "$S"
s1_finish "$rc" "$OUT" "rehG2.pkgcheck.json" 1

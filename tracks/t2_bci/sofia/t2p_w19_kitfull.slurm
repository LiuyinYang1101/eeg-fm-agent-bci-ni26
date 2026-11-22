#!/usr/bin/env bash
# T2-D549 condition K, part K2: the W19 ZIP through the kit's own streaming harness on ALL public streams (t2p_w19_kitfull.py), compared with
# the post-build check's saved predictions. Runs afterok the post-build check (launch_w19.sh).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_kitfull.slurm --time=01:00:00 --job-name=ni26-t2p-w19kit-<name> \
#       -- <W19 zip under runs/ni26> <pkgcheck pred npz under runs/ni26> <name>
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
ZIP="$S1_ROOT/runs/ni26/${1:?zip}"; PRED="$S1_ROOT/runs/ni26/${2:?pred npz}"; NAME=${3:?name}
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
for z in "$ZIP" "$PRED"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
zsha_check() {   # T2-D569 F4 (Track3): print AND check the sha256 of a ZIP this job loads (vs its sibling .sha256 when present, vs an expected value when given)
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
zsha_check "$ZIP" "${ZIP_SHA_EXPECT:-}" || s1_finish 2 "$OUT" none 1
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP" "$OUT/pkg" || s1_finish 2 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w19_kitfull.py" --pkg "$OUT/pkg" --pred "$PRED" --export "$D/export_proteus_v1" --out "$OUT/$NAME.kitfull.json"
rc=$?
rm -rf "$OUT/pkg"
s1_finish "$rc" "$OUT" "$NAME.kitfull.json" 1

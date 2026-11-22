#!/usr/bin/env bash
# T2 post-build check on one H200 (the local 5090 queue is FIFO and shared): unzip OLD and NEW into the job's output dir and run
# t2p_w15_pkgcheck.py (W15 vs its step-A base) or t2p_w17_pkgcheck.py (W17 vs its W15 build) on the sofia PROTEUS export.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_pkgcheck.slurm --time=00:45:00 --job-name=ni26-t2p-pkgcheck-<name> \
#       -- w15|w17 <OLD zip under runs/ni26> <NEW zip under runs/ni26> <name>
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
KIND=${1:?w15|w17}; OLDZ="$S1_ROOT/runs/ni26/${2:?old zip}"; NEWZ="$S1_ROOT/runs/ni26/${3:?new zip}"; NAME=${4:?name}
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
for z in "$OLDZ" "$NEWZ"; do [ -f "$z" ] || { echo "!! no zip $z"; s1_finish 2 "$OUT" none 1; }; done
zsha_check() {   # T2-D569 F4 (Track3): print AND check the sha256 of a ZIP this job loads (vs its sibling .sha256 when present, vs an expected value when given)
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
zsha_check "$OLDZ" "${OLD_SHA_EXPECT:-}" && zsha_check "$NEWZ" "${NEW_SHA_EXPECT:-}" || s1_finish 2 "$OUT" none 1
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2]); zipfile.ZipFile(sys.argv[3]).extractall(sys.argv[4])" \
  "$OLDZ" "$OUT/old" "$NEWZ" "$OUT/new" || s1_finish 2 "$OUT" none 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_${KIND}_pkgcheck.py" --old "$OUT/old" --new "$OUT/new" --export "$D/export_proteus_v1" \
  --out "$OUT/$NAME.pkgcheck.json"
rc=$?
rm -rf "$OUT/old" "$OUT/new"                     # the zips stay where they are; only the check's JSON is kept
s1_finish "$rc" "$OUT" "$NAME.pkgcheck.json" 1

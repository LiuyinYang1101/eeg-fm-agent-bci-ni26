#!/usr/bin/env bash
# T2-D549 (the W19 build; runs ONLY via launch_w19.sh, which refuses without the owner's typed W19 line): one draw's W19 build on one H200,
# three repacks in sequence, each ONE change with the check built for it (DECISIONS T2-D549):
#   R1  t2p_repack --zip <W18 bN> --template t2p_submission_w19.py (refit off = W18 bitwise) --fm-from <W19 merged donor>  -> <name>_r1.zip
#   R2  t2p_repack --zip <name>_r1.zip --template W19 --side-from <side.pt S10>  (Gate T)                                 -> <name>_r2.zip
#   R3  t2p_repack --zip <name>_r2.zip --template W19 --set 'side.refit={"lam": 1.0}' --set-ref                             -> <name>_r3.zip
#   R4  t2p_repack --zip <name>_r3.zip --template W19 --set 'fm_side_refit={"lam": 1.0}' --set-ref  (T2-D553)               -> <name>.zip
#   K1  the official kit (benchopt, worker-image import guard) on <name>.zip: Simulated + the PROTEUS warm-up (t2p_kit_replay.sh), refit ON;
#       the template's side-refit AND FM-refit (T2-D553) log lines are counted (any 'errors' / 'kept_trained' > 0 fails the job).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w19_build.slurm --time=04:00:00 --job-name=ni26-t2p-w19-<name> \
#       -- <W18 zip> <W19 merged donor zip> <side.pt S10> <name>
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
BASE="$1"; DONOR="$2"; SIDE="$3"; NAME="$4"
for z in "$BASE" "$DONOR" "$SIDE"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
sha256sum "$BASE" "$DONOR" "$SIDE" | tee "$OUT/$NAME.inputs.sha256"
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w19.py"
RPK="$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py"
step() {   # name, input zip, extra args...
  local nm=$1 in=$2; shift 2
  echo "### $(date +%T) $nm"
  python -u "$RPK" --zip "$in" --out "$OUT/$nm" --export "$D/export_proteus_v1" --template "$T" "$@"
  local rc=$?
  rm -rf "$OUT/${nm}_ref" "$OUT/${nm}_dref" "$OUT/${nm}_bref" "$OUT/${nm}_old" "$OUT/$nm"     # the check trees (files cap); ZIP + repack.json stay
  [ $rc = 0 ] || { echo "!! $nm failed (rc $rc)"; s1_finish "$rc" "$OUT" "$nm.repack.json" 1; }
  sha256sum "$OUT/$nm.zip" | tee "$OUT/$nm.zip.sha256"
}
step "${NAME}_r1" "$BASE" --fm-from "$DONOR"
step "${NAME}_r2" "$OUT/${NAME}_r1.zip" --side-from "$SIDE"
step "${NAME}_r3" "$OUT/${NAME}_r2.zip" --set 'side.refit={"lam": 1.0}' --set-ref
step "$NAME" "$OUT/${NAME}_r3.zip" --set 'fm_side_refit={"lam": 1.0}' --set-ref                                      # T2-D553
rm -f "$OUT/${NAME}_r1.zip" "$OUT/${NAME}_r2.zip" "$OUT/${NAME}_r3.zip"   # the intermediate zips (their sha256 and repack.json stay)
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
nfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "W19 side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
nok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "W19 side refit at")
echo "### K1: $nok side-refit log lines in the kit runs, $nfb with a fallback / error"
[ "$nfb" = 0 ] || { echo "!! K1: side-refit fallbacks in the kit run"; rc=4; }
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed")
echo "### K1: $fok FM-refit (T2-D553) log lines in the kit runs, $ffb with a fallback / error, $fail refit-failure lines"
[ "$ffb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] || { echo "!! K1: T2-D553 FM-refit fallbacks / failures / none in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

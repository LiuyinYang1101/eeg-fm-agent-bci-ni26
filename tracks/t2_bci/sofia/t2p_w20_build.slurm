#!/usr/bin/env bash
# T2-D560 (the W20 build; runs ONLY via launch_w20.sh, which refuses a full build without the owner's typed W20 line): W20 = W19 b2 + cfg
# fm_proto, two repacks on ONE GPU (zen5_vis by default: a template-only repack loads OLD + NEW = 12 ST-L, which fits 32 GB; the 18-ST-L
# OOM case was OLD + REF + NEW), each ONE change with its check:
#   R5a t2p_repack --zip <W19 b2> --template t2p_submission_w20.py      (fm_proto absent = W19 bitwise: OLD vs NEW equal on every window)
#   R5b t2p_repack --zip <name>_r5a.zip --template W20 --set 'fm_proto={"w": 0.5, "lam_headless": 1.0}' --set-ref   (the edit took)
#   K1  the official kit (t2p_kit_replay.sh) on <name>.zip, refit ON: the D553 FM-refit and D560 proto log lines counted (any
#       'errors' / 'kept_trained' > 0 or none at all fails the job).
#   SMOKE (arg 3 = smoke): R5a only, on 2 streams, no K1 (memory / env / the inert check's start; never a package to gate).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w20_build.slurm --partition=zen5_vis --account=vsc --time=04:00:00 \
#       --job-name=ni26-t2p-w20-<name> -- <W19 b2 zip> <name> [smoke]
#SBATCH --partition=zen5_vis
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
BASE="$1"; NAME="$2"; MODE="${3:-full}"
[ -f "$BASE" ] || { echo "!! no input $BASE"; s1_finish 2 "$OUT" none 1; }
sha256sum "$BASE" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"   # = t2p_w19_build.slurm
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w20.py"
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
if [ "$MODE" = smoke ]; then
  step "${NAME}_r5a" "$BASE" --max-streams 2
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -f "$OUT/${NAME}_r5a.zip"
  s1_finish 0 "$OUT" "${NAME}_r5a.repack.json" 1
fi
step "${NAME}_r5a" "$BASE"
step "$NAME" "$OUT/${NAME}_r5a.zip" --set 'fm_proto={"w": 0.5, "lam_headless": 1.0}' --set-ref                     # T2-D560
rm -f "$OUT/${NAME}_r5a.zip"                     # the intermediate zip (its sha256 and repack.json stay)
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
pfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D560 FM proto at" | grep -cvE ": refits 1, .*kept_trained 0, errors 0 \(")   # 'refits 0' = silently off: fails
pok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D560 FM proto at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed|D560 FM proto failed")
echo "### K1: $fok D553 FM-refit lines ($ffb with a fallback / error), $pok D560 proto lines ($pfb with a fallback / error), $fail failure lines"
[ "$ffb" = 0 ] && [ "$pfb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] && [ "$pok" -gt 0 ] || { echo "!! K1: D553 / D560 fallbacks / failures / none in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

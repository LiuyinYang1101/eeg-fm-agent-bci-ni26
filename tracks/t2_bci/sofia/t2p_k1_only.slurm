#!/usr/bin/env bash
# T2-D652 W30 one-hop: K1 ONLY on an EXISTING ZIP (the build 1649496 wrote the ZIP, then its repack parts check refused an FM swap on a base with
# in-stream FM refits, a check artefact; K1 never ran). = the K1 section of t2p_cfgonly_build.slurm, the ZIP's sha256 pinned.
#   S1_EXPORT="ZIP_SHA_EXPECT=<sha>" S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_k1_only.slurm --partition=zen5_vis --account=vsc \
#       --time=01:00:00 --job-name=ni26-t2p-k1-<name> -- <zip under runs/ni26> <name>
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
ZIP="$S1_ROOT/runs/ni26/${1:?zip}"; NAME=${2:?name}
h=$(sha256sum "$ZIP" | cut -d' ' -f1); echo "### the ZIP $ZIP sha256 $h"
[ -n "${ZIP_SHA_EXPECT:-}" ] && [ "$h" = "$ZIP_SHA_EXPECT" ] || { echo "!! the ZIP sha $h != ZIP_SHA_EXPECT '${ZIP_SHA_EXPECT:-}'"; s1_finish 2 "$OUT" none 1; }
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$ZIP" "$OUT/$NAME"; rc=$?
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
pfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D560 FM proto at" | grep -cvE ": refits 1, .*kept_trained 0, errors 0 \(")   # 'refits 0' = silently off: fails
pok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D560 FM proto at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed|D560 FM proto failed|D569 cnn_lp failed")
lok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D569 cnn_lp at")
lerr=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D569 cnn_lp at" | grep -cvE ", errors 0 \(")
lsm=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -oE "D569 cnn_lp at [0-9]+: smoothed [0-9]+" | awk '{s += $NF} END {print s + 0}')
echo "### K1: $fok D553 FM-refit lines ($ffb with a fallback / error), $pok D560 proto lines ($pfb with a fallback / error), $lok D569 cnn_lp lines ($lsm smoothed, $lerr with errors), $fail failure lines"
[ "$ffb" = 0 ] && [ "$pfb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] && [ "$pok" -gt 0 ] && [ "$lok" -gt 0 ] && [ "$lsm" -gt 0 ] && [ "$lerr" = 0 ] || { echo "!! K1: D553 / D560 / D569 fallbacks / failures / none in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

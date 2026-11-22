#!/usr/bin/env bash
# T2-D595 build (W22 candidate; runs ONLY via launch_w22_d595.sh, which refuses without the owner's typed line naming D595): = t2p_w21lp_build.slurm with the W22 template, R6a inert = the BASE bitwise (W21, or the FM chain's final W22 zip), R6b --set p1_prior_reset=true --set-ref, K1.
# T2-D569 (the W21 LP option's build; runs ONLY via launch_w21lp.sh, which refuses without the owner's typed W21 line naming D569 and
# D569_READS_PASSED=1): W21 = W20+D557 + cfg cnn_lp, two repacks on ONE GPU (zen5_vis), each ONE change with its check (= t2p_w20_build.slurm's
# pattern, the template moved to t2p_submission_w21lp.py):
#   R6a t2p_repack --zip <W20+D557> --template t2p_submission_w21lp.py   (cnn_lp absent = W20 bitwise: OLD vs NEW equal on every window)
#   R6b t2p_repack --zip <name>_r6a.zip --template W21lp --set 'cnn_lp={"alpha": 0.5, "k": 10}' --set-ref   (the edit took)
#   K1  the official kit (t2p_kit_replay.sh) on <name>.zip, refit ON: the D553 / D560 lines as W20's K1 AND the D569 'cnn_lp at' lines
#       (Track3 F1 / F2 on d04af48a): at least one, run-level smoothed > 0, every line errors 0; any 'D569 cnn_lp failed' line fails the job.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w21lp_build.slurm --partition=zen5_vis --account=vsc --time=04:00:00 \
#       --job-name=ni26-t2p-w21lp-<name> -- <W20+D557 zip> <name> [smoke]
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
zsha_check() {   # T2-D569 F4 (Track3): print AND check the sha256 of a ZIP this job loads (vs its sibling .sha256 when present, vs an expected value when given)
  local z=$1 want=${2:-} h
  h=$(sha256sum "$z" | cut -d' ' -f1); echo "### input zip $z sha256 $h"
  if [ -f "$z.sha256" ] && [ "$h" != "$(cut -d' ' -f1 "$z.sha256")" ]; then echo "!! $z sha256 != its .sha256"; return 1; fi
  if [ -n "$want" ] && [ "$h" != "$want" ]; then echo "!! $z sha256 != the expected $want"; return 1; fi
  return 0
}
if [ "${BASE_SHA_EXPECT:-}" = sibling ]; then   # T2 W22 chain (night plan 2026-11-18): the base = the previous chain step's zip; its sha = that step's own .sha256
  [ -f "$BASE.sha256" ] || { echo "!! BASE_SHA_EXPECT=sibling but no $BASE.sha256"; s1_finish 2 "$OUT" none 1; }
  BASE_SHA_EXPECT=$(cut -d' ' -f1 "$BASE.sha256"); echo "### W22 chain: the base sha from its build step's .sha256: $BASE_SHA_EXPECT"
fi
[ -n "${BASE_SHA_EXPECT:-}" ] || { echo "!! BASE_SHA_EXPECT unset (the reviewed W20+D557 sha256)"; s1_finish 2 "$OUT" none 1; }
zsha_check "$BASE" "$BASE_SHA_EXPECT" || s1_finish 2 "$OUT" none 1
sha256sum "$BASE" | tee "$OUT/$NAME.inputs.sha256"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"   # = t2p_w19_build.slurm
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w22.py"   # T2-D595: the W22 template
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
  step "${NAME}_r6a" "$BASE" --max-streams 2
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  rm -f "$OUT/${NAME}_r6a.zip"
  s1_finish 0 "$OUT" "${NAME}_r6a.repack.json" 1
fi
step "${NAME}_r6a" "$BASE"
python -c "import json, sys; r = json.load(open(sys.argv[1])); print('### R6a inert check:', r['clean_windows'], 'windows, differ', r['clean_differ'], 'max dp', r['clean_max_dp']); sys.exit(0 if r['clean_differ'] == 0 and r['clean_max_dp'] == 0.0 else 1)" \
  "$OUT/${NAME}_r6a.repack.json" || { echo "!! R6a: the W22 template without the flag is not the BASE bitwise (clean_max_dp != 0)"; s1_finish 4 "$OUT" "${NAME}_r6a.repack.json" 1; }   # Track3 F3
step "$NAME" "$OUT/${NAME}_r6a.zip" --set 'p1_prior_reset=true' --set-ref                                    # T2-D595
rm -f "$OUT/${NAME}_r6a.zip"                     # the intermediate zip (its sha256 and repack.json stay)
echo "### $(date +%T) kit replay (K1, refit ON)"
bash "$S1_CODE/tracks/t2_bci/sofia/t2p_kit_replay.sh" "$OUT/$NAME.zip" "$OUT/$NAME"; rc=$?
ffb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D553 FM side refit at" | grep -cvE "kept_trained 0, errors 0 \(")
fok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit at")
pfb=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D560 FM proto at" | grep -cvE ": refits 1, .*kept_trained 0, errors 0 \(")   # 'refits 0' = silently off: fails
pok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D560 FM proto at")
fail=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D553 FM side refit failed|W19 side refit failed|D560 FM proto failed|D569 cnn_lp failed")
lok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D569 cnn_lp at")
lerr=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -E "D569 cnn_lp at" | grep -cvE ", errors 0 \(")
lsm=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -oE "D569 cnn_lp at [0-9]+: smoothed [0-9]+" | awk '{s += $NF} END {print s + 0}')
rok=$(cat "$OUT/$NAME".kit_*.log 2>/dev/null | grep -cE "D595 p1 prior reset at")   # T2-D595 (Track3 LOW): the kit run proves the reset fires
echo "### K1: $fok D553 FM-refit lines ($ffb with a fallback / error), $pok D560 proto lines ($pfb with a fallback / error), $lok D569 cnn_lp lines ($lsm smoothed, $lerr with errors), $fail failure lines"
echo "### K1: $rok D595 reset lines"
[ "$ffb" = 0 ] && [ "$pfb" = 0 ] && [ "$fail" = 0 ] && [ "$fok" -gt 0 ] && [ "$pok" -gt 0 ] && [ "$lok" -gt 0 ] && [ "$lsm" -gt 0 ] && [ "$lerr" = 0 ] && [ "$rok" -gt 0 ] || { echo "!! K1: D553 / D560 / D569 fallbacks / failures / none, or no D595 reset, in the kit run"; rc=4; }
s1_finish "$rc" "$OUT" "$NAME.kit_warmup.json" 1

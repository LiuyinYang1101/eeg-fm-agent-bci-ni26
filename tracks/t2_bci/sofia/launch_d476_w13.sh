#!/usr/bin/env bash
# T2-D476 (template W13, the user's "go W13" in T2's shell, 2026-11-13): repack a GATED zip with W13 + the per-family start vectors
# (tracks/t2_bci/submission/configs/t2_d476_start_vectors.json, sha256 f90c94d7..., by T2-D459's rule per family). t2p_repack.slurm:
# weights byte-identical; OLD vs REF (W13 template, old config) must agree on every public window (the template change is inert without
# per_family); the NEW parts must equal REF's and the NEW output must equal sum_f w_f causal_sinkhorn(part_f, start_f) (<= 1e-6);
# robustness cases; then the kit replay under the worker-image guard. Free partition (zen5_vis): no grant GPU-h.
#   mem-guard bash tracks/t2_bci/sofia/launch_d476_w13.sh dry|GO <zip path under runs/ni26> <new name>
set -euo pipefail
cd "$(dirname "$0")/../../.."
mode=${1:?dry|GO}; zip=${2:?zip path under runs/ni26}; name=${3:?new name}
[[ "$mode" == dry || "$mode" == GO ]] || exit 2
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/t2p_repack.slurm tracks/t2_bci/sofia/t2p_kit_replay.sh tracks/t2_bci/sofia/launch_d476_w13.sh)" ] || { echo "commit first"; exit 2; }
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && bash tracks/t2_bci/sofia/t2_sync_if_needed.sh   # the file cap: a new snapshot only when T2 code changed
env "${dry[@]}" S1_PURPOSE="T2-D476 W13 repack ($zip -> $name; per-family decoders, the user's go W13): feeds T2's end-of-window upload" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --partition=zen5_vis --time=03:00:00 --job-name="ni26-t2p-repack-$name" -- \
  "$S1_ROOT/runs/ni26/$zip" "$name" --template tracks/t2_bci/submission/t2p_submission_w13.py \
  --set "tta.per_family=@tracks/t2_bci/submission/configs/t2_d476_start_vectors.json#per_family" 2>&1 | grep -E "SUBMITTED|DRY RUN|to start|refus|rror" | cut -c1-200

#!/usr/bin/env bash
# T2-D603 SMOKE — VALIDATION ONLY, NOT A CANDIDATE (the manager's "smoke OK", 2026-11-19; the real build waits for the owner's line naming D603).
# On 2 public streams: R6a (the W22-D603 template inert on W21: clean_max_dp == 0), R6b (--set p1_prior_informed=true), then the D603 post-build
# check t2p_w22_d603_pkgcheck.py --streams 2 (OLD = W21). Counts the 'D603 p1 informed restart at' lines. EVERY zip and unzipped tree is
# deleted in-job; only the JSONs / logs / sha files stay (never an upload source).
#   S1_EXPORT=BASE_SHA_EXPECT=<W21 sha> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_w22_d603_smoke.slurm --partition=zen5_vis \
#       --account=vsc --time=01:30:00 --job-name=ni26-t2p-SMOKE-d603 -- <W21 zip> t2p_SMOKE_d603
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
BASE="$1"; NAME="$2"
[[ "$NAME" == *SMOKE* ]] || { echo "!! a smoke name must say SMOKE"; s1_finish 2 "$OUT" none 1; }
[ -f "$BASE" ] || { echo "!! no input $BASE"; s1_finish 2 "$OUT" none 1; }
h=$(sha256sum "$BASE" | cut -d' ' -f1); echo "### input zip $BASE sha256 $h"
[ -n "${BASE_SHA_EXPECT:-}" ] && [ "$h" = "$BASE_SHA_EXPECT" ] || { echo "!! BASE sha != BASE_SHA_EXPECT"; s1_finish 2 "$OUT" none 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
T="$S1_CODE/tracks/t2_bci/submission/t2p_submission_w22d603.py"
RPK="$S1_CODE/tracks/t2_bci/scripts/t2p_repack.py"
clean() { rm -f "$OUT"/*.zip; rm -rf "$OUT/old" "$OUT/new" "$OUT"/*_ref "$OUT"/*_dref "$OUT"/*_bref "$OUT"/*_old "$OUT/${NAME}_r6a" "$OUT/$NAME"; }
fin() { clean; ls "$OUT"; s1_finish "$1" "$OUT" "$2" 1; }
echo "### $(date +%T) R6a (inert, 2 streams)"
python -u "$RPK" --zip "$BASE" --out "$OUT/${NAME}_r6a" --export "$D/export_proteus_v1" --template "$T" --max-streams 2 2>&1 | tee "$OUT/r6a.log"
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "!! R6a failed (rc $rc)"; fin "$rc" "${NAME}_r6a.repack.json"; }
python -c "import json, sys; r = json.load(open(sys.argv[1])); print('### R6a inert check:', r['clean_windows'], 'windows, differ', r['clean_differ'], 'max dp', r['clean_max_dp']); sys.exit(0 if r['clean_differ'] == 0 and r['clean_max_dp'] == 0.0 else 1)" \
  "$OUT/${NAME}_r6a.repack.json" || { echo "!! R6a: the template without the flag is not W21 bitwise"; fin 4 "${NAME}_r6a.repack.json"; }
echo "### $(date +%T) R6b (p1_prior_informed, 2 streams)"
python -u "$RPK" --zip "$OUT/${NAME}_r6a.zip" --out "$OUT/$NAME" --export "$D/export_proteus_v1" --template "$T" --set 'p1_prior_informed=true' --set-ref \
  --max-streams 2 2>&1 | tee "$OUT/r6b.log"
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "!! R6b failed (rc $rc)"; fin "$rc" "$NAME.repack.json"; }
echo "### $(date +%T) the D603 post-build check (2 streams; OLD = W21)"
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2]); zipfile.ZipFile(sys.argv[3]).extractall(sys.argv[4])" \
  "$BASE" "$OUT/old" "$OUT/$NAME.zip" "$OUT/new" || fin 2 none
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_w22_d603_pkgcheck.py" --old "$OUT/old" --new "$OUT/new" --export "$D/export_proteus_v1" --streams 2 \
  --out "$OUT/$NAME.pkgcheck.json" 2>&1 | tee "$OUT/pkgcheck.log"
rc=${PIPESTATUS[0]}
echo "### D603 restart lines: r6b $(grep -c 'D603 p1 informed restart at' "$OUT/r6b.log"), check $(grep -c 'D603 p1 informed restart at' "$OUT/pkgcheck.log")"
fin "$rc" "$NAME.pkgcheck.json"

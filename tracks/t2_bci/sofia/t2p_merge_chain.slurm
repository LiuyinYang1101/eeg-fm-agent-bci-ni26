#!/usr/bin/env bash
# T2-D557 build PREPARATION (not a package): W19 b2's merged FM donor with its five headed members replaced by the D557 augmented ones, five
# t2p_merge_donors.py swaps in sequence (--keep-side each; the MI member at slot 0 unchanged): slot 1 <- A fm_0 (621), slot 3 <- B fm_0 (623),
# slot 5 <- C fm_0 (625), slot 2 <- cont-622 fm_0, slot 4 <- cont-624 fm_0. CPU on zen5_vis. Only the final zip is kept (+ sha256).
#   S1_EXPORT=RUN_NAME=<run>,NAME=<zip name> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_merge_chain.slurm --partition=zen5_vis \
#     --account=vsc --time=01:00:00 -- <base donor> <A> <B> <C> <cont-622> <cont-624>
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
BASE=$1; A=$2; B=$3; C=$4; K2=$5; K4=$6
for z in "$BASE" "$A" "$B" "$C" "$K2" "$K4"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
sha256sum "$BASE" "$A" "$B" "$C" "$K2" "$K4" | tee "$OUT/${NAME:?NAME}.inputs.sha256"
M="python -u $S1_CODE/tracks/t2_bci/scripts/t2p_merge_donors.py"
prev="$BASE"; i=0
for sw in "$A:1" "$B:3" "$C:5" "$K2:2" "$K4:4"; do
  z=${sw%:*}; slot=${sw##*:}; i=$((i + 1)); out="$OUT/${NAME}_m$i.zip"; [ $i = 5 ] && out="$OUT/$NAME.zip"
  echo "### $(date +%T) merge $i: slot $slot <- $(basename "$z") fm index 0"
  $M --base "$prev" --swap "$z" --slots "$slot" --swap-index 0 --keep-side --out "$out" || { echo "!! merge $i failed"; s1_finish 3 "$OUT" "$NAME.zip" 1; }
  [ "$prev" != "$BASE" ] && rm -f "$prev"
  prev="$out"
done
python -c "import zipfile, json, sys; c = json.loads(zipfile.ZipFile(sys.argv[1]).read('config.json')); print([(j, m['file'], m.get('seed'), bool(m.get('side')), m.get('aug'), m.get('mixup'), (m.get('cont') or {}).get('decision')) for j, m in enumerate(c['fm_members'])])" "$OUT/$NAME.zip"
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
s1_finish 0 "$OUT" "$NAME.zip" 1

#!/usr/bin/env bash
# T2-D642 x9 (W25-ALL6) build PREPARATION (not a package): W19 b2's merged FM donor with ALL SIX members replaced by the user-balanced
# D642 donors = t2p_merge_chain.slurm (D557's five headed swaps) + the MI member: SIX t2p_merge_donors.py swaps in sequence (--keep-side on the
# five HEADED slots; slot 0 WITHOUT it: Track3's review, --keep-side asserts a head on both members and slot 0 is head-less in W19 b2's donor AND
# the MI donor; the tool's else branch asserts the swap stays head-less): slot 0 <- MI fm_0 (620), slot 1 <- A fm_0 (621), slot 3 <- B fm_0 (623), slot 5 <- C fm_0 (625), slot 2 <-
# cont-622 fm_0, slot 4 <- cont-624 fm_0 (the merge tool refuses a CPT-sha change, T2-D642). CPU on zen5_vis. Only the final zip (+ sha256).
#   S1_EXPORT=RUN_NAME=<run>,NAME=<zip name> S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_merge_chain6.slurm --partition=zen5_vis \
#     --account=vsc --time=01:00:00 -- <base donor> <MI> <A> <B> <C> <cont-622> <cont-624>
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
BASE=$1; MI=$2; A=$3; B=$4; C=$5; K2=$6; K4=$7
for z in "$BASE" "$MI" "$A" "$B" "$C" "$K2" "$K4"; do [ -f "$z" ] || { echo "!! no input $z"; s1_finish 2 "$OUT" none 1; }; done
sha256sum "$BASE" "$MI" "$A" "$B" "$C" "$K2" "$K4" | tee "$OUT/${NAME:?NAME}.inputs.sha256"
M="python -u $S1_CODE/tracks/t2_bci/scripts/t2p_merge_donors.py"
prev="$BASE"; i=0
for sw in "$MI:0" "$A:1" "$B:3" "$C:5" "$K2:2" "$K4:4"; do
  z=${sw%:*}; slot=${sw##*:}; i=$((i + 1)); out="$OUT/${NAME}_m$i.zip"; [ $i = 6 ] && out="$OUT/$NAME.zip"
  echo "### $(date +%T) merge $i: slot $slot <- $(basename "$z") fm index 0"
  ks=--keep-side; [ "$slot" = 0 ] && ks=""          # Track3: the head-less MI slot 0 merges WITHOUT --keep-side
  $M --base "$prev" --swap "$z" --slots "$slot" --swap-index 0 $ks --out "$out" || { echo "!! merge $i failed"; s1_finish 3 "$OUT" "$NAME.zip" 1; }
  [ "$prev" != "$BASE" ] && rm -f "$prev"
  prev="$out"
done
python -c "import zipfile, json, sys; c = json.loads(zipfile.ZipFile(sys.argv[1]).read('config.json')); print([(j, m['file'], m.get('seed'), bool(m.get('side')), m.get('aug'), m.get('mixup'), (m.get('cont') or {}).get('decision')) for j, m in enumerate(c['fm_members'])])" "$OUT/$NAME.zip"
sha256sum "$OUT/$NAME.zip" | tee "$OUT/$NAME.zip.sha256"
s1_finish 0 "$OUT" "$NAME.zip" 1

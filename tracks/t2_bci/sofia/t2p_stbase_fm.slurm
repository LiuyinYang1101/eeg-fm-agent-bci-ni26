#!/usr/bin/env bash
# T2-D356 / T2-FM-D2: ST-EEGFormer-BASE MAE FROM SCRATCH (t2p_cpt.py --random-init --store-split) on the in-house FM project's compliant pool +
# data/fm/ni26_only, read IN PLACE through the in-house FM project's own loader (its snapshot INHOUSE_SRC) with C's frozen split, as-is.
# TAG fm (never the other project's allocation): this job reads ONLY the fm stores below (read-only) and writes ONLY runs/fm/$RUN_NAME.
#   $FM_STORE_ROOT/preproc_v3, $FM_STORE_ROOT/preproc_ext2026   (the in-house FM project's stores: read, never written)
#   $S1_ROOT/data/fm/ni26_only, $S1_ROOT/data/fm/store_extra_ws                    (a root absent or empty is dropped, with a note)
# The submit resolves S1_CODE to code/fm/current (the in-house FM project's tree), so the ni26 code comes from NI26_CODE and PYTHONPATH is set here.
# Required S1_EXPORT: NI26_CODE=<code/ni26/snap_...> INHOUSE_SRC=<code/fm/snap_.../src> FM_SPLIT=<split parquet> FM_SPLIT_SHA256=<sha256>
#   RUN_NAME=<stable out dir name>. Arguments after `--` go to t2p_cpt.py (--out and the store roots are set here unless given).
#   S1_EXPORT=NI26_CODE=$S1/code/ni26/snap_X,INHOUSE_SRC=$S1/code/fm/snap_Y/src,FM_SPLIT=$S1/data/fm/splits/Z.parquet,FM_SPLIT_SHA256=...,\
#     RUN_NAME=fm-t2-stbase S1_PURPOSE="..." s1 submit fm $S1/code/ni26/snap_X/tracks/t2_bci/sofia/t2p_stbase_fm.slurm --time=24:00:00 \
#     --job-name=fm-t2-stbase -- --fm st_base --random-init --lr 1.5e-4 --wd 1e-5 --mask-ratio 0.75 --steps 100000 ...
# Resumable: pass --state-every N; on the walltime signal the trainer writes state-cpt.pth and exits 3; a second segment with the
# same RUN_NAME and the same args + --resume continues (submit it --dependency=afterany:<first>).
#SBATCH --partition=zen4_h200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
[ "${S1_TAG:-}" = fm ] || { echo "!! [stbase] tag ${S1_TAG:-?} != fm: refusing" >&2; exit 2; }
for v in NI26_CODE INHOUSE_SRC FM_SPLIT FM_SPLIT_SHA256 RUN_NAME; do
  [ -n "${!v:-}" ] || { echo "!! [stbase] $v unset (S1_EXPORT)" >&2; exit 2; }
done
[ -r "$NI26_CODE/tracks/t2_bci/scripts/t2p_cpt.py" ] || { echo "!! [stbase] no t2p_cpt.py under NI26_CODE=$NI26_CODE" >&2; exit 2; }
[ -r "$INHOUSE_SRC/inhouse_fm/data/continuous_dataset.py" ] || { echo "!! [stbase] no the in-house FM project loader under INHOUSE_SRC=$INHOUSE_SRC" >&2; exit 2; }
echo "$FM_SPLIT_SHA256  $FM_SPLIT" | sha256sum -c --status || { echo "!! [stbase] $FM_SPLIT sha256 != $FM_SPLIT_SHA256" >&2; exit 2; }
s1_env ni26b || exit 2
export PYTHONPATH="$INHOUSE_SRC:$NI26_CODE/src:$NI26_CODE" INHOUSE_FM_SRC="$INHOUSE_SRC"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_USE_FILE_LOCKING=FALSE INHOUSE_INDEX_WORKERS="${INHOUSE_INDEX_WORKERS:-16}"
python -c "import h5py, scipy, pyarrow, pandas; print('[stbase] h5py', h5py.__version__, 'pyarrow', pyarrow.__version__)" || exit 2
OUT=$(s1_outdir "$RUN_NAME")
ROOTS=$FM_STORE_ROOT/preproc_v3,$FM_STORE_ROOT/preproc_ext2026
for r in "$S1_ROOT/data/fm/ni26_only" "$S1_ROOT/data/fm/store_extra_ws"; do
  if [ -d "$r" ] && [ -n "$(find "$r" -mindepth 1 -maxdepth 1 -type d ! -name '_*' -print -quit 2>/dev/null)" ]; then ROOTS="$ROOTS,$r"
  else echo "### [stbase] NOTE extra root $r absent or without dataset dirs: not a root of this job"; fi
done
grep -q -- "--store-roots" <<< "$*" && RARG=() || RARG=(--store-roots "$ROOTS")
{ echo "split=$FM_SPLIT sha256=$FM_SPLIT_SHA256"; echo "inhouse_src=$INHOUSE_SRC"; grep -E '^(commit|synced_utc)=' "$INHOUSE_SRC/../SYNC_VERSION" 2>/dev/null
  echo "ni26_code=$NI26_CODE"; grep -E '^(commit|synced_utc)=' "$NI26_CODE/SYNC_VERSION" 2>/dev/null; echo "roots=$ROOTS"; } | tee -a "$OUT/stbase_provenance.txt"
python -u "$NI26_CODE/tracks/t2_bci/scripts/t2p_cpt.py" --store-split "$FM_SPLIT" --store-src "$INHOUSE_SRC" "${RARG[@]}" --out "$OUT" "$@" &
PID=$!
trap 'kill -TERM "$PID"' USR1
rc=0; wait "$PID" || rc=$?
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; rc=$?; done
if grep -q -- "--bench" <<< "$*"; then s1_finish "$rc" "$OUT" 'bench.json' 1; fi
s1_finish "$rc" "$OUT" 'checkpoint-cpt.pth' 1

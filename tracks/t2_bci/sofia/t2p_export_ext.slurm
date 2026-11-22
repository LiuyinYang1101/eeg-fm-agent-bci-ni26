#!/usr/bin/env bash
# Track 2 EXTERNAL corpus through the PROTEUS stream pipeline (t2p_export.py --study ...): same filter / 120 Hz / uV /
# 4 s contract as Codabench's PROTEUS windows. zen4_h200 because zen5_vis is often blocked by the per-user CPU cap
# (the export takes minutes; 1 GPU is the smallest legal allocation). Usage:
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_export_ext.slurm --time=00:45:00 --job-name=ni26-t2p-ext \
#       -- <export dir name> --study Scherer2015Individually --start 4.0 --label-map 4:0,2:1,1:2
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
D="$S1_ROOT/data/ni26/t2_proteus"
NAME="$1"; shift
E="$D/$NAME"
export BENCHOPT_DATA_HOME="${T2P_BENCH_HOME:-$D/benchopt}" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0
[ -e "$E/meta.json" ] && { echo "!! $E already exported; refusing to overwrite (new dir = new version)"; s1_finish 4 "$OUT" 'none' 1; }
python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_export.py" --out "$E" "$@" 2>&1 | grep -v -E "S3:|Overall:|FutureWarning|make_standard_montage"; rc=${PIPESTATUS[0]}
cp "$E"/{meta.json,audit.json,index.csv} "$OUT/" 2>/dev/null
du -sh "$BENCHOPT_DATA_HOME"/neural_compet/* "$E" 2>/dev/null
s1_finish "$rc" "$OUT" 'audit.json' 1

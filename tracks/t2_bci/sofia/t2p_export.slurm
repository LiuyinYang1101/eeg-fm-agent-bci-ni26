#!/usr/bin/env bash
# Track 2 PROTEUS data on sofia: the public PROTEUS release (NEMAR nm000290 v1.0.0, ~1.5 GB) -> data/ni26/t2_proteus/benchopt
# (kit layout), then t2p_export.py: every window through the kit's neuralbench `eeg/_motor_imagery_stream` pipeline
# into data/ni26/t2_proteus/export_proteus_v1 (contract window 1-5 s, (N, 41, 480) uV) and, from the warm cache,
# export_proteus_wide_v1 (0-8 s re cue; research: crop augmentation + time-resolved analysis). CPU only -> zen5_vis.
# Env ni26b (neuroai 0.4.0) + the kit at 442edcd (the first kit with PROTEUS; `current` stays 2067faf for T1/T4).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_export.slurm --partition=zen5_vis --time=03:00:00
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir)
D="$S1_ROOT/data/ni26/t2_proteus"
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0
mkdir -p "$BENCHOPT_DATA_HOME"
code=$(curl -s -o /dev/null -m 20 -w '%{http_code}' https://data.nemar.org || echo 000); echo "### net data.nemar.org -> $code"
rc=0
E1="$D/export_proteus_v1"
if [ -e "$E1/meta.json" ]; then echo "!! $E1 already exported; skipping (new dir = new version)"
else python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_export.py" --out "$E1" || rc=$?; fi
cp "$E1"/{meta.json,audit.json,index.csv} "$OUT/" 2>/dev/null; cp -r "$E1/bids" "$OUT/" 2>/dev/null
E2="$D/export_proteus_wide_v1"
if [ "$rc" = 0 ] && [ ! -e "$E2/meta.json" ]; then
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_export.py" --no-download --start 0.0 --duration 8.0 --out "$E2" || rc=$?
  mkdir -p "$OUT/wide"; cp "$E2"/{meta.json,audit.json,index.csv} "$OUT/wide/" 2>/dev/null
fi
du -sh "$BENCHOPT_DATA_HOME"/neural_compet/* "$E1" "$E2" 2>/dev/null
s1_finish "$rc" "$OUT" 'audit.json' 1

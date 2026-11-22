#!/usr/bin/env bash
# Track 2: per-component time of packaged StreamModels (t2p_profile.py), one H200. Args = package dirs (runs-relative or
# absolute), then optional `--` t2p_profile.py options.
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_profile.slurm --time=00:30:00 --job-name=ni26-t2p-profile -- <pkg dir> ...
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
export BENCHOPT_DATA_HOME="$D/benchopt" NI26_KIT="${NI26_KIT:-$S1_ROOT/code/ni26_kit/442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
pk=(); rest=(); sep=0
for a in "$@"; do if [ $sep = 1 ]; then rest+=("$a"); elif [ "$a" = "--" ]; then sep=1; else pk+=("$a"); fi; done
rc=0; k=0
for p in "${pk[@]}"; do
  [[ "$p" = /* ]] || p="$S1_ROOT/runs/ni26/$p"
  k=$((k + 1))
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2p_profile.py" --pkg "$p" --export "$D/export_proteus_v1" --out "$OUT/prof_$k.json" "${rest[@]}" \
    2>&1 | grep -v "Warn\|warn(\|make_standard\|pos = " | tail -25 || rc=3
done
s1_finish "$rc" "$OUT" 'prof_*.json' "${#pk[@]}"

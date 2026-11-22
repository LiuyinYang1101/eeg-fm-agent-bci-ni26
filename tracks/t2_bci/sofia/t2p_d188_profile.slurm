#!/usr/bin/env bash
# T2-D188: the single-pass routed-LoRA profile. (1) t2p_stlora_synth.py synth: T2-D162 build 1's package + the current template + synthetic
# per-user ST-L deltas (K = 10, r16, every Linear in ST-L's blocks); (2) parity: zeroed bank = the original package, graph = eager;
# (3) t2p_profile.py on the original and the synthetic package (B = 1, one H200). Arg 1 = the original package dir (runs-relative or absolute).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_d188_profile.slurm --time=00:45:00 --job-name=ni26-t2p-d188-profile -- <pkg dir>
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
p="$1"; [[ "$p" = /* ]] || p="$S1_ROOT/runs/ni26/$p"
S="$S1_CODE/tracks/t2_bci/scripts"
rc=0
python -u "$S/t2p_stlora_synth.py" synth --pkg "$p" --out "$OUT/pkg_synth" --r 16 --scale 0.5 --amp 0.01 --seed 0 || rc=3
python -u "$S/t2p_stlora_synth.py" parity --orig "$p" --synth "$OUT/pkg_synth" --export "$D/export_proteus_v1" --n 150 --streams 3 \
  --out "$OUT/parity.json" 2>&1 | grep -v "Warn\|warn(\|make_standard\|pos = " | tail -30 || rc=3
k=0
for q in "$p" "$OUT/pkg_synth"; do
  k=$((k + 1))
  python -u "$S/t2p_profile.py" --pkg "$q" --export "$D/export_proteus_v1" --out "$OUT/prof_$k.json" 2>&1 | grep -v "Warn\|warn(\|make_standard\|pos = " | tail -25 || rc=3
done
rm -rf "$OUT/pkg_synth"/weights*.pt                    # keep the config + template, not 0.9 GB of copied weights
s1_finish "$rc" "$OUT" '*.json' 3

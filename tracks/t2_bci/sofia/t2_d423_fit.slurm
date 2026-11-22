#!/usr/bin/env bash
# T2-D423 Stage A on one H200 (pre-registered in docs/DECISIONS_T2.md): fit ReBrain's iEEG codec prior on the prepared ds003029 runs
# (data/ni26/d423_ds003029_512 = t2_d423_prep.py's output; CC0), then re-express the PROTEUS export through it and run the fidelity gate.
# The pinned vector-quantize-pytorch 1.31.0 + einx 0.4.3 + frozendict come from data/ni26/d423_pylib_py313 (pure-Python wheels, --no-deps).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2_d423_fit.slurm --time=16:00:00 --job-name=ni26-t2p-d423-codec [-- --steps N]
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
OUT=$(s1_outdir "${RUN_NAME:-}")
D="$S1_ROOT/data/ni26"
export PYTHONPATH="$D/d423_pylib_py313${PYTHONPATH:+:$PYTHONPATH}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import vector_quantize_pytorch, einx, torchaudio; from importlib.metadata import version; print('vq', version('vector-quantize-pytorch'), 'torch', __import__('torch').__version__)" || s1_finish 3 "$OUT" 'none' 1
python -u "$S1_CODE/tracks/t2_bci/scripts/t2_d423_fit.py" --data "$D/d423_ds003029_512" --out "$OUT" "$@"
rc=$?
if [ $rc -eq 0 ]; then
  python -u "$S1_CODE/tracks/t2_bci/scripts/t2_d423_reexpress.py" --export "$D/t2_proteus/export_proteus_v1" --ckpt "$OUT/codec_last.pt" --out "$OUT/export_rb"
  rc=$?
fi
s1_finish $rc "$OUT" 'export_rb/fidelity.json' 1

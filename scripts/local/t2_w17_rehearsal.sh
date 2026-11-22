#!/usr/bin/env bash
# T2-D517 W17 build REHEARSAL on the local 5090 (never a candidate; nothing here is uploaded): steps E and F of launch_d517_w17.sh run through
# t2p_repack.py directly on a W15 package (the rehearsal step-B' zip t2p_w15d_lite33, fetched), then t2p_w17_pkgcheck (E vs F). Purpose: catch a
# step-F failure (recombination, robustness, GPU memory) before the user's W16 + W17 line puts these steps on the critical path.
#   gpu-submit bash scripts/local/t2_w17_rehearsal.sh <W15 zip>
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_w17_rehearsal
ZIP=${1:?W15 zip}
mkdir -p "$R"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=tracks/t2_bci/submission/t2p_submission_w17.py
echo "### $(date +%T) step E"
$PY -u tracks/t2_bci/scripts/t2p_repack.py --zip "$ZIP" --out "$R/rehE" --export "$EXP" --template $T > "$R/rehE.log" 2>&1 || { echo "!! step E failed"; tail -5 "$R/rehE.log"; exit 1; }
grep -E "all_pass|clean|FAIL" "$R/rehE.log" | tail -3
echo "### $(date +%T) step F"
$PY -u tracks/t2_bci/scripts/t2p_repack.py --zip "$R/rehE.zip" --out "$R/rehF" --export "$EXP" --template $T --set-ref \
  --set "dcfam=@tracks/t2_bci/submission/configs/t2_d516_dcfam.json#dcfam" --set "side=@tracks/t2_bci/submission/configs/t2_d517_side.json#side" \
  --add-file "$HOME/workspace/outputs/neural_interface_challenge/t2_d516/dcfam.npz" --add-file "$HOME/workspace/outputs/neural_interface_challenge/t2_d517_final/side.pt" \
  > "$R/rehF.log" 2>&1 || { echo "!! step F failed"; tail -8 "$R/rehF.log"; exit 1; }
grep -E "all_pass|parts max|T2-D500|FAIL" "$R/rehF.log" | tail -4
echo "### $(date +%T) pkgcheck"
$PY -u tracks/t2_bci/scripts/t2p_w17_pkgcheck.py --old "$R/rehE" --new "$R/rehF" --out "$R/pkgcheck.json" 2>&1 | grep -E "w17 check|Error|assert" | tail -5
echo "### $(date +%T) done"

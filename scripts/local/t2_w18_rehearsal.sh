#!/usr/bin/env bash
# T2-D525 / W18 build REHEARSAL on the local 5090 (PREPARATION; never a candidate; nothing here is uploaded): the W18 build steps run through
# t2p_repack.py on W17 build 1 (t2p_fin1_w17.zip, fetched). Purpose: catch a W18 step failure (template inertness, the imcoh recombination,
# robustness, GPU memory) before an owner line for W18 puts these steps on the critical path.
#   E: W17 zip -> W18 template only (switches off): must equal W17 on every public window (clean equality, 0 differ).
#   F: E -> cfg imcoh on (configs/t2_d525_imcoh.json, imcoh.npz from t2_d525_final.py), --set-ref: the binding checks are the
#      independent recombination (now with the imcoh term), the edit took, robustness.
#   gpu-submit bash scripts/local/t2_w18_rehearsal.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
R=$O/t2_w18_rehearsal
ZIP=$O/sofia_s1/ni26-t2p-repack-t2p_fin1_w17-1619616/t2p_fin1_w17.zip
mkdir -p "$R"
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=tracks/t2_bci/submission/t2p_submission_w18.py
echo "### $(date +%T) step E (W18 template, switches off)"
$PY -u tracks/t2_bci/scripts/t2p_repack.py --zip "$ZIP" --out "$R/rehE" --export "$EXP" --template $T > "$R/rehE.log" 2>&1 || { echo "!! step E failed"; tail -5 "$R/rehE.log"; exit 1; }
grep -E "all_pass|clean:|FAIL" "$R/rehE.log" | tail -3
echo "### $(date +%T) step F (imcoh on)"
$PY -u tracks/t2_bci/scripts/t2p_repack.py --zip "$R/rehE.zip" --out "$R/rehF" --export "$EXP" --template $T --set-ref \
  --set "imcoh=@tracks/t2_bci/submission/configs/t2_d525_imcoh.json#imcoh" --add-file "$O/t2_d525_final/imcoh.npz" \
  > "$R/rehF.log" 2>&1 || { echo "!! step F failed"; tail -8 "$R/rehF.log"; exit 1; }
grep -E "all_pass|parts max|T2-D500|clean:|FAIL" "$R/rehF.log" | tail -5
echo "### $(date +%T) done"

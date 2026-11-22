#!/usr/bin/env bash
# T2-D524 / W18 build REHEARSAL, step G (PREPARATION; never a candidate; nothing here is uploaded): t2p_repack --fm-from a side-head donor on the
# W18 rehearsal's step-F zip (W17 build 1 + W18 template + imcoh). The donor is FAKE (t2_w18_fakedonor.py: W17's own FM members + small random
# heads), so the members' plain mean must equal the donor's exactly and the heads must equal the independent recomputation; the parts check,
# the recombination (with the imcoh term), robustness and the clean pass bind as for any donor.
#   DONOR=<zip> NAME=<name> gpu-submit bash scripts/local/t2_w18_fmside_rehearsal_g2.sh   (the parametrised copy; G2 = DONOR=$R/fakemerged.zip NAME=rehG2)
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_w18_rehearsal
# G's outputs (out + its _ref / _dref / _old trees + ZIP, about 17 GB) go to New Volume2 beside sofia_s1 (the manager 10:1x: New Volume at 98 %)
G=${GOUT:-"~/workspace_offload/outputs/neural_interface_challenge/t2_w18_rehearsal_G"}
mkdir -p "$G"
DONOR=${DONOR:-"$R/fakedonor.zip"}; NAME=${NAME:-rehG}      # G2: DONOR=$R/fakemerged.zip NAME=rehG2 (the W18 build's shape: 5 heads + 1 head-less swapped member)
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=tracks/t2_bci/submission/t2p_submission_w18.py
[ -f "$R/rehF.zip" ] && [ -f "$DONOR" ] || { echo "!! needs rehF.zip and $DONOR"; exit 2; }
echo "### $(date +%T) step $NAME (FM side heads from the donor $DONOR)"
$PY -u tracks/t2_bci/scripts/t2p_repack.py --zip "$R/rehF.zip" --out "$G/$NAME" --export "$EXP" --template $T --fm-from "$DONOR" \
  > "$R/$NAME.log" 2>&1 || { echo "!! step $NAME failed"; tail -8 "$R/$NAME.log"; exit 1; }
grep -E "all_pass|parts max|T2-D524|clean:|FAIL" "$R/$NAME.log" | tail -6
echo "### $(date +%T) done"

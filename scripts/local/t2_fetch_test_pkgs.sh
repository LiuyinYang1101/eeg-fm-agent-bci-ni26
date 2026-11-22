#!/usr/bin/env bash
# Bring sofia-trained submission packages home and contract-test them locally (CPU, incl. warm-up replay).
#   bash scripts/local/t2_fetch_test_pkgs.sh <sofia job dir, e.g. ni26-t2-final-1434380> <name> [<name> ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
JOB=${1:?job dir}; shift
SRC=~/workspace/outputs/neural_interface_challenge/sofia_s1/$JOB/submissions
DST=~/workspace/outputs/neural_interface_challenge/t2/submissions
[ -d "$SRC" ] || s1 fetch ni26 "$JOB/submissions"
mkdir -p "$DST"
export CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
for n in "$@"; do
  [ -e "$DST/$n" ] && { echo "!! $DST/$n exists — refusing to overwrite"; exit 3; }
  cp -r "$SRC/$n" "$SRC/$n.zip" "$SRC/$n.manifest.json" "$SRC/$n.golden.npz" "$DST/"
  (cd "$DST" && sha256sum "$n.zip" | tee -a "$DST/SHA256SUMS")
  $PY scripts/t2_contract_test.py --name "$n" --warmup-replay
done

#!/usr/bin/env bash
# Classical floors (CPU) at T2 on a protocol, then TSF-style late fusions of {eegnet,shallow} with fbts_lr.
#   bash scripts/local/t2_floors_fusion.sh <protocol>
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
export CUDA_VISIBLE_DEVICES="" NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=8
P=${1:?protocol}
for m in logbp_lda ts_lr fbts_lr; do $PY scripts/t2_run.py --protocol "$P" --method $m --tier T2 --device cpu --now; done
for m in fuse_eegnet_fbts fuse_shallow_fbts; do $PY scripts/t2_run.py --protocol "$P" --method $m --tier T2 --device cpu --now; done
echo "floors+fusion done $(date -Is)"

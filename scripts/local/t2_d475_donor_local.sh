#!/usr/bin/env bash
# T2-D475e build preparation on the local 5090 (the manager's re-ordering, 2026-11-14; weights only, NOT an adoption): the same donor
# build as tracks/t2_bci/sofia/launch_d475_prep.sh donor (five CAR ST-L, seeds 600-604, M7's FM recipe at 100 epochs, all users, a
# 1-epoch FBCNet placeholder, --fm-donor), plus --fm-micro 4 (exact gradient accumulation; batch 64 does not fit 32 GB). The d111-all
# CPT is the sofia file fetched locally (sha256 checked). Queued behind the running jobs (FIFO). The sofia copy 1615573 stays queued as
# the redundant copy; whichever finishes second is cancelled.
#   bash scripts/local/t2_d475_donor_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts scripts/local/t2_d475_donor_local.sh)" ] || { echo "commit first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d475/donor_5090
CK=$O/sofia_s1/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth
[ "$(sha256sum "$CK" | cut -c1-16)" = 01a3c54d37a08cb9 ] || { echo "d111-all checkpoint missing or wrong sha"; exit 2; }
KN='{"600":{"car":1},"601":{"car":1},"602":{"car":1},"603":{"car":1},"604":{"car":1}}'
mkdir -p "$R"; job=$R/job.sh
cat > "$job" <<EOF
set -uo pipefail; cd '$(pwd)'
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{ echo "### \$(date +%T) d475 donor on \$(nvidia-smi --query-gpu=name --format=csv,noheader); git \$(git rev-parse --short HEAD)"; } >> '$R/donor.log'
$PY -u tracks/t2_bci/scripts/t2p_final.py --export '$EXP' --out '$R/t2p_d475_car5donor' \\
  --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \\
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 600,601,602,603,604 \\
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt '$CK' --fm-seed-knobs '$KN' --fm-donor --fm-micro 4 >> '$R/donor.log' 2>&1; rc=\$?
[ -f '$R/t2p_d475_car5donor.zip' ] && sha256sum '$R/t2p_d475_car5donor.zip' >> '$R/donor.log'
echo "### done d475 donor \$(date +%T) rc=\$rc" >> '$R/donor.log'
exit \$rc
EOF
if [ "${1:-}" = --dry ]; then cat "$job"; exit 0; fi
[ ! -e "$R/t2p_d475_car5donor.zip" ] || { echo "$R already holds a donor zip"; exit 2; }
gpu-submit -L "neural_interface_challenge:t2-d475-donor" bash "$job"

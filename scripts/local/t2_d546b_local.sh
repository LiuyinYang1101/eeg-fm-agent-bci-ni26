#!/usr/bin/env bash
# T2-D546 (b) build PREPARATION, LOCAL FALLBACK of sofia job 1626636 (the manager ≈ 23:2x: only if 1626636 has not STARTED by Nov 17 09:00;
# the manager sequences it on the local 5090; NEVER in both places: s1 cancel 1626636 first, verified gone, then submit this).
# The same code path (t2p_final --fm-init-zip, commit 93d7ace7 or later with t2p_final / t2p_fm unchanged since), the same inputs: W18 b2's ZIP
# (local copy, sha256 checked against D534's 93a4aad0…5cbe before training), the d111-all CPT (local copy of the sofia run dir), the same
# flags as tracks/t2_bci/sofia/launch_d546_build.sh. One deviation, stated: --fm-micro 2 (gradient accumulation over 2 chunks per batch,
# t2p_fm's exact TA568 path, for the 32 GB 5090; the H200 job runs micro 1). Donor build: the FM files are the product, no kit replay.
#   bash scripts/local/t2_d546b_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_final.py tracks/t2_bci/scripts/t2p_fm.py src/ni26 scripts/local/t2_d546b_local.sh)" ] || { echo "commit first"; exit 2; }
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
B2=$S/ni26-t2p-w18-t2p_fin2_w18-1622525/t2p_fin2_w18.zip
B2SHA=93a4aad05e8da69c7ae349aaf1ab6e7a9b70f7f4d74ccc6e6c2d8d35e3515cbe
CK=$S/ni26-t2p-cpt-d111-all/checkpoint-cpt.pth
D=$O/t2_d546b_local
[ -e "$D" ] && { echo "$D exists: refusing to overwrite"; exit 2; }
KN='{"622":{"tilt":0.15},"624":{"car":1,"tilt":0.15}}'
job=$(mktemp "$O/t2_d546b_local_job.XXXX.sh")
cat > "$job" <<EOF
set -euo pipefail; cd '$(pwd)'; mkdir -p '$D'; git rev-parse HEAD > '$D/git_head.txt'
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "### \$(date +%T) sha256 of W18 b2's ZIP" >> '$D/arm.log'
[ "\$(sha256sum '$B2' | cut -d' ' -f1)" = "$B2SHA" ] || { echo "### W18 b2 ZIP sha mismatch" >> '$D/arm.log'; exit 3; }
echo "### \$(date +%T) t2p_final D546b continuation (local fallback of 1626636)" >> '$D/arm.log'
$PY -u tracks/t2_bci/scripts/t2p_final.py --export '$EXP' --out '$D/t2p_d546b_cont_b2' --model fbcnet_bboth --align none --epochs 1 --seeds 0 --train all \
  --fm st_large --fm-align sz --fm-calib gain --fm-epochs 100 --fm-weight 0.35 --fm-mixup 0.4 --fm-seeds 622,624 \
  --fm-llrd 0.75 --fm-rc 0.75 --fm-ckpt '$CK' --fm-seed-knobs '$KN' --fm-donor --fm-side dcmean --fm-micro 2 \
  --fm-init-zip '$B2' --fm-init-map 622:fm_2.pt,624:fm_4.pt --fm-cont-epochs 25 --fm-cont-lr 0.1 > '$D/t2p_final.log' 2>&1
echo "### done \$(date +%T): \$(ls '$D/t2p_d546b_cont_b2' | tr '\n' ' ')" >> '$D/arm.log'
EOF
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-d546b-local" bash "$job"

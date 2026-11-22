#!/usr/bin/env bash
# Night 2026-10-05, part 2 — new spec system (configs/ + t2_run.py). Each line queues its own gpu-submit job(s).
set -euo pipefail
cd "$(dirname "$0")/../.."
R="$HOME/miniconda3/envs/neural_interface_challenge/bin/python scripts/t2_run.py"
$R --protocol dreyer_xs5   --method eegnet_r1          --tier T2      # Dreyer anchor (duplicate recording excluded)
$R --protocol dreyer_kit   --method eegnet_r1          --tier T2      # warm-up split anchor (cross-subject; warm-up calibration only)
$R --protocol scherer3_xs9 --method eegnet_r1_shuffle  --tier TC      # control: must be at chance
$R --protocol dreyer_xs5   --method eegnet_r1_shuffle  --tier TC      # control: must be at chance
for m in shallow_r1 ifnet_r1 fbcnet_r1; do $R --protocol dreyer_xs5 --method $m --tier T2; done   # Scherer leaders on Dreyer

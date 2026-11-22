#!/usr/bin/env bash
# T2-D444: the menu builds M7r (M7's exact line on fresh seeds) and M8r (the same seeds + tilt 0.15 on all five ST-L members).
#   mem-guard bash scripts/local/t2_d444_launch.sh dry|GO M7r|M8r
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?dry|GO}; name=${2:?M7r|M8r}
S=100,101,102,103,104,105,106,107,108,109,110,111,112,113,114
case $name in
  M7r) K='{"100": {"epochs": 100}, "101": {"epochs": 100}, "102": {"epochs": 100}, "103": {"epochs": 100}, "104": {"lp_epochs": 10}}';;
  M8r) K='{"100": {"epochs": 100, "tilt": 0.15}, "101": {"epochs": 100, "tilt": 0.15}, "102": {"epochs": 100, "tilt": 0.15}, "103": {"epochs": 100, "tilt": 0.15}, "104": {"lp_epochs": 10, "tilt": 0.15}}';;
  *) exit 2;;
esac
SEEDS=$S W3=0.1 X3="--model3 ifnet --align3 ea --seeds3 $S --epochs3 180 --recipe3 r1_pulx_vp2_e180_vr --user-blend3 0.5" \
  TEMPLATE=tracks/t2_bci/submission/t2p_submission_w11.py \
  bash tracks/t2_bci/sofia/launch_maxstack.sh "$mode" "$name" shallow sz 100,101,102,103,104 0.35 "$K" 2>&1 | grep -v "^chmod\|^rm: "

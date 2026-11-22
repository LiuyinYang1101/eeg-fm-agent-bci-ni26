#!/usr/bin/env bash
# T2-D471: M7r2-B3 = M7's exact line (T2-D384 / D444) on a SECOND set of fresh seeds (CNN 200-214 in all three families, ST-L 200-204),
# built through template W12 with the decoder start vector (T2-D459; approved by the user's template line, Nov 12). build -> sk -> tv.
#   mem-guard bash scripts/local/t2_d471_launch.sh dry|GO
set -euo pipefail
cd "$(dirname "$0")/../.."
mode=${1:?dry|GO}
S=200,201,202,203,204,205,206,207,208,209,210,211,212,213,214
K='{"200": {"epochs": 100}, "201": {"epochs": 100}, "202": {"epochs": 100}, "203": {"epochs": 100}, "204": {"lp_epochs": 10}}'
SEEDS=$S W3=0.1 X3="--model3 ifnet --align3 ea --seeds3 $S --epochs3 180 --recipe3 r1_pulx_vp2_e180_vr --user-blend3 0.5" \
  TEMPLATE=tracks/t2_bci/submission/t2p_submission_w12.py TTA_START='[0.4073,0.2541,0.3386]' \
  bash tracks/t2_bci/sofia/launch_maxstack.sh "$mode" M7r2b3 shallow sz 200,201,202,203,204 0.35 "$K" 2>&1 | grep -v "^chmod\|^rm: "

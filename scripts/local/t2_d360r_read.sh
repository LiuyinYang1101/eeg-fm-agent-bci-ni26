#!/usr/bin/env bash
# T2-D360r read (pre-registered 2026-11-05 22:30; this script written 2026-11-06 before any kit number): T2-D360's spec with the EA ST-L
# member re-pointed at the FRESH seeds 12-23 (sofia 1566579 kit / 1566580 seen1; draw i = seed 12 + i), the same arms (swap_lp10 primary,
# swap_r24, swap_vp12 as D360), SKλ rescreen + the user-cluster CI for swap_lp10. Bar (the hedge bar): typical >= 0 AND P04 >= +0.003, >= 8 / 12.
#   mem-guard bash scripts/local/t2_d360r_read.sh <kit run dir> <seen1 run dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
KIT=${1:?kit run dir (with runs/)}; SE1=${2:?seen1 run dir (with runs/)}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d360r
mkdir -p "$R"
$PY - "$O" "$KIT" "$SE1" <<'EOF'
import json, os, sys
O, KIT, SE1 = sys.argv[1:4]
s = json.load(open(f"{O}/t2_d360/spec.json"))
M = dict(s["members"])
stem = "fm-st_large-mean__ea__n00__fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75"
M["stl_ea"] = {"offset": 12, "kit": f"{KIT}/runs/kit__{stem}cptstlarge-d89b__s{{s}}.npz", "seen1": f"{SE1}/runs/seen1__{stem}cptpt-d111-with__s{{s}}.npz"}
for p in ("kit", "seen1"):
    miss = [i for i in range(12) if not os.path.exists(M["stl_ea"][p].format(s=12 + i))]
    assert not miss, (p, miss)
json.dump({**s, "members": M}, open(f"{O}/t2_d360r/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d360r/spec.json": list(s["arms"])}, open(f"{O}/t2_d360r/sk.json", "w"))
print("arms", list(s["arms"]), "| stl_ea -> fresh seeds 12-23")
EOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 2 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: swap_lp10 vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 swap_lp10 > "$R/cluster_swap_lp10.txt" 2>&1; tail -12 "$R/cluster_swap_lp10.txt"

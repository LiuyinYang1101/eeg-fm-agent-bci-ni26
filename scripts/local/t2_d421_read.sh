#!/usr/bin/env bash
# T2-D421 (pre-registered 2026-11-10 before any code or number): RPA's label-free stretch on the FBTS path, zero GPU. Builds the features + fits
# (tracks/t2_bci/scripts/t2_d421_rpa.py; R0 asserted = D328), then reads FBTS alone R1 vs R0 (one-member arms, anchor R0) under causal SKλ
# (t2_d320_rescreen) + the user-cluster CI / W50 (t2_cluster_ci). BAR: typical >= +0.010 AND W50 > 0 -> opens the CNN row; typical <= 0 -> CLOSED;
# in between -> noted, not pursued.
#   mem-guard bash scripts/local/t2_d421_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
R=$HOME/workspace/outputs/neural_interface_challenge/t2_d421
mkdir -p "$R"
OMP_NUM_THREADS=4 $PY -u tracks/t2_bci/scripts/t2_d421_rpa.py --export "$E" --out "$R/runs" > "$R/fit.log" 2>&1
grep '^\[d421\]' "$R/fit.log"
$PY - "$R" <<'PYEOF'
import json, sys
R = sys.argv[1]
M = {a: {"offset": 0, **{p: f"{R}/runs/{p}__fbts_{a}__s{{s}}.npz" for p in ("kit", "seen1")}} for a in ("r0", "r1")}
json.dump({"n": 12, "anchor": "R0", "members": M, "arms": {"R0": {"r0": 1.0}, "R1": {"r1": 1.0}}}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": ["R0", "R1"]}, open(f"{R}/sk.json", "w"))
PYEOF
mkdir -p "$R/out"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R/out" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd
d=pd.read_csv('$R/out/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print('== FBTS ALONE (R1 stretch vs R0 re-centre only)')
print(d[['arm','anchor','typical','typical_up','P04','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" R0 R1 > "$R/cluster.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster.txt"
$PY - "$R" <<'PYEOF'
import re, sys
import pandas as pd
R = sys.argv[1]
d = pd.read_csv(f"{R}/out/rescreen_sk.csv"); r = d[(d.status == "ok") & (d.dec == "sk") & (d.arm == "R1")].iloc[0]
w50 = float(re.search(r"W50 .*?Δ ([+-]\d\.\d+)", open(f"{R}/cluster.txt").read()).group(1))
v = "BAR MET -> opens the CNN row (signal-level stretch for the EA nets)" if r.typical >= 0.010 and w50 > 0 else ("CLOSED (typical <= 0)" if r.typical <= 0 else "IN BETWEEN -> noted, not pursued")
print(f"== VERDICT (pre-registered): {v}  [typical {r.typical:+.4f}, W50 {w50:+.4f}, P04 {r.P04:+.4f}, newT {r.newT:+.4f}, R1 level {r.typ_level:+.4f}]")
PYEOF

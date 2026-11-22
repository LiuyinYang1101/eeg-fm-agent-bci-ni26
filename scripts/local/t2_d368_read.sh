#!/usr/bin/env bash
# T2-D368 read (pre-registered 2026-11-06 06:21; written before any number): the cross-user SupCon ST-L (seeds 24-35) on nm35 of t2_d345r/full,
# SKλ, 12 draws: ADD (primary) = a 6th ST-L (0.35 split evenly over six); SWAP = r24 -> the SupCon twin (paired); + the user-cluster CI.
#   mem-guard bash scripts/local/t2_d368_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d368
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; S = f"{O}/sofia_s1"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
m = {"offset": 24}
for p, d in (("kit", "ni26-t2p-d368-scx-kit-1568652"), ("seen1", "ni26-t2p-d368-scx-seen1-1568654")):
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, st
    m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=24 + i)) for i in range(12)), p
M["scx"] = m
slot = [k for k in ("r0", "r12", "r24", "vp12", "lp10") if k in nm]
add = {k: v for k, v in nm.items() if k not in slot}; add.update({k: sum(nm[x] for x in slot) / (len(slot) + 1) for k in slot + ["scx"]})
swap = dict(nm); swap["scx"] = swap.pop("r24")
arms = {"nm35": nm, "ADD": add, "SWAP": swap, "m_scx": {"scx": 1.0}, "m_r24": {"r24": 1.0}}
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d368/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d368/spec.json": list(arms)}, open(f"{O}/t2_d368/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: ADD vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 ADD > "$R/cluster_ADD.txt" 2>&1; grep -E 'typical:|new \(' "$R/cluster_ADD.txt"

#!/usr/bin/env bash
# T2-D376 read (pre-registered 2026-11-06 08:5x; written before any fresh number): S2 on FRESH seeds = nm35 with r0 -> LP-FT + lh3 seeds 36-47,
# r12 -> LP-FT + lh3 seeds 48-59, + sw2 seeds 36-47 as a 6th ST-L (the 0.35 slot split evenly over six); SKλ, 12 draws; the fresh singles
# (L2f, Caf) + the user-cluster CI. The user's bar (relayed): typical >= +0.002 with >= 9 / 12, P04 >= 0, new >= 0.
#   mem-guard bash scripts/local/t2_d376_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d376
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, sys
O = sys.argv[1]; S = f"{O}/sofia_s1"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
def stl(kd, sd, off):
    m = {"offset": off}
    for p, d in (("kit", kd), ("seen1", sd)):
        st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, (d, st)
        m[p] = st[0] + "__s{s}.npz"; assert all(os.path.exists(m[p].format(s=off + i)) for i in range(12)), (d, p)
    return m
M["lplh36"] = stl("ni26-t2p-d376-lplh_s36-kit-1568928", "ni26-t2p-d376-lplh_s36-seen1-1568929", 36)
M["lplh48"] = stl("ni26-t2p-d376-lplh_s48-kit-1568930", "ni26-t2p-d376-lplh_s48-seen1-1568931", 48)
M["sw2f"] = stl("ni26-t2p-d376-sw2_s36-kit-1568932", "ni26-t2p-d376-sw2_s36-seen1-1568933", 36)
l2 = dict(nm); l2["lplh36"] = l2.pop("r0"); l2["lplh48"] = l2.pop("r12")
def add6(a, new):
    slot = [k for k in ("r0", "r12", "lplh36", "lplh48", "r24", "vp12", "lp10") if k in a]
    tot = sum(a[k] for k in slot); b = {k: v for k, v in a.items() if k not in slot}
    b.update({k: tot / (len(slot) + 1) for k in slot + [new]}); return b
arms = {"nm35": nm, "S2f": add6(l2, "sw2f"), "L2f": l2, "Caf": add6(nm, "sw2f")}
for k, a in arms.items(): assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d376/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d376/spec.json": list(arms)}, open(f"{O}/t2_d376/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')].set_index('arm')
print(d[['typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: S2f vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 S2f > "$R/cluster_S2f.txt" 2>&1; tail -12 "$R/cluster_S2f.txt"

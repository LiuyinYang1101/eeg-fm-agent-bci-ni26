#!/usr/bin/env bash
# T2-D372 read (pre-registered 2026-11-06 07:1x; written before any package number): the LoRA-at-a-LoRA-lr ST-L (seeds 0-5) on nm35 of
# t2_d345r/full under SKλ, n = 6 draws: SWAP = r0 -> the LoRA twin (paired seeds 0-5); ADD = the LoRA ST-L as a 6th member (0.35 split
# evenly over six); + the trigger's full-precision val slope and the user-cluster CIs (--n 6).
#   mem-guard bash scripts/local/t2_d372_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d372; S=$O/sofia_s1
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np
O = sys.argv[1]; S = f"{O}/sofia_s1"
D = {"kit": "ni26-t2p-d372-lora-kit-1568707", "seen1": "ni26-t2p-d372-lora-seen1-1568709"}
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
M["lora"] = {"offset": 0}
Rx = re.compile(r"\]\s+ep (\d+) loss ([\d.]+) val ([\d.na]+)")
for p, d in D.items():
    st = sorted({f.rsplit("__s", 1)[0] for f in glob.glob(f"{S}/{d}/runs/{p}__fm-st_large-mean__*__s*.npz")}); assert len(st) == 1, st
    M["lora"][p] = st[0] + "__s{s}.npz"
    assert all(os.path.exists(M["lora"][p].format(s=i)) for i in range(6)), p
    cur = {}
    for f in glob.glob(f"{S}/{d}/*.log"):
        for line in open(f):
            m = Rx.search(line)
            if m:
                cur.setdefault(int(m.group(1)), []).append((float(m.group(2)), float(m.group(3)) if m.group(3) != "nan" else np.nan))
    v = lambda es: float(np.mean([np.nanmean([x[1] for x in cur[e]]) for e in es]))  # noqa: E731
    l = lambda e: float(np.mean([x[0] for x in cur[e]]))  # noqa: E731
    print(f"curves {p}: val 16-20 {v(range(16, 21)):.5f} -> 26-30 {v(range(26, 31)):.5f} (slope {v(range(26, 31)) - v(range(16, 21)):+.5f}; rising iff > +0.005); "
          f"train loss ep25 {l(25):.4f} -> ep30 {l(30):.4f} (falling iff < -0.02: {l(30) - l(25):+.4f})")
slot = [k for k in ("r0", "r12", "r24", "vp12", "lp10") if k in nm]
swap = dict(nm); swap["lora"] = swap.pop("r0")
add = {k: v for k, v in nm.items() if k not in slot}; add.update({k: sum(nm[x] for x in slot) / (len(slot) + 1) for k in slot + ["lora"]})
arms = {"nm35": nm, "SWAP": swap, "ADD": add}
for k, a in arms.items():
    assert abs(sum(a.values()) - sum(nm.values())) < 1e-9, k
json.dump({"n": 6, "anchor": "nm35", "members": M, "arms": arms}, open(f"{O}/t2_d372/spec.json", "w"), indent=1)
json.dump({f"{O}/t2_d372/spec.json": list(arms)}, open(f"{O}/t2_d372/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
for a in SWAP ADD; do echo "== user-cluster CI: $a vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 $a --n 6 > "$R/cluster_$a.txt" 2>&1; tail -12 "$R/cluster_$a.txt"; done

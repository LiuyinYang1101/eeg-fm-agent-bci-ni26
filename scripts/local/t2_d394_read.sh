#!/usr/bin/env bash
# T2-D394 step 2 read (design amended 2026-11-06 22:55 before any step-2 number; this script written before any step-2 number): G2 B1's
# density-ratio correction applied to nm35's SHIPPED CNN members. sofia 1572725 (t2_d375_member.py --dres) wrote, per family / protocol / seed,
# the bounded per-window log-ratio adj on the SAME nets (recipe + seeds 192-203 / 156-167). Here: p' = p * exp(adj), renormalised, on
# fbc_tv8c / sh_tv8c's own probabilities (test + val rows), -> fbc_dres / sh_dres. nm35 under SKλ, 12 draws:
#   NMdres (THE DECISION ARM) = nm35 with fbc_tv8c -> fbc_dres AND sh_tv8c -> sh_dres; singles NMdf / NMds; anchor nm35; the user-cluster CI.
# Bar: typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0.
#   mem-guard bash scripts/local/t2_d394_read.sh <fetched ni26-t2p-d394-dres-<job> dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?the fetched ni26-t2p-d394-dres-<job> dir}
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d394
mkdir -p "$R"
$PY - "$O" "$RUN" <<'PYEOF'
import json, os, sys
import numpy as np
O, RUN = sys.argv[1], sys.argv[2]; R = f"{O}/t2_d394"
s = json.load(open(f"{O}/t2_d345r/full/spec.json")); M = dict(s["members"]); nm = s["arms"]["nm35"]
for fam, src, off in (("fbc", "fbc_tv8c", 192), ("sh", "sh_tv8c", 156)):
    assert M[src]["offset"] == off, (src, M[src]["offset"])
    d = f"{R}/route_{fam}_dres"; os.makedirs(d, exist_ok=True)
    for p in ("kit", "seen1"):
        for i in range(12):
            z = np.load(M[src][p].format(s=off + i)); a = np.load(f"{RUN}/{fam}_dres/{p}__s{off + i}.npz")
            assert np.array_equal(z["test_idx"], a["test_idx"]) and np.array_equal(z["val_idx"], a["val_idx"]), (fam, p, i)
            out = {}
            for k, ka in (("prob_last", "adj_te"), ("prob_val_last", "adj_va")):
                q = z[k].astype(np.float64) * np.exp(a[ka].astype(np.float64)); out[k] = (q / q.sum(1, keepdims=True)).astype(np.float32)
            np.savez_compressed(f"{d}/{p}__s{off + i}.npz", test_idx=z["test_idx"], val_idx=z["val_idx"], **out)
    M[f"{fam}_dres"] = {"offset": off, **{p: f"{d}/{p}__s{{s}}.npz" for p in ("kit", "seen1")}}
def sw(a, pairs):
    a = dict(a)
    for o, n in pairs: assert o in a and n not in a, (o, n); a[n] = a.pop(o)
    return a
A = {"nm35": nm, "NMdres": sw(nm, [("fbc_tv8c", "fbc_dres"), ("sh_tv8c", "sh_dres")]), "NMdf": sw(nm, [("fbc_tv8c", "fbc_dres")]), "NMds": sw(nm, [("sh_tv8c", "sh_dres")])}
json.dump({"n": 12, "anchor": "nm35", "members": M, "arms": A}, open(f"{R}/spec.json", "w"), indent=1)
json.dump({f"{R}/spec.json": list(A)}, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))"
echo "== user-cluster CI: NMdres vs nm35"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" nm35 NMdres > "$R/cluster.txt" 2>&1; tail -8 "$R/cluster.txt"

#!/usr/bin/env bash
# T2-D408 read (pre-registered 2026-11-07 12:08; this script written before any D408 number): a CLASSICAL member added to M6 — T2-D328's FBTS
# (filter-bank tangent space + LR, causal re-centering; deterministic predictions in outputs/t2_d328/runs) on outputs/t2_endphase's M6 / M7.
#   spec (anchor M6): AD05 = M6 + FBTS at 0.05 (THE DECISION ARM), AD10 at 0.10 (descriptive);  spec_m7 (anchor M7): M7AD05 (the replicate shape).
#   spec_alone (anchor M6): FBTS alone (descriptive).  Error correlation with M6 (SKλ-decoded, held-out windows, 12 draws pooled).
# Bar: AD05 vs M6 typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0, AND M7AD05 vs M7 typical > 0. W50 reported, never judged.
#   mem-guard bash scripts/local/t2_d408_read.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge; R=$O/t2_d408
mkdir -p "$R"
$PY - "$O" <<'PYEOF'
import json, os, sys
O = sys.argv[1]; R = f"{O}/t2_d408"
ep = json.load(open(f"{O}/t2_endphase/spec.json")); M = dict(ep["members"])
fb = json.load(open(f"{O}/t2_d328/spec.json"))["members"]["fbts"]
assert all(os.path.exists(fb[p].format(s=i)) for p in ("kit", "seen1") for i in range(12))
M["fbts"] = fb
def add(a, n, frac):
    tot = sum(a.values()); a = {k: v * (tot - frac) / tot for k, v in a.items()}; a[n] = frac; return a
M6, M7 = ep["arms"]["M6"], ep["arms"]["M7"]
S = {"spec": ("M6", {"M6": M6, "AD05": add(M6, "fbts", 0.05), "AD10": add(M6, "fbts", 0.10)}),
     "spec_m7": ("M7", {"M7": M7, "M7AD05": add(M7, "fbts", 0.05)}),
     "spec_alone": ("M6", {"M6": M6, "FBTS": {"fbts": 1.0}})}
sk = {}
for nm, (anc, arms) in S.items():
    for a in arms.values():
        assert abs(sum(a.values()) - 1) < 1e-9, a
    json.dump({"n": 12, "anchor": anc, "members": M, "arms": arms}, open(f"{R}/{nm}.json", "w"), indent=1)
    sk[f"{R}/{nm}.json"] = list(arms)
json.dump(sk, open(f"{R}/sk.json", "w"))
PYEOF
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -I tracks/t2_bci/scripts/t2_d320_rescreen.py --export "$E" --out "$R" --jobs 4 \
  --specs "$R/spec*.json" --sk-list "$R/sk.json" > "$R/run.log" 2>&1
$PY -c "
import pandas as pd; d=pd.read_csv('$R/rescreen_sk.csv'); d=d[(d.status=='ok')&(d.dec=='sk')]
print(d[['arm','anchor','typical','typical_up','P04','P04_up','P10','newT','typ_level']].to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
a = d[d.arm == 'AD05'].iloc[0]; r = d[d.arm == 'M7AD05'].iloc[0]
ok = a.typical >= 0.003 and a.typical_up >= 9 and a.P04 >= -0.010 and a.newT >= 0 and r.typical > 0
print(f'[D408] DECISION AD05 vs M6: typical {a.typical:+.4f} ({int(a.typical_up)} / 12), P04 {a.P04:+.4f}, new {a.newT:+.4f}; replicate M7AD05 vs M7 {r.typical:+.4f} -> '
      f\"{'BAR MET' if ok else 'bar NOT met'}\")"
for x in AD05 AD10; do
  echo "== user-cluster CI + W50: $x vs M6"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec.json" M6 $x > "$R/cluster_$x.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster_$x.txt"
done
echo "== user-cluster CI + W50: M7AD05 vs M7"; OMP_NUM_THREADS=2 $PY -I scripts/local/t2_cluster_ci.py "$R/spec_m7.json" M7 M7AD05 > "$R/cluster_M7AD05.txt" 2>&1; sed -n '/per user/,$p' "$R/cluster_M7AD05.txt"
# error correlation with M6 (held-out windows, SKλ-decoded, 12 draws x kit / seen1 x test / val pooled)
$PY -I - "$O" "$E" <<'PYEOF'
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "tracks/t2_bci/scripts")
from t2p_report import causal_sinkhorn
O, E = sys.argv[1:3]; R = f"{O}/t2_d408"
J = json.load(open(f"{R}/spec_alone.json")); M = J["members"]
d = pd.read_csv(f"{E}/index.csv", dtype={"subject": str, "session": str})
strm, ordr, y = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy()
em, ef = [], []
for i in range(12):
    for p in ("kit", "seen1"):
        for split in ("test", "val"):
            P = {}
            for arm in ("M6", "FBTS"):
                ix, mix = None, 0.0
                for m, w in J["arms"][arm].items():
                    z = np.load(M[m][p].format(s=M[m].get("offset", 0) + i))
                    ix_, pr = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    assert ix is None or np.array_equal(ix, ix_); ix, mix = ix_, mix + w * pr.astype(np.float64)
                P[arm] = (ix, causal_sinkhorn(mix, (strm[ix], ordr[ix]), 10, 3, 0.95).argmax(1))
            assert np.array_equal(P["M6"][0], P["FBTS"][0])
            ix = P["M6"][0]; em.append(P["M6"][1] != y[ix]); ef.append(P["FBTS"][1] != y[ix])
em, ef = np.concatenate(em), np.concatenate(ef)
phi = float(np.corrcoef(em.astype(float), ef.astype(float))[0, 1])
out = dict(phi=phi, p_fbts_right_given_m6_wrong=float(np.mean(~ef[em])), m6_err=float(em.mean()), fbts_err=float(ef.mean()),
           p_both_wrong=float(np.mean(em & ef)))
print("[D408] error correlation (held-out windows, SKλ):", {k: round(v, 4) for k, v in out.items()})
json.dump(out, open(f"{R}/errcorr.json", "w"), indent=1)
PYEOF

#!/usr/bin/env python
"""T2 gap analysis of W18's form (the owner's T2 direction 2026-11-16 ≈ 20:00, relayed by the manager; descriptive, decides nothing):
where are W18's errors on the local K = 9 protocol (D536's draws 816-818, W18's form = D536's BASE, level 0.7546)?
  - per group: the ten strangers (cross-user, = the sealed new users) vs the returning sessions (= the sealed later sessions of public users);
  - per user, per class (recall), the confusion matrix;
  - by stream position (windows 0-29 / 30-59 / 60-99 / 100-149 / 150+): what the causal decode pays early in a stream;
  - per family alone (argmax after its own corrector) and the family-oracle (any family right) as a complementarity bound.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_gap_w18.py
"""
import json
import time

import numpy as np
import pandas as pd

T0 = time.time()
SRC36 = open("scripts/local/t2_d536_read.py").read()
CUT = '\nW18 = ["D524", "D529"]\n'
assert SRC36.count(CUT) == 1
exec(compile(SRC36.split(CUT)[0], "t2_d536_read_head(gap)", "exec"))

W18 = ["D524", "D529"]
SIDE_MODE = "unc"
BASE = mix(W18)
lv = level(BASE)
print(f"gap: BASE level {lv}", flush=True)

# the families' own decoded outputs (each through its own causal corrector, as inside mix)
FAM = {}
for f in range(10):
    ix = IX[f]
    key = (STREAM[ix], ORD[ix])
    for d in DRAWS:
        P = parts_of(f, d, W18)
        FAM[(f, d)] = {fam: causal_sinkhorn(P[fam], key, 10, 3, 0.95, start=np.ones(3) / 3) for fam in P}

rows = []
for (f, d), (ix, mx) in BASE.items():
    kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
    # position in the stream (0-based, by 'order')
    pos = np.zeros(len(ix), int)
    for s in np.unique(STREAM[ix]):
        j = np.flatnonzero(STREAM[ix] == s)
        j = j[np.argsort(ORD[ix][j], kind="stable")]
        pos[j] = np.arange(len(j))
    fams = FAM[(f, d)]
    rows.append(pd.DataFrame(dict(f=f, d=d, row=ix, user=U[ix], kind=kind, cell=CELL[ix], y=Y[ix], pred=mx.argmax(1), conf=mx.max(1), pos=pos,
                                  **{f"ok_{k}": (v.argmax(1) == Y[ix]) for k, v in fams.items()})))
R = pd.concat(rows, ignore_index=True)
R["ok"] = R.pred == R.y
R = R[R.kind != "other"]


def cell_ba(df, col="pred"):
    out = []
    for (u, c), g in df.groupby(["user", "cell"]):
        ks = np.unique(g.y)
        if len(ks) >= 2:
            out.append(dict(user=u, cell=c, ba=np.mean([np.mean(g[col][g.y == k] == k) for k in ks])))
    return pd.DataFrame(out)


res = {"level_ten_strangers": lv}
for kd in ("stranger", "returning"):
    g = R[R.kind == kd]
    cb = cell_ba(g)
    res[kd] = dict(n_windows=int(len(g) / len(DRAWS)), cell_ba=round(float(cb.groupby("user").ba.mean().mean()), 4),
                   per_user=cb.groupby("user").ba.mean().round(4).to_dict(),
                   recall={int(k): round(float(g.ok[g.y == k].mean()), 4) for k in sorted(g.y.unique())},
                   confusion=pd.crosstab(g.y, g.pred, normalize="index").round(3).to_dict(),
                   by_position={str(b): round(float(g.ok[(g.pos >= lo) & (g.pos < hi)].mean()), 4)
                                for b, (lo, hi) in {"0-29": (0, 30), "30-59": (30, 60), "60-99": (60, 100), "100-149": (100, 150), "150+": (150, 10 ** 6)}.items()},
                   share_by_position={str(b): round(float(((g.pos >= lo) & (g.pos < hi)).mean()), 4)
                                      for b, (lo, hi) in {"0-29": (0, 30), "30-59": (30, 60), "60-99": (60, 100), "100-149": (100, 150), "150+": (150, 10 ** 6)}.items()},
                   family_alone_acc={k[3:]: round(float(g[k].mean()), 4) for k in g.columns if k.startswith("ok_")},
                   stack_acc=round(float(g.ok.mean()), 4),
                   family_oracle_acc=round(float(g[[k for k in g.columns if k.startswith("ok_")]].any(axis=1).mean()), 4),
                   conf_deciles_acc=g.groupby(pd.qcut(g.conf, 5, labels=False, duplicates="drop")).ok.mean().round(3).to_dict())
res["sealed_mix_weights"] = "0.57 strangers + 0.43 returning (D517's reading)"
for k, v in res.items():
    print(f"gap {k}: {v}", flush=True)
json.dump(res, open(O / "t2_gap_w18.json", "w"), indent=1, default=str)
print(f"gap done ({time.time() - T0:.0f} s)", flush=True)

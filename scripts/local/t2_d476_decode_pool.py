#!/usr/bin/env python
"""T2-D476 (written before any number; the fresh-context seat G10's item 3, 2026-11-13): DECODE EACH FAMILY, THEN POOL.
The package pools raw member probabilities and runs ONE causal Sinkhorn (SKλ: iters 10, n0 3, λ 0.95, no start vector, as every shape-C read).
Arms on T2-D459's shape C (M7's weights; CNN files = D419 / D429 control draws, FM = D442 control draws; kit + seen1, test + val, 12 draws):
  base : raw pool (shipped weights) -> SKλ                                     (= shape C as read in T2-D475)
  DP1  : per-family SKλ (FBCNet / ShallowNet / IFNet / the five ST-L) -> pool with the families' shipped weights -> argmax
  DP2  : as DP1, then SKλ again on the pooled balanced posteriors
Within a family the shipped member weights are kept (the five ST-L at equal weight). Typical = non-P04.
BAR (each of DP1, DP2 vs base): typical >= +0.003 with >= 9 / 12; per-user reported (floor -0.005 reported, not a bar).
    mem-guard python scripts/local/t2_d476_decode_pool.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
FAM = {"fbc": ["fbc_tv8c"], "sh": ["sh_tv8c"], "ifn": ["ifn"], "fm": [k for k in C if k.startswith("n_")]}
assert sorted(sum(FAM.values(), [])) == sorted(C), (FAM, C)
idx = pd.read_csv(EXP / "index.csv")
N = 12
# T2-D476c (the manager's ask, pre-registered before its numbers): "b3 fresh" = the same arms on draw set F, which neither lever was read on in
# shape C: FBCNet = T2-D430's control (348-359), ShallowNet = T2-D424's control (336-347), IFNet = M7's stored nets (168-, read only in
# D458), the five ST-L = T2-D455's controls (472 + k, k = 0..4); M7's weights; 10 draws (the ST-L seeds); start vectors cross-fitted on F.
if "fresh" in sys.argv[1:]:
    import glob
    S_ = O / "sofia_s1"
    M["fbc_tv8c"] = json.load(open(O / "t2_d430/spec_member_fbc.json"))["members"]["fbc_ctl"]
    M["sh_tv8c"] = json.load(open(O / "t2_d424/spec_member.json"))["members"]["sh_ctl"]
    M["ifn"] = json.load(open(O / "t2_d408/spec_m7.json"))["members"]["ifn"]
    stl = {}
    for pr in ("kit", "seen1"):
        f = glob.glob(f"{S_}/ni26-t2p-d455-ctl-{pr}-[0-9]*/runs/{pr}__fm-*__s472.npz")
        assert len(f) == 1, f
        stl[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    for k, m in enumerate([m for m in C if m.startswith("n_")]):
        M[m] = {"offset": 472 + k, **stl}
    N = 10
SK = dict(iters=10, n0=3, lam=0.95)


def load(m, proto, i, split):
    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
    return (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])


out = O / "t2_d476"
rows = []
for proto in (("kit", "seen1") if "b3" not in sys.argv[1:] else ()):
    for i in range(N):
        for split in ("test", "val"):
            ix, P = None, {}
            for m in C:
                ix_, p = load(m, proto, i, split)
                assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                ix, P[m] = ix_, p.astype(np.float64)
            keys = (idx.stream.to_numpy()[ix], idx.order.to_numpy()[ix])
            sk = lambda q: causal_sinkhorn(q, keys, SK["iters"], SK["n0"], SK["lam"])  # noqa: E731
            raw = sum(C[m] * P[m] for m in C)
            fam_w = {f: sum(C[m] for m in ms) for f, ms in FAM.items()}
            fam_q = {f: sk(sum(C[m] * P[m] for m in ms) / fam_w[f]) for f, ms in FAM.items()}
            pooled = sum(fam_w[f] * fam_q[f] for f in FAM)
            pred = {"base": sk(raw).argmax(1), "DP1": pooled.argmax(1), "DP2": sk(pooled).argmax(1)}
            y, cells = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix]
            subj = idx.subject.str[-3:].to_numpy()[ix]
            for c in np.unique(cells):
                mk = cells == c
                r = dict(proto=proto, split=split, i=i, cell=c, subj=subj[mk][0])
                for a, pr in pred.items():
                    r[a] = np.mean([np.mean(pr[mk][y[mk] == k] == k) for k in np.unique(y[mk])])
                rows.append(r)
        print(f"{proto} draw {i} done", flush=True)
if "b3" not in sys.argv[1:]:
  R = pd.DataFrame(rows)
  out.mkdir(exist_ok=True)
  R.to_csv(out / "cells.csv", index=False)
  t = R[R.subj != "P04"]
  print(f"base typical level {t.base.mean():.4f}")
  for a in ("DP1", "DP2"):
    d = (t[a] - t.base).groupby(t.i).mean()
    u = (R[a] - R.base).groupby(R.subj).mean()
    pp = (t[a] - t.base).groupby(t.proto).mean()
    print(f"{a} vs base: typical {d.mean():+.4f} ({int((d > 0).sum())}/{N}) level {t[a].mean():.4f}  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items())
          + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in pp.items()), flush=True)


# T2-D476b (pre-registered in the D476 row before its numbers): the same arms under the shipped B3 decoder (start vector, T2-D459).
# Start vector for a split = mean of the normalised raw mix rows over the OTHER split (disjoint users), all draws and protocols; per family
# the same with the family's own mix. base_B3 = raw pool -> SKλ(start); DP1_B3 = per-family SKλ(start_f) -> pool -> argmax.
if "b3" in sys.argv[1:]:
    MIX = {}
    for proto in ("kit", "seen1"):
        for i in range(N):
            for split in ("test", "val"):
                ix, P = None, {}
                for m in C:
                    ix_, p = load(m, proto, i, split)
                    assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                    ix, P[m] = ix_, p.astype(np.float64)
                fam = {f: sum(C[m] * P[m] for m in ms) for f, ms in FAM.items()}
                fam["all"] = sum(C[m] * P[m] for m in C)
                MIX[(proto, i, split)] = (ix, {f: v / v.sum(1, keepdims=True) for f, v in fam.items()})
    VEC = {(f, split): np.concatenate([v[1][f] for k, v in MIX.items() if k[2] != split]).mean(0) for f in list(FAM) + ["all"] for split in ("test", "val")}
    U = idx.subject.to_numpy()
    assert not (set(U[np.concatenate([v[0] for k, v in MIX.items() if k[2] == "test"])]) & set(U[np.concatenate([v[0] for k, v in MIX.items() if k[2] == "val"])]))
    print("start vectors:", {f"{f}/{sp}": np.round(v, 3).tolist() for (f, sp), v in VEC.items()}, flush=True)
    # D476b gate (i): "b3 shuf" = windows in a random order within each stream (seeded; the T2.json stress case), same start vectors
    ORD = idx.order.to_numpy().copy()
    if "shuf" in sys.argv[1:]:
        rng = np.random.default_rng(476)
        ORD = rng.permutation(len(idx)).astype(np.int64)      # a random rank; causal_sinkhorn sorts within each stream by it
    rows = []
    for (proto, i, split), (ix, mx) in MIX.items():
        keys = (idx.stream.to_numpy()[ix], (ORD if "shuf" in sys.argv[1:] else idx.order.to_numpy())[ix])
        fam_w = {f: sum(C[m] for m in ms) for f, ms in FAM.items()}
        pooled_b3 = sum(fam_w[f] * causal_sinkhorn(mx[f], keys, SK["iters"], SK["n0"], SK["lam"], start=VEC[(f, split)]) for f in FAM)
        pooled = sum(fam_w[f] * causal_sinkhorn(mx[f], keys, SK["iters"], SK["n0"], SK["lam"]) for f in FAM)
        pred = {"base_B3": causal_sinkhorn(mx["all"], keys, SK["iters"], SK["n0"], SK["lam"], start=VEC[("all", split)]).argmax(1),
                "DP1_B3": pooled_b3.argmax(1), "DP1": pooled.argmax(1)}
        y, cells = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix]
        subj = idx.subject.str[-3:].to_numpy()[ix]
        for c in np.unique(cells):
            mk = cells == c
            r = dict(proto=proto, split=split, i=i, cell=c, subj=subj[mk][0])
            for a, pr in pred.items():
                r[a] = np.mean([np.mean(pr[mk][y[mk] == k] == k) for k in np.unique(y[mk])])
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(out / ("cells_b3" + ("_fresh" if "fresh" in sys.argv[1:] else "") + ("_shuf" if "shuf" in sys.argv[1:] else "") + ".csv"), index=False)
    old = pd.read_csv(out / "cells.csv")
    chk = R.merge(old[["proto", "split", "i", "cell", "DP1", "base"]], on=["proto", "split", "i", "cell"], suffixes=("", "_old"))
    assert "shuf" in sys.argv[1:] or "fresh" in sys.argv[1:] or np.allclose(chk.DP1, chk.DP1_old), "DP1 differs from the D476 read"
    t = R[R.subj != "P04"]
    print(f"base_B3 typical level {t.base_B3.mean():.4f}" + ("" if "fresh" in sys.argv[1:] else f" (no-start base {t.merge(old, on=['proto','split','i','cell']).base.mean():.4f})"))
    for a in ("DP1_B3", "DP1"):
        d = (t[a] - t.base_B3).groupby(t.i).mean()
        u = (R[a] - R.base_B3).groupby(R.subj).mean()
        pp = (t[a] - t.base_B3).groupby(t.proto).mean()
        sp = (t[a] - t.base_B3).groupby(t.split).mean()
        print(f"{a} vs base_B3: typical {d.mean():+.4f} ({int((d > 0).sum())}/{N}) level {t[a].mean():.4f}  " + " ".join(f"{s_} {v:+.4f}" for s_, v in u.items())
              + "  | " + " ".join(f"{s_} {v:+.4f}" for s_, v in pp.items()) + "  | " + " ".join(f"{s_} {v:+.4f}" for s_, v in sp.items()), flush=True)

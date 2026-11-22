"""Zero-GPU regime diagnostics for the 0.57 -> 0.81 question (T2, 2026-11-05).
A. composition; B. label order inside recordings; C. package-level (c_nm blend) cell BA by user / session / interface / run."""
import json
import sys
import numpy as np
import pandas as pd

E = "~/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
d = pd.read_csv(f"{E}/index.csv")
print("A. tasks x interface:", d.groupby(["task", "interface"]).size().to_dict())
print("   sessions per subject:", d.groupby("subject").session.unique().apply(lambda s: sorted(s)).to_dict())
print("   windows per cell: median", int(d.groupby("cell").size().median()), "min", d.groupby("cell").size().min(),
      "max", d.groupby("cell").size().max(), "n cells", d.cell.nunique())
print("   runs per cell (median):", d.groupby("cell").run.nunique().median(), "; windows per run (median):", d.groupby(["cell", "run"]).size().median())

# B. label order: P(next label == this label) inside a stream (kit order), vs the shuffle expectation
same, exp, runs = [], [], []
for s, g in d.sort_values("order").groupby("stream"):
    y = g.y.to_numpy()
    if len(y) < 3:
        continue
    same.append(np.mean(y[1:] == y[:-1]))
    p = np.bincount(y, minlength=3) / len(y)
    exp.append(((p * len(y) - 1).clip(0) / (len(y) - 1) * p).sum() if False else (p ** 2).sum())
    r, m = 1, 1
    for a, b in zip(y[:-1], y[1:]):
        r = r + 1 if a == b else 1
        m = max(m, r)
    runs.append(m)
print(f"B. adjacent same-label rate {np.mean(same):.3f} vs iid {np.mean(exp):.3f}; max run median {np.median(runs):.0f} (n streams {len(same)})")
print("   class share overall:", (d.y.value_counts(normalize=True).sort_index().round(3)).to_dict())

J = json.load(open("~/workspace/outputs/neural_interface_challenge/t2_d253/spec.json"))
W = {"fbc_rc75": 0.4 / 0.9, "sh_vr": 0.3 / 0.9, "stl_rcll": 0.2 / 0.9}


def bal(y, p):
    cl = np.unique(y)
    return np.mean([np.mean(p[y == c] == c) for c in cl])


rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        P = {}
        for split in ("test", "val"):
            acc = None
            for k, w in W.items():
                m = J["members"][k]
                z = np.load(m[proto].format(s=m["offset"] + i))
                ix = z["test_idx"] if split == "test" else z["val_idx"]
                pr = z["prob_last"] if split == "test" else z["prob_val_last"]
                acc = (ix, w * pr) if acc is None else (acc[0], acc[1] + w * pr)
                assert (acc[0] == ix).all()
            ix, pr = acc
            g = d.iloc[ix].copy()
            g["pred"] = pr.argmax(1)
            g["conf"] = pr.max(1)
            for c, gc in g.groupby("cell"):
                rows.append(dict(proto=proto, split=split, seed=i, cell=c, subject=gc.subject.iat[0][-3:], session=gc.session.iat[0],
                                 interface=gc.interface.iat[0], n=len(gc), ba=bal(gc.y.to_numpy(), gc.pred.to_numpy()),
                                 conf=gc.conf.mean()))
R = pd.DataFrame(rows)
C = R.groupby(["proto", "split", "cell", "subject", "session", "interface"]).agg(ba=("ba", "mean"), n=("n", "first"), conf=("conf", "mean")).reset_index()
pd.set_option("display.width", 200)
print("C. package (c_nm blend, raw argmax, mean of 12 draws) cell BA:")
print(C.sort_values(["proto", "split", "subject", "session", "interface"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
for (pr, sp), g in C.groupby(["proto", "split"]):
    print(f"   {pr}/{sp}: cell-mean {g.ba.mean():.3f}; by interface", g.groupby("interface").ba.mean().round(3).to_dict(),
          "; by session", g.groupby("session").ba.mean().round(3).to_dict())
C.to_csv(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", index=False)

# D. decoders: raw / prior (1, 3) / SK10 / SK10 lambda 0.95, per cell, on the same blends (kit + seen1, test + val)
sys.path.insert(0, "~/workspace/projects/neural_interface_challenge/tracks/t2_bci/scripts")
from t2p_report import causal_prior_correction, causal_sinkhorn  # noqa: E402
strm, ordr = d.stream.to_numpy(), d.order.to_numpy()
rows = []
for proto in ("kit", "seen1"):
    for i in range(12):
        for split in ("test", "val"):
            acc = None
            for k, w in W.items():
                m = J["members"][k]
                z = np.load(m[proto].format(s=m["offset"] + i))
                ix = z["test_idx"] if split == "test" else z["val_idx"]
                pr = z["prob_last"] if split == "test" else z["prob_val_last"]
                acc = (ix, w * pr) if acc is None else (acc[0], acc[1] + w * pr)
            ix, mix = acc
            ok = (strm[ix], ordr[ix])
            dec = {"raw": mix, "prior": causal_prior_correction(mix, ok), "sk10": causal_sinkhorn(mix, ok, 10, 3, 1.0),
                   "skl95": causal_sinkhorn(mix, ok, 10, 3, 0.95)}
            g = d.iloc[ix]
            for nm, P in dec.items():
                pred = P.argmax(1)
                for c in np.unique(g.cell):
                    mk = (g.cell == c).to_numpy()
                    rows.append(dict(proto=proto, split=split, seed=i, cell=c, dec=nm, ba=bal(g.y.to_numpy()[mk], pred[mk])))
                for tk in ("Acq", "Onl"):
                    mk = g.task.str.startswith("Acquisition" if tk == "Acq" else "Online").to_numpy()
                    rows.append(dict(proto=proto, split=split, seed=i, cell="task:" + tk, dec=nm,
                                     ba=np.mean([bal(g.y.to_numpy()[mk & (g.cell == c).to_numpy()], pred[mk & (g.cell == c).to_numpy()])
                                                 for c in np.unique(g.cell[mk])])))
D = pd.DataFrame(rows).groupby(["proto", "split", "cell", "dec"]).ba.mean().unstack("dec")[["raw", "prior", "sk10", "skl95"]]
print("D. decoder levels (mean of 12 draws):")
print(D.to_string(float_format=lambda v: f"{v:.3f}"))
cc = D[~D.index.get_level_values("cell").str.startswith("task:")]
print("   group means:", cc.groupby(level=["proto", "split"]).mean().round(3).to_string())
lo = cc[cc.raw < 0.6]; hi = cc[cc.raw >= 0.6]
print(f"   skl95 - raw: cells raw < 0.6 (n {len(lo)}) {(lo.skl95 - lo.raw).mean():+.3f}; raw >= 0.6 (n {len(hi)}) {(hi.skl95 - hi.raw).mean():+.3f}")
print(f"   skl95 - prior: low {(lo.skl95 - lo.prior).mean():+.3f}; high {(hi.skl95 - hi.prior).mean():+.3f}")

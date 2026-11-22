"""T2-D310's arms (A0, tv8, tv8c, pv10, S1, S2, mor) re-read per cell under the shipped decoder (SK10, lambda 0.95), split
P04 cells vs the other users' cells; plus draw-to-draw sd of A0 per group. Zero GPU. Descriptive (not a selector)."""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "~/workspace/projects/neural_interface_challenge/tracks/t2_bci/scripts")
from t2p_report import causal_sinkhorn  # noqa: E402

E = "~/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
d = pd.read_csv(f"{E}/index.csv")
J = json.load(open("~/workspace/outputs/neural_interface_challenge/t2_d310/spec.json"))
strm, ordr, y, cells = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy(), d.cell.to_numpy()


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


cache = {}


def load(m, proto, i, split):
    key = (m, proto, i, split)
    if key not in cache:
        M = J["members"][m]
        z = np.load(M[proto].format(s=M["offset"] + i))
        cache[key] = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
    return cache[key]


rows = []
for proto in ("kit", "seen1"):
    for split in ("test", "val"):
        for i in range(J["n"]):
            for arm, wts in J["arms"].items():
                ix, mix = None, 0
                for m, w in wts.items():
                    ix_, p = load(m, proto, i, split)
                    assert ix is None or (ix == ix_).all()
                    ix, mix = ix_, mix + w * p
                pr = causal_sinkhorn(mix, (strm[ix], ordr[ix]), 10, 3, 0.95).argmax(1)
                for c in np.unique(cells[ix]):
                    mk = cells[ix] == c
                    rows.append(dict(proto=proto, split=split, i=i, arm=arm, cell=c, ba=bal(y[ix][mk], pr[mk])))
R = pd.DataFrame(rows)
R["grp"] = np.where(R.cell.str.contains("/P04\\|"), "P04", "others")
base = R[R.arm == "A0"].set_index(["proto", "split", "i", "cell"]).ba
out = []
for arm in J["arms"]:
    if arm == "A0":
        continue
    a = R[R.arm == arm].set_index(["proto", "split", "i", "cell"])
    dd = (a.ba - base).rename("d").reset_index().merge(a.reset_index()[["proto", "split", "i", "cell", "grp"]])
    per_i = dd.groupby(["grp", "i"]).d.mean().unstack("grp")
    out.append(dict(arm=arm, P04=per_i.P04.mean(), P04_up=int((per_i.P04 > 0).sum()), others=per_i.others.mean(),
                    others_up=int((per_i.others > 0).sum())))
print(pd.DataFrame(out).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
A0 = R[R.arm == "A0"]
lv = A0.groupby(["grp", "i"]).ba.mean().unstack("grp")
print("A0 level P04 {:.3f} / others {:.3f}; draw sd (12 draws, cell-mean of the group) P04 {:.4f} / others {:.4f}".format(
    lv.P04.mean(), lv.others.mean(), lv.P04.std(ddof=1), lv.others.std(ddof=1)))
print("n cells: P04", A0[A0.i == 0].grp.eq("P04").sum(), "others", A0[A0.i == 0].grp.eq("others").sum())

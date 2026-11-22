#!/usr/bin/env python
"""T2-D491 read (written before any number): the TSMNet-form SPD member (tracks/t2_bci/scripts/t2_d491_tsmnet.py), on T2-D475's
A13 harness (W13's decode: per-family causal SK 10 it / n0 3 / lambda 0.95, each family's start vector cross-fitted over the other
split; 10 draws; the A13 base = FBCNet D419 / ShallowNet + IFNet D429 controls, CNN slot seed = offset + i; FM = W14's shape:
5 plain ST-L at seed index i .. i+4 + 3 CAR at i+5 .. i+7, each w_fm / 8; ST-L seed of index k in draw i = 472 + (i + k) mod 14).
The SPD member: seed 360 + i in draw i (pooled net only: no copies, no views).
 STEP 1 (KILL): alone, decoded, typical (non-P04) level < 0.70 -> KILLED, nothing else is read.
 STEP 2 (only if >= 0.70): a FOURTH family at w in {0.03, 0.05, 0.08, 0.12} taken from the CNN share pro rata (D487's form);
   multiplicity: w chosen per split by the OTHER split's typical level (cross-fitted, 't_cf'). BAR (t_cf vs jW_ref): typical >= +0.006
   with >= 8 / 10, no typical user below -0.005; GATE: D476b's within-stream shuffle (rng 476): t_cf loses <= 0.003 more than jW_ref.
   Reported: P05, each fixed w (descriptive), the error correlation with jW_ref, ms / window from the trainer's --time line.
    mem-guard python scripts/local/t2_d491_read.py        (D491_FAKE=1: the SPD files = the base's ShallowNet -> a_tsm = a_sh exactly)
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
FAKE = bool(os.environ.get("D491_FAKE"))
VAR = os.environ.get("D491_VARIANT", "")      # T2-D491b: "nore" = the net without SPD re-centring (tsmnetnore files)
assert VAR in ("", "nore"), VAR
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
FM5 = [k for k in C if k.startswith("n_")]
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
N13 = 10


def stl(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{S}/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        f = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472*.npz")
        assert len(f) == 1, f
        out[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    return out


pp, pc = stl("ni26-t2p-d455-ctl-{pr}"), stl("ni26-t2p-d475-stlcar-{pr}")
w_fm = sum(C[m] for m in FM5)
CNN = [("fbc", "fbc_tv8c"), ("sh", "sh_tv8c"), ("ifn", "ifn")]
cnn_share = sum(C[m] for _, m in CNN)
WW = (0.03, 0.05, 0.08, 0.12)
M["tsm"] = dict(M["sh_tv8c"]) if FAKE else {"offset": 360, **{p: str(O / f"t2_d491/runs/{p}__tsmnet{VAR}__ea__s{{s}}.npz") for p in ("kit", "seen1")}}


def st(k):
    return lambda i: 472 + (i + k) % 14


FMW = [("fm", w_fm / 8, ("pl", st(k))) for k in range(5)] + [("fm", w_fm / 8, ("car", st(k))) for k in (5, 6, 7)]


def with_t(w):
    return [(f, C[m] * (cnn_share - w) / cnn_share, ("M", m)) for f, m in CNN] + FMW + ([("tsm", w, ("M", "tsm"))] if w > 0 else [])


keys_ = idx.stream.to_numpy(), idx.order.to_numpy()
ORD_S = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order


def build(A13):
    MIX = {}
    for proto in ("kit", "seen1"):
        for i in range(N13):
            cache = {}

            def get(src):
                kind, x = src
                f = M[x][proto].format(s=M[x].get("offset", 0) + i) if kind == "M" else (pp if kind == "pl" else pc)[proto].format(s=x(i))
                if f not in cache:
                    z = np.load(f)
                    cache[f] = {"test": (z["test_idx"], z["prob_last"].astype(np.float64)), "val": (z["val_idx"], z["prob_val_last"].astype(np.float64))}
                return cache[f]
            for a, mem in A13.items():
                for split in ("test", "val"):
                    ix, fam, fw = None, {}, {}
                    for f, w, src in mem:
                        ix_, p = get(src)[split]
                        assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                        ix, fam[f], fw[f] = ix_, fam.get(f, 0.0) + w * p, fw.get(f, 0.0) + w
                    MIX[(a, proto, i, split)] = (ix, {f: v / v.sum(1, keepdims=True) for f, v in fam.items()}, fw)
    assert all(abs(sum(v[2].values()) - 1.0) < 1e-9 for v in MIX.values())
    VEC = {}
    for a in A13:
        for sp in ("test", "val"):
            other = [v for k, v in MIX.items() if k[0] == a and k[3] != sp]
            VEC[(a, sp)] = {f: np.concatenate([o[1][f] for o in other]).mean(0) for f in other[0][1]}
    return MIX, VEC


def decode(MIX, VEC, a, shuf=False):
    out = {}
    for (a_, proto, i, split), (ix, mx, fw) in MIX.items():
        if a_ == a:
            kk = (keys_[0][ix], (ORD_S if shuf else keys_[1])[ix])
            out[(proto, i, split)] = (ix, sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1))
    return out


def cells(dec, a):
    rows = []
    for (proto, i, split), (ix, P) in dec.items():
        y, cl, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
        for c in np.unique(cl):
            mk = cl == c
            rows.append(dict(arm=a, proto=proto, split=split, i=i, cell=c, subj=subj[mk][0],
                             ba=np.mean([np.mean(P[mk][y[mk] == k] == k) for k in np.unique(y[mk])])))
    return rows


def cmp(R, a, b, bar):
    m = R[R.arm == a].merge(R[R.arm == b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    ok = d.mean() >= bar and (d > 0).sum() >= 8 and u.drop("P04", errors="ignore").min() >= -0.005
    return (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N13}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
            + " ".join(f"{s} {v:+.4f}" for s, v in u.items()) + f"  -> {'PASS' if ok else 'no'}")


def main():
    alone = {"a_tsm": [("tsm", 1.0, ("M", "tsm"))], **{f"a_{f}": [(f, 1.0, ("M", m))] for f, m in CNN}}
    MIX, VEC = build(alone)
    R = pd.DataFrame(sum((cells(decode(MIX, VEC, a), a) for a in alone), []))
    lv = {}
    for a in alone:
        r = R[R.arm == a]
        lv[a] = r[r.subj != "P04"].ba.mean()
        print(f"D491 alone {a}: typical level {lv[a]:.4f}  per user " + " ".join(f"{u} {v:.3f}" for u, v in r.groupby('subj').ba.mean().items()), flush=True)
    (O / "t2_d491").mkdir(exist_ok=True)
    R.to_csv(O / f"t2_d491/cells_alone{VAR}{'_fake' if FAKE else ''}.csv", index=False)
    if lv["a_tsm"] < 0.70:
        print(f"D491 KILL: the SPD member alone {lv['a_tsm']:.4f} < 0.70 -> KILLED; the stack is not read", flush=True)
        return
    print(f"D491 alone {lv['a_tsm']:.4f} >= 0.70 -> step 2 (the stack)", flush=True)
    A13 = {"jW_ref": with_t(0.0), **{f"t_{w}": with_t(w) for w in WW}}
    MIX, VEC = build(A13)
    R = pd.DataFrame(sum((cells(decode(MIX, VEC, a), a) for a in A13), []))
    parts, wsel = [], {}
    for sp, other in (("test", "val"), ("val", "test")):
        l_ = {w: R[(R.arm == f"t_{w}") & (R.split == other) & (R.subj != "P04")].ba.mean() for w in WW}
        wsel[sp] = max(l_, key=l_.get)
        parts.append(R[(R.arm == f"t_{wsel[sp]}") & (R.split == sp)])
    R = pd.concat([R, pd.concat(parts).assign(arm="t_cf")])
    R.to_csv(O / f"t2_d491/cells_stack{VAR}{'_fake' if FAKE else ''}.csv", index=False)
    print(f"D491 BAR (w* test {wsel['test']} / val {wsel['val']}) t_cf vs jW_ref:", cmp(R, "t_cf", "jW_ref", 0.006), flush=True)
    t = R[(R.arm == "t_cf") & (R.subj != "P04")].ba.mean() - R[(R.arm == "jW_ref") & (R.subj != "P04")].ba.mean()
    print(f"D491 no-harm line (typical >= -0.002): {t:+.4f} -> " + ("ok" if t >= -0.002 else "HARM"), flush=True)

    def shuf_level(a, splits=("test", "val")):
        v = [r["ba"] for r in cells({k: x for k, x in decode(MIX, VEC, a, True).items() if k[2] in splits}, a) if r["subj"] != "P04"]
        return float(np.mean(v))
    sh_cf = np.mean(sum(([r["ba"] for r in cells({k: x for k, x in decode(MIX, VEC, f"t_{wsel[sp]}", True).items() if k[2] == sp}, "x")
                          if r["subj"] != "P04"] for sp in ("test", "val")), []))
    lose_cf = R[(R.arm == "t_cf") & (R.subj != "P04")].ba.mean() - sh_cf
    lose_ref = R[(R.arm == "jW_ref") & (R.subj != "P04")].ba.mean() - shuf_level("jW_ref")
    print(f"D491 shuffle gate: t_cf loses {lose_cf:+.4f}, jW_ref {lose_ref:+.4f} -> " + ("ok" if lose_cf - lose_ref <= 0.003 else "FAILS"), flush=True)
    for w in WW:
        print(f"   (descriptive) t_{w} vs jW_ref:", cmp(R, f"t_{w}", "jW_ref", 0.006).split("  ->")[0], flush=True)
    MIXa, VECa = build({"a_tsm": alone["a_tsm"]})
    da, dr, cs = decode(MIXa, VECa, "a_tsm"), decode(MIX, VEC, "jW_ref"), []
    for k_, (ix, P) in dr.items():
        y, subj = idx.y.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
        m_ = subj != "P04"
        cs.append(np.corrcoef((P != y)[m_].astype(float), (da[k_][1] != y)[m_].astype(float))[0, 1])
    print(f"D491 error correlation with jW_ref (typical windows): {np.mean(cs):+.3f}", flush=True)


if __name__ == "__main__":
    main()

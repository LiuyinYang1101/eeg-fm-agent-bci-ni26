#!/usr/bin/env python
"""T2-D481 (registered 2026-11-14 04:08 before any number; zero GPU): the pool interface after W13 / W14, re-fitted and cross-fitted.
Members = W14's set on A13's draws (shape-C CNN draws D419 / D429, D455 plain ST-L at seed index i + k, the D475 pair's CAR nets at i + 5 .. i + 7),
or on A13r's draws with `rep` (set F's CNNs, D442's plain ST-L). Each family is decoded ONCE per knob (causal SKλ, iters 10, its own start vector
cross-fitted by D476b's rule), so a weighting is a sum of decoded parts. Base = W14 as specified (M7's family weights, CAR inside fm at 0.35 / 8,
λ 0.95 / n0 3 everywhere).
  (1) weights: CAR inside fm: fm share x IFNet share grid; CAR as its own family: plain-fm share x car share grid.
  (2) knobs: per family λ x n0, chosen family by family (others at the shipped knob), then combined.
Cross-fit: for each split, the configuration maximising the OTHER split's typical level is scored on this split.
BAR: typical >= +0.003 with >= 8 / 10, no typical user below -0.005 (and >= 0 on the rep set); gate on a pass: under the within-stream shuffle the
pick must not lose more than base by > 0.003.
    mem-guard python scripts/local/t2_d481_decode_fit.py [rep]
"""
import glob
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
REP = "rep" in sys.argv[1:]
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
FM5 = [k for k in C if k.startswith("n_")]
if REP:
    M["fbc_tv8c"] = json.load(open(O / "t2_d430/spec_member_fbc.json"))["members"]["fbc_ctl"]
    M["sh_tv8c"] = json.load(open(O / "t2_d424/spec_member.json"))["members"]["sh_ctl"]
    M["ifn"] = json.load(open(O / "t2_d408/spec_m7.json"))["members"]["ifn"]
idx = pd.read_csv(EXP / "index.csv")
N = 10


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


def files(proto, i):
    """family -> list of npz paths for draw i (equal weights inside a family)."""
    cnn = {f: [M[m][proto].format(s=M[m].get("offset", 0) + i)] for f, m in (("fbc", "fbc_tv8c"), ("sh", "sh_tv8c"), ("ifn", "ifn"))}
    plain = ([M[m][proto].format(s=M[m].get("offset", 0) + i) for m in FM5] if REP
             else [pp[proto].format(s=472 + (i + k) % 14) for k in range(5)])
    car = [pc[proto].format(s=472 + (i + k) % 14) for k in (5, 6, 7)]
    return dict(cnn, fmin=plain + car, fmpl=plain, car=car)


FAMS = ["fbc", "sh", "ifn", "fmin", "fmpl", "car"]
KNOBS = [(lam, n0) for lam in (0.90, 0.95, 0.98, 1.0) for n0 in (1, 3, 6)]
BASEK = (0.95, 3)
ORD_SHUF = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order

MIX = {}
cache = {}
for proto in ("kit", "seen1"):
    for i in range(N):
        fl = files(proto, i)
        for split in ("test", "val"):
            ix, mx = None, {}
            for fam, fs in fl.items():
                acc = 0.0
                for f in fs:
                    if f not in cache:
                        z = np.load(f)
                        cache[f] = {"test": (z["test_idx"], z["prob_last"].astype(np.float64)), "val": (z["val_idx"], z["prob_val_last"].astype(np.float64))}
                    ix_, p = cache[f][split]
                    assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                    ix, acc = ix_, acc + p / len(fs)
                mx[fam] = acc / acc.sum(1, keepdims=True)
            MIX[(proto, i, split)] = (ix, mx)
    cache.clear()
    print(f"{proto} loaded", flush=True)
VEC = {(f, sp): np.concatenate([v[1][f] for k, v in MIX.items() if k[2] != sp]).mean(0) for f in FAMS for sp in ("test", "val")}


def decode(shuf, knobs_needed):
    """Q[(proto, i, split)][fam][knob] = the family's decoded posteriors."""
    Q = {}
    order = ORD_SHUF if shuf else idx.order.to_numpy()
    for key, (ix, mx) in MIX.items():
        kk = (idx.stream.to_numpy()[ix], order[ix])
        Q[key] = {f: {kn: causal_sinkhorn(mx[f], kk, 10, kn[1], kn[0], start=VEC[(f, key[2])]) for kn in knobs_needed.get(f, [BASEK])} for f in FAMS}
    return Q


Y, CELL, SUBJ = idx.y.to_numpy(), idx.cell.to_numpy(), idx.subject.str[-3:].to_numpy()


def rows(Q, cfg):
    """cfg = {fam: (weight, knob)} -> DataFrame of per-cell BA."""
    out = []
    for (proto, i, split), (ix, _) in MIX.items():
        P = sum(w * Q[(proto, i, split)][f][kn] for f, (w, kn) in cfg.items()).argmax(1)
        y, cells, subj = Y[ix], CELL[ix], SUBJ[ix]
        for c in np.unique(cells):
            mk = cells == c
            out.append((proto, split, i, c, subj[mk][0], np.mean([np.mean(P[mk][y[mk] == k] == k) for k in np.unique(y[mk])])))
    return pd.DataFrame(out, columns=["proto", "split", "i", "cell", "subj", "ba"])


def level(R, split):
    t = R[(R.split == split) & (R.subj != "P04")]
    return t.ba.mean()


def compare(R, Rb):
    key = ["proto", "split", "i", "cell"]
    m = R.merge(Rb[key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    ok = d.mean() >= 0.003 and (d > 0).sum() >= 8 and u.drop("P04", errors="ignore").min() >= -0.005
    return ok, (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
                + " ".join(f"{s} {v:+.4f}" for s, v in u.items()))


def cnn_w(rest, ifn):
    """FBCNet : ShallowNet 4 : 3 of rest - ifn (M7: 0.3143 / 0.2357 / 0.1 of 0.65)."""
    return {"fbc": (rest - ifn) * 4 / 7, "sh": (rest - ifn) * 3 / 7, "ifn": ifn}


def cfg_in(fm, ifn, kn=None):
    kn = kn or {}
    c = {f: (w, kn.get(f, BASEK)) for f, w in cnn_w(1 - fm, ifn).items()}
    c["fmin"] = (fm, kn.get("fmin", BASEK))
    return c


def cfg_own(pl, car, kn=None):
    kn = kn or {}
    rest = 1 - pl - car
    c = {f: (w * rest / 0.65, kn.get(f, BASEK)) for f, w in (("fbc", C["fbc_tv8c"]), ("sh", C["sh_tv8c"]), ("ifn", C["ifn"]))}
    c["fmpl"], c["car"] = (pl, kn.get("fmpl", BASEK)), (car, kn.get("car", BASEK))
    return c


def crossfit(Q, cands):
    """cands: name -> cfg. For each split, pick the cfg maximising the OTHER split's typical level; return (rows, picks)."""
    R = {n: rows(Q, c) for n, c in cands.items()}
    parts, picks = [], {}
    for sp, other in (("test", "val"), ("val", "test")):
        best = max(R, key=lambda n: level(R[n], other))
        picks[sp] = best
        parts.append(R[best][R[best].split == sp])
    return pd.concat(parts), picks, R


base = cfg_in(0.35, 0.10)
assert abs(sum(w for w, _ in base.values()) - 1) < 1e-9
Q = decode(False, {f: KNOBS for f in FAMS})
Rb = rows(Q, base)
print(f"base (W14 as specified) typical level {Rb[Rb.subj != 'P04'].ba.mean():.4f} (A13's add3: 0.8396 on this set; rep 0.8433); "
      f"test {level(Rb, 'test'):.4f} val {level(Rb, 'val'):.4f}", flush=True)
# (1) weights
cands = {f"in fm {fm} ifn {ifn}": cfg_in(fm, ifn) for fm in (0.25, 0.30, 0.35, 0.40, 0.45) for ifn in (0.05, 0.10, 0.15)}
cands.update({f"own pl {pl} car {car}": cfg_own(pl, car) for pl in (0.18, 0.22, 0.26) for car in (0.09, 0.13, 0.17)})
Rw, pw, _ = crossfit(Q, cands)
okw, sw = compare(Rw, Rb)
print(f"(1) WEIGHTS cross-fitted (test <- {pw['test']} | val <- {pw['val']}):", sw, "->", "PASS" if okw else "no", flush=True)
# (2) knobs, family by family on the CAR-inside base, then combined per split
parts, pk = [], {}
for sp, other in (("test", "val"), ("val", "test")):
    kn = {}
    for f in ("fbc", "sh", "ifn", "fmin"):
        lv = {k_: level(rows(Q, cfg_in(0.35, 0.10, {f: k_})), other) for k_ in KNOBS}
        kn[f] = max(lv, key=lv.get)
    pk[sp] = kn
    R_ = rows(Q, cfg_in(0.35, 0.10, kn))
    parts.append(R_[R_.split == sp])
Rk = pd.concat(parts)
okk, sk = compare(Rk, Rb)
print(f"(2) KNOBS cross-fitted (test <- {pk['test']} | val <- {pk['val']}):", sk, "->", "PASS" if okk else "no", flush=True)
# the shuffle gate (reported for every pick, binding on a pass)
need = {f: sorted({BASEK} | {pk[s][f] for s in pk if f in pk[s]}) for f in FAMS}
Qs = decode(True, need)
Rbs = rows(Qs, base)
for name, cfgs in (("weights", {sp: cands[pw[sp]] for sp in pw}), ("knobs", {sp: cfg_in(0.35, 0.10, pk[sp]) for sp in pk})):
    if name == "weights" and any(any(kn != BASEK for _, kn in c.values()) for c in cfgs.values()):
        continue
    R_ = pd.concat([rows(Qs, cfgs[sp])[lambda d, sp=sp: d.split == sp] for sp in cfgs])
    t, tb = R_[R_.subj != "P04"].ba.mean(), Rbs[Rbs.subj != "P04"].ba.mean()
    lose_pick, lose_base = (Rw if name == "weights" else Rk)[lambda d: d.subj != "P04"].ba.mean() - t, Rb[Rb.subj != "P04"].ba.mean() - tb
    print(f"shuffle gate {name}: shuffled level {t:.4f} vs base {tb:.4f}; the pick loses {lose_pick:+.4f}, base {lose_base:+.4f} -> "
          + ("ok" if lose_pick - lose_base <= 0.003 else "FAILS"), flush=True)

#!/usr/bin/env python
"""T2-D320 (descriptive, post hoc, NOT a selector): re-screen every saved package-shape spec (outputs/t2_d*/spec*.json, the
t2p_pkgshape format) with each arm's paired delta vs its own anchor split by user group: P04 cells (the warm-up's hard user) vs
the typical users (P05 P08 P09 P10; 16 cells over kit + seen1, test + val), also P10 (returning user) and newT (P05 P08 P09).
Decoder: the causal prior correction per stream with the arm's (alpha, n0) (t2p_pkgshape's own; default 1, 3); --sk adds SK10
lambda 0.95 (the shipped decoder) for the arms named in --sk-arms. Arms with arm_gate or a per-cell (oracle) prior are skipped.
  python t2_d320_rescreen.py --export E --out DIR [--jobs 8] [--specs glob] [--sk-list file]
"""
import argparse
import glob
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2p_report import causal_prior_correction, causal_sinkhorn  # noqa: E402

G = {}


def init(export):
    d = pd.read_csv(f"{export}/index.csv")
    G.update(strm=d.stream.to_numpy(), ordr=d.order.to_numpy(), y=d.y.to_numpy(), cells=d.cell.to_numpy(),
             subj=d.subject.str[-3:].to_numpy())


def bal(yy, pp):
    return np.mean([np.mean(pp[yy == c] == c) for c in np.unique(yy)])


def temper(P, T):
    if T == 1:
        return P
    Q = np.power(np.clip(P, 1e-12, None), 1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def screen(args):
    path, sk_arms = args
    try:
        J = json.load(open(path))
        if "members" not in J or "arms" not in J or "anchor" not in J:
            return [dict(spec=path, status="not pkgshape")]
        gate, pkey, ppar = J.get("arm_gate", {}), J.get("arm_prior_key", {}), J.get("arm_prior", {})
        arms = {a: w for a, w in J["arms"].items() if a not in gate and pkey.get(a, "stream") == "stream"}
        if J["anchor"] not in arms:
            return [dict(spec=path, status="anchor skipped")]
        n = int(J.get("n", 12))
        cache, rows = {}, []
        for proto in ("kit", "seen1"):
            for i in range(n):
                for split in ("test", "val"):
                    for a, wts in arms.items():
                        ix, mix = None, 0.0
                        for m, w in wts.items():
                            M = J["members"][m]
                            f = M[proto].format(s=M.get("offset", 0) + i)
                            key = (f, split)
                            if key not in cache:
                                if not os.path.exists(f):
                                    return [dict(spec=path, status=f"missing {f}")]
                                z = np.load(f)
                                cache[key] = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                            ix_, p = cache[key]
                            if ix is not None and not (len(ix) == len(ix_) and (ix == ix_).all()):
                                return [dict(spec=path, status="index mismatch")]
                            ix, mix = ix_, mix + w * temper(p.astype(np.float64), M.get("temp", 1))
                        ok = (G["strm"][ix], G["ordr"][ix])
                        al, n0 = ppar.get(a, (1.0, 3.0))
                        decs = {"prior": causal_prior_correction(mix, ok, float(al), float(n0)) if al else mix}
                        if a in sk_arms or a == J["anchor"] and sk_arms:      # T2-D343: optional per-arm SK lambda / n0 (absent = 0.95 / 3)
                            decs["sk"] = causal_sinkhorn(mix, ok, 10, J.get("arm_sk_n0", {}).get(a, 3), J.get("arm_sk_lam", {}).get(a, 0.95))
                        for dn, P in decs.items():
                            pr = P.argmax(1)
                            for c in np.unique(G["cells"][ix]):
                                mk = G["cells"][ix] == c
                                rows.append(dict(proto=proto, split=split, i=i, arm=a, dec=dn, cell=c,
                                                 subj=G["subj"][ix][mk][0], ba=bal(G["y"][ix][mk], pr[mk])))
        R = pd.DataFrame(rows)
        R["grp"] = np.select([R.subj == "P04", R.subj == "P10"], ["P04", "P10"], "newT")
        out = []
        for dn, Rd in R.groupby("dec"):
            per = {}
            for name, mask in (("typical", Rd.grp != "P04"), ("P04", Rd.grp == "P04"), ("P10", Rd.grp == "P10"), ("newT", Rd.grp == "newT")):
                per[name] = Rd[mask].groupby(["arm", "i"]).ba.mean().unstack("arm")
            anc = J["anchor"]
            for a in per["typical"].columns:
                if a == anc:
                    continue
                r = dict(spec=path, status="ok", dec=dn, anchor=anc, arm=a, n=n)
                for name, T in per.items():
                    dd = T[a] - T[anc]
                    r[name] = dd.mean()
                    r[name + "_up"] = int((dd > 0).sum())
                r["typ_level"] = per["typical"][a].mean()
                r["anc_typ_level"] = per["typical"][anc].mean()
                out.append(r)
        return out or [dict(spec=path, status="no arms")]
    except Exception as e:  # noqa: BLE001
        return [dict(spec=path, status=f"error {type(e).__name__}: {e}", tb=traceback.format_exc()[-400:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--specs", default=os.path.expanduser("~/workspace/outputs/neural_interface_challenge/t2_d*/**/spec*.json"))
    ap.add_argument("--sk-list", default="", help="json {spec path: [arms]} for the SK10 lambda 0.95 re-read")
    a = ap.parse_args()
    specs = sorted(set(glob.glob(a.specs, recursive=True)))
    sk = json.load(open(a.sk_list)) if a.sk_list else {}
    if sk:
        specs = [s for s in specs if s in sk]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.jobs, initializer=init, initargs=(a.export,)) as P:
        res = [r for rr in P.imap_unordered(screen, [(s, set(sk.get(s, []))) for s in specs]) for r in rr]
    df = pd.DataFrame(res)
    tag = "sk" if sk else "prior"
    df.to_csv(f"{a.out}/rescreen_{tag}.csv", index=False)
    ok = df[df.status == "ok"]
    print(f"{len(specs)} specs; {ok.spec.nunique()} read; {len(ok)} arm rows; skipped:", df[df.status != "ok"].status.str[:40].value_counts().to_dict())
    top = ok.sort_values("typical", ascending=False)
    pd.set_option("display.width", 250)
    cols = ["spec", "dec", "anchor", "arm", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]
    show = top[cols].copy()
    show["spec"] = show.spec.str.replace(os.path.expanduser("~/workspace/outputs/neural_interface_challenge/"), "", regex=False)
    print(show.head(40).to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.3f}"))


if __name__ == "__main__":
    main()

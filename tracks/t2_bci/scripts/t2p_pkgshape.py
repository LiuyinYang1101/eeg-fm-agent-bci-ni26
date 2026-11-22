#!/usr/bin/env python
"""Track 2: several package-shape arms vs an anchor, seed-paired, on the kit split AND the seen-user protocol (seen1), from
per-seed bench result files on disk (T2-D137, plan 3 T2 item 2). Zero compute.

A member = one bench result-file pattern per protocol (`{s}` = the seed) and a seed offset: index i reads seed offset + i;
optional `temp` T (T2-D244): its probabilities enter the arms as p^(1/T) renormalised (default 1 = unchanged).
Optional `arm_gate` (T2-D254): {arm: {"on": {m: w}, "tau": t, "low": {m: w}}}: a window whose `on` blend (renormalised) has top
probability < t takes the `low` weights instead of the arm's own (before the prior correction); absent = unchanged.
An arm = member weights summing to 1; p = sum_m w_m p_m, then the causal prior correction (1, 3) per stream on the mix, as
in the package; one seed per member per index (the t2p_paired --pkg shape). Groups (t2p_paired):
  kit    test = P04 + P09 (the leaderboard), val = P05 + P08
  seen1  seen = P04 + P10 sessions 2-3, p09 = P09, val = P05 + P08, unseen = P05 P08 P09
plus every participant. Derived (T2-D124c definitions): new = mean(kit test, seen1 val), mix = (seen + new) / 2.

  python t2p_pkgshape.py --export E --spec spec.json [--out res.json]

spec = {"n": 12, "anchor": "base", "bar": "d137",
        "members": {"fbc": {"offset": 12, "kit": ".../kit__fbcnet_bboth__none__n00__r1__s{s}.npz", "seen1": "..."}, ...},
        "arms": {"base": {"fbc": 0.6, "sh": 0.2, "stl": 0.2}, "a": {...}}}
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402

PROTOS = ("kit", "seen1")
BARS = {  # T2-D137 (plan 3): seen >= +0.010 with >= 8 / 12 up AND new >= -0.005 AND mix >= +0.005
    "d137": dict(seen=0.010, seen_up=8, new=-0.005, mix=0.005),
    # T2-D141 (plan 3 addendum, T2 item 5): mix >= +0.005 with >= 9 / 12 up AND kit test >= -0.005
    "d141": dict(mix=0.005, mix_up=9, kit_test=-0.005),
    # T2-D143 (addendum T2 item 6b, stricter after D142's look): mix >= +0.005 with >= 10 / 12 up AND kit test >= -0.003
    # AND new >= -0.003
    "d143": dict(mix=0.005, mix_up=10, kit_test=-0.003, new=-0.003),
}


def temper(P, T):
    """T2-D244: a member's temperature, p^(1/T) renormalised per row (= softmax(logit / T)); T = 1 returns P itself (off, bitwise)."""
    if T == 1.0:
        return P
    Q = P ** (1.0 / T)
    return Q / Q.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--protos", default="kit,seen1", help="kit only = a check against kit numbers on record")
    args = ap.parse_args()
    protos = tuple(args.protos.split(","))
    spec = json.loads(Path(args.spec).read_text())
    n, anchor = int(spec["n"]), spec["anchor"]
    mems, arms = spec["members"], spec["arms"]
    # T2-D186 (oracle read): an arm's causal prior may run per CELL (stream x the TRUE interface) instead of per stream;
    # not deployable as is (predict() sees no interface), so such an arm is an upper bound only. Default: per stream.
    prior_key = spec.get("arm_prior_key", {})
    assert set(prior_key) <= set(arms) and set(prior_key.values()) <= {"stream", "cell"}, prior_key
    prior_par = spec.get("arm_prior", {})        # T2-D212: {arm: [alpha, n0]} (default the package's 1, 3); [0, 0] = no correction
    assert set(prior_par) <= set(arms), prior_par
    for a, w in arms.items():
        assert abs(sum(w.values()) - 1) < 1e-9 and set(w) <= set(mems), (a, w)
    gate = spec.get("arm_gate", {})               # T2-D254: the CNN-confidence gate
    assert set(gate) <= set(arms), gate
    for a, g in gate.items():
        assert abs(sum(g["low"].values()) - 1) < 1e-9 and set(g["low"]) | set(g["on"]) <= set(mems), (a, g)
    need = set().union(*arms.values(), *(set(g["low"]) | set(g["on"]) for g in gate.values()))
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj, ses = idx["subject"].to_numpy(), idx["session"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()

    rows, used = [], {}
    for proto in protos:
        te = va = None
        for i in range(n):
            P = {}
            for m, d in mems.items():
                if m not in need:
                    continue
                f = d[proto].format(s=int(d.get("offset", 0)) + i)
                z = np.load(f)
                used.setdefault(m, {}).setdefault(proto, []).append(f)
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), f"{f}: windows differ"
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = temper(Pm, float(d.get("temp", 1.0)))
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
            tm[te], vm[va] = True, True
            if proto == "seen1":
                masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), p09=tm & (subj == "P09"), val=vm,
                             unseen=vm | (tm & (subj == "P09")))
                assert set(ses[masks["seen"]]) <= {"2", "3"}
            else:
                masks = dict(test=tm, val=vm)
            ok = tm | vm
            for a, w in arms.items():
                mix = sum(wm * P[m] for m, wm in w.items())
                r = dict(proto=proto, i=i, arm=a)
                if a in gate:
                    g = gate[a]
                    q = sum(wm * P[m] for m, wm in g["on"].items()) / sum(g["on"].values())
                    low = ok & (np.where(ok, q.max(1), np.inf) < float(g["tau"]))
                    mix = np.where(low[:, None], sum(wm * P[m] for m, wm in g["low"].items()), mix)
                    r["gated"] = float(low[ok].mean())
                assert not np.isnan(mix[ok]).any(), (proto, i, a)
                Pc = np.zeros_like(mix)
                key = cells if prior_key.get(a, "stream") == "cell" else strm
                al_, n0_ = prior_par.get(a, (1.0, 3.0))
                Pc[ok] = causal_prior_correction(mix[ok], (key[ok], ordr[ok]), float(al_), float(n0_)) if al_ else mix[ok]
                pr = Pc.argmax(1)
                for g, mk in masks.items():
                    r[g] = metrics.cell_balanced_accuracy(y[mk], pr[mk], cells[mk])[0]
                for p_ in np.unique(subj[ok]):
                    mk = ok & (subj == p_)
                    r[p_] = metrics.cell_balanced_accuracy(y[mk], pr[mk], cells[mk])[0]
                rows.append(r)
    df = pd.DataFrame(rows)

    def wide(proto, col):
        d = df[df.proto == proto].pivot_table(index="i", columns="arm", values=col)
        return d

    # per index: the derived groups
    kt, kv = wide("kit", "test"), wide("kit", "val")
    if "seen1" not in protos:                                   # a kit-only check: report the kit levels and stop
        lv = {a: dict(kit_test=float(kt[a].mean()), kit_val=float(kv[a].mean())) for a in arms}
        print(json.dumps(lv))
        if args.out:
            Path(args.out).write_text(json.dumps(dict(spec=spec, files=used, levels=lv), indent=1))
        return
    ss, sp, sv, su = wide("seen1", "seen"), wide("seen1", "p09"), wide("seen1", "val"), wide("seen1", "unseen")
    out = dict(spec=spec, files=used, arms={})
    bar = BARS.get(spec.get("bar", ""))
    print(f"anchor {anchor}: kit test {kt[anchor].mean():.4f} val {kv[anchor].mean():.4f} | seen1 seen {ss[anchor].mean():.4f} "
          f"p09 {sp[anchor].mean():.4f} val {sv[anchor].mean():.4f} ({n} indices)")
    for a in arms:
        if a == anchor:
            continue
        D = dict(kit_test=kt[a] - kt[anchor], kit_val=kv[a] - kv[anchor], seen=ss[a] - ss[anchor], p09=sp[a] - sp[anchor],
                 seen1_val=sv[a] - sv[anchor], unseen=su[a] - su[anchor])
        D["new"] = (D["kit_test"] + D["seen1_val"]) / 2
        D["mix"] = (D["seen"] + D["new"]) / 2
        r = {k: dict(d=float(v.mean()), up=int((v > 0).sum()), down=int((v < 0).sum()), sd=float(v.std(ddof=1))) for k, v in D.items()}
        parts = {}
        for proto in protos:
            sub = df[df.proto == proto]
            for p_ in sorted(c for c in sub.columns if c.startswith("P") and sub[c].notna().any()):
                w_ = sub.pivot_table(index="i", columns="arm", values=p_)
                parts[f"{proto}:{p_}"] = float((w_[a] - w_[anchor]).mean())
        r["participants"] = parts
        r["level"] = dict(kit_test=float(kt[a].mean()), kit_val=float(kv[a].mean()), seen=float(ss[a].mean()), seen1_val=float(sv[a].mean()))
        if bar:
            r["bar"] = {k: r[k]["d"] >= v and r[k]["up"] >= bar.get(f"{k}_up", 0) for k, v in bar.items() if not k.endswith("_up")}
            r["met"] = all(r["bar"].values())
        out["arms"][a] = r
        print(f"\n{a} vs {anchor}  {arms[a]}")
        for k in ("seen", "new", "mix", "kit_test", "kit_val", "seen1_val", "p09", "unseen"):
            print(f"  {k:9s} {r[k]['d']:+.4f}  ({r[k]['up']} up / {r[k]['down']} down, sd {r[k]['sd']:.4f})")
        print("  per participant: " + ", ".join(f"{k} {v:+.4f}" for k, v in parts.items()))
        if bar:
            print(f"  bar {spec['bar']}: " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in r["bar"].items())
                  + f" -> {'MET' if r['met'] else 'not met'}")
    if args.out:
        out["rows"] = rows
        Path(args.out).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

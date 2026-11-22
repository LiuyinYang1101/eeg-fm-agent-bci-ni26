#!/usr/bin/env python
"""T2-D290 (zero GPU; pre-registered in docs/DECISIONS_T2.md): would a different stream structure on the worker reproduce T2-W7's pattern
(the prior path unchanged, SK10 -0.0055 vs the replay)? T2-D289's inputs unperturbed (T2-D253's spec, 12 indices, w4321), the kit protocol's
scored windows; SK1 (= the prior) and SK10 decoded under
  R0  the export's order, one stream per session (= the replay)
  RP  each session's recordings (`timeline`) in a random order, each in time order (20 draws)
  RR  the recordings reversed
  RJ  one stream per recording (stream_id = record_id, the kit's default)
  RN  no reset: one stream over all scored windows in the export's order
Rule (fixed before any number): a scenario REPRODUCES THE WORKER iff SK1's mean kit-test delta vs R0 is within +-0.0010 AND SK10's <= -0.0030.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d290_order.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import metrics  # noqa: E402
from t2_d289_bf16 import sk_batch, W, OFFS, O, E  # noqa: E402

N, NDRAW, RNG = 12, 20, 20261028
ARMS = {"SK1": 1, "SK10": 10}


def decode(mix, streams, it):
    pred = np.full(len(mix), -1)
    for r_ in streams:
        pred[r_] = sk_batch(mix[r_][None], it)[0].argmax(1)
    return pred


def main():
    s = json.load(open(O / "t2_d253" / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    rec = idx["timeline"].to_numpy()
    ordr = idx["order"].to_numpy()
    rows = []
    for i in range(N):
        P, te = {}, None
        for m, d in mem.items():
            z = np.load(d["kit"].format(s=OFFS[m] + i))
            if te is None:
                te, va = z["test_idx"], z["val_idx"]
            Pm = np.full((len(idx), 3), np.nan)
            Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
            P[m] = Pm
        mix = sum(W[m] * P[m] for m in W)
        ok = np.zeros(len(idx), bool)
        ok[te] = ok[va] = True
        tm = np.zeros(len(idx), bool)
        tm[te] = True
        okw = np.flatnonzero(ok)
        okw = okw[np.lexsort((ordr[okw], strm[okw]))]
        sess = {}
        for st in dict.fromkeys(strm[okw]):
            r_ = okw[strm[okw] == st]
            r_ = r_[np.argsort(ordr[r_], kind="stable")]
            recs = list(dict.fromkeys(rec[r_]))                      # recordings in the export's order
            sess[st] = [r_[rec[r_] == rc] for rc in recs]
        rng = np.random.default_rng([RNG, i])
        scen = {"R0": [[np.concatenate(v)] for v in sess.values()],
                "RR": [[np.concatenate(v[::-1])] for v in sess.values()],
                "RJ": [list(v) for v in sess.values()],
                "RN": [[np.concatenate([np.concatenate(v) for v in sess.values()])]]}
        for dr in range(NDRAW):
            scen[f"RP{dr}"] = [[np.concatenate([v[k] for k in rng.permutation(len(v))])] for v in sess.values()]
        base = {}
        for name, groups in scen.items():
            streams = [g for grp in groups for g in grp]
            assert np.array_equal(np.sort(np.concatenate(streams)), np.sort(okw)), name
            for k, it in ARMS.items():
                pr = decode(mix, streams, it)
                ba = metrics.cell_balanced_accuracy(y[tm], pr[tm], cells[tm])[0]
                if name == "R0":
                    base[k] = (pr, ba)
                rows.append(dict(i=i, scen="RP" if name.startswith("RP") else name, draw=name, arm=k, kit_test=ba,
                                 d=ba - base[k][1], flips=int((pr[tm] != base[k][0][tm]).sum())))
        print(f"i={i}: done", flush=True)
    df = pd.DataFrame(rows)
    g = df.groupby(["scen", "arm"]).agg(d=("d", "mean"), flips=("flips", "mean"), kit_test=("kit_test", "mean")).unstack("arm")
    res = {}
    print(f"kit test: {int(tm.sum())} windows; stream structures vs R0 (mean over {N} indices; RP also over {NDRAW} draws)")
    for sc in ("R0", "RP", "RR", "RJ", "RN"):
        d1, d10 = g.loc[sc, ("d", "SK1")], g.loc[sc, ("d", "SK10")]
        rep = bool(abs(d1) <= 0.0010 and d10 <= -0.0030)
        gain = g.loc[sc, ("kit_test", "SK10")] - g.loc[sc, ("kit_test", "SK1")]
        res[sc] = dict(d_sk1=float(d1), d_sk10=float(d10), flips_sk1=float(g.loc[sc, ("flips", "SK1")]),
                       flips_sk10=float(g.loc[sc, ("flips", "SK10")]), gain_kit_test=float(gain), reproduces=rep)
        print(f"{sc}: SK1 d {d1:+.4f} (flips {res[sc]['flips_sk1']:.1f}), SK10 d {d10:+.4f} (flips {res[sc]['flips_sk10']:.1f}); "
              f"SK10 - SK1 kit test {gain:+.4f}; reproduces the worker: {rep}")
    reps = [k for k, v in res.items() if v["reproduces"]]
    print("verdict: " + (f"REPRODUCED by {', '.join(reps)}" if reps else "no stream structure reproduces the worker"))
    out = O / "t2_d290"
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(dict(scenarios=res, reproduced_by=reps), indent=1))
    df.to_csv(out / "rows.csv", index=False)


if __name__ == "__main__":
    main()

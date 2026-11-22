#!/usr/bin/env python
"""T2-D304 (zero GPU; pre-registered in docs/DECISIONS_T2.md): SKλ, a forgetting causal Sinkhorn. T2-D253's members at index i (w4321), kit + seen1,
test + val windows; each session (subject | session) decoded causally. SKλ = T2-D289's sk_batch with row j of window t's matrix (3 pseudo-rows + p_1..p_t)
weighted λ^(t+3-j) in the column balance (λ 1.0 = sk_batch bitwise, asserted). Orders: R0 (the export's order) and RP (each session's recordings in a
random order, 20 draws, as T2-D290). Metrics = T2-D285's (new = (kit test + seen1 val) / 2, mix = (seen + new) / 2).
Bar: a λ < 1 passes iff mix(SKλ) − mix(SK10) >= +0.003 with >= 9 / 12 up under R0 AND the mean RP Δ >= 0.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d304_skl.py
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

N, NDRAW, RNG, ITERS = 12, 20, 20261030, 10
LAMS = (0.90, 0.95, 0.98, 0.99, 1.0)


def sk_lambda(p, iters, lam):
    """sk_batch with forgetting: p (D, T, 3) -> (D, T, 3); window t weighs row j (0-2 pseudo, 3.. = p_1..) by lam^(t + 3 - j)."""
    D, T, K = p.shape
    rows = np.concatenate([np.full((D, 3, K), 1.0 / K), p], 1)
    age = (np.arange(T)[:, None] + 3) - np.arange(T + 3)[None, :]            # (T, T+3)
    w = np.where(age >= 0, lam ** np.maximum(age, 0), 0.0)
    nw = w.sum(1)[None, :, None, None]
    M = rows[:, None] * (age >= 0)[None, :, :, None]
    for _ in range(iters):
        M = M * (nw / K) / np.maximum((M * w[None, :, :, None]).sum(2, keepdims=True), 1e-300)
        rs = M.sum(3, keepdims=True)
        M = M / np.where(rs > 0, rs, 1.0)
    return M[:, np.arange(T), np.arange(T) + 3]


def main():
    global LAMS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="t2_d253")
    ap.add_argument("--offsets", default="fbc_rc75=204,sh_vr=168,stl_rcll=0,mor=0")
    ap.add_argument("--lams", default=",".join(str(x) for x in LAMS), help="T2-D304r: 0.95,1.0 (the candidate only)")
    ap.add_argument("--out", default="t2_d304")
    a = ap.parse_args()
    LAMS = tuple(float(x) for x in a.lams.split(","))
    assert LAMS[-1] == 1.0
    OFFS.clear(); OFFS.update({k: int(v) for k, v in (x.split("=") for x in a.offsets.split(","))})
    s = json.load(open(O / a.spec / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    rec, ordr = idx["timeline"].to_numpy(), idx["order"].to_numpy()
    chk = sk_batch(np.random.default_rng(0).dirichlet(np.ones(3), (2, 40)), ITERS)
    assert np.array_equal(chk, sk_lambda(np.random.default_rng(0).dirichlet(np.ones(3), (2, 40)), ITERS, 1.0)), "λ 1.0 must be sk_batch bitwise"
    rows = []
    for pi, proto in enumerate(("kit", "seen1")):
        for i in range(N):
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=OFFS[m] + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            mix = sum(W[m] * P[m] for m in W)
            ok = np.zeros(len(idx), bool); ok[te] = ok[va] = True
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
            if proto == "seen1":
                masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm)
            else:
                masks = dict(test=tm, val=vm)
            rng = np.random.default_rng([RNG, pi, i])
            sess = []
            for st in dict.fromkeys(strm[ok]):
                r_ = np.flatnonzero(ok & (strm == st)); r_ = r_[np.argsort(ordr[r_], kind="stable")]
                assert (strm == st).sum() == len(r_)
                sess.append([r_[rec[r_] == rc] for rc in dict.fromkeys(rec[r_])])
            orders = {"R0": [[np.concatenate(v) for v in sess]]}
            orders["RP"] = [[np.concatenate([v[k] for k in rng.permutation(len(v))]) for v in sess] for _ in range(NDRAW)]
            for oname, draws in orders.items():
                for lam in LAMS:
                    sc = {g: [] for g in masks}
                    preds = [np.full(len(idx), -1) for _ in draws]
                    for si in range(len(sess)):
                        seqs = np.stack([dr[si] for dr in draws])                     # (D, T) window indices
                        out = sk_lambda(mix[seqs], ITERS, lam).argmax(-1)              # (D, T)
                        for d_ in range(len(draws)):
                            preds[d_][seqs[d_]] = out[d_]
                    for pr in preds:
                        for g, mk in masks.items():
                            sc[g].append(metrics.cell_balanced_accuracy(y[mk], pr[mk], cells[mk])[0])
                    rows.append(dict(proto=proto, i=i, order=oname, lam=lam, **{g: float(np.mean(v)) for g, v in sc.items()}))
            print(f"{proto} i={i}: done", flush=True)
    df = pd.DataFrame(rows)
    out = O / a.out; out.mkdir(exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    res = {}
    for oname in ("R0", "RP"):
        k_ = df[(df.proto == "kit") & (df.order == oname)].pivot_table(index="i", columns="lam", values="test")
        sv = df[(df.proto == "seen1") & (df.order == oname)].pivot_table(index="i", columns="lam", values="val")
        ss = df[(df.proto == "seen1") & (df.order == oname)].pivot_table(index="i", columns="lam", values="seen")
        for lam in LAMS[:-1]:
            dk, dv, ds = k_[lam] - k_[1.0], sv[lam] - sv[1.0], ss[lam] - ss[1.0]
            new = (dk + dv) / 2
            mx = (ds + new) / 2
            res[(oname, lam)] = dict(mix=float(mx.mean()), up=int((mx > 0).sum()), new=float(new.mean()), seen=float(ds.mean()), kit_test=float(dk.mean()))
    print("SKλ − SK10 (mean over 12 indices; RP also over 20 draws):")
    passing = []
    for lam in LAMS[:-1]:
        r0, rp = res[("R0", lam)], res[("RP", lam)]
        ok = r0["mix"] >= 0.003 and r0["up"] >= 9 and rp["mix"] >= 0
        if ok:
            passing.append((r0["mix"], lam))
        print(f"  λ {lam:.2f}: R0 mix {r0['mix']:+.4f} ({r0['up']} / 12), new {r0['new']:+.4f}, seen {r0['seen']:+.4f}, kit test {r0['kit_test']:+.4f} | "
              f"RP mix {rp['mix']:+.4f} (new {rp['new']:+.4f}, seen {rp['seen']:+.4f}) -> {'PASS' if ok else 'no'}")
    verdict = f"CANDIDATE λ {max(passing)[1]:.2f}" if passing else "NOT ADOPTED (no λ passes): the item closes"
    print("VERDICT:", verdict)
    (out / "result.json").write_text(json.dumps({f"{k[0]}_{k[1]}": v for k, v in res.items()} | {"verdict": verdict}, indent=1))


if __name__ == "__main__":
    main()

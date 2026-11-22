#!/usr/bin/env python
"""T2-D292 (zero GPU; pre-registered in docs/DECISIONS_T2.md): a per-stream adaptive LDA on log band power, fitted causally on
the stream's own past windows, multiplied into the package mix before SK10.
Inputs = T2-D285's (T2-D253's spec, index i: fbc_rc75 204+i, sh_vr 168+i, stl_rcll i, mor i; w4321; kit + seen1; replicate:
--spec t2_d247r --offsets fbc_rc75=192,sh_vr=156,stl_rcll=12,mor=0). Features = t2p_route.band5_log (log-variance per channel in
4-8 / 8-13 / 13-20 / 20-30 / 30-45 Hz, 41 ch x 5 = 205) of the export's X (120 Hz, the template's front end).
At window t of a stream (n = t past windows), once n >= 15: LDA on the past windows with label weights L (n, 3): class means
mu_c = weighted means, pooled within-class scatter S, shrunk toward its diagonal by lam = d / (d + n), uniform class prior ->
p_lda(t). Combined row r_t = p_t * p_lda(t)^b, b = n / (n + 60), renormalised; else r_t = p_t. The decoder = SK10 over r_1..r_t
(n0 3, = the template's SinkhornCorrector).
  B  = SK10 over the raw mix (the shipped decoder, anchor)
  OR = L = one-hot TRUE labels of the past windows (the ceiling; not shippable)
  PL = L = B's own SK10 outputs of the past windows (soft; label-free, causal: shippable)
Bar (PL, fixed before any number): T2-D285's: mix >= +0.005, >= 9 / 12 up, kit test >= -0.005, new >= 0, >= 3 / 4 kit participants
up -> the replicate (adopt iff its mix > 0 and new >= 0). Kill (the line): OR mix < +0.010.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d292_streamlda.py [--spec ... --offsets ... --out ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "local"))
sys.path.insert(0, str(ROOT / "tracks" / "t2_bci" / "scripts"))
from ni26 import metrics  # noqa: E402
import t2_d285_quota as D285  # noqa: E402
import t2p_route  # noqa: E402

O, E, W, N = D285.O, D285.E, D285.W, D285.N
MIN_N, HALF = 15, 60.0
ARMS = ("B", "OR", "PL")


def lda_proba(F, L, x):
    """F (n, d) past features, L (n, 3) label weights, x (d,) -> class probabilities (3,), or None if a class has < 1 mass."""
    n, d = F.shape
    wc = L.sum(0)
    if (wc < 1.0).any():
        return None
    mu = (L.T @ F) / wc[:, None]
    S = np.zeros((d, d))
    for c in range(3):
        R = F - mu[c]
        S += (R * L[:, c:c + 1]).T @ R
    S /= n
    lam = d / (d + n)
    S = (1 - lam) * S + lam * np.diag(np.diag(S))
    A = np.linalg.solve(S, mu.T)                               # (d, 3)
    g = x @ A - 0.5 * np.einsum("cd,dc->c", mu, A)
    p = np.exp(g - g.max())
    return p / p.sum()


def combine(p, F, L_all):
    """r_t = p_t * p_lda(t)^b (renormalised) with the LDA fitted on rows < t of L_all; L_all None -> r = p."""
    r = p.copy()
    if L_all is None:
        return r
    for t in range(MIN_N, len(p)):
        pl = lda_proba(F[:t], L_all[:t], F[t])
        if pl is None:
            continue
        b = t / (t + HALF)
        v = np.log(np.maximum(p[t], 1e-12)) + b * np.log(np.maximum(pl, 1e-12))
        v = np.exp(v - v.max())
        r[t] = v / v.sum()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="t2_d253")
    ap.add_argument("--offsets", default="fbc_rc75=204,sh_vr=168,stl_rcll=0,mor=0")
    ap.add_argument("--out", default="t2_d292")
    ap.add_argument("--quick", action="store_true", help="indices 0-1 only (a smoke test, no verdict)")
    a = ap.parse_args()
    offs = {k: int(v) for k, v in (x.split("=") for x in a.offsets.split(","))}
    out = O / a.out
    out.mkdir(exist_ok=True)
    s = json.load(open(O / a.spec / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    assert {m: d["offset"] for m, d in mem.items()} == offs, ({m: d["offset"] for m, d in mem.items()}, offs)
    meta = json.load(open(E / "meta.json"))
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    X = np.load(E / "X.npy", mmap_mode="r")
    feats = np.concatenate([t2p_route.band5_log(np.asarray(X[k:k + 256], np.float64), float(meta["sfreq"]))
                            for k in range(0, len(X), 256)])
    assert feats.shape == (len(idx), 5 * int(meta["n_chans"])) and np.isfinite(feats).all(), feats.shape
    n_idx = 2 if a.quick else N
    rows, or_cache = [], {}
    for proto in ("kit", "seen1"):
        for i in range(n_idx):
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=offs[m] + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (m, proto, i)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            mix = sum(W[m] * P[m] for m in W)
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            pred = {k: np.full(len(idx), -1) for k in ARMS}
            for st in np.unique(strm[ok]):
                r_ = np.flatnonzero(ok & (strm == st))
                r_ = r_[np.argsort(ordr[r_], kind="stable")]
                assert (strm == st).sum() == len(r_), f"{st}: a scored stream is not whole"
                p, F = mix[r_], feats[r_]
                qB = D285.sinkhorn(p, 10)
                key = (proto, st)
                if key not in or_cache:                                  # OR's LDA term does not depend on the index
                    or_cache[key] = combine(np.full_like(p, 1 / 3), F, np.eye(3)[y[r_]])
                rO = or_cache[key] * p
                rO = rO / rO.sum(1, keepdims=True)
                D = {"B": qB, "OR": D285.sinkhorn(rO, 10), "PL": D285.sinkhorn(combine(p, F, qB), 10)}
                for k_, v in D.items():
                    pred[k_][r_] = v.argmax(1)
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
            tm[te], vm[va] = True, True
            if proto == "seen1":
                masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), p09=tm & (subj == "P09"), val=vm)
            else:
                masks = dict(test=tm, val=vm)
            for k_ in ARMS:
                r = dict(proto=proto, i=i, arm=k_)
                for g, mk in masks.items():
                    r[g] = metrics.cell_balanced_accuracy(y[mk], pred[k_][mk], cells[mk])[0]
                for u in np.unique(subj[ok]):
                    mk = ok & (subj == u)
                    r[u] = metrics.cell_balanced_accuracy(y[mk], pred[k_][mk], cells[mk])[0]
                rows.append(r)
            print(f"{proto} i={i} done", flush=True)
    df = pd.DataFrame(rows)

    def wide(proto, col):
        return df[df.proto == proto].pivot_table(index="i", columns="arm", values=col)
    kt, kv, ss, sv = wide("kit", "test"), wide("kit", "val"), wide("seen1", "seen"), wide("seen1", "val")
    res = {}
    print(f"B levels: kit test {kt['B'].mean():.4f}, kit val {kv['B'].mean():.4f}, seen1 seen {ss['B'].mean():.4f}, seen1 val {sv['B'].mean():.4f}")
    for k_ in ("OR", "PL"):
        D = dict(kit_test=kt[k_] - kt["B"], kit_val=kv[k_] - kv["B"], seen=ss[k_] - ss["B"], seen1_val=sv[k_] - sv["B"])
        D["new"] = (D["kit_test"] + D["seen1_val"]) / 2
        D["mix"] = (D["seen"] + D["new"]) / 2
        r = {g: dict(d=float(v.mean()), up=int((v > 0).sum()), down=int((v < 0).sum())) for g, v in D.items()}
        kp = {u: float((wide("kit", u)[k_] - wide("kit", u)["B"]).mean()) for u in ("P04", "P05", "P08", "P09")}
        r["kit_parts"] = kp
        r["bar"] = (r["mix"]["d"] >= 0.005 and r["mix"]["up"] >= 9 and r["kit_test"]["d"] >= -0.005 and r["new"]["d"] >= 0
                    and sum(v > 0 for v in kp.values()) >= 3)
        res[k_] = r
        print(f"{k_:3s} vs B: mix {r['mix']['d']:+.4f} ({r['mix']['up']} / {n_idx} up), kit test {r['kit_test']['d']:+.4f}, kit val "
              f"{r['kit_val']['d']:+.4f}, seen {r['seen']['d']:+.4f}, seen1 val {r['seen1_val']['d']:+.4f}, new {r['new']['d']:+.4f}; kit "
              "participants " + ", ".join(f"{u} {v:+.4f}" for u, v in kp.items()) + (f"; bar {'MET' if r['bar'] else 'not met'}" if k_ == "PL" else ""))
    if a.quick:
        print("QUICK (indices 0-1): no verdict")
        return
    if res["OR"]["mix"]["d"] < 0.010:
        verdict = f"KILLED (the line): OR ceiling mix {res['OR']['mix']['d']:+.4f} < +0.010"
    elif res["PL"]["bar"]:
        verdict = "CANDIDATE PL -> the replicate (adopt iff its mix > 0 and new >= 0)"
    else:
        verdict = "NOT ADOPTED (PL misses the bar)"
    print("VERDICT:", verdict)
    (out / "read.json").write_text(json.dumps(dict(arms=res, verdict=verdict, spec=a.spec, offsets=offs, min_n=MIN_N, half=HALF,
                                                   b_levels=dict(kit_test=float(kt["B"].mean()), kit_val=float(kv["B"].mean()),
                                                                 seen=float(ss["B"].mean()), seen1_val=float(sv["B"].mean())),
                                                   rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()

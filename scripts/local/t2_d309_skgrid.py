#!/usr/bin/env python
"""T2-D309 (zero GPU; pre-registered in docs/DECISIONS_T2.md): (a) an SK knob grid with an honest holdout and (b) an interface-aware oracle.
Members as T2-D304 (w4321 mix of the spec's four members at index i), kit + seen1, test + val windows; each stream decoded causally in `order`.
Decoder = SKλ with n0 uniform pseudo-rows and `iters` rounds (T2-D304's sk_lambda generalised to n0; λ 1 = SK10's math).
  --mode grid      : λ ∈ {0.90, 0.93, 0.95, 0.97} × n0 ∈ {1, 3, 10} × iters ∈ {5, 10, 20}, streams = sessions, order R0 → mix etc. vs the shipped
                     λ 0.95 / n0 3 / iters 10 (on the selection seeds, T2-D253's spec); --configs restricts it (the holdout runs one config + the reference)
  --mode interface : λ ∈ {1.0, 0.95} (n0 3, iters 10), streams = subject|session|interface (ORACLE interface labels) vs streams = sessions
Metrics = T2-D285's (new = (kit test + seen1 val) / 2, mix = (seen + new) / 2).
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d309_skgrid.py --mode grid [--spec t2_d247r --offsets ... --configs ...]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import metrics  # noqa: E402
from t2_d289_bf16 import sk_batch, W, O, E  # noqa: E402

N = 12
REF = (0.95, 3, 10)


def sk(p, iters, lam, n0):
    """causal SKλ for one stream p (T, 3) with n0 pseudo-rows; window t weighs row j by lam^(t + n0 - j)."""
    T, K = p.shape
    rows = np.concatenate([np.full((n0, K), 1.0 / K), p], 0)
    age = (np.arange(T)[:, None] + n0) - np.arange(T + n0)[None, :]
    w = np.where(age >= 0, lam ** np.maximum(age, 0), 0.0)
    nw = w.sum(1)[:, None, None]
    M = rows[None] * (age >= 0)[:, :, None]
    for _ in range(iters):
        M = M * (nw / K) / np.maximum((M * w[:, :, None]).sum(1, keepdims=True), 1e-300)
        M = M / np.maximum(M.sum(2, keepdims=True), 1e-300)
    return M[np.arange(T), np.arange(T) + n0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["grid", "interface"], required=True)
    ap.add_argument("--spec", default="t2_d253")
    ap.add_argument("--offsets", default="fbc_rc75=204,sh_vr=168,stl_rcll=0,mor=0")
    ap.add_argument("--configs", default="", help="lam:n0:iters,... (default: the full grid + the reference)")
    ap.add_argument("--out", default="t2_d309")
    a = ap.parse_args()
    offs = {k: int(v) for k, v in (x.split("=") for x in a.offsets.split(","))}
    chk = np.random.default_rng(0).dirichlet(np.ones(3), 40)
    assert np.allclose(sk(chk, 10, 1.0, 3), sk_batch(chk[None], 10)[0], atol=1e-12), "λ 1, n0 3 must be SK10"
    if a.mode == "grid":
        cfgs = [tuple(float(c) if i == 0 else int(c) for i, c in enumerate(x.split(":"))) for x in a.configs.split(",")] if a.configs else \
            list(itertools.product((0.90, 0.93, 0.95, 0.97), (1, 3, 10), (5, 10, 20)))
        if REF not in cfgs:
            cfgs.append(REF)
        groupings = ["session"]
    else:
        cfgs, groupings = [(1.0, 3, 10), (0.95, 3, 10)], ["session", "interface"]
    s = json.load(open(O / a.spec / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    keys = {"session": (idx["subject"] + "|" + idx["session"]).to_numpy(), "interface": cells}
    ordr = idx["order"].to_numpy()
    rows = []
    for proto in ("kit", "seen1"):
        for i in range(N):
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=offs[m] + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            mix = sum(W[m] * P[m] for m in W)
            ok = np.zeros(len(idx), bool); ok[te] = ok[va] = True
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool); tm[te] = True; vm[va] = True
            masks = dict(seen=tm & np.isin(subj, ["P04", "P10"]), val=vm) if proto == "seen1" else dict(test=tm, val=vm)
            for g in groupings:
                streams = []
                for st in dict.fromkeys(keys[g][ok]):
                    r_ = np.flatnonzero(ok & (keys[g] == st))
                    streams.append(r_[np.argsort(ordr[r_], kind="stable")])
                for c in cfgs:
                    pred = np.full(len(idx), -1)
                    for r_ in streams:
                        pred[r_] = sk(mix[r_], c[2], c[0], c[1]).argmax(1)
                    rows.append(dict(proto=proto, i=i, group=g, cfg=f"{c[0]:.2f}:{c[1]}:{c[2]}",
                                     **{k: metrics.cell_balanced_accuracy(y[mk], pred[mk], cells[mk])[0] for k, mk in masks.items()}))
            print(f"{proto} i={i}: done", flush=True)
    df = pd.DataFrame(rows)
    out = O / a.out; out.mkdir(exist_ok=True)
    df.to_csv(out / f"rows_{a.mode}.csv", index=False)

    def mixv(g, cfg):
        k_ = df[(df.proto == "kit") & (df.group == g) & (df.cfg == cfg)].set_index("i")["test"]
        sv = df[(df.proto == "seen1") & (df.group == g) & (df.cfg == cfg)].set_index("i")["val"]
        ss = df[(df.proto == "seen1") & (df.group == g) & (df.cfg == cfg)].set_index("i")["seen"]
        new = (k_ + sv) / 2
        return (ss + new) / 2, new, ss, k_
    res = {}
    if a.mode == "grid":
        ref = f"{REF[0]:.2f}:{REF[1]}:{REF[2]}"
        r0, n0_, s0, k0 = mixv("session", ref)
        for c in sorted(set(df.cfg)):
            m_, n_, s_, k_ = mixv("session", c)
            d = m_ - r0
            res[c] = dict(mix=float(d.mean()), up=int((d > 0).sum()), new=float((n_ - n0_).mean()), seen=float((s_ - s0).mean()), kit_test=float((k_ - k0).mean()))
        best = max((c for c in res if c != ref), key=lambda c: res[c]["mix"])
        print(f"vs the shipped {ref} (mean over {N} indices, R0):")
        for c in sorted(res, key=lambda c: -res[c]["mix"])[:8]:
            print(f"  {c}: mix {res[c]['mix']:+.4f} ({res[c]['up']} / {N}), new {res[c]['new']:+.4f}, seen {res[c]['seen']:+.4f}, kit test {res[c]['kit_test']:+.4f}")
        print("BEST (selection):", best)
    else:
        for c in sorted(set(df.cfg)):
            ms, ns, ss_, _ = mixv("session", c)
            mi, ni, si, _ = mixv("interface", c)
            d = mi - ms
            res[c] = dict(mix=float(d.mean()), up=int((d > 0).sum()), new=float((ni - ns).mean()), seen=float((si - ss_).mean()))
            print(f"ORACLE interface streams − session streams at {c}: mix {res[c]['mix']:+.4f} ({res[c]['up']} / {N}), new {res[c]['new']:+.4f}, seen {res[c]['seen']:+.4f}")
    (out / f"result_{a.mode}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

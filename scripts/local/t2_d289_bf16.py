#!/usr/bin/env python
"""T2-D289 (zero GPU; pre-registered in docs/DECISIONS_T2.md): is SK10's replay-vs-worker gap (T2-W7 0.57445 on the worker vs 0.57993 in the
replay, while the prior path of T2-W6 reproduced exactly) bf16-scale noise that SK10 amplifies?
T2-D285's inputs (T2-D253's spec, index i: fbc_rc75 204+i, sh_vr 168+i, stl_rcll i, mor i; w4321; kit + seen1). One draw: every member's
probability row x exp(eps), eps ~ N(0, (2^-8)^2) i.i.d. per window x class x member, rows renormalised, then w4321 -> mix -> per-stream causal
decode. 20 draws per (protocol, i), paired across arms. Pass "all" (primary) perturbs the four members, pass "fm" (descriptive) only ST-L + MOR.
Arms SK1 (= the shipped prior), SK3, SK5, SK10 (SK3 / SK5 read only if the trigger fires).
  (a) flips: windows whose perturbed argmax differs from the same arm's unperturbed argmax (kit test = the warm-up size; all scored windows)
  (b) spread: sd over draws of the perturbed - unperturbed cell BA (kit test direct; sealed size by T2-D280's cell model, U 10 / C 20, C 240)
  (c) gain: SK10 - SK1 package-shape mix per index, averaged over draws; mean over the 12 indices, indices up
Rule (fixed before any number): ">>" = SK10's mean kit-test flips >= 3 x SK1's and >= 1. SHIPS AS PLANNED iff >> AND the perturbed gain >= +0.005;
gain < +0.003 -> SK3 / SK5 under the same test to the user before Nov 5; otherwise reported to the user (SK10 stays the plan unless the user
decides).
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d289_bf16.py [--draws 20] [--quick]
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
from ni26 import metrics  # noqa: E402
import t2_d285_quota as D285  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}
OFFS = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
FM = ("stl_rcll", "mor")
ARMS = {"SK1": 1, "SK3": 3, "SK5": 5, "SK10": 10}
SIG, RNG, N = 2.0 ** -8, 20261027, 12
SCEN = [(10, 20), (10, 240)]
GAP = -0.0055                                                       # T2-W7 worker - replay (SK10); T2-W6's prior path: 0


def sk_batch(p, iters):
    """causal Sinkhorn for D draws of one stream at once: p (D, T, 3) -> (D, T, 3); row t = D285.sinkhorn(p[d], iters)[t]."""
    D, T, K = p.shape
    rows = np.concatenate([np.full((D, 3, K), 1.0 / K), p], 1)
    mask = np.arange(T + 3)[None, :] <= np.arange(T)[:, None] + 3   # (T, T+3): window t sees 3 pseudo-rows + p_1..p_t
    n = mask.sum(1).astype(float)[None, :, None, None]
    M = rows[:, None] * mask[None, :, :, None]
    for _ in range(iters):
        M = M * (n / K) / np.maximum(M.sum(2, keepdims=True), 1e-300)
        rs = M.sum(3, keepdims=True)
        M = M / np.where(rs > 0, rs, 1.0)
    return M[:, np.arange(T), np.arange(T) + 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="index 0 only, 3 draws (a smoke; no verdict)")
    a = ap.parse_args()
    nd, idxs = (3, [0]) if a.quick else (a.draws, list(range(N)))
    out = O / "t2_d289"
    out.mkdir(exist_ok=True)
    s = json.load(open(O / "t2_d253" / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    assert {m: d["offset"] for m, d in mem.items()} == OFFS
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj = idx["subject"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    recs, cellD, chk, n_kt, max_T = [], {}, [], 0, 0
    for pid, proto in enumerate(("kit", "seen1")):
        for i in idxs:
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=OFFS[m] + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (m, proto, i)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            tm, vm = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
            tm[te], vm[va] = True, True
            masks = (dict(test=tm, val=vm) if proto == "kit"
                     else dict(seen=tm & np.isin(subj, ["P04", "P10"]), p09=tm & (subj == "P09"), val=vm))
            fmask = tm if proto == "kit" else tm & np.isin(subj, ["P04", "P10"])
            okw = np.flatnonzero(ok)
            if proto == "kit":
                n_kt = int(tm.sum())
            rng =np.random.default_rng([RNG, pid, i])
            eps = {m: rng.normal(0.0, SIG, size=(nd, len(okw), 3)) for m in W}        # the same eps for both passes (paired)
            streams = []
            for st in np.unique(strm[ok]):
                r_ = np.flatnonzero(ok & (strm == st))
                streams.append(r_[np.argsort(ordr[r_], kind="stable")])
            max_T = max(max_T, max(len(r_) for r_ in streams))
            pos =np.full(len(idx), -1)
            pos[okw] = np.arange(len(okw))
            for pas in ("all", "fm"):
                mix = np.zeros((nd + 1, len(okw), 3))
                for m in W:
                    base = P[m][okw][None].repeat(nd + 1, 0)
                    if pas == "all" or m in FM:
                        base[1:] = base[1:] * np.exp(eps[m])
                        base[1:] /= base[1:].sum(2, keepdims=True)
                    mix += W[m] * base
                pred = {k: np.full((nd + 1, len(idx)), -1) for k in ARMS}
                for r_ in streams:
                    pp = mix[:, pos[r_]]
                    for k, it in ARMS.items():
                        q = sk_batch(pp, it)
                        pred[k][:, r_] = q.argmax(2)
                        if pas == "all" and k in ("SK1", "SK10") and len(chk) < 200:
                            ref = D285.prior(pp[0]) if k == "SK1" else D285.sinkhorn(pp[0], it)
                            chk.append(float(np.abs(q[0] - ref).max()))
                for k in ARMS:
                    for dr in range(nd + 1):
                        r = dict(proto=proto, i=i, pas=pas, arm=k, draw=dr - 1)
                        for g, mk in masks.items():
                            ba, per = metrics.cell_balanced_accuracy(y[mk], pred[k][dr][mk], cells[mk])
                            r[g] = ba
                            for c, v in per.items():
                                cellD[(proto, i, pas, k, dr - 1, c)] = v
                        if dr > 0:
                            r["flips_warm"] = int((pred[k][dr][fmask] != pred[k][0][fmask]).sum())
                            r["flips_all"] = int((pred[k][dr][ok] != pred[k][0][ok]).sum())
                        recs.append(r)
            print(f"{proto} i={i}: done", flush=True)
    print(f"check: batched decode vs T2-D285's prior / sinkhorn (unperturbed), max abs diff {max(chk):.2e} over {len(chk)} streams")
    assert max(chk) < 1e-9, max(chk)
    df = pd.DataFrame(recs)
    res = dict(sigma=SIG, draws=nd, indices=idxs, check_max_abs=max(chk), n_kit_test=int(n_kt), max_stream=int(max_T), passes={})
    print(f"kit test (the warm-up size): {n_kt} windows; longest stream {max_T} windows")
    for pas in ("all", "fm"):
        d = df[df.pas == pas]
        u = d[d.draw < 0].set_index(["proto", "i", "arm"])
        pr = d[d.draw >= 0].copy()
        for g in ("test", "val", "seen", "p09"):
            if g in pr:
                pr["d_" + g] = pr.apply(lambda x: x[g] - u.loc[(x.proto, x.i, x.arm), g] if pd.notna(x[g]) else np.nan, axis=1)
        R = {}
        for k in ARMS:
            kp = pr[(pr.arm == k) & (pr.proto == "kit")]
            ap_ = pr[pr.arm == k]
            sd_warm = float(np.sqrt(kp.groupby("i")["d_test"].var(ddof=1).mean()))
            comps = []
            for proto in ("kit", "seen1"):
                for i in idxs:
                    cl = sorted({c for (p_, i_, pa_, k_, dr_, c) in cellD if p_ == proto and i_ == i and pa_ == pas and k_ == k and dr_ == -1})
                    base = np.array([cellD[(proto, i, pas, k, -1, c)] for c in cl])
                    S = np.array([[cellD[(proto, i, pas, k, dr, c)] for c in cl] for dr in range(nd)]) - base
                    cov = np.cov(S, rowvar=False)
                    user = np.array([c.split("|")[0] for c in cl])
                    same = (user[:, None] == user[None, :]) & ~np.eye(len(cl), dtype=bool)
                    diff = user[:, None] != user[None, :]
                    comps.append((np.diag(cov).mean(), cov[same].mean() if same.any() else 0.0, cov[diff].mean()))
            v, cu, cx = np.mean(comps, 0)
            sealed = {f"U{U}_C{C}": float(np.sqrt(max(v / C + cu * (1 / U - 1 / C) + cx * (1 - 1 / U), 0))) for U, C in SCEN}
            dt = kp["d_test"].to_numpy()
            R[k] = dict(flips_warm=float(kp.flips_warm.mean()), flips_all=float(ap_.flips_all.mean()),
                        p0_flips_warm=float((kp.flips_warm == 0).mean()), sd_warm=sd_warm, sealed=sealed,
                        mean_d_warm=float(dt.mean()), p_shift_le_gap=float((dt <= GAP).mean()))
        # (c) the gain SK_k - SK1, package shape, per index averaged over draws (and the unperturbed gain as a check of T2-D285)
        def shape(frame):
            w = lambda proto, col: frame[frame.proto == proto].pivot_table(index=["i", "draw"], columns="arm", values=col)
            kt, ss, sv = w("kit", "test"), w("seen1", "seen"), w("seen1", "val")
            return kt, ss, sv
        kt, ss, sv = shape(d)
        G = {}
        for k in ("SK3", "SK5", "SK10"):
            new = ((kt[k] - kt["SK1"]) + (sv[k] - sv["SK1"])) / 2
            mixd = ((ss[k] - ss["SK1"]) + new) / 2
            mixd = mixd.reset_index()
            mixd.columns = ["i", "draw", "mix"]
            pert = mixd[mixd.draw >= 0].groupby("i")["mix"].mean()
            unp = mixd[mixd.draw < 0].set_index("i")["mix"]
            ktd = (kt[k] - kt["SK1"]).reset_index()
            ktd.columns = ["i", "draw", "kt"]
            G[k] = dict(gain=float(pert.mean()), up=int((pert > 0).sum()), gain_unperturbed=float(unp.mean()),
                        up_unperturbed=int((unp > 0).sum()), gain_draw_sd=float(mixd[mixd.draw >= 0].groupby("i")["mix"].std(ddof=1).mean()),
                        kit_test_gain=float(ktd[ktd.draw >= 0]["kt"].mean()), kit_test_gain_sd_warm=float(ktd[ktd.draw >= 0].groupby("i")["kt"].std(ddof=1).mean()))
        res["passes"][pas] = dict(arms=R, gains=G)
        print(f"\n== pass {pas} ({'all four members' if pas == 'all' else 'ST-L + MOR only'} perturbed, sigma 2^-8, {nd} draws x {len(idxs)} indices)")
        for k in ARMS:
            r = R[k]
            print(f"{k:5s} flips: kit test {r['flips_warm']:.2f} (P(0) {r['p0_flips_warm']:.2f}), all scored {r['flips_all']:.2f}; "
                  f"d cell BA kit test mean {r['mean_d_warm']:+.4f} sd {r['sd_warm']:.4f} (P(<= {GAP}) {r['p_shift_le_gap']:.3f}); sealed sd "
                  + ", ".join(f"{s_} {v_:.4f}" for s_, v_ in r["sealed"].items()))
        for k, g in G.items():
            print(f"{k} - SK1 mix: perturbed {g['gain']:+.4f} ({g['up']} / {len(idxs)} up; per-index draw sd {g['gain_draw_sd']:.4f}); unperturbed "
                  f"{g['gain_unperturbed']:+.4f} ({g['up_unperturbed']} up); kit test {g['kit_test_gain']:+.4f} (draw sd {g['kit_test_gain_sd_warm']:.4f})")
    if not a.quick:
        A = res["passes"]["all"]
        f1, f10, g = A["arms"]["SK1"]["flips_warm"], A["arms"]["SK10"]["flips_warm"], A["gains"]["SK10"]["gain"]
        much = f10 >= 3 * f1 and f10 >= 1
        v = ("SK10 SHIPS AS PLANNED" if much and g >= 0.005 else "TRIGGER: SK3 / SK5 to the user before Nov 5" if g < 0.003
             else "REPORT to the user (SK10 stays the plan unless the user decides)")
        res.update(much=much, verdict=v)
        print(f"\nverdict: SK10 kit-test flips {f10:.2f} vs SK1 {f1:.2f} -> >> {much}; SK10 perturbed mix gain {g:+.4f} -> {v}"
              + ("" if much else "; flips not >> SK1: bf16-scale noise does not explain the gap"))
    (out / ("result_quick.json" if a.quick else "result.json")).write_text(json.dumps(res, indent=1))
    df.to_csv(out / ("rows_quick.csv" if a.quick else "rows.csv"), index=False)


if __name__ == "__main__":
    main()

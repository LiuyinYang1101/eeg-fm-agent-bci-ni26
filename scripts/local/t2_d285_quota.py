#!/usr/bin/env python
"""T2-D285 (zero GPU; pre-registered in docs/DECISIONS_T2.md): count-aware causal decoding after the shipped prior.
T2-D253's members at index i (fbc_rc75 204+i, sh_vr 168+i, stl_rcll i, mor i; w4321) -> the package mix p_t; per stream (the scored
sessions are whole streams, in `order`) each arm decodes causally:
  A0   the shipped prior (1, 3)                                   (anchor)
  QL   oracle-label quota (CEILING): true runs, true per-run counts, R_c = n_c - the run's earlier TRUE labels
  QH   self-count quota, hard: true runs, R_c = L/3 - the arm's own argmax counts so far in the run
  QS   run quota, soft, oracle boundaries: true runs, R_c = L/3 - the sum of the arm's own decoded probabilities so far
  QI   run quota, soft, count-inferred boundaries: a window counter with the schedule [69, 30, 30, ...]   (shippable)
  QW   QI with the wrong Online length: [69, 36, 36, ...]
  SK3 / SK10  causal Sinkhorn: raw p_1..p_t + 3 uniform pseudo-rows, 3 / 10 iterations (columns to equal mass, rows normalised), row t
A quota multiplies the prior-corrected q_t by pi_c = (max(R_c, 0) + 1) / (W + 3) (W = windows left in the run including t) and renormalises.
Bar (QI; also SK3 / SK10): d141 (mix >= +0.005, >= 9 / 12 up, kit test >= -0.005) AND new >= 0 AND >= 3 / 4 kit participants up.
Kill: QL mix < +0.010; QI kit test <= 0; QI mix < 0.5 x QS mix; QW mix < -0.010.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d285_quota.py [--spec t2_d253] [--offsets ...] [--out t2_d285]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import metrics  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}
ARMS = ("A0", "QL", "QH", "QS", "QI", "QW", "SK3", "SK10")
SHIP = ("QI", "SK3", "SK10")
N = 12


def prior(p, n0=3.0):
    t = np.arange(1, len(p) + 1)[:, None]
    m = (n0 / 3 + np.cumsum(p, 0)) / (n0 + t)
    q = p / m
    return q / q.sum(1, keepdims=True)


def segments_true(run):
    """per window: (run index, position in run, run length) from the true run labels (in stream order)."""
    b = np.r_[0, np.flatnonzero(run[1:] != run[:-1]) + 1, len(run)]
    seg = np.zeros(len(run), int)
    pos = np.zeros(len(run), int)
    ln = np.zeros(len(run), int)
    for k in range(len(b) - 1):
        seg[b[k]:b[k + 1]] = k
        pos[b[k]:b[k + 1]] = np.arange(b[k + 1] - b[k])
        ln[b[k]:b[k + 1]] = b[k + 1] - b[k]
    return seg, pos, ln


def segments_sched(T, first, rest):
    seg, pos, ln = np.zeros(T, int), np.zeros(T, int), np.zeros(T, int)
    t, k = 0, 0
    while t < T:
        L = first if k == 0 else rest
        e = min(t + L, T)
        seg[t:e], pos[t:e], ln[t:e] = k, np.arange(e - t), L          # the decoder only knows the scheduled length
        t, k = e, k + 1
    return seg, pos, ln


def quota(q, seg, pos, ln, mode, y=None):
    out = np.zeros_like(q)
    cnt = np.zeros(3)
    for t in range(len(q)):
        if pos[t] == 0:
            cnt = np.zeros(3)
            if mode == "label":
                nc = np.bincount(y[seg == seg[t]], minlength=3).astype(float)
        L = ln[t]
        n = nc if mode == "label" else np.full(3, L / 3.0)
        R = np.maximum(n - cnt, 0.0)
        Wt = L - pos[t]
        pi = (R + 1.0) / (max(Wt, 0) + 3.0)
        r = q[t] * pi
        r = r / r.sum()
        out[t] = r
        if mode == "label":
            cnt[y[t]] += 1
        elif mode == "hard":
            cnt[int(r.argmax())] += 1
        else:
            cnt += r
    return out


def sinkhorn(p, iters):
    out = np.zeros_like(p)
    for t in range(len(p)):
        M = np.vstack([np.full((3, 3), 1 / 3), p[:t + 1]])
        for _ in range(iters):
            M = M * (M.shape[0] / 3) / M.sum(0, keepdims=True)
            M = M / M.sum(1, keepdims=True)
        out[t] = M[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="t2_d253")
    ap.add_argument("--offsets", default="fbc_rc75=204,sh_vr=168,stl_rcll=0,mor=0")
    ap.add_argument("--out", default="t2_d285")
    ap.add_argument("--weights", default="", help="T2-D303: member=weight,... (a subset of W; renormalised to sum 1), e.g. the MOR-free mix")
    a = ap.parse_args()
    if a.weights:                                     # T2-D303: a member subset, renormalised (the arms below decode this mix)
        w_ = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
        assert set(w_) <= set(W) and all(v > 0 for v in w_.values()), w_
        W.clear(); W.update({k: v / sum(w_.values()) for k, v in w_.items()})
        print("weights:", {k: round(v, 4) for k, v in W.items()}, flush=True)
    offs = {k: int(v) for k, v in (x.split("=") for x in a.offsets.split(",")) if k in W}
    out = O / a.out
    out.mkdir(exist_ok=True)
    s = json.load(open(O / a.spec / "spec.json"))
    mem = {m: s["members"][m] for m in W}
    assert {m: d["offset"] for m, d in mem.items()} == offs, ({m: d["offset"] for m, d in mem.items()}, offs)
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    subj, ses = idx["subject"].to_numpy(), idx["session"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    run = (idx["task"] + "#" + idx["run"].astype(str)).to_numpy()
    ordr = idx["order"].to_numpy()
    rows, chk, pos_gain = [], [], []
    for proto in ("kit", "seen1"):
        for i in range(N):
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
            pred = {k: np.full(len(idx), -1) for k in ARMS + ("SK1",)}
            for st in np.unique(strm[ok]):
                r_ = np.flatnonzero(ok & (strm == st))
                r_ = r_[np.argsort(ordr[r_], kind="stable")]
                assert (strm == st).sum() == len(r_), f"{st}: a scored stream is not whole"
                p = mix[r_]
                q = prior(p)
                sg, ps, ln = segments_true(run[r_])
                D = {"A0": q, "QL": quota(q, sg, ps, ln, "label", y[r_]), "QH": quota(q, sg, ps, ln, "hard"), "QS": quota(q, sg, ps, ln, "soft")}
                D["QI"] = quota(q, *segments_sched(len(r_), 69, 30), "soft")
                D["QW"] = quota(q, *segments_sched(len(r_), 69, 36), "soft")
                for k_ in (1, 3, 10):
                    D[f"SK{k_}"] = sinkhorn(p, k_)
                for k_, v in D.items():
                    pred[k_][r_] = v.argmax(1)
                if proto == "kit" and np.isin(subj[r_[0]], ["P04", "P09"]):        # descriptive: QI's window gain by position in a true run
                    late = ps >= (2 * ln) // 3
                    for nm, msk in (("last_third", late), ("rest", ~late)):
                        pos_gain.append(dict(i=i, part=nm, d=float((D["QI"].argmax(1)[msk] == y[r_][msk]).mean() - (q.argmax(1)[msk] == y[r_][msk]).mean()), n=int(msk.sum())))
            chk.append(float((pred["SK1"][ok] == pred["A0"][ok]).mean()))
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
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)                # T2-D296sk: per-index levels, for arm-vs-arm comparisons under one decoder

    def wide(proto, col):
        return df[df.proto == proto].pivot_table(index="i", columns="arm", values=col)
    kt, kv, ss, sv = wide("kit", "test"), wide("kit", "val"), wide("seen1", "seen"), wide("seen1", "val")
    res = {}
    print(f"validity: SK1 argmax = A0 argmax on {100 * np.mean(chk):.2f} % of windows (mean over 24 proto x index); "
          f"A0 levels kit test {kt['A0'].mean():.4f}, kit val {kv['A0'].mean():.4f}, seen1 seen {ss['A0'].mean():.4f}, seen1 val {sv['A0'].mean():.4f}")
    for k_ in ARMS[1:]:
        D = dict(kit_test=kt[k_] - kt["A0"], kit_val=kv[k_] - kv["A0"], seen=ss[k_] - ss["A0"], seen1_val=sv[k_] - sv["A0"])
        D["new"] = (D["kit_test"] + D["seen1_val"]) / 2
        D["mix"] = (D["seen"] + D["new"]) / 2
        r = {g: dict(d=float(v.mean()), up=int((v > 0).sum()), down=int((v < 0).sum())) for g, v in D.items()}
        kp = {}
        for u in ("P04", "P05", "P08", "P09"):
            w_ = wide("kit", u)
            kp[u] = float((w_[k_] - w_["A0"]).mean())
        r["kit_parts"] = kp
        r["bar"] = (r["mix"]["d"] >= 0.005 and r["mix"]["up"] >= 9 and r["kit_test"]["d"] >= -0.005 and r["new"]["d"] >= 0
                    and sum(v > 0 for v in kp.values()) >= 3)
        res[k_] = r
        print(f"{k_:5s} vs A0: mix {r['mix']['d']:+.4f} ({r['mix']['up']} / {N} up), kit test {r['kit_test']['d']:+.4f}, kit val {r['kit_val']['d']:+.4f}, "
              f"seen {r['seen']['d']:+.4f}, seen1 val {r['seen1_val']['d']:+.4f}, new {r['new']['d']:+.4f}; kit participants "
              + ", ".join(f"{u} {v:+.4f}" for u, v in kp.items()) + (f"; bar {'MET' if r['bar'] else 'not met'}" if k_ in SHIP else ""))
    pg = pd.DataFrame(pos_gain).groupby("part").apply(lambda g: np.average(g.d, weights=g.n))
    print("QI window-accuracy gain by position in the true run (kit test): " + ", ".join(f"{k} {v:+.4f}" for k, v in pg.items()))
    kills = []
    if res["QL"]["mix"]["d"] < 0.010:
        kills.append(f"ceiling QL mix {res['QL']['mix']['d']:+.4f} < +0.010")
    if res["QI"]["kit_test"]["d"] <= 0:
        kills.append(f"QI kit test {res['QI']['kit_test']['d']:+.4f} <= 0")
    if res["QI"]["mix"]["d"] < 0.5 * res["QS"]["mix"]["d"]:
        kills.append(f"QI mix {res['QI']['mix']['d']:+.4f} < 0.5 x QS {res['QS']['mix']['d']:+.4f}")
    if res["QW"]["mix"]["d"] < -0.010:
        kills.append(f"QW mix {res['QW']['mix']['d']:+.4f} < -0.010")
    cands = [k_ for k_ in SHIP if res[k_]["bar"] and not (k_ == "QI" and kills)]
    verdict = ("KILLED (the quota line): " + "; ".join(kills) + (f"; label-free Sinkhorn candidates: {cands}" if cands else "")) if kills else \
        (f"CANDIDATE(S) {cands} -> the fresh-seed replicate (T2-D247r's set)" if cands else "NOT ADOPTED (no shippable arm meets the bar)")
    print("VERDICT:", verdict)
    (out / "read.json").write_text(json.dumps(dict(arms=res, verdict=verdict, kills=kills, candidates=cands, validity_sk1_eq_a0=float(np.mean(chk)),
                                                   a0=dict(kit_test=float(kt["A0"].mean()), kit_val=float(kv["A0"].mean()), seen=float(ss["A0"].mean()), seen1_val=float(sv["A0"].mean())),
                                                   pos_gain=pg.to_dict(), rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()

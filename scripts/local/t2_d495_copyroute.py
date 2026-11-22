#!/usr/bin/env python
"""T2-D495 (seat G14 item A; registered 2026-11-14 before any number; zero GPU): route a STRANGER's stream to the per-user copy that
explains it best by label-free model evidence, instead of the fingerprint's identity match.
Files: the per-copy outputs (t2p_bench --peruser 'pu' twins: prob_last = the pooled net, prob_users_test = one copy per training user) of
A13's CNN controls (FBCNet T2-D419 seeds 324-335, r 1/3; ShallowNet / IFNet T2-D429 seeds 348-359, r 0.5) and their shipped routed files
(t2p_fpcalib blend: fingerprint-nearest copy). Strangers = test streams whose user is not a training user of the protocol (kit: P04 P05 P08 P09;
seen1: P09). Every arm is decoded per stream by the shipped causal Sinkhorn (10 it, n0 3, lambda 0.95, uniform start); score = the stream's
mean cell balanced accuracy; per user = the mean over its streams and the 12 seeds; typical = non-P04.
Arms per family and for the CNN mix (M7's CNN weights, normalised):
  shipped  the routed file (fingerprint route)            pooled   the pooled net alone (r 0)
  cu_r     copy u blended at r (the family's shipped r) and at r 1, for every training user u
  oracle_s the best copy per STREAM (hindsight; optimistic: max of 6-7 noisy scores)   oracle_u the best copy per USER (fixed over streams / seeds)
  mi       causal: pooled for t < 5, then the copy u*(t) = argmax_u [H(mean_{1..t} q_u) - mean_{1..t} H(q_u)], q_u = the copy blended at r
KILL (the seat's): oracle_u - shipped < +0.02 on typical strangers at the CNN-mix level, OR mi recovers under half of (oracle_u - shipped).
    mem-guard python scripts/local/t2_d495_copyroute.py
"""
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
FAM = {"fbc": (S / "ni26-t2p-d419-fbc_e180_s324-5090/runs", "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cpu", 324, 1 / 3,
               O / "t2_d419/route_fbc_e180/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz"),
       "sh": (S / "ni26-t2p-d429-sh-ctl-5090/runs", "shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 348, 0.5,
              O / "t2_d429/route_sh_ctl/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"),
       "ifn": (S / "ni26-t2p-d429-ifn-ctl-5090/runs", "ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cpu", 348, 0.5,
               O / "t2_d429/route_ifn_ctl/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz")}
J = json.load(open(O / "t2_d442/spec.json"))
C = J["arms"]["M7f"]
WC = {"fbc": C["fbc_tv8c"], "sh": C["sh_tv8c"], "ifn": C["ifn"]}
WC = {k: v / sum(WC.values()) for k, v in WC.items()}
idx = pd.read_csv(EXP / "index.csv")
U = idx.subject.str[-3:].to_numpy()
T0 = 5


def ent(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(-1)


def ba_stream(P, rows):
    key = (idx.stream.to_numpy()[rows], idx.order.to_numpy()[rows])
    pr = causal_sinkhorn(P, key, 10, 3, 0.95).argmax(1)
    y, cells = idx.y.to_numpy()[rows], idx.cell.to_numpy()[rows]
    return float(np.mean([np.mean([np.mean(pr[(cells == c) & (y == k)] == k) for k in np.unique(y[cells == c])]) for c in np.unique(cells)]))


def mi_route(Q):
    """Q (n_users, T, 3) blended copies of one stream in order -> (T, 3): pooled-free causal selection by mutual information."""
    nU, T, _ = Q.shape
    out = np.empty((T, 3))
    cs, ch = np.zeros((nU, 3)), np.zeros(nU)
    for t in range(T):
        cs += Q[:, t]
        ch += ent(Q[:, t])
        if t + 1 < T0:
            out[t] = np.nan
            continue
        mi = ent(cs / (t + 1)) - ch / (t + 1)
        out[t] = Q[int(np.argmax(mi)), t]
    return out


rows_out = []
SPL = {"test": ("test_idx", "prob_last", "prob_users_test"), "val": ("val_idx", "prob_val_last", "prob_users_val")}   # both held-out splits
for proto, split in (("kit", "test"), ("kit", "val"), ("seen1", "test"), ("seen1", "val")):
    for i in range(12):
        seq = {}
        ki, kp, ku = SPL[split]
        for fam, (rd, stem, off, r, routed) in FAM.items():
            z = np.load(rd / f"{proto}__{stem}__s{off + i}.npz")
            zr = np.load(str(routed).format(p=proto, s=off + i))
            assert (z[ki] == zr[ki]).all()
            seq[fam] = dict(te=z[ki], pooled=z[kp].astype(np.float64), Pu=z[ku].astype(np.float64),
                            users=[str(u) for u in z["users"]], ship=zr[kp].astype(np.float64), r=r)
        te = seq["fbc"]["te"]
        users = seq["fbc"]["users"]
        assert all((s_["te"] == te).all() and s_["users"] == users for s_ in seq.values())
        pos = {int(x): k for k, x in enumerate(te)}
        for st_, g in idx.iloc[te].groupby("stream", sort=False):
            u = g.subject.iloc[0][-3:]
            if u in users:
                continue
            rows = g.sort_values("order")["i"].to_numpy()
            k_ = np.array([pos[int(x)] for x in rows])
            arms = {}
            for fam, s_ in seq.items():
                sh, Pu, r = s_["pooled"][k_], s_["Pu"][:, k_], s_["r"]
                arms[(fam, "shipped")] = s_["ship"][k_]
                arms[(fam, "pooled")] = sh
                Qr = (1 - r) * sh[None] + r * Pu
                for j, cu in enumerate(users):
                    arms[(fam, f"c{cu}_r")] = Qr[j]
                    arms[(fam, f"c{cu}_1")] = Pu[j]
                mr = mi_route(Qr)
                mr[:T0 - 1] = sh[:T0 - 1]
                arms[(fam, "mi")] = mr
            for name in {a for _, a in arms}:                 # the CNN mix: the same copy / rule in every family
                arms[("mix", name)] = sum(WC[f] * arms[(f, name)] for f in FAM)
            for (fam, name), P in arms.items():
                rows_out.append(dict(proto=proto + "_" + split, i=i, stream=st_, user=u, fam=fam, arm=name, ba=ba_stream(P / P.sum(1, keepdims=True), rows)))
    print(f"[d495] {proto} {split} done", flush=True)
R = pd.DataFrame(rows_out)
out = O / "t2_d495"
out.mkdir(exist_ok=True)
R.to_csv(out / "cells.csv", index=False)
res = {}
for fam in list(FAM) + ["mix"]:
    D = R[R.fam == fam]
    per_u = D.groupby(["user", "arm"]).ba.mean().unstack()
    copies_r = [c for c in per_u.columns if c.startswith("c") and c.endswith("_r")]
    copies_1 = [c for c in per_u.columns if c.startswith("c") and c.endswith("_1")]
    ps = D.groupby(["proto", "i", "stream", "arm"]).ba.mean().unstack()
    o_s = ps[copies_r].max(1).groupby(level=[0, 1, 2]).first()
    ps = ps.assign(oracle_s=ps[copies_r].max(1))
    per_u["oracle_s"] = ps.reset_index().merge(D[["proto", "i", "stream", "user"]].drop_duplicates(), on=["proto", "i", "stream"]).groupby("user").oracle_s.mean()
    per_u["oracle_u"] = per_u[copies_r].max(1)
    per_u["best_copy_r"] = per_u[copies_r].idxmax(1)
    per_u["oracle_u1"] = per_u[copies_1].max(1)
    per_u["best_copy_1"] = per_u[copies_1].idxmax(1)
    typ = per_u.drop("P04", errors="ignore")
    num = ["shipped", "pooled", "oracle_s", "oracle_u", "oracle_u1", "mi"]
    res[fam] = dict(per_user={u: {k: (round(float(per_u.loc[u, k]), 4) if k in num else per_u.loc[u, k]) for k in num + ["best_copy_r", "best_copy_1"]}
                              for u in per_u.index},
                    typical={k: round(float(typ[k].mean()), 4) for k in num})
    t = res[fam]["typical"]
    print(f"D495 {fam}: typical strangers shipped {t['shipped']:.4f} pooled {t['pooled']:.4f} | oracle per user {t['oracle_u']:.4f} "
          f"({t['oracle_u'] - t['shipped']:+.4f}) per stream {t['oracle_s']:.4f} ({t['oracle_s'] - t['shipped']:+.4f}) r1 {t['oracle_u1']:.4f} | "
          f"MI rule {t['mi']:.4f} ({t['mi'] - t['shipped']:+.4f})", flush=True)
    for u in per_u.index:
        pu_ = res[fam]["per_user"][u]
        print(f"   {u}: shipped {pu_['shipped']:.4f} pooled {pu_['pooled']:.4f} oracle_u {pu_['oracle_u']:.4f} ({pu_['best_copy_r']}) "
              f"r1 {pu_['oracle_u1']:.4f} ({pu_['best_copy_1']}) mi {pu_['mi']:.4f}", flush=True)
m = res["mix"]["typical"]
gain, rec = m["oracle_u"] - m["shipped"], m["mi"] - m["shipped"]
kill = gain < 0.02 or rec < 0.5 * gain
print(f"D495 KILL TEST (CNN mix, typical strangers): oracle_u - shipped {gain:+.4f} (needs >= +0.02); MI - shipped {rec:+.4f} "
      f"(needs >= half of the oracle gain) -> {'KILLED' if kill else 'PASSES to a stack read'}", flush=True)
json.dump(res, open(out / "read.json", "w"), indent=1, default=str)

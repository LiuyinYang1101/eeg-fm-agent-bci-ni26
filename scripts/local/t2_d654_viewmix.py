#!/usr/bin/env python
"""T2-D654 (FM view-TTA) LAB VIEW-MIXING plumbing (Track3 draft for Track2; design pinned in REGISTRATION_D654.md, the manager 15:33:36).

The member files of the value runs (NI26_FEAT_DUMP=1 NI26_VIEW_DUMP=4) carry the identity's fm_feat_te + the three fixed views' fm_view_feat.
An ARM turns them into ONE final pooled feature per window, F = the view mean over the first K views (identity included) when the window uses
views, else the identity, and EVERYTHING downstream is recomputed from F through the member's own head: logits L = F W^T + b, prob =
softmax(L + side) (side vectors of the identity window). Writing F / L / prob into the per-row arrays the lab reads (fm_feat_te, side_logit_last,
prob_last, prob_best) makes SK, the in-stream LP / prototype refits, the teachers and the students all see the SAME final feature (the pinned
consistency): the lab chain is unchanged.

ARMS (pinned): v1 = the PAIRED BASE (identity, every scored row recomputed through the head: the recompute differs from the network's bf16
prob_last by <= ~3e-3, so every arm, the base included, goes through the same recompute); f2 / f4 = every window, K = 2 / 4; g2q20 / g2q30 /
g4q20 (+ g2q10, added before any number after the 5090 smoke's r2 1.42) = gated (K, q), with the gate source suffix c (the CNN-family blend: the cell's pooled route nets fbc / sh / ifn, W renormalised, raw, the
mean over the three draws: draw-independent, available before the FM) or m (the FM family's identity blend: the mean of the six members' v1
probabilities; the fallback arm), e.g. g2q30c.
GATE (pinned, causal, tuning-free): per stream in recording order, window t uses the views iff margin_t <= the running q-quantile of the
stream's margins m_0..m_t (numpy 'linear') AND n_gated_so_far + 1 <= q (t + 1) + 1 (the token bucket); margin = top1 - top2; window 0 is gated.

  python t2_d654_viewmix.py selftest                                   # synthetic + real-file checks (CPU, seconds)
  python t2_d654_viewmix.py member --pre d654a --seed0 1600 --rtl-pre d654rl --f1 <ms> --r2 <x> --r3 <x> --r4 <x> [--cell bw21]
                                                                       # STAGE 1: every arm vs v1 + its extra ms + THE PINNED PICK
  python t2_d654_viewmix.py fmspl <newtag> <basetag> <arm> <cell>      # STAGE 2: the transformed copy of setread/<cell>/fmspl_<basetag>/
Run from the ni26 repo root (threads capped by the caller).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src"); sys.path.insert(0, "tracks/t2_bci/scripts")
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S1 = O / "sofia_s1"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
I = pd.read_csv(EXP / "index.csv", low_memory=False)
U, SES, Y = I["subject"].astype(str).str[-3:].to_numpy(), I["session"].astype(str).to_numpy(), I["y"].to_numpy()
CELLC, ST, OD, RUN = I["cell"].astype(str).to_numpy(), I["stream"].to_numpy(), I["order"].to_numpy(), I["run"].to_numpy()
USERS = [f"P{k:02d}" for k in range(1, 11)]
STRANGERS = [u for u in USERS if u not in ("P04", "P10")]
SPL = [USERS.index(u) for u in STRANGERS]
SLOTS = ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]
CONT = {"s2_plaincont", "s4_carcont"}
DRAWS = (860, 861, 862)
VIEW_SHIFTS = (15, -15, 30)                      # == t2p_fm.VIEW_SHIFTS (asserted against every file's fm_view_shift)
W_CNN = {"fbc": 0.3143, "sh": 0.2357, "ifn": 0.10}   # W21's CNN family weights (t2_d532_read.W), renormalised below
RT = {"fbc": "route_fbc/lc9_{f}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__s{s}.npz",
      "sh": "route_sh/lc9_{f}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz",
      "ifn": "route_ifn/lc9_{f}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s{s}.npz"}   # == t2_d532_read.RT (the pooled route nets, FM-free)
ARMS = {"v1": (1, None), "f2": (2, None), "f4": (4, None), "g2q10": (2, 0.1), "g2q20": (2, 0.2), "g2q30": (2, 0.3), "g4q20": (4, 0.2),
        "p2q20": (2, 0.2), "p2q30": (2, 0.3)}   # g2q10: added before any number (r2 1.42); p2q20 / p2q30 = the owner's probabilistic gate (16:4x)
PROB_SEED = 65460                                   # T2-D654 prob gate: ONE torch.Generator(65460) per stream, re-seeded at the stream's first window


def arm_of(name):
    """(K, q, gate source) of an arm name: v1 / f2 / f4, or a gated base + 'c' (CNN) / 'm' (FM identity); 'p…c' = the probabilistic CNN gate."""
    if name in ("v1", "f2", "f4"):
        return ARMS[name] + (None,)
    assert name[:-1] in ARMS and ARMS[name[:-1]][1] is not None and name[-1] in "cm", name
    assert not (name.startswith("p") and name[-1] != "c"), (name, "the prob gate reads the CNN margin only")
    return ARMS[name[:-1]] + ({"c": "cnn", "m": "fm"}[name[-1]],)


ALL_ARMS = ["v1", "f2", "f4"] + [g + s for g in ("g2q10", "g2q20", "g2q30", "g4q20") for s in "cm"] + ["p2q20c", "p2q30c"]


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def margin(p):
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def causal_gate(m, streams, order, q):
    """the pinned gate over rows (any order): per stream in recording order, row t gated iff m_t <= quantile_q(m_0..m_t) and the token bucket
    (n_gated + 1 <= q (t + 1) + 1) allows it. Causal: row t depends on rows 0..t of its own stream only."""
    g = np.zeros(len(m), bool)
    for s_ in np.unique(streams):
        r = np.flatnonzero(streams == s_)
        r = r[np.argsort(order[r], kind="stable")]
        n_g = 0
        for t, i in enumerate(r):
            if m[i] <= np.quantile(m[r[:t + 1]], q) and n_g + 1 <= q * (t + 1) + 1:
                g[i] = True
                n_g += 1
    return g


def prob_gate(m, streams, order, q, seed=PROB_SEED):
    """the owner's probabilistic gate (pinned 2026-11-20 16:4x): per stream in recording order, ONE torch.Generator(seed) re-seeded at the stream's
    first window; window t draws u_t = torch.rand(1, generator=g) (float32 -> float; ONE draw per window, gated or not), r_t = #{s < t: m_s < m_t} / t
    (r_0 = 0; 0 = the least confident so far), p_t = min(1, 2q (1 - r_t)); viewed iff u_t < p_t AND n_gated + 1 <= q (t + 1) + 1 (the same bucket)."""
    import torch
    g = np.zeros(len(m), bool)
    for s_ in np.unique(streams):
        r = np.flatnonzero(streams == s_)
        r = r[np.argsort(order[r], kind="stable")]
        gen = torch.Generator().manual_seed(int(seed))
        n_g = 0
        for t, i in enumerate(r):
            u = float(torch.rand(1, generator=gen)[0])
            rk = float(np.sum(m[r[:t]] < m[i])) / t if t > 0 else 0.0
            if u < min(1.0, 2.0 * q * (1.0 - rk)) and n_g + 1 <= q * (t + 1) + 1:
                g[i] = True
                n_g += 1
    return g


def final_rows(z, sel, K, use):
    """the final (F, L, prob) of the rows sel of a member file z: F = the mean of the first K views (identity + K - 1 fixed views) where use,
    else the identity; L = F W^T + b; prob = softmax(L + side) (identity side vectors)."""
    F0 = z["fm_feat_te"][sel].astype(np.float64)
    F = F0
    if K > 1:
        assert "fm_view_feat" in z.files and z["fm_view_feat"].shape[0] >= K - 1, (K, z.files)
        assert tuple(int(v) for v in z["fm_view_shift"][:K - 1]) == VIEW_SHIFTS[:K - 1], z["fm_view_shift"]
        Fv = (F0 + z["fm_view_feat"][:K - 1, sel].astype(np.float64).sum(0)) / K
        F = np.where(np.asarray(use, bool)[:, None], Fv, F0)
    L = F @ z["fm_out_w"].astype(np.float64).T + z["fm_out_b"].astype(np.float64)
    side = (z["side_te"][sel].astype(np.float64) @ z["side_w"].astype(np.float64).T + z["side_b"].astype(np.float64)) if "side_w" in z.files else 0.0
    return F, L, softmax(L + side)


_RT = {}


def route_blend(cell, f):
    """the CNN-family gate blend of lc9 fold f: sum_fam w_fam * mean_draws route prob (raw), over the fold's test rows -> {row: prob}."""
    k = (cell, f)
    if k not in _RT:
        wt = sum(W_CNN.values())
        acc, te0 = None, None
        for fam, w in W_CNN.items():
            for d in DRAWS:
                z = np.load(O / f"t2_d638_{cell}_860" / RT[fam].format(f=f, s=d))
                te, p = z["test_idx"].astype(np.int64), z["prob_last"].astype(np.float64)
                o = np.argsort(te)
                te, p = te[o], p[o] / p[o].sum(1, keepdims=True)
                if te0 is None:
                    te0, acc = te, np.zeros_like(p)
                assert np.array_equal(te, te0), (fam, d, f)
                acc += (w / wt) * p / len(DRAWS)
        _RT[k] = dict(zip(te0.tolist(), acc))
    return _RT[k]


def cnn_prob(cell, rows, folds):
    """the CNN gate blend on rows, averaged over the lc9 folds given (one fold for a stranger's rows; the 8 spliced folds for P04 / P10 rows)."""
    return np.mean([np.stack([route_blend(cell, f)[int(r)] for r in rows]) for f in folds], 0)


def gate_for(src, rows, arm, cell, fm_fam_prob=None, fold=None):
    K, q, gs = arm_of(arm)
    if q is None:
        return np.ones(len(rows), bool) if K > 1 else np.zeros(len(rows), bool)
    if gs == "cnn":
        folds = [fold] if fold is not None else ([USERS.index(src.split("_", 1)[1])] if src.startswith(("ss8_", "rtl_")) else SPL)
        mg = margin(cnn_prob(cell, rows, folds))
    else:
        assert fm_fam_prob is not None
        mg = margin(fm_fam_prob)
    if arm.startswith("p"):
        return prob_gate(mg, ST[rows], OD[rows], q)
    return causal_gate(mg, ST[rows], OD[rows], q)


def scored_rows(src, te):
    kind, x = src.split("_", 1)
    if kind == "ss8":
        return U[te] == x
    if kind == "rt1":
        return (U[te] == x) & np.isin(SES[te], ("2", "3"))
    if kind == "rt2":
        return (U[te] == x) & (SES[te] == "3")
    assert kind == "rtl", src
    runs = np.unique(RUN[U == x])
    return (U[te] == x) & np.isin(RUN[te], runs[-2:])


def find(pre, slot, src, seed):
    fs = [f for d in glob.glob(str(S1 / f"ni26-t2p-{pre}-{slot}_C-*")) for f in glob.glob(f"{d}/runs/{src}__fm-*__s{seed}.npz")
          if ("cb__" in os.path.basename(f)) == (slot in CONT) and "__ubw" in os.path.basename(f)
          and not any(t in os.path.basename(f) for t in ("grl", "vp2"))]
    assert len(fs) == 1, f"REFUSED: {pre} {slot} {src} s{seed}: {[os.path.basename(x) for x in fs]}"
    return fs[0]


LINE_W25_S, LINE_W25_MAX, LINE_SLOPE = 3242.6, 3273.0, 62.0   # D517: s = 62.0 ms - 163; W25's reps 3,273.0 / 3,235.2 / 3,219.7 s (mean 3,242.6)


def headroom_ms(mean_bar=3480.0, rep_bar=3550.0):
    """the promotion timing gates (the manager 2026-11-20 16:29, = W25's): mean <= mean_bar AND every rep <= rep_bar on the line -> the extra ms /
    window an arm may add = min((mean_bar - 3242.6) / 62, (rep_bar - 3273.0) / 62). The OWNER raised the D654 bar in writing (2026-11-20 16:4x:
    "3400-3500 is okay") -> the manager's pin mean <= 3,480 / every rep <= 3,550 -> 3.829 ms (it was 0.926 at W25's 3,300 / 3,400)."""
    return min((mean_bar - LINE_W25_S) / LINE_SLOPE, (rep_bar - LINE_W25_MAX) / LINE_SLOPE)


HEADROOM_MS = headroom_ms()


def extra_ms(arm, tm):
    """the pinned eligibility arithmetic: tm = dict(f1 = the six FM members at B = 1 on the D517 line's ms scale, r2 / r3 / r4 = B = K / B = 1)."""
    K, q, gs = arm_of(arm)
    r = {1: 1.0, 2: tm["r2"], 3: tm["r3"], 4: tm["r4"]}
    if K == 1:
        return 0.0
    if q is None:
        return (r[K] - 1.0) * tm["f1"]
    return q * ((r[K] - 1.0) if gs == "cnn" else r[K - 1]) * tm["f1"]


def cmd_member(pre, seed0, cell, rtl_pre=None, tm=None):
    """STAGE 1, member alone: per slot x source (ss8_u, rt1_x, rt2_x, rtl_u when present) the arm's prob on the scored rows -> causal SK
    (10 / 3 / 0.95) -> cell BA, minus v1's (paired: the same fit). Pooled: strangers = mean over 6 slots x 8; rt1 / rt2 = over 6 x 2; RL = 6 x 8."""
    srcs = [f"ss8_{u}" for u in STRANGERS] + [f"rt{j}_{x}" for j in (1, 2) for x in ("P04", "P10")]
    rtl_pre = rtl_pre or pre                             # Track2's naming: d654a = the 1600 ship set, d654rl = its rtl set, d654b = 1700
    has_rtl = bool(glob.glob(str(S1 / f"ni26-t2p-{rtl_pre}-{SLOTS[1]}_C-*/runs/rtl_P01__*__s{seed0 + 1}.npz")))
    srcs += [f"rtl_{u}" for u in STRANGERS] if has_rtl else []
    lev = {a: {} for a in ALL_ARMS}
    dg = dict(margin=[], agree=[], spread=[], correct=[], user=[])   # DIAGNOSTIC (label-using; the manager 16:2x): view agreement as a confidence signal
    gfrac = {a: [] for a in ALL_ARMS}
    for src in srcs:
        Z = {k: np.load(find(rtl_pre if src.startswith("rtl_") else pre, SLOTS[k], src, seed0 + k)) for k in range(6)}
        te = Z[0]["test_idx"].astype(np.int64)
        for k in range(1, 6):
            assert np.array_equal(Z[k]["test_idx"].astype(np.int64), te), (src, k, "the six slots' test rows differ")
        sel = np.flatnonzero(scored_rows(src, te))
        rows = te[sel]
        fam = np.mean([final_rows(Z[k], sel, 1, None)[2] for k in range(6)], 0)   # the FM family's identity (v1) blend: the m gate
        if src.startswith("ss8_"):                       # DIAGNOSTIC only: per (slot, window) the identity's raw correctness vs label-free signals
            for k in range(6):
                pv = [final_rows(Z[k], sel, 1, None)[2]]
                for v in range(1, 4):                    # view v alone through the head (the identity window's side vectors)
                    zz = {key: Z[k][key] for key in Z[k].files}
                    zz["fm_feat_te"] = Z[k]["fm_view_feat"][v - 1]
                    zz = type("Zv", (dict,), {"files": property(lambda self_: list(self_.keys()))})(zz)
                    pv.append(final_rows(zz, sel, 1, None)[2])
                pv = np.stack(pv)
                a0 = pv[0].argmax(1)
                dg["margin"] += margin(pv[0]).tolist()
                dg["agree"] += (pv.argmax(2) == a0[None]).all(0).astype(float).tolist()
                dg["spread"] += np.abs(pv[1:] - pv[0][None]).sum(2).mean(0).tolist()
                dg["correct"] += (a0 == Y[rows]).astype(int).tolist()
                dg["user"] += [src[4:]] * len(rows)
        for a in ALL_ARMS:
            K, q, gs = arm_of(a)
            use = gate_for(src, rows, a, cell, fm_fam_prob=fam)
            gfrac[a].append(float(use.mean()))
            for k in range(6):
                p = final_rows(Z[k], sel, K, use)[2]
                D = causal_sinkhorn(p, (ST[rows], OD[rows]), iters=10, n0=3, lam=0.95)
                lev[a][(SLOTS[k], src)] = cell_balanced_accuracy(Y[rows], D.argmax(1), CELLC[rows])[0]
    out = {}
    for a in ALL_ARMS:
        d = {key: lev[a][key] - lev["v1"][key] for key in lev[a]}
        pick = lambda pref: [v for (s, src), v in d.items() if src.startswith(pref)]   # noqa: E731
        out[a] = dict(dS=float(np.mean(pick("ss8_"))), n_ge0_S=int(sum(v >= 0 for v in pick("ss8_"))), d_rt1=float(np.mean(pick("rt1_"))),
                      d_rt2=float(np.mean(pick("rt2_"))), dRL=(float(np.mean(pick("rtl_"))) if has_rtl else None),
                      n_rl=len(pick("rtl_")), gated_frac=float(np.mean(gfrac[a])),
                      per_slot_dS={s: float(np.mean([v for (s_, src), v in d.items() if s_ == s and src.startswith("ss8_")])) for s in SLOTS})
    for kfull, gated in (("f2", ("g2q10", "g2q20", "g2q30")), ("f4", ("g4q20",))):   # the gate's efficiency: the share of the full-K gain captured
        for g in gated:
            for sfx in "cm":
                fv = out[kfull]["dS"]
                out[g + sfx]["share_of_full"] = (out[g + sfx]["dS"] / fv) if fv > 0.001 else None
    diag = view_agreement_diag(dg)
    OUTd = O / "t2_d654"
    OUTd.mkdir(exist_ok=True)
    assert len(srcs) == (20 if has_rtl else 12), srcs
    L_ = [f"# T2-D654 STAGE 1 — member alone, every arm − v1 (paired within the fit), {pre} at {seed0} + slot", ""]
    pick = None
    if tm is not None:                                   # the pinned pick: the eligible arm (extra <= 2.55 ms) with the largest pooled dS passing the kill
        for a, v in out.items():
            v["extra_ms"] = extra_ms(a, tm)
            v["line_s"] = LINE_W25_S + LINE_SLOPE * v["extra_ms"]
            v["shippable"] = not a.endswith("m")          # W31v ships the c / p gates and full K only (an m gate needs FM@1 first: no template path)
            v["eligible"] = bool(v["shippable"] and v["extra_ms"] <= HEADROOM_MS + 1e-12)
            v["eligible_at_w25_gates"] = bool(v["extra_ms"] <= headroom_ms(3300.0, 3400.0) + 1e-12)   # descriptive (W25's own promotion gates)
            v["passes_kill"] = bool(v["dS"] > 0)         # the owner's inclusion rule (16:31:32): the member gate = the SIGN (rt legs -> the full chain)
        cand = [a for a, v in out.items() if a != "v1" and v["eligible"] and v["passes_kill"]]
        pick = None
        if cand:                                         # the largest member dS; within 0.001 of it, the CHEAPEST arm (the manager 16:4x)
            top = max(out[a]["dS"] for a in cand)
            pick = min((a for a in cand if out[a]["dS"] >= top - 0.001), key=lambda a: (out[a]["extra_ms"], -out[a]["dS"]))
        L_ += [f"**THE PICK (pinned: eligible arms only, member ΔS > 0, max ΔS; within 0.001 the cheapest): {pick or 'NONE → D654 CLOSED at member level'}** "
               f"(timing f1 {tm['f1']:.2f} ms, r2 {tm['r2']:.3f}, r3 {tm['r3']:.3f}, r4 {tm['r4']:.3f}; headroom {HEADROOM_MS:.3f} ms at the owner's mean ≤ 3,480 / "
               f"rep ≤ 3,550; {headroom_ms(3300.0, 3400.0):.3f} ms at W25's own 3,300 / 3,400)", ""]
    L_ += [f"- {a:7s}: ΔS {v['dS']:+.4f} ({v['n_ge0_S']} / 48 ≥ 0), rt1 {v['d_rt1']:+.4f}, rt2 {v['d_rt2']:+.4f}, "
           f"RL {('%+.4f' % v['dRL']) if v['dRL'] is not None else 'n/a'}; views on {v['gated_frac']:.0%} of the windows"
           + (f"; +{v['extra_ms']:.2f} ms → {v['line_s']:.0f} s on the line ({'ELIGIBLE' if v['eligible'] else 'not eligible'}; "
              f"{'; also inside W25 gates' if v['eligible_at_w25_gates'] else ''}{'' if v['shippable'] else '; NOT shippable (m gate: no W31v path)'}; "
              f"kill {'pass' if v['passes_kill'] else 'fail'})" if tm is not None else "") for a, v in out.items()]
    (OUTd / f"members_d654_{pre}_{seed0}.json").write_text(json.dumps(dict(pre=pre, rtl_pre=rtl_pre if has_rtl else None, seed0=seed0, cell=cell,
                                                                      timing=tm, pick=pick, arms=out, diagnostic_view_agreement=diag), indent=1, default=float))
    L_ += ["", "DIAGNOSTICS (label-using; no ship path):",
           "- the gate's efficiency (share of the full-K member gain): " + ", ".join(
               f"{a} {v['share_of_full']:.0%}" if v.get("share_of_full") is not None else f"{a} n/a" for a, v in out.items() if "share_of_full" in v),
           f"- view agreement as a confidence signal (the identity's raw correctness, {diag['n']} slot x window instances, strangers): AUC margin "
           f"{diag['auc_margin']:.3f}, agreement {diag['auc_agree']:.3f}, -spread {diag['auc_spread']:.3f}, the two combined (leave-one-user-out "
           f"logistic) {diag['auc_combined']:.3f}; WITHIN margin quintiles, -spread {diag['auc_spread_within_margin_q5']:.3f} (0.5 = nothing beyond the margin)"]
    (OUTd / f"RESULT_D654_members_{pre}_{seed0}.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def view_agreement_diag(dg):
    """DIAGNOSTIC: AUC (predicting the identity's raw correctness) of the identity margin, the all-views-agree flag, -spread (mean L1 between a
    view's and the identity's probabilities), the margin + -spread + agree combined (logistic, leave-one-user-out), and -spread within margin
    quintiles (the information BEYOND the margin)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    m, ag, sp, c, us = (np.asarray(dg[k]) for k in ("margin", "agree", "spread", "correct", "user"))
    X = np.column_stack([m, -sp, ag])
    oof = np.empty(len(c))
    for u in np.unique(us):
        h = us == u
        oof[h] = LogisticRegression(max_iter=1000).fit(X[~h], c[~h]).decision_function(X[h])
    qb = np.quantile(m, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(m, qb)
    within = [roc_auc_score(c[strata == b], -sp[strata == b]) for b in range(5) if len(np.unique(c[strata == b])) == 2]
    return dict(n=int(len(c)), acc=float(c.mean()), auc_margin=float(roc_auc_score(c, m)), auc_agree=float(roc_auc_score(c, ag)),
                auc_spread=float(roc_auc_score(c, -sp)), auc_combined=float(roc_auc_score(c, oof)),
                auc_spread_within_margin_q5=float(np.mean(within)), agree_rate=float(ag.mean()))


def cmd_fmspl(newtag, basetag, arm, cell):
    """STAGE 2: setread/<cell>/fmspl_<newtag>/ = fmspl_<basetag>/ with each file's SCORED rows rewritten from the manifest's source member file
    under the arm (F / L / prob through the head); unscored rows (set E's) untouched; never overwrites. The gate: per fold, the scored rows'
    streams; the c gate uses THIS lc9 fold's routes (the fold the lab mixes), the m gate the six slots' v1 blend of this fold's sources."""
    SR = O / "t2_d642" / "setread" / cell
    bdir, out = SR / f"fmspl_{basetag}", SR / f"fmspl_{newtag}"
    assert not out.exists(), f"never overwrite a read's inputs: {out}"
    man = json.loads((bdir / "manifest.json").read_text())
    assert man["slots"] == SLOTS, ("D654 transforms ALL six slots", man["slots"])
    K, q, gs = arm_of(arm)
    out.mkdir(parents=True)
    new_man, folds = [], sorted({e["fold"] for e in man["files"]})
    for f in folds:
        ents = {e["slot"]: e for e in man["files"] if e["fold"] == f}
        assert sorted(ents) == sorted(SLOTS), (f, sorted(ents))
        zb = {s: np.load(bdir / ents[s]["file"]) for s in SLOTS}
        zs = {s: np.load(ents[s]["source"]) for s in SLOTS}
        te = zb[SLOTS[0]]["test_idx"].astype(np.int64)
        src = ents[SLOTS[0]]["scored"]
        m = scored_rows(src, te)
        rows = te[m]
        sels = {}
        for s in SLOTS:
            assert np.array_equal(zb[s]["test_idx"].astype(np.int64), te) and ents[s]["scored"] == src, (f, s)
            pos = {int(r): i for i, r in enumerate(zs[s]["test_idx"].astype(np.int64))}
            sels[s] = np.array([pos[int(r)] for r in rows])
            assert np.array_equal(zb[s]["fm_feat_te"][m], zs[s]["fm_feat_te"][sels[s]]), (f, s, "the base splice != its source")
        fam = np.mean([final_rows(zs[s], sels[s], 1, None)[2] for s in SLOTS], 0)
        use = gate_for(src, rows, arm, cell, fm_fam_prob=fam, fold=f)
        for s in SLOTS:
            F, L, p = final_rows(zs[s], sels[s], K, use)
            new = {key: np.array(zb[s][key], copy=True) for key in zb[s].files}
            new["fm_feat_te"][m] = F.astype(new["fm_feat_te"].dtype)
            new["prob_last"][m] = p.astype(new["prob_last"].dtype)
            new["prob_best"][m] = p.astype(new["prob_best"].dtype)       # the lab never reads prob_best; kept consistent
            if "side_logit_last" in new:
                new["side_logit_last"][m] = L.astype(new["side_logit_last"].dtype)
            res = json.loads(str(zb[s]["res"])) if "res" in zb[s].files else {}
            res["d654"] = dict(arm=arm, K=K, q=q, gate=gs, base=basetag, gated_frac=float(np.mean(use)), n_scored=int(m.sum()))
            new["res"] = np.array(json.dumps(res))
            np.savez(out / ents[s]["file"], **new)
            new_man.append(dict(ents[s], d654_arm=arm, gated_frac=float(np.mean(use))))
    (out / "manifest.json").write_text(json.dumps(dict(man, tag=newtag, d654=dict(arm=arm, base=basetag), files=new_man), indent=1))
    print(f"[d654 viewmix] {out.name}: {len(new_man)} files from fmspl_{basetag} under arm {arm} (K {K}, q {q}, gate {gs}); "
          f"views on {np.mean([e['gated_frac'] for e in new_man]):.0%} of the scored windows", flush=True)


def selftest():
    rng = np.random.default_rng(654)
    n = 400
    st, od = rng.integers(0, 5, n), rng.permutation(n)
    mg = rng.random(n)
    for q in (0.2, 0.3):
        g = causal_gate(mg, st, od, q)
        for s_ in range(5):
            r = np.flatnonzero(st == s_)
            r = r[np.argsort(od[r])]
            assert g[r[0]], "window 0 must be gated"
            assert all(g[r[:t + 1]].sum() <= q * (t + 1) + 1 for t in range(len(r))), "the token bucket"
            cut = r[: len(r) // 2]                         # causal: the gate of a prefix == the full stream's gate on that prefix
            sub = np.zeros(n, bool)
            sub[cut] = True
            g2 = causal_gate(np.where(sub, mg, 9.0), np.where(sub, st, -1), od, q)
            assert np.array_equal(g2[cut], g[cut]), "the gate is not causal"
        assert abs(g.mean() - q) < 0.08, g.mean()
    assert causal_gate(mg, st, od, 1.0).all()
    print(f"1. the gate: window 0 gated, the token bucket holds at every t, causal (prefix-invariant), realised fraction ~ q, q = 1 -> all")
    N, D = 30, 16
    F0, FV = rng.standard_normal((N, D)), rng.standard_normal((3, N, D))
    Wm, bm, Sx, Ws, bs = rng.standard_normal((3, D)), rng.standard_normal(3), rng.standard_normal((N, 5)), rng.standard_normal((3, 5)), rng.standard_normal(3)
    z = dict(fm_feat_te=F0, fm_view_feat=FV, fm_view_shift=np.array([15, -15, 30]), fm_out_w=Wm, fm_out_b=bm, side_te=Sx, side_w=Ws, side_b=bs)

    class Z(dict):
        files = property(lambda self: list(self.keys()))
    z = Z(z)
    sel, use = np.arange(N), rng.random(N) < 0.4
    F1, L1, p1 = final_rows(z, sel, 1, None)
    assert np.array_equal(F1, F0) and np.allclose(p1, softmax(F0 @ Wm.T + bm + Sx @ Ws.T + bs))
    F4, _, _ = final_rows(z, sel, 4, np.ones(N, bool))
    assert np.allclose(F4, (F0 + FV.sum(0)) / 4)
    Fg, Lg, pg = final_rows(z, sel, 2, use)
    assert np.allclose(Fg[use], (F0[use] + FV[0, use]) / 2) and np.array_equal(Fg[~use], F0[~use])
    assert np.allclose(Lg, Fg @ Wm.T + bm) and np.allclose(pg, softmax(Lg + Sx @ Ws.T + bs))
    print("2. final rows: v1 = the identity through the head; f4 = the 4-view mean; gated K = 2 = the view mean on gated rows only; L / prob from F")
    m0 = json.loads((O / "t2_d642/setread/bw21/fmspl_ks/manifest.json").read_text())["files"]
    e = [x for x in m0 if x["slot"] == "s1_stlsd"][0]
    zs = np.load(e["source"])
    te = zs["test_idx"].astype(np.int64)
    sel = np.flatnonzero(scored_rows(e["scored"], te))
    _, _, pv = final_rows(zs, sel, 1, None)
    dv = float(np.abs(pv - zs["prob_last"][sel]).max())
    assert dv < 1e-2, dv
    pc = cnn_prob("bw21", te[sel][:20], [USERS.index(e["scored"].split("_", 1)[1])])
    assert np.allclose(pc.sum(1), 1) and pc.shape == (20, 3)
    print(f"3. a real D642 member: the v1 recompute vs the network's prob_last max |d| {dv:.2g} (bf16; why every arm incl. the base recomputes); "
          f"the CNN route blend loads (fold {e['fold']})")
    tm = dict(f1=30.0, r2=1.42, r3=2.12, r4=2.48)
    ex = {a: extra_ms(a, tm) for a in ALL_ARMS}
    assert ex["v1"] == 0 and abs(ex["f2"] - 12.6) < 1e-9 and abs(ex["g2q30c"] - 0.3 * 0.42 * 30) < 1e-9 and abs(ex["g2q20m"] - 0.2 * 30) < 1e-9
    assert abs(ex["g4q20c"] - 0.2 * 1.48 * 30) < 1e-9 and abs(ex["g4q20m"] - 0.2 * 2.12 * 30) < 1e-9
    rd = np.random.default_rng(1)
    nd = 600
    mm, sp_ = rd.random(nd), rd.random(nd)
    cc = (rd.random(nd) < 0.3 + 0.4 * mm).astype(int)
    dd = view_agreement_diag(dict(margin=list(mm), agree=list((sp_ < 0.5).astype(float)), spread=list(sp_), correct=list(cc),
                                  user=[f"P{i % 8}" for i in range(nd)]))
    assert dd["auc_margin"] > 0.6 and abs(dd["auc_spread"] - 0.5) < 0.1 and abs(dd["auc_spread_within_margin_q5"] - 0.5) < 0.12, dd
    print(f"5. the diagnostic: a margin-only synthetic signal -> AUC margin {dd['auc_margin']:.2f}, -spread {dd['auc_spread']:.2f}, within {dd['auc_spread_within_margin_q5']:.2f}")
    st6 = np.repeat(np.arange(4), 300)
    od6 = np.tile(np.arange(300), 4)
    m6 = rd.random(1200)
    for q in (0.2, 0.3):
        a1, a2 = prob_gate(m6, st6, od6, q), prob_gate(m6, st6, od6, q)
        assert np.array_equal(a1, a2), "the prob gate is not reproducible"
        for s_ in range(4):
            r = np.flatnonzero(st6 == s_)
            assert all(a1[r[:t + 1]].sum() <= q * (t + 1) + 1 for t in range(len(r))), "the prob gate's bucket"
        assert abs(a1.mean() - q) < 0.06 and m6[a1].mean() < m6[~a1].mean(), (a1.mean(), "gated windows must be the less confident")
    print(f"6. the prob gate: reproducible (Generator {PROB_SEED} per stream), the bucket holds, fraction ~ q, the gated windows are the low-margin ones")
    assert abs(headroom_ms() - (3480 - 3242.6) / 62) < 1e-12 and abs(headroom_ms(3300.0, 3400.0) - (3300 - 3242.6) / 62) < 1e-12
    print("4. the eligibility arithmetic (f1 30, r2 1.42): " + ", ".join(f"{a} +{v:.2f}" for a, v in ex.items()))
    print("OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    kv = dict(zip(a[1::2], a[2::2])) if a and a[0] == "member" else {}
    if a[0] == "selftest":
        selftest()
    elif a[0] == "member":
        tm_ = {k: float(kv[f"--{k}"]) for k in ("f1", "r2", "r3", "r4")} if "--f1" in kv else None
        cmd_member(kv["--pre"], int(kv["--seed0"]), kv.get("--cell", "bw21"), kv.get("--rtl-pre"), tm_)
    elif a[0] == "fmspl":
        cmd_fmspl(a[1], a[2], a[3], a[4])
    else:
        raise SystemExit(__doc__)

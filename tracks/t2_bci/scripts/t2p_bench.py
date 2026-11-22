#!/usr/bin/env python
"""Track 2 PROTEUS bench: every (protocol, fold, model, align, seed) task -> test probabilities + curves.

Data = the kit-contract export (``t2p_export.py``): (N, 41, 480) uV windows, index.csv with subject / session /
interface / stream order. Scoring = the sealed metric (balanced accuracy per subject x session x interface cell,
averaged over cells; ``ni26.metrics``), computed by ``t2p_report.py`` from the saved probabilities.

Protocols (the sealed phase mixes later sessions of represented participants with NEW participants, FAQ 2026-10-13):
  loso   new participant: test = every session of subject s; inner val = the next subject in a fixed rotation
         (epoch choice only, "best"); train = the remaining subjects. One fold per subject.
  xsess  represented participant, later sessions: for each subject with >= 2 sessions, test = its sessions after
         the first; train = its FIRST session + every other subject (val = rotation subject as in loso).
  kit    the NeuralBench / Codabench warm-up split (train / val / test subjects of SklearnSplit seed 33).

Alignment (``ni26.stream``) is causal per stream for EVERY window, training sessions included (same recursion as
``predict``); the EA prior (if n0 > 0) is the trace-normalised mean covariance of the fold's TRAINING windows only.

Usage:
  python t2p_bench.py --export <dir> --out <dir> --protocols loso,xsess,kit --models eegnet,shallow \
      --aligns none,wz,sz,ea --seeds 0,1,2 [--shard i --nshards K]
"""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ni26 import stream  # noqa: E402

DEEP = ("eegnet", "shallow", "deep4", "conformer", "atcnet", "eegnex", "ifnet", "fbcnet", "eegtcnet")
CLASSICAL = ("fbts", "lvlda")


# ----------------------------------------------------------------------------------------------- data / folds
def softclip(X, mode):
    """T2-D306 (the 2025 report's Challenge 2 preprocessing): per window x <- tanh(x / q99(|x|)), q99 per channel over time ('ch') or over
    all channels and time ('glob'); applied to every window before alignment / augmentation (the copies' routing fingerprints stay raw)."""
    X = np.asarray(X, dtype=np.float32)
    q = np.quantile(np.abs(X), 0.99, axis=(2,) if mode == "ch" else (1, 2), keepdims=True)
    return np.tanh(X / np.maximum(q, 1e-6)).astype(np.float32)


def vs_bank(XA, rows, idx, shrink=1e-3):
    """T2-D308: same-user drift maps from the training rows only. Per (subject, session) the covariance of its demeaned windows (+ shrink x the
    mean eigenvalue); the Bures map T_ab (T C_a T = C_b, as ni26.train.rc_bank) for every ordered pair of sessions of the same user, and between
    the first and second half (by `order`) of every session, both directions -> (K, C, C) float32."""
    import torch
    rows = np.asarray(rows)
    sub, ses = idx["subject"].to_numpy()[rows], idx["session"].astype(str).to_numpy()[rows]
    ordr = idx["order"].to_numpy()[rows]

    def cov(r):
        x = np.asarray(XA[rows[r]], dtype=np.float64)
        x = x - x.mean(-1, keepdims=True)
        C = np.einsum("ncl,ndl->cd", x, x) / (x.shape[0] * x.shape[-1])
        return C + shrink * np.trace(C) / len(C) * np.eye(len(C))

    def sq(A, pw):
        w, V = np.linalg.eigh(A)
        return (V * np.clip(w, 1e-12, None) ** pw) @ V.T

    def ot(Ca, Cb):
        h, hi = sq(Ca, 0.5), sq(Ca, -0.5)
        return hi @ sq(h @ Cb @ h, 0.5) @ hi

    covs, M = {}, []
    for key in sorted(set(zip(sub, ses))):
        r = np.flatnonzero((sub == key[0]) & (ses == key[1]))
        covs[key] = cov(r)
        r = r[np.argsort(ordr[r], kind="stable")]
        h = len(r) // 2
        if h >= 10:
            A, Bm = cov(r[:h]), cov(r[h:])
            M += [ot(A, Bm), ot(Bm, A)]
    for (sa, ea), Ca in covs.items():
        for (sb, eb), Cb in covs.items():
            if sa == sb and ea != eb:
                M.append(ot(Ca, Cb))
    return torch.as_tensor(np.stack(M), dtype=torch.float32)


def load_export(path):
    path = Path(path)
    X = np.load(path / "X.npy", mmap_mode="r")
    idx = pd.read_csv(path / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]           # "Dreyer2026Proteus/P01" -> "P01" (file-name safe)
    idx["stream"] = idx["subject"] + "|" + idx["session"]
    meta = json.loads((path / "meta.json").read_text())
    return X, idx, meta


def long_check(Lpath, X, idx, meta):
    """T2-D486: the long export (D473's, -3 .. +8 s re cue) row-aligned with the contract export, and its contract slice equal to the contract
    windows BITWISE. Returns (Lraw memmap, offset of the contract window in samples, the long window's start re cue in s)."""
    XL, iL, mL = load_export(Lpath)
    assert float(mL["sfreq"]) == float(meta["sfreq"]) and list(mL["ch_names"]) == list(meta["ch_names"]) and len(iL) == len(idx), "long export: rate / montage / rows"
    for c in ("subject", "session", "stream", "order", "y"):
        assert (iL[c].astype(str).to_numpy() == idx[c].astype(str).to_numpy()).all(), f"long export: column {c} not row-aligned"
    ls, cs, fs = float(mL["window_start"]), float(meta["window_start"]), float(meta["sfreq"])
    off = int(round((cs - ls) * fs))
    T_ = X.shape[-1]
    for a in range(0, len(idx), 256):
        assert np.array_equal(np.asarray(XL[a:a + 256, :, off:off + T_]), np.asarray(X[a:a + 256])), f"long export: contract slice differs (rows {a}+)"
    return XL, off, ls


def long_aligned(X, XA, XL, rows, tol=1e-3):
    """T2-D486: each row's alignment operator M (C x C) recovered from its contract window (XA[r] = M (X[r] - mean_t X[r]); every mode of
    stream.align_stream is such a per-window linear map of the demeaned window), applied to the whole long window: M XL[r] (not demeaned: the
    crop is demeaned after cropping, which commutes with M). Asserts the recovered M reproduces XA[r] (relative residual <= tol)."""
    out = np.empty((len(rows), XL.shape[1], XL.shape[2]), np.float32)
    worst = 0.0
    for k, r in enumerate(rows):
        x = np.asarray(X[r], np.float64)
        x = x - x.mean(-1, keepdims=True)
        a = np.asarray(XA[r], np.float64)
        Mt, *_ = np.linalg.lstsq(x.T, a.T, rcond=None)
        res = float(np.abs(x.T @ Mt - a.T).max() / max(np.abs(a).max(), 1e-12))
        worst = max(worst, res)
        out[k] = (Mt.T @ np.asarray(XL[r], np.float64)).astype(np.float32)
    assert worst <= tol, f"T2-D486: the recovered alignment operator misses its contract window (relative residual {worst:.2e})"
    return out, worst


LC_BASE = {"lc3": (0, 1, 3), "lc5": (0, 1, 2, 4, 7), "lc7": (1, 3, 4, 6, 7, 8, 9), "lc9": (1, 2, 3, 4, 5, 6, 7, 8, 9)}   # T2-D410


def folds(idx, protocol):
    """-> list of (fold_name, train_mask, val_mask, test_mask) boolean arrays over idx rows."""
    subs = sorted(idx["subject"].unique())
    nxt = {s: subs[(i + 1) % len(subs)] for i, s in enumerate(subs)}
    out = []
    if protocol.split("_")[0] in LC_BASE and protocol.count("_") == 1 and protocol.split("_")[1].isdigit():
        # T2-D498: ONE fold of a learning-curve protocol (lc9_<f>), so per-fold tools (t2p_fpcalib blend) can take it as a protocol
        one = [f_ for f_ in folds(idx, protocol.split("_")[0]) if f_[0] == protocol]
        assert len(one) == 1, protocol
        return one
    if protocol in ("ss8", "rt", "rtl") or protocol.startswith(("ss8_", "rt1_", "rt2_", "rtl_")):
        # T2-D630 (the pooled-training composition): every fold trains on EVERY session of its training users (the shipped composition, t2p_final
        # --train all) minus the scored part; no val. ss8_<u> = a single-session user u held out (test = all u's windows; the 8 strangers);
        # rt1_<X> = X in {P04, P10} trains on session 1 only, test = X's sessions 2-3; rt2_<X> = X trains on sessions 1-2, test = session 3.
        # 'ss8' = the 8 ss8_<u> folds, 'rt' = rt1_P04, rt1_P10, rt2_P04, rt2_P10.
        sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
        ses = idx["session"].astype(str).to_numpy()
        none = np.zeros(len(idx), bool)
        nses = {u: len(set(ses[sub == u])) for u in np.unique(sub)}
        if protocol == "ss8":
            return [f_ for u in sorted(u for u, n in nses.items() if n == 1) for f_ in folds(idx, f"ss8_{u}")]
        if protocol == "rt":
            return [f_ for k in ("rt1", "rt2") for x in ("P04", "P10") for f_ in folds(idx, f"{k}_{x}")]
        if protocol == "rtl":                      # T2-D651 (Track3): the 8 pseudo-returning folds rtl_<u>
            return [f_ for u in sorted(u for u, n in nses.items() if n == 1) for f_ in folds(idx, f"rtl_{u}")]
        kind, u = protocol.split("_", 1)
        assert u in nses, protocol
        if kind == "ss8":
            assert nses[u] == 1, (protocol, nses[u])
            te = sub == u
            trn = ~te
        elif kind == "rtl":
            # T2-D651 (Track3; a DIAGNOSTIC tie-breaker, NEVER a sealed-split validation: it knowingly trains on the scored session's early
            # runs, like sd46): single-session user u stays SEEN through its early runs; test = u's LAST TWO runs (by run number)
            assert nses[u] == 1, (protocol, nses[u])
            run_ = idx["run"].to_numpy()
            runs_ = np.unique(run_[sub == u])
            assert len(runs_) >= 4, (protocol, runs_.tolist())
            te = (sub == u) & np.isin(run_, runs_[-2:])
            trn = ~te
        else:
            assert u in ("P04", "P10") and nses[u] == 3, protocol
            te = (sub == u) & (np.isin(ses, ["2", "3"]) if kind == "rt1" else (ses == "3"))
            trn = ~te
        assert not (trn & te).any() and te.any()
        return [(protocol, trn, none, te)]
    if protocol == "sd46":
        # T2-D499: the same-day returning panel: session 1 of every user, runs 1-4 train, runs 5-6 test (session distance 0); no val
        s1m = (idx["session"].astype(str) == "1").to_numpy()
        run = idx["run"].to_numpy()
        out.append(("sd46", s1m & (run <= 4), np.zeros(len(idx), bool), s1m & (run >= 5)))
        return out
    if protocol == "loso":
        for s in subs:
            te = (idx["subject"] == s).to_numpy()
            va = (idx["subject"] == nxt[s]).to_numpy()
            out.append((f"loso_{s}", ~te & ~va, va, te))
    elif protocol == "xsess":
        for s in subs:
            sess = sorted(idx.loc[idx["subject"] == s, "session"].unique())
            if len(sess) < 2:
                continue
            own = (idx["subject"] == s).to_numpy()
            first = own & (idx["session"] == sess[0]).to_numpy()
            te = own & ~first
            v = nxt[s]
            va = (idx["subject"] == v).to_numpy()
            out.append((f"xsess_{s}", (~own | first) & ~va, va, te))
    elif protocol == "kit":
        out.append(("kit", (idx["split"] == "train").to_numpy(), (idx["split"] == "val").to_numpy(),
                    (idx["split"] == "test").to_numpy()))
    elif protocol == "kitp10":
        # seen-user later-session DIAGNOSTIC (user-approved 2026-10-14; never a selector): the sealed set holds later
        # sessions of the public users, the kit test does not. P10 sessions 2-3 leave training and become the test;
        # P10 session 1 stays in training (the user is "seen"); val = the kit val (P05 P08)
        late = (idx["subject"].astype(str).str.endswith("P10") & idx["session"].astype(str).isin(["2", "3"])).to_numpy()
        assert late.sum() > 0, "kitp10: no P10 session 2/3 windows"
        out.append(("kitp10", (idx["split"] == "train").to_numpy() & ~late, (idx["split"] == "val").to_numpy(), late))
    elif protocol == "seen1":
        # T2-D111 seen-user protocol (the sealed half the kit cannot measure: LATER sessions of represented users):
        # train = session 1 of P01 P02 P03 P04 P06 P07 P10, val = the kit val (P05 P08), test = P04 + P10 sessions 2-3
        # (seen) and P09 (unseen)
        sub, ses = idx["subject"].astype(str).str.split("/").str[-1], idx["session"].astype(str)
        seen = sub.isin(["P04", "P10"]).to_numpy()
        late = seen & ses.isin(["2", "3"]).to_numpy()
        va = (idx["split"] == "val").to_numpy()
        te = late | (sub == "P09").to_numpy()
        trn = (ses == "1").to_numpy() & ~va & ~te
        assert late.sum() > 0 and not (trn & te).any() and set(sub[trn]) == {"P01", "P02", "P03", "P04", "P06", "P07", "P10"}
        out.append(("seen1", trn, va, te))
    elif protocol == "l2o":
        # TSF cross-fitting (D87): for the loso outer fold k, every other subject j gets out-of-fold backbone outputs
        # from a backbone trained WITHOUT k and j (8 subjects, = the loso backbone's size; no val, `last`)
        none = np.zeros(len(idx), bool)
        for k in subs:
            for j in subs:
                if j != k:
                    te = (idx["subject"] == j).to_numpy()
                    out.append((f"l2o_{k}_{j}", ~te & (idx["subject"] != k).to_numpy(), none, te))
    elif protocol == "kit2":
        # the same for the warm-up mirror: kit2_outer = backbone on train + val (8 subjects) -> test P04 + P09;
        # kit2_<j> = backbone on (train + val) minus j -> j's out-of-fold outputs
        none = np.zeros(len(idx), bool)
        trv = idx["split"].isin(["train", "val"]).to_numpy()
        out.append(("kit2_outer", trv, none, (idx["split"] == "test").to_numpy()))
        for j in sorted(idx.loc[trv, "subject"].unique()):
            te = (idx["subject"] == j).to_numpy()
            out.append((f"kit2_{j}", trv & ~te, none, te))
    elif protocol in LC_BASE:
        # T2-D410 learning curve over the NUMBER OF TRAINING USERS (the user's question via the manager, 2026-11-07): K = int(protocol[2:]) of
        # the 10 public users train (their session 1 only), 10 cyclic folds per K (fold i trains users (b + i) mod 10 for b in LC_BASE[K], users
        # sorted), so every user trains in exactly K of the 10 folds; test = every window of the other 10 - K users (NEW) + sessions 2-3 of a
        # TRAINED P04 / P10 (RETURNING); no val (`last`)
        none = np.zeros(len(idx), bool)
        sub, ses = idx["subject"].astype(str).str.split("/").str[-1], idx["session"].astype(str)
        short = [s.split("/")[-1] for s in subs]
        assert len(short) == 10, short
        for i in range(10):
            tu = {short[(b + i) % 10] for b in LC_BASE[protocol]}
            intr = sub.isin(tu).to_numpy()
            trn = intr & (ses == "1").to_numpy()
            te = ~intr | (intr & ses.isin(["2", "3"]).to_numpy())
            assert not (trn & te).any() and set(sub[trn]) == tu
            out.append((f"{protocol}_{i}", trn, none, te))
    elif protocol in ("kitoof", "seen1oof"):
        # T2-D253: out-of-fold outputs on the TRAINING windows of kit / seen1: per training user j, a net trained on the
        # protocol's other training users predicts j's training windows (no val; `last`); oof_pcnn reads them back
        _, trn, _, _ = folds(idx, protocol[:-3])[0]
        none = np.zeros(len(idx), bool)
        sub = idx["subject"].astype(str).str.split("/").str[-1]
        for j in sorted(sub[trn].unique()):
            own = trn & (sub == j).to_numpy()
            out.append((f"{protocol}_{j}", trn & ~own, none, own))
    elif protocol.startswith("lc9_") and protocol.endswith("oof") and protocol[4:-3].isdigit() and protocol.count("_") == 1:
        # T2-D530: NESTED out-of-fold teachers on lc9 fold f ('lc9_<f>oof'): per training user j of that fold, a net trained on the
        # fold's training rows minus j's predicts j's training windows (no val; `last`); oof_pcnn(dir, 'lc9_<f>', ...) reads them back.
        # No teacher sees fold f's stranger / test rows or the user it labels.
        ((_, trn, _, tem),) = folds(idx, protocol[:-3])
        none = np.zeros(len(idx), bool)
        sub = idx["subject"].astype(str).str.split("/").str[-1]
        tu, stranger = set(sub[trn]), set(sub) - set(sub[trn])
        assert len(tu) == 9 and len(stranger) == 1, (protocol, sorted(tu))
        for j in sorted(tu):
            own = trn & (sub == j).to_numpy()
            tr_ = trn & ~own
            assert set(sub[tr_]) == tu - {j} and not set(sub[tr_]) & stranger, (protocol, j)
            assert not (tr_ & tem).any() and not (tr_ & own).any(), (protocol, j)
            out.append((f"{protocol}_{j}", tr_, none, own))
    else:
        raise ValueError(protocol)
    return out


def aligned_all(X, idx, mode, train_mask, n0, shrink, scale):
    """Causally align every stream (in kit order) -> float32 array aligned with idx rows."""
    prior = None
    if mode in ("ea", "fbea") and n0 > 0:
        tr = np.flatnonzero(train_mask)
        sel = tr[np.linspace(0, len(tr) - 1, min(len(tr), 3000)).astype(int)]
        xs = np.asarray(X[sel], dtype=np.float64)
        if mode == "ea":
            prior = stream.trace_normalised_mean(stream.window_covs(xs - xs.mean(-1, keepdims=True)))
        else:                                                          # T2-D424: one EA prior per band
            prior = stream.fb_prior(xs - xs.mean(-1, keepdims=True))
    out = np.empty(X.shape, dtype=np.float32)
    for _, g in idx.groupby("stream", sort=False):
        rows = g.sort_values("order")["i"].to_numpy()
        out[rows] = stream.align_stream(np.asarray(X[rows]), mode, scale=scale, prior=prior, n0=n0, shrink=shrink)
    return out


def bn_stream_probs(net, XA, idx, rows, momentum, dev):
    """T2-D100 causal within-session BatchNorm-statistics adaptation, streamed like Codabench: per stream (participant x
    session, kit order) a fresh copy of the trained net; for each window, ONE forward with only the BN layers in train
    mode updates their running mean / var from that window (momentum m, no gradient), then an eval forward predicts it
    with the updated statistics. Label-free and causal (window t sees windows <= t of its own session)."""
    import copy
    import torch
    import torch.nn.functional as F
    out = np.zeros((len(rows), 3), np.float32)
    if not len(rows):
        return out
    pos = {int(r): k for k, r in enumerate(rows)}
    src = copy.deepcopy(net).eval()
    sub = idx.iloc[rows]
    with torch.no_grad():
        for _, g in sub.groupby("stream", sort=False):
            mm = copy.deepcopy(src)
            bns = [b for b in mm.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
            for b in bns:
                b.momentum = float(momentum)
            for r in g.sort_values("order")["i"].to_numpy():
                x = torch.as_tensor(np.asarray(XA[int(r)])[None], device=dev)
                for b in bns:
                    b.train()
                mm(x)
                for b in bns:
                    b.eval()
                out[pos[int(r)]] = F.softmax(mm(x).float(), 1)[0].cpu().numpy()
    return out


def protocol_test_streams(idx, protocol):
    """(subject, session) of every TEST window of `protocol`, over all its folds (review T2 P2e: the CPT-pool check)."""
    sub, ses = idx["subject"].astype(str).str.split("/").str[-1].to_numpy(), idx["session"].astype(str).to_numpy()
    out = set()
    for _, _, _, te in folds(idx, protocol):
        out |= set(zip(sub[te], ses[te]))
    return out


def _sqrt_spd(R):
    w, V = np.linalg.eigh(R)
    return (V * np.sqrt(np.maximum(w, w.max() * 1e-8))) @ V.T


def recolour_own(XA, idx, trm, tem, fp, n0=10.0, shrink=0.05):
    """T2-D347 (outlier bet B2): the TEST rows only, causally per stream. Window t is mapped from the stream's running mean
    covariance R_t onto the mean training-window covariance C_u of the user u the package's fingerprint router names at t
    (t2p_fpcalib.fit_space on this fold's training fingerprints + stream_route_every, the package's route):
    x_t' = C_u^1/2 R_t^-1/2 x_t, R_t = (n0 C_u + sum_{i<=t} C_i) / (n0 + t) (C_u as the warm start). No route yet = unchanged.
    C_u and C_i are taken after the member's own alignment (XA). Train / val rows are returned unchanged."""
    import t2p_fpcalib as FC
    subj = idx["subject"].to_numpy()
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    proj, mu, users = FC.fit_space(fp[tr], subj[tr])
    Cu, Su = {}, {}
    for u in users:
        xs = np.asarray(XA[tr[subj[tr] == u]], np.float64)
        Cu[u] = stream.window_covs(xs - xs.mean(-1, keepdims=True)).mean(0)
        Su[u] = _sqrt_spd(Cu[u])
    out = np.array(XA, dtype=np.float32, copy=True)
    for g in stream_groups(idx, te):
        k = FC.stream_route_every(proj(fp[g]), mu, 1)
        x = np.asarray(XA[g], np.float64)
        x = x - x.mean(-1, keepdims=True)
        cum = np.cumsum(stream.window_covs(x), 0)
        for t in range(len(g)):
            if k[t] < 0:
                continue
            u = users[k[t]]
            R = (n0 * Cu[u] + cum[t]) / (n0 + t + 1.0)
            out[g[t]] = (Su[u] @ stream._inv_sqrt(R, shrink) @ x[t]).astype(np.float32)
    return out


def stream_groups(idx, rows):
    """rows (idx positions) -> list of row arrays, one per stream (participant x session), each in kit order."""
    sub = idx.iloc[rows]
    return [g.sort_values("order")["i"].to_numpy() for _, g in sub.groupby("stream", sort=False)]


def pl_stream_probs(net, XA, idx, rows, variant, dev):
    """T2-D113 causal self-training inside predict, streamed like Codabench: per stream a fresh copy of the trained net;
    window t is predicted by the current weights, then ONE plain-SGD step on window t (BN layers stay in eval mode =
    running statistics; every trainable weight adapts). variant = (kind 'ent' | 'pl', lr, threshold); T2-D405: kind 'im' = Adam on mean H(p_i) - H(mean p) over the last thr windows, from window IM_MIN on."""
    import copy
    import torch
    import torch.nn.functional as F
    kind, lr, thr = variant
    if kind == "anc":
        return anc_stream_probs(net, XA, idx, rows, lr, int(thr), dev)
    out = np.zeros((len(rows), 3), np.float32)
    pos = {int(r): k for k, r in enumerate(rows)}
    src = copy.deepcopy(net).eval()
    for g in stream_groups(idx, rows):
        mm = copy.deepcopy(src).eval()
        params = [p for p in mm.parameters() if p.requires_grad]
        opt = torch.optim.SGD(params, lr=lr) if kind not in ("im", "eata") else torch.optim.Adam(params, lr=lr)
        buf, hbuf = [], []
        for r in g:
            x = torch.as_tensor(np.asarray(XA[int(r)])[None], device=dev)
            lo = mm(x).float()
            p = F.softmax(lo, 1)
            out[pos[int(r)]] = p[0].detach().cpu().numpy()
            if kind == "eata":              # T2-D420 (backlog 104): the IM step over the buffer windows that were RELIABLE when predicted (H < EATA_E0)
                buf = (buf + [x])[-int(thr):]
                hbuf = (hbuf + [float(-(p * F.log_softmax(lo, 1)).sum().detach())])[-int(thr):]
                rel = [b for b, h in zip(buf, hbuf) if h < EATA_E0]
                if len(buf) < IM_MIN or len(rel) < EATA_MIN:
                    continue
                lb = mm(torch.cat(rel)).float()
                pb = F.softmax(lb, 1)
                pm = pb.mean(0)
                loss = -(pb * F.log_softmax(lb, 1)).sum(1).mean() + (pm * torch.log(pm.clamp_min(1e-8))).sum()
            elif kind == "im":              # T2-D405: T-TIME-style information maximisation over the causal buffer (thr = its size)
                buf = (buf + [x])[-int(thr):]
                if len(buf) < IM_MIN:
                    continue
                lb = mm(torch.cat(buf)).float()
                pb = F.softmax(lb, 1)
                pm = pb.mean(0)
                loss = -(pb * F.log_softmax(lb, 1)).sum(1).mean() + (pm * torch.log(pm.clamp_min(1e-8))).sum()   # mean H(p_i) - H(mean p)
            elif kind == "ent":
                loss = -(p * F.log_softmax(lo, 1)).sum()
            elif float(p.max()) >= thr:
                loss = F.cross_entropy(lo, p.argmax(1).detach())
            else:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return out


def anc_stream_probs(net, XA, idx, rows, lr, nbuf, dev):
    """T2-D420 ANCHORED multi-step TTT, streamed like Codabench (per stream a fresh student AND teacher = copies of the trained net;
    BN in eval mode). Window t is predicted by the TEACHER (an EMA of the student). From window IM_MIN on, over the causal buffer of
    the last nbuf windows (incl. t): the teacher's probabilities are Sinkhorn-balanced to uniform class marginals (ANC['sk'] iterations)
    = soft pseudo-labels q; ANC['steps'] Adam steps on CE(student(buffer), q) + ANC['lam'] * sum (theta - theta_0)^2 (an L2 pull to the
    source weights); then teacher <- ANC['ema'] teacher + (1 - ANC['ema']) student. COLLAPSE GUARD: if one class takes more than
    ANC['guard'] of the buffer's teacher argmaxes, student, teacher and optimiser are reset to the source weights."""
    import copy
    import torch
    import torch.nn.functional as F
    out = np.zeros((len(rows), 3), np.float32)
    pos = {int(r): k for k, r in enumerate(rows)}
    src = copy.deepcopy(net).eval()
    th0 = [p.detach().clone() for p in src.parameters() if p.requires_grad]

    def fresh():
        st, te = copy.deepcopy(src).eval(), copy.deepcopy(src).eval()
        for p in te.parameters():
            p.requires_grad_(False)
        ps = [p for p in st.parameters() if p.requires_grad]
        return st, te, ps, torch.optim.Adam(ps, lr=lr)
    for g in stream_groups(idx, rows):
        st, te, ps, opt = fresh()
        buf = []
        for r in g:
            x = torch.as_tensor(np.asarray(XA[int(r)])[None], device=dev)
            with torch.no_grad():
                out[pos[int(r)]] = F.softmax(te(x).float(), 1)[0].cpu().numpy()
            buf = (buf + [x])[-nbuf:]
            if len(buf) < IM_MIN:
                continue
            xb = torch.cat(buf)
            with torch.no_grad():
                pt = F.softmax(te(xb).float(), 1)
                if float(torch.bincount(pt.argmax(1), minlength=pt.shape[1]).max()) > ANC["guard"] * len(xb):
                    st, te, ps, opt = fresh()           # collapse guard: back to the source weights
                    continue
                q = pt.clone()
                for _ in range(ANC["sk"]):              # Sinkhorn: columns -> n / C each, rows -> 1
                    q = q / q.sum(0, keepdim=True) * (len(q) / q.shape[1])
                    q = q / q.sum(1, keepdim=True)
            for _ in range(ANC["steps"]):
                loss = -(q * F.log_softmax(st(xb).float(), 1)).sum(1).mean() + ANC["lam"] * sum(((p - p0) ** 2).sum() for p, p0 in zip(ps, th0))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                for pt_, ps_ in zip(te.parameters(), st.parameters()):
                    pt_.mul_(ANC["ema"]).add_(ps_.detach(), alpha=1 - ANC["ema"])
    return out


TTA_PL = {"V1": ("ent", 1e-4, 0.0), "V2": ("ent", 1e-3, 0.0), "V3": ("pl", 1e-4, 0.6), "V4": ("pl", 1e-3, 0.6),
          "IM1": ("im", 1e-4, 16), "IM2": ("im", 1e-5, 16),     # T2-D405: Adam on mean H(p_i) - H(mean p) over the last 16 windows
          "EA1": ("eata", 1e-5, 16), "EA2": ("eata", 3e-5, 16),  # T2-D420: IM over the reliable buffer windows only (EATA's E0)
          "AT1": ("anc", 1e-5, 16), "AT2": ("anc", 3e-5, 16)}    # T2-D420: anchored mean-teacher TTT on Sinkhorn-balanced pseudo-labels
IM_MIN = 8                                                         # T2-D405: no update before a stream has 8 windows
EATA_E0 = 0.4 * float(np.log(3))                                   # T2-D420: EATA's reliability threshold 0.4 ln C (Niu et al. 2022), fixed a priori
EATA_MIN = 4                                                       # T2-D420: no EATA step on fewer than 4 reliable buffer windows
ANC = dict(steps=2, ema=0.99, lam=1.0, sk=3, guard=0.8)           # T2-D420: fixed a priori (2 steps / window, EMA 0.99, L2 1.0, 3 SK its, reset at > 80 % one class)


def mae_tta_probs(net, mae, XA, idx, rows, lr, dev, ratio=0.75, seed=0):
    """T2-D125 test-time MAE CPT of an ST member, streamed like Codabench: per stream a fresh copy of the fine-tuned
    classifier; window t is predicted by the current weights, then ONE masked-reconstruction step on window t (the CPT
    input: per-window z of the 4 s window resampled to 512 samples; mask ratio 0.75; the CPT MAE's decoder, frozen;
    AdamW lr, LayerNorms and the classifier head frozen, encoder weights adapt). Returns (probs, s/window plain, s/window TTA)."""
    import copy
    import torch
    import torch.nn.functional as F
    import t2p_cpt
    import t2p_fm
    S_ = t2p_fm.t1.STEEG
    prep_z = t2p_fm.STEEG4.prep.__get__(type("P", (), dict(FS=S_.FS, calib="zwin", CLIP_GAIN=S_.CLIP_GAIN))())
    out = np.zeros((len(rows), 3), np.float32)
    pos = {int(r): k for k, r in enumerate(rows)}
    src = copy.deepcopy(net).eval()
    mae = mae.eval()
    for q in mae.parameters():
        q.requires_grad_(False)
    chan = src.enc.chan_idx.to(dev)
    t_plain = t_all = 0.0
    n = 0
    for g in stream_groups(idx, rows):
        mm = copy.deepcopy(src).eval()
        params = [q for n_, q in mm.enc.vit.named_parameters() if q.requires_grad and "norm" not in n_]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for r in g:
            x = torch.as_tensor(np.asarray(XA[int(r)])[None], device=dev)
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                out[pos[int(r)]] = F.softmax(mm(x).float(), 1)[0].cpu().numpy()
            t1_ = time.time()
            xz = prep_z(x)
            noise = torch.rand(1, (xz.shape[-1] // 16) * xz.shape[1], generator=gen).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                loss = t2p_cpt.mae_loss_split(mm.enc.vit, mae, xz, chan, ratio, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if x.is_cuda:
                torch.cuda.synchronize()
            t_plain += t1_ - t0
            t_all += time.time() - t0
            n += 1
    return out, t_plain / max(n, 1), t_all / max(n, 1)


def mae_tta_ln_probs(net, mae, XA, idx, rows, lr, dev, arm, ratio=0.75, seed=0, prefix=30, buf=8, restore=0.99):
    """T2-D283 (the TTT panel's Step 1, lead/PANEL_2026-10-27_TTT_CONCLUSION.md): test-time MAE of an ST member on its encoder's
    LayerNorm affine ONLY, streamed like Codabench: per stream a fresh copy of the fine-tuned classifier (reset per stream); every
    window is predicted with the current weights BEFORE any step that uses it; masks from a generator re-seeded at every stream
    start (so a stream's output does not depend on the streams before it); the CPT
    MAE's decoder frozen; AdamW(lr, wd 0); mask ratio 0.75.
      arm P: after window `prefix` is predicted, ONE step on the `prefix` buffered windows, then frozen for the rest of the stream;
      arm K: after every `buf`-th window, ONE step on the last `buf` windows, then theta <- theta0 + restore (theta - theta0).
    Returns (probs, s/window plain, s/window with TTA, per-stream relative LN drift ||theta - theta0|| / ||theta0||)."""
    import copy
    import torch
    import torch.nn.functional as F
    import t2p_cpt
    import t2p_fm
    assert arm in ("P", "K"), arm
    S_ = t2p_fm.t1.STEEG
    prep_z = t2p_fm.STEEG4.prep.__get__(type("P", (), dict(FS=S_.FS, calib="zwin", CLIP_GAIN=S_.CLIP_GAIN))())
    out = np.zeros((len(rows), 3), np.float32)
    pos = {int(r): k for k, r in enumerate(rows)}
    src = copy.deepcopy(net).eval()
    mae = mae.eval()
    for q in mae.parameters():
        q.requires_grad_(False)
    chan = src.enc.chan_idx.to(dev)
    t_plain = t_all = 0.0
    n, drift = 0, []
    for g in stream_groups(idx, rows):
        mm = copy.deepcopy(src).eval()
        for q in mm.parameters():
            q.requires_grad_(False)
        ln = [q for n_, q in mm.enc.vit.named_parameters() if "norm" in n_]
        assert ln, "no LayerNorm parameters found in the ST encoder"
        for q in ln:
            q.requires_grad_(True)
        q0 = [q.detach().clone() for q in ln]
        opt = torch.optim.AdamW(ln, lr=lr, weight_decay=0.0)
        gen = torch.Generator(device="cpu").manual_seed(seed)        # the same seed at every stream start: a template can do this
        bx = []

        def step():
            xz = prep_z(torch.cat(bx, 0))
            noise = torch.rand(xz.shape[0], (xz.shape[-1] // 16) * xz.shape[1], generator=gen).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xz.is_cuda):
                loss = t2p_cpt.mae_loss_split(mm.enc.vit, mae, xz, chan, ratio, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        for t, r in enumerate(g):
            x = torch.as_tensor(np.asarray(XA[int(r)])[None], device=dev)
            if x.is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                out[pos[int(r)]] = F.softmax(mm(x).float(), 1)[0].cpu().numpy()
            t1_ = time.time()
            if arm == "P" and t < prefix:
                bx.append(x)
                if t + 1 == prefix:
                    step()
                    bx = []
            elif arm == "K":
                bx.append(x)
                if len(bx) == buf:
                    step()
                    with torch.no_grad():
                        for q, a in zip(ln, q0):
                            q.copy_(a + restore * (q - a))
                    bx = []
            if x.is_cuda:
                torch.cuda.synchronize()
            t_plain += t1_ - t0
            t_all += time.time() - t0
            n += 1
        with torch.no_grad():
            num = sum(float(((q.float() - a.float()) ** 2).sum()) for q, a in zip(ln, q0))
            den = sum(float((a.float() ** 2).sum()) for a in q0)
        drift.append((num / max(den, 1e-30)) ** 0.5)
    return out, t_plain / max(n, 1), t_all / max(n, 1), np.array(drift)


def gs_probs(net, XA, idx, rows, dev):
    """T2-D111 (ii-gs): the soft user-expert head with its gate replaced by the causal running mean of the per-window gate
    posteriors over windows 1..t of the stream."""
    import torch
    import torch.nn.functional as F
    out = np.zeros((len(rows), 3), np.float32)
    pos = {int(r): k for k, r in enumerate(rows)}
    net.eval()
    with torch.no_grad():
        for g in stream_groups(idx, rows):
            x = torch.as_tensor(np.asarray(XA[g]), device=dev)
            u = F.softmax(net.forward_all(x)[1].float(), 1)
            gm = torch.cumsum(u, 0) / torch.arange(1, len(g) + 1, device=dev)[:, None]
            out[[pos[int(r)] for r in g]] = F.softmax(net(x, gate=gm).float(), 1).cpu().numpy()
    return out


def interp_matrix(src, dst):
    """(len(dst), len(src)) map from a source montage onto `dst`: shared channels copied exactly, the others by MNE's
    spherical-spline interpolation (standard_1005 positions, origin 'auto') = what interpolate_bads computes."""
    import mne
    names = list(src) + [c for c in dst if c not in src]
    raw = mne.io.RawArray(np.vstack([np.eye(len(src)), np.zeros((len(names) - len(src), len(src)))]),
                          mne.create_info(names, 100.0, "eeg"), verbose=False)
    raw.set_montage("standard_1005", verbose=False)
    raw.info["bads"] = names[len(src):]
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, origin="auto", verbose=False)
    M = raw.get_data()
    return M[[names.index(c) for c in dst]]


THREE_COMMAND = {"Scherer2015Individually": "4:0,2:1,1:2"}   # studies whose export maps onto MI / SUB / WORD (hub page)


def load_extra(path):
    X, idx, meta = load_export(path)
    lm = meta.get("label_map") or THREE_COMMAND.get(meta.get("study"))
    if not lm:   # e.g. Dreyer2023 (left / right MI): its y is NOT a PROTEUS class -> unlabelled use only (CPT)
        raise SystemExit(f"--extra {path}: study {meta.get('study')} has no 3-command label map; refusing it as labelled data")
    keep = np.flatnonzero(idx["y"].to_numpy() >= 0)
    return dict(name=Path(path).name, X=X, idx=idx, keep=keep, names=list(meta["ch_names"]), sha=meta.get("x_sha256"))


def extra_users(ext, users):
    """T2-D453: keep only the labelled windows of the listed external users (the last path part of `subject`, e.g. '3' for
    'Scherer2015Individually/3'); '' = every user = the dict unchanged. Alignment still runs over each user's whole stream."""
    if not users:
        return ext
    want = set(users.split("+"))
    u = ext["idx"]["subject"].astype(str).str.rsplit("/", n=1).str[-1].to_numpy()
    assert want <= set(u), (sorted(want), sorted(set(u)))
    keep = ext["keep"][np.isin(u[ext["keep"]], sorted(want))]
    assert len(keep), users
    return dict(ext, keep=keep)


KD_RNG = 20261028   # backlog 69: the distillation windows' draw (the teacher dump and every student use the same windows)


def kd_select(pools, n):
    """backlog 69: the distillation windows of unlabelled external cpt_pools: n // len(pools) per pool, split evenly over its subjects, drawn
    uniformly without replacement (rng KD_RNG), each subject's kept in index (acquisition) order. Returns [dict(name, X, idx, names, rows)]."""
    rng = np.random.default_rng(KD_RNG)
    out = []
    for d in pools:
        d = Path(d)
        meta = json.loads((d / "meta.json").read_text())
        idx = pd.read_csv(d / "index.csv", dtype=str, keep_default_na=False)
        X = np.load(d / "X.npy", mmap_mode="r")
        assert X.shape[-1] >= 480, (d.name, X.shape)
        subj = idx["subject"].to_numpy()
        subs = pd.unique(subj)
        per = max(1, (n // len(pools)) // len(subs))
        rows = [np.sort(rng.choice(r, min(per, len(r)), replace=False)) for r in (np.flatnonzero(subj == u) for u in subs)]
        out.append(dict(name=d.name, X=X, idx=idx, names=list(meta["ch_names"]), rows=np.concatenate(rows)))
    return out


def kd_aligned(sel, mode, shrink, dst_names):
    """backlog 69: the kd_select windows mapped onto dst_names (interp_matrix over the standard_1005 channels both share, as T2-D124b), cut
    to the contract's 480 samples and aligned per stream (subject | session, index order) with `mode`, each pool at its own scale (as
    aligned_extra). Returns (X (N, len(dst_names), 480) float32, keys 'pool:row')."""
    import mne
    known = set(mne.channels.make_standard_montage("standard_1005").ch_names)
    outs, keys = [], []
    for p in sel:
        keep = [i for i, c in enumerate(p["names"]) if c in known]
        W = interp_matrix([p["names"][i] for i in keep], dst_names)
        X, rows = p["X"], p["rows"]
        scale = float(np.median(np.asarray(X[rows[::7]], np.float64)[:, keep, :480].std(-1)))
        sub = p["idx"].iloc[rows]
        strm = (sub["subject"] + "|" + sub["session"]).to_numpy()
        o = np.empty((len(rows), len(dst_names), 480), np.float32)
        for st in pd.unique(strm):
            j = np.flatnonzero(strm == st)
            xs = np.asarray(X[rows[j]], np.float64)[:, keep, :480]
            o[j] = stream.align_stream(np.einsum("dc,nct->ndt", W, xs), mode, scale=scale, shrink=shrink)
        outs.append(o)
        keys += [f"{p['name']}:{r}" for r in rows]
    return np.concatenate(outs), np.array(keys)


def aligned_extra(ext, mode, shrink, dst_names=None, frac=1.0, car=False):
    """Causally align an external corpus per stream (its own subject|session order). With dst_names (CNNs): first map
    every window onto that montage (interp_matrix), then align; else (FMs) align in the native montage. T2-D219: frac < 1
    keeps an evenly spaced share of each stream's labelled windows in acquisition order (fixed, no seed), after the alignment."""
    X, idx = ext["X"], ext["idx"]
    W = interp_matrix(ext["names"], dst_names) if dst_names is not None else None
    C = len(dst_names) if dst_names is not None else X.shape[1]
    keep = ext["keep"]
    scale = float(np.median(np.asarray(X[keep[::7]]).std(-1)))
    out = np.empty((len(idx), C, X.shape[2]), dtype=np.float32)
    for _, g in idx.groupby("stream", sort=False):
        rows = g.sort_values("order")["i"].to_numpy()
        xs = np.asarray(X[rows], dtype=np.float64)
        if W is not None:
            xs = np.einsum("dc,nct->ndt", W, xs)
        if car:                                         # T2-D507: the PROTEUS input's CAR (T2-D475) on the mapped montage, before alignment
            xs = stream.car_windows(xs).astype(np.float64)
        out[rows] = stream.align_stream(xs, mode, scale=scale, shrink=shrink)
    Xk, yk = out[keep], idx["y"].to_numpy()[keep].astype(int)
    if frac < 1:
        g = idx.iloc[keep].reset_index(drop=True)
        j = g.sort_values("order").groupby("stream", sort=False).cumcount().sort_index().to_numpy()
        sel = np.floor((j + 1) * frac) > np.floor(j * frac)
        Xk, yk = Xk[sel], yk[sel]
    return Xk, yk


def setval_mask(pp, ya, thr, K=3):
    """T2-D594b: the label sets of the P1-selected rows: a row whose teacher top-2 classes are {MI 0, WORD 2} and p_top1 - p_top2 < thr gets
    {MI, WORD}; every other row its own label (a one-hot row). pp (N, K) the teacher posterior, ya (N,) its argmax."""
    pp = np.asarray(pp, np.float64)
    m = np.zeros((len(pp), K), bool)
    m[np.arange(len(pp)), np.asarray(ya, np.int64)] = True
    if len(pp):
        o = np.sort(pp, 1)
        top2 = np.sort(np.argsort(-pp, 1)[:, :2], 1)
        rule = (top2 == np.array([0, 2])).all(1) & ((o[:, -1] - o[:, -2]) < thr)
        m[rule, 0] = m[rule, 2] = True
    return m


def side_raw(X, kind):
    """T2-D517: per-window side features from the DELIVERED windows (before any alignment)."""
    Xd = np.asarray(X, dtype=np.float64)
    if kind == "dcmean":
        m = Xd.mean(-1)
        return np.hstack([m, m - m.mean(1, keepdims=True)])
    if kind == "dc8":
        N, C, T_ = Xd.shape
        return Xd.reshape(N, C, 8, T_ // 8).mean(-1).reshape(N, C * 8)
    if kind in ("lvwin", "lvstream"):
        return np.log(Xd.var(-1) + 1e-6)
    raise ValueError(kind)


def temporal_view(X, view, chunk=512):
    """T2-D519: a per-member temporal input view of the (aligned) windows, the hall-of-fame's preprocessing-view family scaled to 120 Hz:
    ma6 = centred 6-sample moving average (their 5 at 100 Hz); ema3 = forward EMA alpha 0.3 started at the first sample; dt37 = the window minus
    its centred 37-sample moving average (their 31-sample detrend at 100 Hz). float32 in, float32 out; edges by 'nearest'."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import lfilter
    out = np.empty(X.shape, np.float32)
    for a in range(0, len(X), chunk):
        x = np.asarray(X[a:a + chunk], np.float64)
        if view == "ma6":
            y = uniform_filter1d(x, size=6, axis=-1, mode="nearest")
        elif view == "ema3":
            al = 0.3
            y = lfilter([al], [1.0, -(1.0 - al)], x, axis=-1, zi=(1.0 - al) * x[..., :1])[0]
        elif view == "dt37":
            y = x - uniform_filter1d(x, size=37, axis=-1, mode="nearest")
        else:
            raise ValueError(view)
        out[a:a + chunk] = y
    return out


def side_z(F, idx, tr, kind, n0=10.0):
    """T2-D517: causal per-stream z of the side features (running mean / sd of windows <= t, shrunk to the training rows' mean / sd); lvstream =
    the stream's causal running MEAN instead, z-scored against the training rows' population (the stream-level part)."""
    mu0, sd0 = np.nanmean(F[tr], 0), np.nanstd(F[tr], 0) + 1e-6
    st, od = idx["stream"].to_numpy(), idx["order"].to_numpy()
    Z = np.empty_like(F)
    for s_ in np.unique(st):
        r = np.flatnonzero(st == s_)
        r = r[np.argsort(od[r], kind="stable")]
        x = np.where(np.isfinite(F[r]), F[r], mu0)
        n = n0 + np.arange(1, len(r) + 1)[:, None]
        m = (n0 * mu0 + np.cumsum(x, 0)) / n
        if kind == "lvstream":
            Z[r] = (m - mu0) / sd0
        else:
            v = (n0 * (sd0 ** 2 + mu0 ** 2) + np.cumsum(x * x, 0)) / n - m ** 2
            Z[r] = (x - m) / np.sqrt(np.maximum(v, 1e-12))
    return Z.astype(np.float32)


# ----------------------------------------------------------------------------------------------- classical
def _bandpass(X, lo, hi, fs):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1)


def _spd_fn(Cs, fn):
    w, V = np.linalg.eigh(Cs)
    return np.einsum("...ij,...j,...kj->...ik", V, fn(np.maximum(w, 1e-10)), V)


class FBTS:
    """Filter-bank tangent space (log-Euclidean reference) + logistic regression. Window-local filtering."""

    def __init__(self, fs, bands=((4, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 45)), C=1.0):
        self.fs, self.bands, self.C = fs, bands, C

    def _covs(self, X, band):
        Xf = _bandpass(X, *band, self.fs)
        Cs = np.einsum("nct,ndt->ncd", Xf, Xf) / Xf.shape[-1]
        tr = np.trace(Cs, axis1=1, axis2=2)[:, None, None] / Cs.shape[1]
        return 0.95 * Cs + 0.05 * tr * np.eye(Cs.shape[1])

    def _feats(self, X, fit):
        F = []
        iu = None
        for k, b in enumerate(self.bands):
            Cs = self._covs(X, b)
            if fit:
                self.ref_[k] = _spd_fn(_spd_fn(Cs, np.log).mean(0), lambda w: np.exp(w) ** -0.5)
            W = self.ref_[k]
            L = _spd_fn(W @ Cs @ W, np.log)
            iu = np.triu_indices(L.shape[1])
            coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
            F.append(L[:, iu[0], iu[1]] * coef)
        return np.concatenate(F, 1)

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.ref_ = {}
        self.clf_ = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=3000))
        self.clf_.fit(self._feats(X, True), y)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._feats(X, False))


class LVLDA:
    def __init__(self, fs, bands=((4, 8), (8, 13), (13, 20), (20, 30), (30, 45))):
        self.fs, self.bands = fs, bands

    def _f(self, X):
        return np.concatenate([np.log(_bandpass(X, *b, self.fs).var(-1) + 1e-10) for b in self.bands], 1)

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.clf_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(self._f(X), y)
        return self

    def predict_proba(self, X):
        return self.clf_.predict_proba(self._f(X))


# ----------------------------------------------------------------------------------------------- main
def cpt_tag(path):
    """the --fm-ckpt part of a result stem: the checkpoint dir's last 12 characters; for an fm_pilot checkpoint
    (<run>/pilot/ckpt_s<step>/, T2-D141) the run's last 5 + _s<step>, so the TP-1 arms do not share a stem."""
    p = Path(path)
    if p.parent.name.startswith("ckpt_s") and p.parents[1].name == "pilot":
        return p.parents[2].name[-5:] + p.parent.name[4:]
    return p.parent.name[-12:]


def cap_rotations(K, s, seed):
    """backlog 65 (b): K rotation matrices exp([w]_x) (Rodrigues), w ~ N(0, s^2 I3), from the generator `seed`."""
    out = []
    for w in np.random.default_rng(seed).normal(0.0, s, (K, 3)):
        th = np.linalg.norm(w)
        a, b, c = w / th
        Kx = np.array([[0.0, -c, b], [c, 0.0, -a], [-b, a, 0.0]])
        out.append(np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx)
    return np.stack(out)


def vp_bank(names, K=64, sigma=0.005, gain=0.1, shift=1, seed=0, vref=0.0, cap=0.0):
    """T2-D193: K "virtual participants" for the montage `names`: electrode positions (standard_1005) re-centred on a fitted sphere,
    jittered tangentially by N(0, sigma) metres and put back on the sphere; mats[k] = MNE's spherical-spline map from the real to the
    jittered positions (C x C, near identity; spline regularisation 1e-8); gains[k] ~ U(1 - gain, 1 + gain) per channel; shifts[k] ~ U{-shift..shift} samples.
    T2-D215: vref > 0 -> mats[k] = (I - a 1 w^T) mats[k], a partial re-reference to a Dirichlet(1) mix w of 1-4 random electrodes, a ~ U(0, vref).
    backlog 65 (b): cap > 0 -> draw k's jittered positions are also rotated rigidly about the sphere centre by cap_rotations(K, s, seed + 3),
    s = cap / (sqrt(2) R) with R the mean radius, so the common scalp displacement has RMS ~cap metres (0 = the bank as before, bitwise)."""
    import mne
    import torch
    from mne.channels.interpolation import _make_interpolation_matrix
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    P = np.array([pos[c] for c in names], np.float64)
    sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P ** 2).sum(1), rcond=None)[0]
    Q = P - sol[:3]
    nrm = np.linalg.norm(Q, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    rots = cap_rotations(K, cap / (np.sqrt(2) * nrm.mean()), seed + 3) if cap > 0 else None
    mats, gains, shifts = [], [], []
    for k in range(K):
        d = rng.normal(0.0, sigma, Q.shape)
        d -= (d * (Q / nrm)).sum(1, keepdims=True) * (Q / nrm)
        Qj = Q + d
        Qj *= nrm / np.linalg.norm(Qj, axis=1, keepdims=True)
        if rots is not None:                                          # backlog 65 (b): the rigid cap shift (norms kept: still on the sphere)
            Qj = Qj @ rots[k].T
        mats.append(_make_interpolation_matrix(Q, Qj, alpha=1e-8))      # MNE's default 1e-5 smooths this dense montage (diag 0.48 at zero jitter)
        gains.append(rng.uniform(1 - gain, 1 + gain, len(names)))
        shifts.append(int(rng.integers(-1, 2)))                       # T2-D214: this draw stays the +-1 one (the maps / gains of every bank unchanged)
    if shift != 1:                                                    # T2-D214: another range from its own generator
        shifts = [int(v) for v in np.random.default_rng(seed + 1).integers(-shift, shift + 1, K)]
    if vref > 0:                                                      # T2-D215: virtual references, from their own generator
        rr = np.random.default_rng(seed + 2)
        for k in range(K):
            m = int(rr.integers(1, 5))
            w = np.zeros(len(names))
            w[rr.choice(len(names), m, replace=False)] = rr.dirichlet(np.ones(m))
            mats[k] = (np.eye(len(names)) - rr.uniform(0.0, vref) * np.outer(np.ones(len(names)), w)) @ mats[k]
    return dict(mats=torch.tensor(np.stack(mats), dtype=torch.float32), gains=torch.tensor(np.stack(gains), dtype=torch.float32),
                shifts=torch.tensor(shifts))


def window_iaf(X, fs, names):
    """T2-D191: per-window IAF as t2p_route.fingerprint's first feature (peak of the mean posterior-channel Welch PSD in 7-13 Hz)."""
    from scipy.signal import welch
    post = [i for i, c in enumerate(names) if c.upper().startswith("P")]
    out = np.empty(len(X))
    for i0 in range(0, len(X), 256):
        xb = np.asarray(X[i0:i0 + 256], np.float64)
        xb = xb - xb.mean(-1, keepdims=True)
        f, P = welch(xb, fs=fs, nperseg=min(240, xb.shape[-1]), axis=-1)
        a = (f >= 7) & (f <= 13)
        out[i0:i0 + 256] = f[a][np.argmax(np.log(P[:, post][:, :, a].mean(1) + 1e-20), 1)]
    return out


def iaf_factors(iaf, idx, trm):
    """T2-D191: the IAF each window is warped by. Training windows: their user's median over the protocol's training windows; every
    other window: the causal running mean of its stream's IAF (the windows so far, in order) = the router's running statistic."""
    subj, strm, order = idx["subject"].to_numpy(), idx["stream"].to_numpy(), idx["order"].to_numpy()
    fac = np.full(len(iaf), np.nan)
    for u in np.unique(subj[trm]):
        m = trm & (subj == u)
        fac[m] = np.median(iaf[m])
    for s_ in np.unique(strm[~trm]):
        g = np.flatnonzero((strm == s_) & ~trm)
        g = g[np.argsort(order[g], kind="stable")]
        fac[g] = np.cumsum(iaf[g]) / np.arange(1, len(g) + 1)
    assert np.isfinite(fac).all()
    return fac


def iaf_warp(XA, fac, ref=10.0):
    """T2-D191: FFT-resample each window from L to round(L * IAF / ref) samples, then reflect-pad / centre-crop back to L: at the same
    sampling rate this maps a frequency f to f * ref / IAF, so every window's alpha peak lands at `ref` Hz."""
    from scipy.signal import resample
    L = XA.shape[-1]
    out = np.empty(XA.shape, np.float32)
    for i in range(len(XA)):
        n_new = int(round(L * fac[i] / ref))
        w = np.asarray(XA[i], np.float64) if n_new == L else resample(np.asarray(XA[i], np.float64), n_new, axis=-1)
        if n_new < L:
            p_ = L - n_new
            w = np.pad(w, ((0, 0), (p_ // 2, p_ - p_ // 2)), mode="reflect")
        elif n_new > L:
            c_ = (n_new - L) // 2
            w = w[:, c_:c_ + L]
        out[i] = w
    return out


SW_W = {"fbcnet_bboth": 4 / 7, "shallow": 3 / 7}           # T2-D253: the package's CNN ratio (w4321's 0.4 : 0.3)


def oof_pcnn(oof_dir, proto, y, rows, full=False):
    """T2-D253: the out-of-fold CNN probability of the true class for the training windows `rows` of protocol `proto`
    (kit | seen1), from the <proto>oof_<user>__<model>__...__s<seed>.npz files in oof_dir: per model the mean over its seeds,
    then FBCNet 4/7 + ShallowNet 3/7. Each model has one recipe and one seed set; every (model, seed) covers each row
    exactly once and covers nothing but `rows`. full: the (len(rows), 3) blend instead."""
    y, rows = np.asarray(y).astype(int), np.asarray(rows)
    P = 0.0
    for m, w in SW_W.items():
        fs_ = sorted(Path(oof_dir).glob(f"{proto}oof_*__{m}__*.npz"))
        assert fs_, f"no {proto}oof {m} files in {oof_dir}"
        assert len({"__".join(f.stem.split("__")[2:-1]) for f in fs_}) == 1, f"{m}: one align / recipe per model"
        by = {}
        for f in fs_:
            z = np.load(f)
            a = by.setdefault(int(f.stem.rsplit("__s", 1)[1]), np.full((len(y), 3), np.nan))
            te = z["test_idx"]
            assert np.isnan(a[te]).all(), f"{f.name}: a row twice"
            a[te] = z["prob_last"]
        cov = [np.flatnonzero(~np.isnan(a[:, 0])) for a in by.values()]
        assert all(np.array_equal(c, np.sort(rows)) for c in cov), f"{m}: the OOF rows != the protocol's training rows"
        P = P + w * np.mean([a[rows] for a in by.values()], 0)
    return P if full else P[np.arange(len(rows)), y[rows]]


def fm_aug(s):
    """T2-D257: --fm-aug 'g:s' -> (gain_jit, tshift) for t2p_fm.fit_eval_fm; T2-D260: 'g:s:c' -> (gain_jit, tshift, chdrop); '' -> None (off)."""
    if not s:
        return None
    f = s.split(":")
    assert len(f) in (2, 3), s
    g, t, c = float(f[0]), int(f[1]), float(f[2]) if len(f) == 3 else 0.0
    canon = f"{g:g}:{t}" + (f":{c:g}" if len(f) == 3 else "")
    assert g >= 0 and t >= 0 and 0 <= c < 1 and (g > 0 or t > 0 or c > 0) and (len(f) == 2 or c > 0) and canon == s, s
    return (g, t) if len(f) == 2 else (g, t, c)


def task_tag(args, fname, model, align, seed, xtag):
    """the result-file stem of one task (t2p_report / t2p_paired parse it; --list-outputs writes it)."""
    return f"{fname}__{model.replace(':', '-')}__{align}__n0{args.n0:g}__{task_rec(args, model)}{xtag}__s{seed}"


def task_rec(args, model):
    """the recipe field of a task's stem (its twins -- pu / gs / bn<m> -- insert their suffix right after it)."""
    if model.startswith("fm:"):
        rec = (f"fmE{args.fm_epochs}b{args.fm_bs}le{args.fm_lr_enc:g}lh{args.fm_lr_head:g}wd{args.fm_wd:g}"
               f"ll{args.fm_llrd:g}ema{args.fm_ema:g}dp{args.fm_drop_path:g}"
               + (f"L{args.fm_layer}" if args.fm_layer != -1 else "") + (f"lora{args.fm_lora}" if args.fm_lora else "") + (f"mb{args.fm_micro}" if args.fm_micro > 1 else "") + (f"a{args.fm_lora_alpha:g}" if args.fm_lora and args.fm_lora_alpha else "") + (f"ld{args.fm_lora_dropout:g}" if args.fm_lora and args.fm_lora_dropout != 0.2 else "")
               + (f"mx{args.fm_mixup:g}" if args.fm_mixup else "") + ({"gain": "cg", "tp": "ctp"}.get(args.fm_calib, ""))
               + (f"vp{args.fm_vp}" if args.fm_vp else "") + (f"rc{args.fm_rc:g}" if args.fm_rc else "") + (f"tl{args.fm_tilt:g}" if getattr(args, "fm_tilt", 0) else "")
               + (f"aug{args.fm_aug.replace(':', '-')}" if args.fm_aug else "")
               + (f"rb{args.fm_recomb.replace(':', '-')}" if getattr(args, "fm_recomb", "") else "")
               + (f"ph{args.fm_phase.replace(':', '-')}" if getattr(args, "fm_phase", "") else "")
               + (f"mr{args.fm_mirror:g}" if getattr(args, "fm_mirror", 0.0) else "")
               + (f"cs{args.fm_cons:g}" if getattr(args, "fm_cons", 0.0) else "")
               + (f"wm{args.fm_wmargin:g}" if getattr(args, "fm_wmargin", 0.0) else "")
               + (f"ls{args.fm_ls:g}" if getattr(args, "fm_ls", 0.1) != 0.1 else "")
               + (f"lp{args.fm_lp_epochs}" if getattr(args, "fm_lp_epochs", 0) else "")
               + (f"ri{args.fm_reinit_top}" if getattr(args, "fm_reinit_top", 0) else "")
               + ("hf" if getattr(args, "fm_head_freeze", False) else "") + (f"hwd{args.fm_head_wd:g}" if getattr(args, "fm_head_wd", None) is not None else "")
               + (f"td{args.fm_token_drop:g}" if getattr(args, "fm_token_drop", 0.0) else "")
               + (f"ws{args.fm_wise:g}" if getattr(args, "fm_wise", 0.0) else "")
               + (f"cpt{cpt_tag(args.fm_ckpt)}" if args.fm_ckpt else "")
               + (f"fu{args.fm_fuse}" if args.fm_fuse else "") + (f"bp{args.fm_band.replace(':', '-')}" if args.fm_band else "")
               + (f"sc{args.fm_supcon:g}{args.fm_supcon_mode}t{args.fm_supcon_tau:g}" if args.fm_supcon else "")
               + (f"qkv{args.fm_qkv_lr_mult:g}" if args.fm_qkv_lr_mult != 1.0 else "") + ("lclip" if args.fm_layer_clip else "")
               + (f"li{args.fm_lora_in}{'role' if args.fm_lora_in_roles else ''}d{args.fm_lora_in_dropout:g}" if args.fm_lora_in else "")
               + (f"mr{args.fm_mae_retain}" if args.fm_mae_retain else "") + (f"sd{args.fm_side}" if args.fm_side else "")
               + (f"sw{args.fm_sw:g}" if args.fm_sw else "")
               + (f"pk{args.fm_pkg:g}w{args.fm_pkg_w:g}" if getattr(args, "fm_pkg", 0) else "")   # backlog 96
               + (f"ce{args.fm_chemb_lr:g}" if getattr(args, "fm_chemb_lr", None) else "")   # T2-D644 (W27)
               + (f"dro{args.fm_dro:g}" if getattr(args, "fm_dro", 0.0) else "")   # T2-D645 (FM-DRO)
               + (f"grl{args.fm_grl:g}" if getattr(args, "fm_grl", None) is not None else ""))   # T2-D649 (W29)
    else:
        rec = f"{args.recipe}{'e' + str(args.epochs) if args.epochs else ''}" + ({"users": "bagu", "seeds": "bags", "uw1": "bagw1", "uw4": "bagw4", "runs": "bagr", "ub2": "bagb2", "ub5": "bagb5", "s3": "bags3", "ovr": "bagovr", "pair": "bagpw", "ovc": "bagovc"}[args.bag] if args.bag else "")
        rec += f"cr{args.crop.replace(':', '-')}" if args.crop else ""
        rec += "rco" if getattr(args, "recolour_own", "") else ""      # T2-D347
        rec += f"ini{args.cnn_init_tag}" if getattr(args, "cnn_init", "") else ""   # T2-D367: an SSL-pretrained init of the pooled CNN
    return rec


def bag_rows(mode, utr, grp, seed):
    """T2-D437: the training rows (indices, with repeats) of one bagged net; len = len(utr), every user present.
    uw1 / uw4: user shares ~ Dirichlet(a) (a = 1 / 4), floored at a quarter of the user's own share, rows drawn with replacement inside the user.
    runs: per user, its (session | run) groups resampled with replacement, each drawn group contributing all its rows."""
    rng = np.random.default_rng(seed)
    users = np.unique(utr)
    out = []
    if mode in ("uw1", "uw4"):
        base = np.array([(utr == u).mean() for u in users])
        w = rng.dirichlet(np.full(len(users), 1.0 if mode == "uw1" else 4.0))
        w = np.maximum(w, 0.25 * base)
        cnt = rng.multinomial(len(utr), w / w.sum())
        for u, c in zip(users, cnt):
            out.append(rng.choice(np.flatnonzero(utr == u), size=max(int(c), 1), replace=True))
    else:
        assert mode == "runs", mode
        for u in users:
            rows = np.flatnonzero(utr == u)
            gs = np.unique(grp[rows])
            for g in rng.choice(gs, size=len(gs), replace=True):
                out.append(rows[grp[rows] == g])
    return np.sort(np.concatenate(out))


EXPERT_PAIRS = ((0, 1), (0, 2), (1, 2))      # T2-D440: mi-sub, mi-word, sub-word (index.y: 0 mi, 1 sub, 2 word)
EXPERT_CH = (("C3", "C4", "C1", "C2", "Cz", "CP3", "CP4", "FC3", "FC4", "CP1", "CP2", "CPz"),            # mi: sensorimotor
             ("Fz", "FCz", "F1", "F2", "FC1", "FC2", "AF3", "AF4"),                                       # sub: frontal midline
             ("F5", "FC5", "C5", "CP5", "TP7", "F6", "FC6", "C6", "CP6", "TP8"))                          # word: fronto-temporal, both sides


def expert_mask(ch_names, k):
    """T2-D440 (ovc): a (C, 1) 0 / 1 mask keeping expert k's channel set (applied after alignment; the other channels are zeroed)."""
    m = np.array([c in EXPERT_CH[k] for c in ch_names], np.float32)
    assert int(m.sum()) == len(EXPERT_CH[k]), (k, m.sum())
    return m[:, None]


def expert_combine(S, mode):
    """T2-D440: (n, 3) expert scores -> (n, 3) class probabilities. ovr / ovc: S[:, c] = p(class c vs rest), normalised.
    pair: S[:, k] = p(second class of EXPERT_PAIRS[k]); Price et al.'s coupling p_i = 1 / (sum_j 1 / r_ij - (K - 2)), normalised."""
    S = np.clip(np.asarray(S, np.float64), 1e-6, 1 - 1e-6)
    if mode != "pair":
        return S / S.sum(1, keepdims=True)
    r = np.zeros((len(S), 3, 3))
    for k, (a, b) in enumerate(EXPERT_PAIRS):
        r[:, b, a], r[:, a, b] = S[:, k], 1 - S[:, k]
    P = np.stack([1.0 / (sum(1.0 / r[:, i, j] for j in range(3) if j != i) - 1.0) for i in range(3)], 1)
    P = np.clip(P, 1e-6, None)
    return P / P.sum(1, keepdims=True)


def boost_mult(err, eta, cap=3.0):
    """T2-D438: per-user multipliers min(exp(eta (e_u - min e)), cap) from the users' leave-one-user-out errors."""
    return np.minimum(np.exp(eta * (np.asarray(err, float) - np.min(err))), cap)


def boost_rows(utr, mult, seed):
    """T2-D438: training rows (with repeats, len = len(utr)) whose user shares are the users' own shares x mult, renormalised; user ids = 0..U-1."""
    rng = np.random.default_rng(seed)
    users = np.unique(utr)
    w = np.array([(utr == u).mean() * mult[int(u)] for u in users])
    cnt = rng.multinomial(len(utr), w / w.sum())
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(utr == u), size=max(int(c), 1), replace=True) for u, c in zip(users, cnt)]))


def user_balance_weights(sub_tr, ses_tr, power=1.0):
    """T2-D630 / D642 (power 1: 1 / the user's number of TRAINING sessions, exactly as before) and T2-D647 W28 (power 0.5: sessions^-0.5); the
    loss self-normalises per batch, so a global scale is irrelevant. -> (per-window weights float32, {user: weight})"""
    ns_ = {u: len(set(ses_tr[sub_tr == u])) for u in np.unique(sub_tr)}
    wu = {u: (1.0 / n if power == 1.0 else float(n) ** (-float(power))) for u, n in ns_.items()}
    return np.array([wu[u] for u in sub_tr], np.float32), wu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocols", default="loso,xsess,kit")
    ap.add_argument("--models", default="eegnet")
    ap.add_argument("--aligns", default="none,wz,sz,ea")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--recipe", default="r1")
    ap.add_argument("--cnn-init", default="", help="T2-D367: a state_dict path pattern with {protocol} (t2p_cnn_ssl.py): the pooled CNN starts from "
                    "every tensor of it but final_layer.*; stems gain ini<--cnn-init-tag>; copies unchanged (they fine-tune the pooled net)")
    ap.add_argument("--cnn-init-tag", default="", help="T2-D367: the stem tag of --cnn-init (required with it)")
    ap.add_argument("--epochs", type=int, default=0, help="override the recipe's epochs")
    ap.add_argument("--n0", type=float, default=0.0, help="EA population-prior strength (windows)")
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--reverse", action="store_true", help="work the task list from the end (a second job resuming "
                    "into the same --out meets the first in the middle; finished tasks are skipped)")
    ap.add_argument("--folds", default="", help="comma list of fold names to keep (debug)")
    # foundation models: --models fm:<fm>:<readout>, e.g. fm:st_base:mean, fm:mor_p1v2:attn, fm:rand_st_base:mean
    ap.add_argument("--fm-epochs", type=int, default=30)
    ap.add_argument("--fm-bs", type=int, default=64)
    ap.add_argument("--fm-lr-enc", type=float, default=1e-4)
    ap.add_argument("--fm-lr-head", type=float, default=1e-3)
    ap.add_argument("--fm-wd", type=float, default=0.05)
    ap.add_argument("--fm-llrd", type=float, default=1.0)
    ap.add_argument("--fm-ema", type=float, default=0.0)
    ap.add_argument("--fm-drop-path", type=float, default=0.0)
    ap.add_argument("--tta-bn", default="", help="CNN models: comma list of BatchNorm momenta for causal within-session "
                    "BN-statistics adaptation (T2-D100); each writes its own result file, recipe suffix bn<m>")
    ap.add_argument("--tta-vp", type=int, default=0, help="backlog 71: CNN models with a vp recipe: also score the pooled net as the mean "
                    "over the identity + the first K-1 vp-bank draws (T.predict_proba_vp) and write the twin <recipe>tv<K> (+ <recipe>tv<K>pu with "
                    "--peruser: the same copies); 0 = off")
    ap.add_argument("--tta-shift", type=int, default=0, help="T2-D302 (a): with --tta-vp K, also write <recipe>tv<K>s<S> (+ pu): the pooled "
                    "TTA over the K vp views + S-1 shift-only views (T.predict_proba_vp shift_only); 0 = off")
    ap.add_argument("--tta-copies", action="store_true", help="T2-D302 (c): with --tta-vp K and --peruser, also write <recipe>tv<K>c (+ pu): "
                    "the pooled TTA + the per-user copies through the same K views")
    ap.add_argument("--user-balance-pow", type=float, default=1.0, help="T2-D647 (W28): with --user-balance, the weight = sessions^-p (1 = D630's "
                    "1 / sessions, bitwise; 0.5 = ALL6-HALF); tag __ubwp<p>")
    ap.add_argument("--user-balance", action="store_true", help="T2-D630: CNN pooled nets: every training window's CE weighs 1 / (its user's number "
                    "of training sessions) (P04 / P10 at 1/3 under --train-all-like folds; each user sums to one session), self-normalised per batch; tag __ubw")
    ap.add_argument("--test-mirror", action="store_true", help="T2-D619 (diagnostic): CNN pooled nets: also write <recipe>mr = the trained net on "
                    "the TEST windows with the homolog left-right channel swap (t2p_fm.mirror_setup's map), and with --tta-vp K <recipe>tv<K>mr; "
                    "training unchanged; the unmirrored re-prediction is asserted == prob_last")
    ap.add_argument("--drop-train", default="", help="T2-D307: a JSON {protocol: [export window indices]} of TRAINING windows to drop "
                    "(the pooled net and its copies; test / val untouched); empty = off")
    ap.add_argument("--side", default="", choices=("", "dcmean", "dc8", "lvwin", "lvstream"), help="T2-D517: a per-window side vector at the "
                    "logits (pooled nets only): dcmean = the 82 window means (delivered + CAR), dc8 = the 8 half-second means (328), lvwin = the window's "
                    "per-channel log-variance (window part), lvstream = the stream's running log-variance level (stream part); causal per-stream z")
    ap.add_argument("--view", default="", choices=("", "ma6", "ema3", "dt37"), help="T2-D519: a temporal input view of the aligned windows (pooled nets "
                    "only; train and test alike): ma6 / ema3 / dt37 (t2p_bench.temporal_view)")
    ap.add_argument("--keep-mean", action="store_true", help="T2-D517: the aligners keep each window's per-channel time mean (statistics on the "
                    "de-meaned windows, the transform applied to the window with its mean); none / sz / ea only")
    ap.add_argument("--car", action="store_true", help="T2-D475: common average reference per window (good channels only) before everything")
    ap.add_argument("--lap", action="store_true", help="T2-D480: nearest-neighbour (k 4) surface Laplacian per window (good channels only) before "
                    "everything: the view without far-field sources (reference, ocular)")
    ap.add_argument("--softclip", default="", choices=["", "ch", "glob"], help="T2-D306: tanh(x / q99(|x|)) per window before everything "
                    "(q99 per channel / over the window); empty = off")
    ap.add_argument("--tta-pl", default="", help="CNN models: comma list of test-time variants: T2-D113 V1..V4, T2-D405 IM1 / IM2, T2-D420 EA1 / EA2 (EATA) and AT1 / AT2 (anchored); "
                    "each writes its own result file, recipe suffix pl<V>")
    ap.add_argument("--iaf-warp", action="store_true", help="T2-D191: CNN models: warp every input window in time so its IAF maps to 10 Hz "
                    "(training windows: the user's median IAF; others: the stream's causal running mean)")
    ap.add_argument("--fm-peruser-lora", action="store_true", help="T2-D189: FM (ST) models: per training user, LoRA r16 / alpha 32 / "
                    "dropout 0.1 on every Linear in the encoder blocks of the trained member (all else frozen), AdamW lr 1e-3, 10 epochs of "
                    "batches 16 own + 16 other-user windows (CE, ls 0.1, cosine to 0, seed x 100 + k); evaluated with the delta at r 1/2 in weight "
                    "space; writes the 'pu' twin (blend it with --r 1)")
    ap.add_argument("--fm-soup", type=int, default=0, help="T2-D176: FM models: also train K-1 more fine-tunes of the same recipe and "
                    "init at seeds seed + 1000 j, average the K float states uniformly (the main file stays the seed's single member) and write "
                    "the 'soupK' twin; 0 / 1 = off")
    ap.add_argument("--fm-soup-members", action="store_true", help="T2-D493: with --fm-soup K, also write each extra member's own predictions "
                    "as the 'soupKm<j>' twin (j = 1 .. K-1, seed + 1000 j; member 0 = the main file), so the ensemble of exactly the souped nets "
                    "can be read beside the soup; the soup twin and the member twins enter --list-outputs; off = unchanged")
    ap.add_argument("--fm-peruser-heads", action="store_true", help="T2-D166: FM models: per training user, the member's head "
                    "fine-tuned on that user's windows with the encoder frozen (PERUSER at lr 1e-3, 40 epochs, from the pooled head); "
                    "writes the 'pu' twin like --peruser")
    ap.add_argument("--instream-ft", default="", help="T2-D509 (a ceiling, TRUE labels): CNN models: for every test stream of a HELD-OUT user, "
                    "fine-tune a copy of the trained net (the copies' recipe) on the stream's first N windows (acquisition order) and predict the rest; "
                    "comma list of N; writes suffix is<N> (prob_last = the adapted net on the scored rows, the pooled net elsewhere; scored = the mask)")
    ap.add_argument("--instream-setval", type=float, default=0.0, help="T2-D594b: the confR P1 rows whose teacher top-2 are {MI, WORD} with "
                    "p_top1 - p_top2 < this get the set loss -log(p_MI + p_WORD) (ni26.train.finetune yset); 0 = off (bitwise)")
    ap.add_argument("--instream-all", action="store_true", help="T2-D510: adapt EVERY test stream (returning users too), not only the held-out user's")
    ap.add_argument("--instream-sched", default="", help="T2-D510: the ship schedule: comma list of checkpoints (e.g. 30,60,100,150); suffix isS<code>")
    ap.add_argument("--instream-reps", type=int, default=1, help="T2-D510: repeats of the schedule arm (fine-tune seed + 1000 k) for --instream-rep-sel")
    ap.add_argument("--instream-rep-sel", default="conf", help="T2-D510: the label forms whose schedule arm is repeated (T2-D511: confR = the prefix re-decoded labels)")
    ap.add_argument("--instream-sel", default="true", help="T2-D510: comma list of the in-stream labels: true (T2-D509, suffix is<N>), soft = the teacher posterior (ps), all = "
                    "the teacher's decoded labels on all N windows (is<N>pa), conf = on the confident half per predicted class (is<N>pc)")
    ap.add_argument("--instream-pl", default="", help="T2-D510: the teacher file pattern ({protocol}, {seed}): npz rows (global index) + post "
                    "(the package's causal decoded posterior on the held-out user's windows)")
    ap.add_argument("--peruser", action="store_true", help="CNN models (T2-D111 iii): also fine-tune the trained net on "
                    "every training user's windows (ni26.train.PERUSER) and write their val / test probabilities (suffix pu)")
    ap.add_argument("--recolour-own", default="", help="T2-D347 (B2): CNN models: the fingerprint file (t2p_fpcalib export, e.g. "
                    "outputs/t2_d124b/fp_proteus.npy); TEST rows re-aligned per stream to the routed user's training covariance "
                    "(recolour_own); stem + 'rco'; '' = off")
    ap.add_argument("--crop", default="", help="CNN models (T2-D122): train / predict on window samples a:b only (after alignment)")
    ap.add_argument("--long-export", default="", help="T2-D486: the long export (D473's -3 .. +8 s re cue; row-aligned, its contract slice = "
                    "this export bitwise) whose real neighbouring EEG feeds the training windows")
    ap.add_argument("--long-jitter", default="", help="T2-D486: 'a,b' = the training windows' start range in s re cue (the contract window starts "
                    "at 1.0): every training batch (pooled net and copies) crops each window at a random start in [a, b]; val / test stay contract")
    ap.add_argument("--bag", default="", choices=("", "users", "seeds", "uw1", "uw4", "runs", "ub2", "ub5", "s3", "ovr", "pair", "ovc"), help="[T2-D440 class experts, three nets per seed: "
                    "s3 = three plain 3-class nets (anchor), ovr = one-vs-rest, pair = pairwise + coupling, ovc = ovr on class-specific channel sets] [T2-D438: ub2 / ub5 = user-level boosting, K nets on rows drawn "
                    "with user shares x min(exp(eta (e_u - min e)), 3), e_u = user u's error under a net trained without u, eta 2 / 5] CNN models (T2-D117): per seed, K nets (K = the "
                    "fold's training users) averaged; users = net k trains without user k, seeds = all windows (seed 100 s + k); T2-D437 (bag_rows): "
                    "uw1 / uw4 = a Bayesian bootstrap over users (Dirichlet(1) / Dirichlet(4) user shares, every user kept), runs = runs resampled "
                    "with replacement within each user x session")
    ap.add_argument("--tta-mae", default="", help="ST models (T2-D125): comma lrs of the streamed test-time MAE CPT; each "
                    "writes its own result file (suffix mae<lr>); needs --fm-ckpt (the CPT checkpoint holds the decoder)")
    ap.add_argument("--tta-arms", default="", help="T2-D283, with --tta-mae: comma arms of the LayerNorm-affine test-time MAE "
                    "(P = prefix 30, one step, then frozen; K = one step per 8-window buffer, 0.99 restoration); each writes "
                    "suffix mae<lr><arm>; '' = T2-D125's full-encoder per-window path")
    ap.add_argument("--fm-layer", type=int, default=-1, help="1-indexed encoder exit; -1 = final layer")
    ap.add_argument("--fm-lora", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    ap.add_argument("--fm-micro", type=int, default=1, help="TA568 screen: gradient accumulation over k chunks per batch (exact; memory / k); 1 = off")
    ap.add_argument("--fm-lora-alpha", type=float, default=0.0, help="T2-D313b: weight-parametrised LoRA alpha (0 = 4 r, the old default)")
    ap.add_argument("--fm-lora-dropout", type=float, default=0.2, help="T2-D313b: dropout on the LoRA delta (0.2 = the old default)")
    ap.add_argument("--fm-mixup", type=float, default=0.0)
    ap.add_argument("--fm-tilt", type=float, default=0.0, help="backlog 94: the CNN recipes' random spectral tilt (sd of the exponent) on the FM's "
                    "training windows (after vp; 0 = off = bitwise); stems gain tl<x>")
    ap.add_argument("--fm-rc", type=float, default=0.0, help="T2-D235: re-colour the FM's target training windows toward a random other training user (strength; 0 = off)")
    ap.add_argument("--fm-vp", type=int, default=0, help="T2-D198: virtual participants on the FM's target windows (k = k x 5 mm; 0 = off)")
    ap.add_argument("--fm-side", default="", choices=["", "amp", "dcmean"], help="T2-X1: amp = the per-channel log-std side input into the FM head; "
                    "T2-D524: dcmean = the 82 causal-z window means at the FM logits (needs --side dcmean, which builds the vectors)")
    ap.add_argument("--fm-sw", type=float, default=0.0, help="T2-D253: gamma of the FM's per-window loss weight 1 + gamma (1 - p_CNN(y)) "
                    "(mean-normalised over the training windows; p_CNN from --fm-sw-oof); 0 = off")
    ap.add_argument("--fm-wmargin", type=float, default=0.0, help="T2-D596: weight of the WORD-margin loss (softplus(0.5 - (z_y - z_c)) over the "
                    "WORD / MI / SUB confuser pairs, ramped over the first 10 % of the steps); 0 = off (bitwise)")
    ap.add_argument("--fm-cons", type=float, default=0.0, help="T2-D588: weight of the symmetric KL between two independently augmented views "
                    "of each FM target window (loss 0.5 (CE1 + CE2) + w KL_sym); 0 = off")
    ap.add_argument("--fm-mirror", type=float, default=0.0, help="T2-D618: probability p per FM target training window of the left-right "
                    "mirror (homologous channels swapped by name, midline fixed; the dcmean side z permuted with it); 0 = off")
    ap.add_argument("--fm-phase", default="", help="T2-D586: 'p:amax' = the shared-phase Fourier surrogate of the FM target windows "
                    "(probability p, phase ~ U(-amax, +amax) per bin, shared by all channels; DC / Nyquist kept); '' = off")
    ap.add_argument("--fm-recomb", default="", help="T2-D583: 'S:p' = within-user, within-class segment recombination of the FM target windows (S segments, probability p; 50 ms cross-fades); '' = off")
    ap.add_argument("--fm-aug", default="", help="T2-D257: 'g:s' = gain jitter x exp(U(-g, g)) per window and channel + a circular time "
                    "shift U{-s..s} samples (export rate) on the FM's target training windows (T2-D112 G1 / G2; tag aug<g>-<s>; '' = off); T2-D260: "
                    "'g:s:c' adds G4's channel dropout with probability c (tag aug<g>-<s>-<c>)")
    ap.add_argument("--fm-ls", type=float, default=0.1, help="T2-D261: label smoothing of the FM fine-tune's cross-entropy (0.1 = every FM "
                    "fine-tune since T2-D82, stem unchanged; other values tag ls<v> after aug, before cpt)")
    ap.add_argument("--fm-head-freeze", action="store_true", help="T2-D349b: --models fm:...:grid: the random grid head frozen (LayerNorm trains)")
    ap.add_argument("--fm-head-wd", type=float, default=None, help="T2-D349b: the FM head's own weight decay (default: --fm-wd)")
    ap.add_argument("--fm-token-drop", type=float, default=0.0, help="T2-D349b: whole-token dropout before the grid head (training only)")
    ap.add_argument("--fm-cont-epochs", type=int, default=0, help="T2-D546 (the owner's warm-start continuation): after the base fit, continue from its converged state for N epochs (every lr x --fm-cont-lr, no warm-up, cosine to 0, data seed + 50000); twins c<arm>; 0 = off")
    ap.add_argument("--fm-cont-lr", type=float, default=0.1, help="T2-D546: the continuation's lr multiplier on every group")
    ap.add_argument("--fm-cont-arms", default="a,b", help="T2-D546: a = head + side head + top ST block re-drawn first; b = nothing re-drawn")
    ap.add_argument("--fm-lp-epochs", type=int, default=0, help="T2-D263: LP-FT = the first N FM epochs train the head only (encoder frozen), "
                    "then the encoder re-warms (warm-up + cosine over the remaining steps); tag lp<N> after ls, before cpt; 0 = off")
    ap.add_argument("--fm-reinit-top", type=int, default=0, help="T2-D264: re-draw the top N ST encoder blocks (+ its final norm) with the MAE's "
                    "own init before the fine-tune; tag ri<N> after lp, before cpt; 0 = off")
    ap.add_argument("--fm-wise", type=float, default=0.0, help="T2-D270: WiSE-FT on LP-FT (needs --fm-lp-epochs): the evaluated weights = "
                    "(1 - w) x the fine-tune's end + w x the probe's end; tag ws<w> after ri, before cpt; 0 = off")
    ap.add_argument("--kd-pools", default="", help="backlog 69: comma list of unlabelled external cpt_pools dirs = the distillation windows "
                    "(kd_select / kd_aligned; the FM teacher dump's and the kd_ext students')")
    ap.add_argument("--kd-n", type=int, default=12000, help="backlog 69: distillation windows drawn over --kd-pools")
    ap.add_argument("--fm-kd-dump", action="store_true", help="backlog 69 (FM runs): after the fine-tune, write <out>/kdteacher__<protocol>__s<seed>.npz "
                    "= the FM's probabilities of the protocol's training windows (train_idx, p_train) and of the --kd-pools windows (kd_key, p_ext)")
    ap.add_argument("--syn-bank", default="", help="T2-D413: the MeanFlow bank npz of syn recipes (t2p_meanflow.py); '{protocol}' is replaced per task")
    ap.add_argument("--rc-ext-bank", default="", help="backlog 65 (a): t2p_covbank.py's npz = the external re-colouring targets of rc_ext recipes")
    ap.add_argument("--kd-teacher", default="", help="backlog 69 (CNN recipes with kd): the teacher dump; '{protocol}' is replaced per task (T2-D530: '{fold}' = the fold name, e.g. lc9_3)")
    ap.add_argument("--fm-pkg", type=float, default=0.0, help="backlog 96 (G3, P3 v2): beta of the FM's PACKAGE loss, + beta NLL((1 - w) p_CNN_oof "
                    "+ w p_FM) on its target training windows (p_CNN_oof from --fm-sw-oof, oof_pcnn's FBCNet 4/7 + ShallowNet 3/7 blend); 0 = off")
    ap.add_argument("--fm-pkg-w", type=float, default=0.35, help="backlog 96: w, the FM share of that package mix (M6: the ST-L slot 0.35)")
    ap.add_argument("--fm-sw-oof", default="", help="T2-D253: the directory of the kitoof / seen1oof CNN result files (oof_pcnn)")
    ap.add_argument("--fm-train-eval", action="store_true", help="T2-D468: log a clean per-epoch training-set cell BA (diagnostic)")
    ap.add_argument("--fm-ckpt", default="", help="fine-tune from a t2p_cpt / t2p_preft checkpoint (ST; MOR from t2p_preft)")
    ap.add_argument("--fm-mae-retain", type=int, default=0, help="T2-D132: keep the CPT MAE objective jointly every N steps (0 = off)")
    ap.add_argument("--fm-lora-in", type=int, default=0, help="T2-D130: input-dropout LoRA rank on every block Linear (alpha 2r)")
    ap.add_argument("--fm-lora-in-roles", action="store_true", help="T2-D130: role-specific LoRA on the MOR's shared attention")
    ap.add_argument("--fm-lora-in-dropout", type=float, default=0.1)
    ap.add_argument("--fm-qkv-lr-mult", type=float, default=1.0, help="T2-D128 S1: lr multiplier of the attention qkv weights")
    ap.add_argument("--fm-grl", type=float, default=None, help="T2-D649 (W29): a subject-adversarial head through a GRL at this lambda max "
                    "(0 = the lambda-0 twin: the adversary trained, no reversal; absent = off, bitwise)")
    ap.add_argument("--fm-grl-lr", type=float, default=1e-3, help="T2-D649: the adversary's own (constant) lr")
    ap.add_argument("--fm-dro", type=float, default=0.0, help="T2-D645 (FM-DRO): online group DRO over the training users at this eta "
                    "(Track3's registration; 0 = off, bitwise)")
    ap.add_argument("--fm-chemb-lr", type=float, default=None, help="T2-D644 (W27): the ST channel-embedding table in its own group at this "
                    "peak lr, weight decay 0 (absent = LLRD's depth-0 group, bitwise)")
    ap.add_argument("--fm-layer-clip", action="store_true", help="T2-D128 S3: per-encoder-layer gradient clip (norm 1)")
    ap.add_argument("--fm-supcon", type=float, default=0.0, help="T2-D127: lambda of the joint own-cue SupCon term (0 = off)")
    ap.add_argument("--fm-supcon-mode", default="xuser", choices=("xuser", "xsession"))
    ap.add_argument("--fm-supcon-tau", type=float, default=0.1)
    ap.add_argument("--fm-band", default="", help="per-window band-pass of the FM input, 'lo:hi' Hz (e.g. 4:40)")
    ap.add_argument("--fm-fuse", default="", help="joint FM + CNN feature fusion (a ni26.train CNN name, e.g. fbcnet)")
    ap.add_argument("--fm-calib", default="zwin", choices=("zwin", "gain", "tp"),
                    help="FM input: per-window z (default), fixed gain, or tp = TP-1's pretraining front end (MOR, T2-D147)")
    ap.add_argument("--extra", default="", help="comma list of EXTERNAL exports (t2p_export --study) added to every "
                    "fold's TRAINING set (never val / test); CNNs: interpolated onto the PROTEUS montage, FMs: native")
    ap.add_argument("--extra-users", default="", help="T2-D453: '+'-joined external user ids kept from every --extra export ('' = all)")
    ap.add_argument("--extra-head", action="store_true", help="T2-D507: the --extra rows train their OWN classifier head (shared trunk; "
                    "the head is discarded after training; CNN pooled nets only)")
    ap.add_argument("--extra-w", type=float, default=1.0, help="T2-D507: the --extra rows' loss weight in the pooled net (1 = as PROTEUS)")
    ap.add_argument("--extra-dsbn", action="store_true", help="T2-D507: the --extra rows use their own BatchNorm statistics in training "
                    "(the PROTEUS BatchNorm is the one kept)")
    ap.add_argument("--extra-frac", type=float, default=1.0, help="T2-D219: the share of each --extra stream's labelled windows "
                    "kept (evenly spaced in acquisition order, fixed); < 1 adds f<frac> to the tag")
    ap.add_argument("--list-outputs", default="", help="review T2 P3j: append the expected result-file names of these "
                    "arguments (all shards) to this file and exit (t2p_multi's artefact gate)")
    ap.add_argument("--skip-leaky-protocols", action="store_true", help="review T2 P2e: a protocol whose TEST streams are in "
                    "the --fm-ckpt checkpoint's pools is dropped with a LEAKY_<protocol>.json marker (default: refuse)")
    args = ap.parse_args()
    assert args.user_balance or args.user_balance_pow == 1.0, "T2-D647 (Track3): --user-balance-pow without --user-balance would be ignored silently"
    assert bool(args.fm_sw or args.fm_pkg) == bool(args.fm_sw_oof), "--fm-sw / --fm-pkg need --fm-sw-oof (and it needs one of them)"
    assert args.fm_pkg >= 0 and 0 < args.fm_pkg_w <= 1, (args.fm_pkg, args.fm_pkg_w)   # backlog 96
    fm_aug(args.fm_aug)                                  # T2-D257: a malformed --fm-aug fails before any training
    assert 0 <= args.fm_ls < 1, args.fm_ls                 # T2-D261
    assert 0 <= args.fm_lp_epochs < args.fm_epochs, args.fm_lp_epochs   # T2-D263
    assert args.fm_reinit_top >= 0, args.fm_reinit_top                  # T2-D264
    assert 0 <= args.fm_wise < 1 and (args.fm_wise == 0 or args.fm_lp_epochs > 0), args.fm_wise   # T2-D270

    X, idx, meta = load_export(args.export)
    if args.softclip:                      # T2-D306: the windows the nets see (train and test alike)
        X = softclip(X, args.softclip)
    assert not (args.car and args.lap), "--car and --lap are two views of the input; pick one"
    X_delivered = X                        # T2-D524: the side vectors come from the DELIVERED window (a template computes them so), before CAR / Laplacian
    if args.lap:                           # T2-D480: surface Laplacian per window (predict-safe), before alignment, train and test alike
        X = stream.lap_windows(X, list(meta["ch_names"]))
    if args.car:                           # T2-D475: common average reference per window (predict-safe), before alignment, train and test alike
        X = stream.car_windows(X)
    from ni26 import train as _T                       # row 77 / T2-D305: the MIRepNet member's montage (unused by every other model)
    _T.MIREP_CH[:] = list(meta["ch_names"])
    if args.fm_ckpt:                       # review T2 P2e: no window of a scored session in the checkpoint's (pre)training pools
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import t2p_fm
        keep = []
        for proto in args.protocols.split(","):
            leak = t2p_fm.cpt_leak(args.fm_ckpt, protocol_test_streams(idx, proto))
            if leak is None:
                print(f"[t2p_bench] FLAG {proto}: --fm-ckpt has no pool record (pinned root / contract-only): leak UNVERIFIED",
                      flush=True)
            elif leak:
                msg = f"{proto}: the --fm-ckpt pools hold {len(leak)} of its test streams {leak}"
                if not args.skip_leaky_protocols:
                    raise SystemExit(f"!! {msg} (A10); --skip-leaky-protocols drops the protocol instead")
                print(f"[t2p_bench] FLAG {msg}: protocol SKIPPED", flush=True)
                Path(args.out).mkdir(parents=True, exist_ok=True)
                (Path(args.out) / f"LEAKY_{proto}.json").write_text(json.dumps(dict(protocol=proto, fm_ckpt=args.fm_ckpt,
                                                                                     leaked_test_streams=leak), indent=1))
                continue
            keep.append(proto)
        if not keep:
            print("[t2p_bench] every protocol skipped (leaky --fm-ckpt pools): nothing to run", flush=True)
            return
        args.protocols = ",".join(keep)
    extras = [extra_users(load_extra(d), args.extra_users) for d in args.extra.split(",") if d]
    xtag = ("__x" + "+".join(e["name"].replace("export_", "") for e in extras)
            + (f"f{args.extra_frac:g}" if args.extra_frac < 1 else "") + (f"u{args.extra_users}" if args.extra_users else "")) if extras else ""
    xtag += "__car" if args.car else ""                                 # T2-D475
    xtag += "__km" if args.keep_mean else ""                            # T2-D517
    xtag += f"__side{args.side}" if args.side else ""
    xtag += f"__view{args.view}" if args.view else ""                   # T2-D519
    assert not args.view or not (args.peruser or args.tta_vp or args.instream_ft or args.instream_sched), "T2-D519: --view is for pooled nets only"
    assert not args.side or not (args.peruser or args.tta_vp or args.instream_ft or args.instream_sched), "T2-D517: --side is for pooled nets only"
    assert args.fm_side != "dcmean" or args.side == "dcmean", "T2-D524: --fm-side dcmean needs --side dcmean (the side vectors)"
    assert not args.fm_phase or not args.fm_cont_epochs, "T2-D586: --fm-phase is registered for the plain non-continuation SD line only"
    assert not args.fm_mirror or (not args.fm_cont_epochs and 0 < args.fm_mirror <= 1), "T2-D618: --fm-mirror p in (0, 1], the plain non-continuation SD line only"
    SIDE_RAW = side_raw(X_delivered, args.side) if args.side else None   # T2-D524: = side_raw(X) without --car / --lap (bitwise as D517)
    if args.keep_mean:
        stream.KEEP_MEAN["on"] = True
    if extras and (args.extra_head or args.extra_w != 1.0 or args.extra_dsbn):   # T2-D507
        xtag += "__xd" + ("h" if args.extra_head else "") + (f"w{args.extra_w:g}".replace(".", "") if args.extra_w != 1.0 else "") \
            + ("b" if args.extra_dsbn else "")
    xtag += "__lap" if args.lap else ""                                 # T2-D480
    xtag += "__ubw" if args.user_balance else ""                        # T2-D630
    xtag += f"p{args.user_balance_pow:g}".replace(".", "") if args.user_balance and args.user_balance_pow != 1.0 else ""   # T2-D647 (__ubwp05)
    LJ = None
    if args.long_jitter:                                                 # T2-D486: real out-of-window training data
        assert args.long_export, "--long-jitter needs --long-export"
        assert not (args.car or args.lap or args.softclip or args.crop or args.iaf_warp or args.recolour_own or args.bag or extras or args.drop_train), \
            "T2-D486: --long-jitter is wired for the plain contract input (no other input transform, no bags, no extras)"
        ja, jb = (float(v) for v in args.long_jitter.split(","))
        XL_, off_, ls_ = long_check(args.long_export, X, idx, meta)
        LJ = dict(XL=XL_, off=off_, lo=int(round((ja - ls_) * float(meta["sfreq"]))), hi=int(round((jb - ls_) * float(meta["sfreq"]))))
        assert 0 <= LJ["lo"] <= LJ["off"] <= LJ["hi"] <= XL_.shape[-1] - X.shape[-1], ("T2-D486: the range must hold the contract start", LJ["lo"], LJ["off"], LJ["hi"])
        xtag += "__lj" + f"{ja:g}-{jb:g}".replace(".", "")
        print(f"[t2p_bench] T2-D486 long jitter: starts {ja:g} .. {jb:g} s re cue = samples {LJ['lo']} .. {LJ['hi']} of {XL_.shape[-1]} "
              f"(contract {LJ['off']}); contract slice bitwise = the export, rows aligned", flush=True)
    ext_cache = {}
    fs = float(meta["sfreq"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    drop = json.loads(Path(args.drop_train).read_text()) if args.drop_train else {}
    for proto in args.protocols.split(","):
        for fname, trm, vam, tem in folds(idx, proto):
            if drop and fname == proto:                 # T2-D307: training-set cleaning (only the protocol's own training rows)
                dr = np.asarray(drop[proto], dtype=int)
                assert trm[dr].all(), f"{proto}: a dropped window is not a training window"
                trm = trm.copy(); trm[dr] = False
                print(f"[t2p_bench] {proto}: {len(dr)} training windows dropped ({args.drop_train}); {int(trm.sum())} kept", flush=True)
            if args.folds and fname not in args.folds.split(","):
                continue
            for model, align, seed in itertools.product(args.models.split(","), args.aligns.split(","),
                                                        [int(s) for s in args.seeds.split(",")]):
                if model in CLASSICAL and seed != int(args.seeds.split(",")[0]):
                    continue  # deterministic
                tasks.append((proto, fname, trm, vam, tem, model, align, seed))
    if args.reverse:
        tasks = tasks[::-1]
    if args.list_outputs:                  # review T2 P3j: t2p_multi's artefact gate = every file these arguments produce
        with open(args.list_outputs, "a") as fh:
            for (proto, fname, trm, vam, tem, model, align, seed) in tasks:
                fh.write(f"{task_tag(args, fname, model, align, seed, xtag)}.npz\n")
                if args.tta_arms and args.tta_mae and model.startswith("fm:st_"):    # T2-D283: the TTA twins are gated too
                    t_, rec_ = task_tag(args, fname, model, align, seed, xtag), task_rec(args, model)
                    for lr_ in [float(v) for v in args.tta_mae.split(",")]:
                        for arm_ in args.tta_arms.split(","):
                            fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}mae{lr_:g}{arm_}{xtag}__") + ".npz\n")
                if args.tta_vp and not model.startswith("fm:") and model not in CLASSICAL:    # backlog 71: the TTA twin is gated too
                    t_, rec_ = task_tag(args, fname, model, align, seed, xtag), task_rec(args, model)
                    fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}tv{args.tta_vp}{xtag}__") + ".npz\n")
                    if args.tta_shift:
                        fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}tv{args.tta_vp}s{args.tta_shift}{xtag}__") + ".npz\n")
                    if args.tta_copies:
                        fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}tv{args.tta_vp}c{xtag}__") + ".npz\n")
                if args.test_mirror and not model.startswith("fm:") and model not in CLASSICAL:    # T2-D619: the mirrored twins are gated too
                    t_, rec_ = task_tag(args, fname, model, align, seed, xtag), task_rec(args, model)
                    for sfx_ in ["mr"] + ([f"tv{args.tta_vp}mr"] if args.tta_vp else []):
                        fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}{sfx_}{xtag}__") + ".npz\n")
                if args.fm_soup_members and args.fm_soup > 1 and model.startswith("fm:"):    # T2-D493: the soup + its members are gated
                    t_, rec_ = task_tag(args, fname, model, align, seed, xtag), task_rec(args, model)
                    for sfx_ in [f"soup{args.fm_soup}"] + [f"soup{args.fm_soup}m{j_}" for j_ in range(1, args.fm_soup)]:
                        fh.write(t_.replace(f"__{rec_}{xtag}__", f"__{rec_}{sfx_}{xtag}__") + ".npz\n")
        print(f"[t2p_bench] {len(tasks)} expected result files -> {args.list_outputs}", flush=True)
        return
    mine = tasks[args.shard::args.nshards]
    print(f"[t2p_bench] {len(tasks)} tasks, shard {args.shard}/{args.nshards}: {len(mine)}", flush=True)

    import torch
    from ni26 import train as T
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recipe = T.RECIPES[args.recipe]
    if args.epochs:
        from dataclasses import replace
        recipe = replace(recipe, epochs=args.epochs)
    if recipe.rc_ext:                                    # backlog 65 (a): the external covariance bank (rc_bank's targets)
        assert args.rc_ext_bank, "an rc_ext recipe needs --rc-ext-bank"
        bk = np.load(args.rc_ext_bank)
        assert list(bk["ch_names"]) == list(meta["ch_names"]), "the bank must be on the export's montage"
        T.RC_EXT_BANK["C"] = bk["C"]
        print(f"[t2p_bench] external re-colouring bank: {len(bk['C'])} subjects ({', '.join(f'{c} {int((bk['corpus'] == c).sum())}' for c in np.unique(bk['corpus']))})", flush=True)
    vpk = recipe.vp or args.fm_vp                        # T2-D198: the FM members' --fm-vp uses the same bank
    assert not (recipe.vp and args.fm_vp and recipe.vp != args.fm_vp), "one bank per job"
    if vpk:                                              # T2-D193: the virtual-participant bank (fixed seed: every job draws from the same bank)
        T.VP_BANK.update({k_: v_.to(dev) for k_, v_ in vp_bank(list(meta["ch_names"]), K=recipe.vp_k, sigma=0.005 * vpk, gain=recipe.vp_g, shift=recipe.vp_s, vref=recipe.vp_r, cap=recipe.vp_cap).items()})   # T2-D195: vp k = k x 5 mm; T2-D202: K
        print(f"[t2p_bench] virtual participants: {len(T.VP_BANK['mats'])} draws, |M - I| max "
              f"{float((T.VP_BANK['mats'] - torch.eye(T.VP_BANK['mats'].shape[1], device=dev)).abs().max()):.3f}", flush=True)
    cache = {}
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    y = idx["y"].to_numpy().astype(int)
    for (proto, fname, trm, vam, tem, model, align, seed) in mine:
        tag = task_tag(args, fname, model, align, seed, xtag)
        rec = task_rec(args, model)                                   # the twins' stems (f7b0afb left it undefined here)
        fp = out / f"{tag}.npz"
        claim = out / f".{tag}.claim"
        if fp.exists() or (claim.exists() and time.time() - claim.stat().st_mtime < 3 * 3600):   # stale claim = killed job
            continue
        claim.write_text(os.environ.get("SLURM_JOB_ID", "local"))     # another job / shard skips this task
        t0 = time.time()
        key = (fname, align)
        if key not in cache:
            cache.clear()
            scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
            cache[key] = aligned_all(X, idx, align, trm, args.n0, args.shrink, scale)
        XA = cache[key]
        if args.iaf_warp and not model.startswith("fm:"):           # T2-D191: the IAF warp (per protocol: the training split sets the factors)
            wkey = ("iaf", key, proto, fname)
            if wkey not in cache:
                if "iaf_win" not in cache:
                    cache["iaf_win"] = window_iaf(X, fs, list(meta["ch_names"]))
                cache[wkey] = iaf_warp(XA, iaf_factors(cache["iaf_win"], idx, trm))
                print(f"[t2p_bench] IAF warp ({proto}/{fname}): window IAF median {np.median(cache['iaf_win']):.2f} Hz", flush=True)
            XA = cache[wkey]
        if args.crop and not model.startswith("fm:"):
            a_, b_ = (int(v) for v in args.crop.split(":"))
            XA = XA[..., a_:b_]
        if args.recolour_own and not model.startswith("fm:"):       # T2-D347: test rows re-aligned to the routed user's training geometry
            rkey = ("rco", key, proto, fname)
            if rkey not in cache:
                cache[rkey] = recolour_own(XA, idx, trm, tem, np.load(args.recolour_own))
                print(f"[t2p_bench] recolour-own ({proto}/{fname}): {int(tem.sum())} test windows", flush=True)
            XA = cache[rkey]
        if args.view and not model.startswith("fm:"):               # T2-D519: the member's temporal view (train and test alike)
            vkey = ("view", key, args.view)
            if vkey not in cache:
                cache[vkey] = temporal_view(XA, args.view)
            XA = cache[vkey]
        tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
        res = dict(protocol=proto, fold=fname, model=model, align=align, n0=args.n0, shrink=args.shrink,
                   recipe=args.recipe, epochs=args.epochs or recipe.epochs, seed=seed, export=str(args.export),
                   export_sha=meta.get("x_sha256"), n_train=len(tr), n_val=len(va), n_test=len(te),
                   extra=[dict(name=e["name"], sha=e["sha"], n=int(len(e["keep"]))) for e in extras])
        ext_native, ext_interp = [], []
        for e in extras:
            kind = "native" if model.startswith("fm:") else "interp"
            k = (e["name"], align, kind)
            if k not in ext_cache:
                ext_cache[k] = aligned_extra(e, align, args.shrink, None if kind == "native" else list(meta["ch_names"]),
                                             frac=args.extra_frac, car=bool(args.car and kind != "native"))
            res["extra"][extras.index(e)].update(frac=args.extra_frac, n_used=int(len(ext_cache[k][1])))
            (ext_native if kind == "native" else ext_interp).append((*ext_cache[k], e["names"]))
        Xtr_, ytr_ = XA[tr], y[tr]
        ltr_ = None
        if LJ is not None:                                # T2-D486: this task's long aligned training windows (rows = Xtr_)
            assert not model.startswith("fm:") and model not in CLASSICAL, "T2-D486: CNN members first (the ST-L only on a pass)"
            lk_ = ("long", fname, align)
            if lk_ not in cache:
                cache[lk_] = long_aligned(X, XA, LJ["XL"], tr)
                print(f"[{tag}] T2-D486: {len(tr)} long training windows aligned (operator residual {cache[lk_][1]:.1e})", flush=True)
            ltr_ = cache[lk_][0]
            T.LONG.update(lo=LJ["lo"], hi=LJ["hi"])
        tr_users = sorted(idx.iloc[tr]["subject"].unique())
        utr_ = idx.iloc[tr]["subject"].map({u: k for k, u in enumerate(tr_users)}).to_numpy()
        sw_ = None
        if args.fm_sw and model.startswith("fm:"):       # T2-D253: up-weight the training windows the out-of-fold CNNs get wrong
            pc_ = oof_pcnn(args.fm_sw_oof, proto, y, tr)
            sw_ = 1.0 + args.fm_sw * (1.0 - pc_)
            sw_ = (sw_ / sw_.mean()).astype(np.float32)
            print(f"[{tag}] OOF CNN p(y) on {len(tr)} training windows: mean {pc_.mean():.3f}, < 1/3 for {(pc_ < 1 / 3).mean():.1%}; "
                  f"loss weights {sw_.min():.2f}-{sw_.max():.2f} (mean 1)", flush=True)
        pk_ = None
        if args.fm_pkg and model.startswith("fm:"):      # backlog 96: the package loss through the fixed out-of-fold CNN mix
            assert not extras, "--fm-pkg: no --extra (the OOF rows are the target training windows)"
            pk_ = (oof_pcnn(args.fm_sw_oof, proto, y, tr, full=True).astype(np.float32), float(args.fm_pkg_w), float(args.fm_pkg))
            print(f"[{tag}] package loss: OOF CNN mix on {len(tr)} training windows (p(y) mean {pk_[0][np.arange(len(tr)), y[tr]].mean():.3f}); "
                  f"beta {args.fm_pkg:g}, w {args.fm_pkg_w:g}", flush=True)
        dtr_ = None
        if ext_interp and (args.extra_head or args.extra_w != 1.0 or args.extra_dsbn):   # T2-D507: one domain id per training row
            assert not model.startswith("fm:") and model not in CLASSICAL, "T2-D507: CNN pooled nets only"
            dtr_ = np.concatenate([np.zeros(len(Xtr_), np.int64)] + [np.full(len(b), 1, np.int64) for _, b, _ in ext_interp])
            T.XDOM.clear()
            T.XDOM.update(head=bool(args.extra_head), w=float(args.extra_w), dsbn=bool(args.extra_dsbn))
            res.update(xdom=dict(T.XDOM))
        if ext_interp:
            Xtr_ = np.concatenate([Xtr_] + [a for a, _, _ in ext_interp])
            ytr_ = np.concatenate([ytr_] + [b for _, b, _ in ext_interp])
        if recipe.vs and not model.startswith("fm:") and model not in CLASSICAL:   # T2-D308: this task's own training rows only
            T.VS_BANK.clear()
            T.VS_BANK["mats"] = vs_bank(XA, tr, idx).to(dev)
            print(f"[{tag}] virtual sessions: {len(T.VS_BANK['mats'])} same-user drift maps, |M - I| mean "
                  f"{float((T.VS_BANK['mats'] - torch.eye(T.VS_BANK['mats'].shape[1], device=dev)).abs().mean()):.4f}", flush=True)
        if recipe.syn and not model.startswith("fm:") and model not in CLASSICAL:   # T2-D413: this protocol's MeanFlow bank, on the task's scale
            assert args.syn_bank and align == "none" and not (ext_interp or args.crop or args.iaf_warp or args.recolour_own or args.softclip), \
                "T2-D413: syn needs --syn-bank, align none and no other input transform"
            sp_ = args.syn_bank.format(protocol=proto)
            zb = np.load(sp_)
            assert str(zb["protocol"]) == proto and np.array_equal(zb["train_idx"], tr) and list(zb["users"]) == tr_users, f"T2-D413: {sp_} is not this task's bank"
            sc_ = float(np.median(np.asarray(X[tr[::7]]).std(-1)))   # aligned_all's align-none scale for this training split
            import hashlib
            T.SYN_BANK.clear()
            T.SYN_BANK.update(X=(torch.as_tensor(zb["X"], device=dev).float() / sc_).half(), y=torch.as_tensor(zb["y"].astype(np.int64), device=dev),
                              u=torch.as_tensor(zb["u"].astype(np.int64), device=dev))
            res.update(syn=dict(bank=sp_, sha256=hashlib.sha256(Path(sp_).read_bytes()).hexdigest(), n=int(len(zb["y"])), ratio=recipe.syn, scale=sc_))
            print(f"[{tag}] MeanFlow bank {sp_}: {len(zb['y'])} windows, classes {np.bincount(zb['y']).tolist()}, scale {sc_:.3f}, "
                  f"sha256 {res['syn']['sha256'][:12]}", flush=True)
        if model in CLASSICAL:
            clf = (FBTS(fs) if model == "fbts" else LVLDA(fs)).fit(Xtr_, ytr_)
            pte = clf.predict_proba(XA[te])
            np.savez_compressed(str(fp) + '.tmp.npz', test_idx=te, prob_last=pte.astype(np.float32), prob_best=pte.astype(np.float32),
                                res=json.dumps(res))
        else:
            data = dict(Xtr=Xtr_, ytr=ytr_, Xva=XA[va], yva=y[va], cva=cells[va], Xte=XA[te], yte=y[te],
                        cte=cells[te], ext=ext_native, utr=utr_, str=idx.iloc[tr]["session"].astype(str).to_numpy(), ltr=ltr_)
            if dtr_ is not None:                          # T2-D507
                data["dtr"] = dtr_
            if args.user_balance:                         # T2-D630: 1 / (the user's number of TRAINING sessions) per training window
                assert len(Xtr_) == len(tr) and not args.drop_train and not args.bag, \
                    "T2-D630: --user-balance = the pooled nets on the protocol's own rows only (T2-D642: FM members too, via fit_eval_fm's data['wtr'])"
                sub_tr = idx.iloc[tr]["subject"].astype(str).to_numpy()
                ses_tr = idx.iloc[tr]["session"].astype(str).to_numpy()
                data["wtr"], wu_ = user_balance_weights(sub_tr, ses_tr, args.user_balance_pow)   # T2-D647: the power knob (1 = D630 bitwise)
                res.update(user_balance={u.split("/")[-1]: round(w_, 4) for u, w_ in wu_.items()},
                           **({"user_balance_pow": args.user_balance_pow} if args.user_balance_pow != 1.0 else {}))
                print(f"[{tag}] T2-D630 user-balanced loss weights: {res['user_balance']}", flush=True)
            if SIDE_RAW is not None:                      # T2-D517: the fold's side vectors (prior = this fold's training rows)
                Sz = side_z(SIDE_RAW, idx, tr, args.side)
                data.update(side_tr=Sz[tr], side_va=Sz[va], side_te=Sz[te])
            assert not ext_interp or model not in ("fbcnet_uid", "fbcnet_moe"), "user heads: no --extra"
            if recipe.kd > 0 and not model.startswith("fm:") and model not in CLASSICAL:   # backlog 69: the FM teacher of this protocol
                assert args.kd_teacher and not ext_interp, "kd needs --kd-teacher and no --extra"
                tp = np.load(args.kd_teacher.format(protocol=proto, fold=fname))
                assert np.array_equal(tp["train_idx"], tr), "the teacher's training windows must be this task's"
                data["ptr"] = tp["p_train"]
                if recipe.kd_ext:
                    assert args.kd_pools, "kd_ext needs --kd-pools (the teacher dump's)"
                    if ("kd", align) not in ext_cache:
                        if "kd_sel" not in ext_cache:
                            ext_cache["kd_sel"] = kd_select(args.kd_pools.split(","), args.kd_n)
                        ext_cache[("kd", align)] = kd_aligned(ext_cache["kd_sel"], align, args.shrink, list(meta["ch_names"]))
                    Xk, kk = ext_cache[("kd", align)]
                    assert np.array_equal(kk, tp["kd_key"]), "the distillation windows must be the teacher dump's"
                    data["Xkd"], data["pkd"] = Xk, tp["p_ext"]
                res.update(kd=dict(teacher=args.kd_teacher.format(protocol=proto, fold=fname), alpha=recipe.kd, t=recipe.kd_t, ext=recipe.kd_ext,
                                   n_ext=int(len(tp["kd_key"])) if recipe.kd_ext else 0))
            r = None
            fm_cont = {}                     # T2-D546: the continuation arms' results (twins c<arm>)
            for attempt in range(4):        # several shards share one GPU: an OOM waits and retries, never kills the shard
                try:
                    if model.startswith("fm:"):
                        sys.path.insert(0, str(Path(__file__).resolve().parent))
                        import t2p_fm
                        _, fmname, readout = model.split(":")
                        fm_kw = dict(epochs=args.fm_epochs, bs=args.fm_bs,
                                               lr_enc=args.fm_lr_enc, lr_head=args.fm_lr_head, wd=args.fm_wd,
                                               llrd=args.fm_llrd, readout=readout, drop_path=args.fm_drop_path,
                                               ema=args.fm_ema, device=dev, log=lambda m: print(f"[{tag}] {m}", flush=True),
                                               layer=args.fm_layer, lora=args.fm_lora, lora_alpha=args.fm_lora_alpha, lora_dropout=args.fm_lora_dropout,
                                               mixup=args.fm_mixup,
                                               calib=args.fm_calib, fm_ckpt=args.fm_ckpt, fuse=args.fm_fuse,
                                               band=tuple(float(v) for v in args.fm_band.split(":")) if args.fm_band else None,
                                               return_state=bool(args.tta_mae), supcon=args.fm_supcon,
                                               supcon_mode=args.fm_supcon_mode, supcon_tau=args.fm_supcon_tau,
                                               qkv_lr_mult=args.fm_qkv_lr_mult, layer_clip=args.fm_layer_clip,
                                               **({"chemb_lr": args.fm_chemb_lr} if args.fm_chemb_lr else {}),   # T2-D644
                                               **({"dro_eta": args.fm_dro} if args.fm_dro else {}),   # T2-D645
                                               **({"grl": args.fm_grl, "grl_lr": args.fm_grl_lr} if args.fm_grl is not None else {}),   # T2-D649
                                               lora_in=args.fm_lora_in, lora_in_roles=args.fm_lora_in_roles,
                                               lora_in_dropout=args.fm_lora_in_dropout, mae_retain_every=args.fm_mae_retain, vp=args.fm_vp,
                                               side=args.fm_side, rc=args.fm_rc, tilt=args.fm_tilt, sw=sw_, pkg=pk_, aug=fm_aug(args.fm_aug), ls=args.fm_ls,
                                               recomb=(tuple(float(v) for v in args.fm_recomb.split(":")) if args.fm_recomb else None),
                                               phase=(tuple(float(v) for v in args.fm_phase.split(":")) if args.fm_phase else None),
                                               cons=args.fm_cons, wmargin=args.fm_wmargin, mirror=args.fm_mirror,
                                               lp_epochs=args.fm_lp_epochs, reinit_top=args.fm_reinit_top, wise=args.fm_wise, micro=args.fm_micro,
                                               head_freeze=args.fm_head_freeze, head_wd=args.fm_head_wd, token_drop=args.fm_token_drop)
                        if args.fm_train_eval:          # T2-D468: per-epoch clean training accuracy (diagnostic; default off = unchanged)
                            assert len(Xtr_) == len(tr), "--fm-train-eval: no --extra windows"
                            fm_kw["train_eval"] = True
                            data = dict(data, ctr=cells[tr])
                        r = t2p_fm.fit_eval_fm(fmname, data, seed, meta["ch_names"], **dict(
                            fm_kw, return_state=fm_kw["return_state"] or args.fm_soup > 1 or args.fm_cont_epochs > 0,
                            return_model=bool(args.fm_peruser_heads) or args.fm_soup > 1 or bool(args.fm_peruser_lora) or args.fm_kd_dump))
                        if args.fm_cont_epochs:          # T2-D546: both arms from the SAME converged base state (paired within the job)
                            assert r.get("state_last") is not None and set(args.fm_cont_arms.split(",")) <= {"a", "b"}, args.fm_cont_arms
                            for arm_ in args.fm_cont_arms.split(","):
                                kw_ = dict(fm_kw, epochs=args.fm_cont_epochs, lr_enc=args.fm_lr_enc * args.fm_cont_lr, lr_head=args.fm_lr_head * args.fm_cont_lr,
                                           warm=0.0, return_state=False, init_state=r["state_last"], cont_reinit=(arm_ == "a"))
                                kw_["log"] = lambda m, a_=arm_: print(f"[{tag} c{a_}] {m}", flush=True)
                                fm_cont[arm_] = t2p_fm.fit_eval_fm(fmname, data, seed + 50000, meta["ch_names"], **kw_)
                            r["state_last"] = None
                        res.update(fm_args=dict(epochs=args.fm_epochs, bs=args.fm_bs, lr_enc=args.fm_lr_enc,
                                                lr_head=args.fm_lr_head, wd=args.fm_wd, llrd=args.fm_llrd, ema=args.fm_ema,
                                                drop_path=args.fm_drop_path, layer=args.fm_layer, lora=args.fm_lora, lora_alpha=args.fm_lora_alpha, lora_dropout=args.fm_lora_dropout,
                                                mixup=args.fm_mixup, calib=args.fm_calib, fm_ckpt=args.fm_ckpt,
                                                fuse=args.fm_fuse, band=args.fm_band,
                                                # review T2 P3k: every knob of the result-file tag is recorded
                                                supcon=args.fm_supcon, supcon_mode=args.fm_supcon_mode,
                                                supcon_tau=args.fm_supcon_tau, lora_in=args.fm_lora_in,
                                                lora_in_roles=args.fm_lora_in_roles, lora_in_dropout=args.fm_lora_in_dropout,
                                                mae_retain=args.fm_mae_retain, qkv_lr_mult=args.fm_qkv_lr_mult,
                                                **({"chemb_lr": args.fm_chemb_lr} if args.fm_chemb_lr else {}),   # T2-D644: recorded only when on
                                                **({"dro_eta": args.fm_dro} if args.fm_dro else {}),   # T2-D645: recorded only when on
                                                **({"grl": args.fm_grl, "grl_lr": args.fm_grl_lr} if args.fm_grl is not None else {}),   # T2-D649
                                                layer_clip=args.fm_layer_clip, cpt=t2p_fm.t1.CPT_INFO.get(fmname), vp=args.fm_vp,
                                                side=args.fm_side, rc=args.fm_rc, tilt=args.fm_tilt, sw=args.fm_sw, sw_oof=args.fm_sw_oof, pkg=args.fm_pkg, pkg_w=args.fm_pkg_w, aug=args.fm_aug, recomb=args.fm_recomb, phase=args.fm_phase, mirror=args.fm_mirror, cons=args.fm_cons, wmargin=args.fm_wmargin, ls=args.fm_ls,
                                                lp_epochs=args.fm_lp_epochs, reinit_top=args.fm_reinit_top, wise=args.fm_wise, micro=args.fm_micro,
                                               head_freeze=args.fm_head_freeze, head_wd=args.fm_head_wd, token_drop=args.fm_token_drop,
                                               cont_epochs=args.fm_cont_epochs, cont_lr=args.fm_cont_lr, cont_arms=args.fm_cont_arms if args.fm_cont_epochs else ""),
                                   provenance=r["provenance"])
                    elif args.bag:
                        rs = []
                        if args.bag in ("s3", "ovr", "pair", "ovc"):   # T2-D440: class experts, three nets per seed
                            assert not ext_interp, "T2-D440: no --extra"
                            has_va = len(va) > 0
                            pe_, pv_, cur_ = [], [], []
                            for k in range(3):
                                if args.bag == "s3":
                                    rk = T.fit_eval(model, data, recipe, seed * 100 + k, device=dev, sfreq=fs, log_every=recipe.epochs)
                                    pe_.append(rk["prob_last"]); pv_.append(rk["prob_val_last"]); cur_.append(rk["curves"][-1])
                                    continue
                                if args.bag == "pair":
                                    a_, b_ = EXPERT_PAIRS[k]
                                    rw = np.flatnonzero((ytr_ == a_) | (ytr_ == b_))
                                    lab = lambda v: (v == b_).astype(v.dtype)                     # noqa: E731
                                else:                                                              # class k vs rest, positives drawn twice (balanced)
                                    rw = np.sort(np.concatenate([np.arange(len(ytr_)), np.flatnonzero(ytr_ == k)]))
                                    lab = lambda v: (v == k).astype(v.dtype)                      # noqa: E731
                                msk = expert_mask(list(meta["ch_names"]), k) if args.bag == "ovc" else 1.0
                                dk = dict(data, Xtr=Xtr_[rw] * msk, ytr=lab(ytr_[rw]), utr=utr_[rw], str=data["str"][rw], Xva=data["Xva"] * msk, yva=lab(data["yva"]),
                                          Xte=data["Xte"] * msk, yte=lab(data["yte"]))
                                rk = T.fit_eval(model, dk, recipe, seed * 100 + k, device=dev, sfreq=fs, log_every=recipe.epochs)
                                pe_.append(rk["prob_last"][:, 1]); pv_.append(rk["prob_val_last"][:, 1] if has_va and rk["prob_val_last"] is not None else None)
                                cur_.append(rk["curves"][-1])
                            comb = (lambda L: np.mean(L, 0)) if args.bag == "s3" else (lambda L: expert_combine(np.stack(L, 1), args.bag))   # noqa: E731
                            r = dict(rk, prob_last=comb(pe_), prob_val_last=comb(pv_) if pv_[0] is not None else None,
                                     prob_best=None, best_epoch=-1, best_val=float("nan"), curves=cur_)
                            res.update(bag=args.bag, bag_k=3)
                            break
                        if args.bag in ("ub2", "ub5"):               # T2-D438: round 1 = leave-one-user-out errors of the training users
                            assert not ext_interp, "T2-D438: no --extra"
                            e_ = []
                            for k in range(len(tr_users)):
                                ho = utr_ == k
                                d1 = dict(data, Xtr=Xtr_[~ho], ytr=ytr_[~ho], utr=utr_[~ho], str=data["str"][~ho], Xva=Xtr_[:0], yva=ytr_[:0], cva=cells[va][:0],
                                          Xte=Xtr_[ho], yte=ytr_[ho], cte=np.full(int(ho.sum()), "oof"))
                                e_.append(float((T.fit_eval(model, d1, recipe, seed * 100 + 50 + k, device=dev, sfreq=fs, log_every=recipe.epochs)["prob_last"].argmax(1) != ytr_[ho]).mean()))
                            bm_ = boost_mult(np.array(e_), 2.0 if args.bag == "ub2" else 5.0)
                            res.update(boost_err=e_, boost_mult=bm_.tolist())
                            print(f"[{tag}] T2-D438 leave-one-user-out errors {np.round(e_, 3).tolist()} -> user multipliers {np.round(bm_, 2).tolist()}", flush=True)
                        for k in range(len(tr_users)):
                            if args.bag in ("ub2", "ub5"):
                                keep = boost_rows(utr_, bm_, seed * 100 + k)
                                dk = dict(data, Xtr=Xtr_[keep], ytr=ytr_[keep], utr=utr_[keep], str=data["str"][keep])
                                rs.append(T.fit_eval(model, dk, recipe, seed * 100 + k, device=dev, sfreq=fs, log_every=recipe.epochs))
                                continue
                            if args.bag in ("uw1", "uw4", "runs"):        # T2-D437: resampled rows, no user removed
                                assert not ext_interp, "T2-D437 bags: no --extra"
                                it_ = idx.iloc[tr]
                                keep = bag_rows(args.bag, utr_, (it_["session"].astype(str) + "|" + it_["run"].astype(str)).to_numpy(), seed * 100 + k)
                                dk = dict(data, Xtr=Xtr_[keep], ytr=ytr_[keep], utr=utr_[keep], str=data["str"][keep])
                                rs.append(T.fit_eval(model, dk, recipe, seed * 100 + k, device=dev, sfreq=fs, log_every=recipe.epochs))
                                continue
                            keep = (utr_ != k) if args.bag == "users" else np.ones(len(utr_), bool)
                            dk = dict(data, Xtr=Xtr_[keep], ytr=ytr_[keep], utr=utr_[keep])
                            rs.append(T.fit_eval(model, dk, recipe, seed * 100 + k, device=dev, sfreq=fs, log_every=recipe.epochs))
                        r = dict(rs[0], prob_last=np.mean([x["prob_last"] for x in rs], 0),
                                 prob_val_last=np.mean([x["prob_val_last"] for x in rs], 0) if rs[0]["prob_val_last"] is not None else None,
                                 prob_best=None, best_epoch=-1, best_val=float("nan"), curves=[x["curves"][-1] for x in rs])
                        er_ = np.stack([x["prob_last"].argmax(1) != y[te] for x in rs]).astype(float)      # T2-D437: diversity inside the bag
                        cc_ = np.corrcoef(er_)
                        res.update(bag=args.bag, bag_k=len(rs), bag_users=tr_users, bag_phi=float(cc_[np.triu_indices(len(rs), 1)].mean()),
                                   bag_acc=float(1 - er_.mean()))
                    else:
                        T.CNN_INIT.clear()
                        if args.cnn_init:                            # T2-D367: this protocol's SSL-pretrained init
                            assert args.cnn_init_tag and not model.startswith("fm:") and model not in CLASSICAL, "--cnn-init: a CNN model + --cnn-init-tag"
                            T.CNN_INIT["sd"] = torch.load(args.cnn_init.format(protocol=proto), map_location="cpu", weights_only=True)
                            print(f"[{tag}] pooled CNN init: {args.cnn_init.format(protocol=proto)}", flush=True)
                        r = T.fit_eval(model, data, recipe, seed, device=dev, sfreq=fs, log_every=5,
                                       return_state=bool(args.tta_bn or args.tta_pl or args.peruser or args.tta_vp or args.instream_ft or args.instream_sched
                                                         or args.test_mirror) or model == "fbcnet_moe")
                    break
                except torch.OutOfMemoryError:
                    r = None
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(f"[t2p_bench] {tag}: CUDA OOM (attempt {attempt + 1}/4); retry in 120 s", flush=True)
                    time.sleep(120)
            if r is None:
                claim.unlink(missing_ok=True)   # left for another job / a resume (RUN_NAME)
                print(f"[t2p_bench] {tag}: SKIPPED after 4 OOMs", flush=True)
                continue
            res.update(best_epoch=r["best_epoch"], best_val=r["best_val"], n_params=r["n_params"], curves=r["curves"])
            if "grl_curve" in r:                          # T2-D649: the engagement record (also inside curves) + its chance level
                res.update(grl_curve=r["grl_curve"], grl_chance=r["grl_chance"], grl_ramp_end_step=r["grl_ramp_end_step"])
            if r.get("uid_test") is not None:
                res.update(train_users=tr_users)
            pv = r.get("prob_val_last")
            uids = {k: r[k].astype(np.float32) for k in ("uid_test", "uid_val") if r.get(k) is not None}
            sdump = {}
            if r.get("side_dump") is not None and not args.bag:   # T2-D533 (NI26_SIDE_DUMP=1): the side head's pre-head logits, weights and the test z
                sdump = dict(side_logit_last=r["side_dump"]["logit_last"].astype(np.float32), side_w=r["side_dump"]["w"].astype(np.float32),
                             side_b=r["side_dump"]["b"].astype(np.float32), side_te=np.asarray(data["side_te"], np.float32))
            if r.get("feat_dump") is not None and not args.bag:   # T2-D555 (NI26_FEAT_DUMP=1): the FM member's pooled features + its main head
                sdump.update(fm_feat_te=r["feat_dump"]["feat"].astype(np.float32), fm_out_w=r["feat_dump"]["w"].astype(np.float32),
                             fm_out_b=r["feat_dump"]["b"].astype(np.float32))
            if r.get("view_dump") is not None and not args.bag:   # T2-D654 (NI26_VIEW_DUMP=K): the fixed test-time views (features + probabilities + spec)
                sdump.update(fm_view_feat=r["view_dump"]["feat"].astype(np.float32), fm_view_prob=r["view_dump"]["prob"].astype(np.float32),
                             fm_view_gain=r["view_dump"]["gain"].astype(np.float32), fm_view_shift=r["view_dump"]["shift"].astype(np.int64))
            np.savez_compressed(str(fp) + '.tmp.npz', test_idx=te, prob_last=r["prob_last"].astype(np.float32),
                                val_idx=va, prob_val_last=(pv if pv is not None else np.zeros((0, 3))).astype(np.float32),
                                prob_best=(r["prob_best"] if r["prob_best"] is not None else r["prob_last"]).astype(np.float32),
                                res=json.dumps(res), **uids, **sdump)
        os.replace(str(fp) + ".tmp.npz", fp)
        claim.unlink(missing_ok=True)
        net = None
        if not model.startswith("fm:") and model not in CLASSICAL and r.get("state_last") is not None:
            net = T.build_model(model, X.shape[1], 3, XA.shape[2], fs, n_users=len(tr_users) if "_" in model else 0).to(dev)
            net.load_state_dict(r["state_last"])
            net.eval()

        def twin(suffix, pte_m, pva_m, **extra):
            fpm = out / f"{tag.replace(f'__{rec}{xtag}__', f'__{rec}{suffix}{xtag}__')}.npz"
            np.savez_compressed(str(fpm) + ".tmp.npz", test_idx=te, prob_last=pte_m, val_idx=va, prob_val_last=pva_m,
                                prob_best=pte_m, res=json.dumps(dict(res, twin=fp.name, twin_suffix=suffix)), **extra)
            os.replace(str(fpm) + ".tmp.npz", fpm)
            print(f"[t2p_bench] {tag} {suffix}: written ({time.time() - t0:.0f}s)", flush=True)

        if net is not None and model == "fbcnet_moe":
            twin("gs", gs_probs(net, XA, idx, te, dev), gs_probs(net, XA, idx, va, dev))
        for arm_, rc_ in fm_cont.items():             # T2-D546: the continuation twins, their own curves inside
            pv_c = rc_["prob_val_last"] if rc_["prob_val_last"] is not None else np.zeros((0, 3))
            sd_c = {}
            if rc_.get("side_dump") is not None:      # T2-D553 (NI26_SIDE_DUMP=1): the continued member's pre-head logits, head and the test z
                sd_c = dict(side_logit_last=rc_["side_dump"]["logit_last"].astype(np.float32), side_w=rc_["side_dump"]["w"].astype(np.float32),
                            side_b=rc_["side_dump"]["b"].astype(np.float32), side_te=np.asarray(data["side_te"], np.float32))
            if rc_.get("feat_dump") is not None:      # T2-D555 (NI26_FEAT_DUMP=1)
                sd_c.update(fm_feat_te=rc_["feat_dump"]["feat"].astype(np.float32), fm_out_w=rc_["feat_dump"]["w"].astype(np.float32),
                            fm_out_b=rc_["feat_dump"]["b"].astype(np.float32))
            if rc_.get("view_dump") is not None:      # T2-D654 (NI26_VIEW_DUMP=K)
                sd_c.update(fm_view_feat=rc_["view_dump"]["feat"].astype(np.float32), fm_view_prob=rc_["view_dump"]["prob"].astype(np.float32),
                            fm_view_gain=rc_["view_dump"]["gain"].astype(np.float32), fm_view_shift=rc_["view_dump"]["shift"].astype(np.int64))
            twin(f"c{arm_}", rc_["prob_last"].astype(np.float32), np.asarray(pv_c, np.float32),
                 cont_curves=np.array(json.dumps(dict(arm=arm_, epochs=args.fm_cont_epochs, lr_mult=args.fm_cont_lr, curves=rc_["curves"]))), **sd_c)
        if args.tta_mae and model.startswith("fm:st_") and r.get("state_last") is not None:
            import torch
            import t2p_cpt
            import t2p_fm
            _, fmname, readout = model.split(":")
            assert args.fm_ckpt, "--tta-mae needs --fm-ckpt (the CPT MAE decoder)"
            enc = t2p_fm.make_encoder(fmname, meta["ch_names"], lambda m_: None, args.fm_drop_path, args.fm_calib)
            fnet = t2p_fm.FMClf(enc, enc.n_layers if args.fm_layer in (-1, None) else args.fm_layer, readout, 3)
            fnet.load_state_dict(r["state_last"])
            fnet = fnet.to(dev).eval()
            mae = t2p_cpt.steeg_mae.__dict__[t2p_cpt.MAE_ARCH[fmname]](norm_pix_loss=False)
            mae.load_state_dict(torch.load(args.fm_ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
            mae = mae.to(dev)
            for lr_ in [float(v) for v in args.tta_mae.split(",")]:
                if args.tta_arms:                # T2-D283: LayerNorm-affine arms P / K (D125's path below stays as it was)
                    for arm_ in args.tta_arms.split(","):
                        pte_m, tp, ta, dte = mae_tta_ln_probs(fnet, mae, XA, idx, te, lr_, dev, arm_, seed=seed)
                        pva_m, _, _, dva = mae_tta_ln_probs(fnet, mae, XA, idx, va, lr_, dev, arm_, seed=seed)
                        twin(f"mae{lr_:g}{arm_}", pte_m, pva_m, tta_s_per_window=np.array([tp, ta]), ln_drift_test=dte, ln_drift_val=dva)
                        print(f"[t2p_bench] {tag} mae{lr_:g}{arm_}: plain {tp * 1e3:.1f} ms / window, with TTA {ta * 1e3:.1f} ms "
                              f"({ta / max(tp, 1e-9):.1f}x); LN drift per stream median {np.median(dte):.2e} (max {dte.max():.2e})", flush=True)
                    continue
                pte_m, tp, ta = mae_tta_probs(fnet, mae, XA, idx, te, lr_, dev)
                pva_m, _, _ = mae_tta_probs(fnet, mae, XA, idx, va, lr_, dev)
                twin(f"mae{lr_:g}", pte_m, pva_m, tta_s_per_window=np.array([tp, ta]))
                print(f"[t2p_bench] {tag} mae{lr_:g}: plain {tp * 1e3:.1f} ms / window, with TTA {ta * 1e3:.1f} ms ({ta / max(tp, 1e-9):.1f}x)", flush=True)
            del fnet, mae
            torch.cuda.empty_cache()
        if net is not None and args.tta_pl:
            for v in args.tta_pl.split(","):
                twin(f"pl{v}", pl_stream_probs(net, XA, idx, te, TTA_PL[v], dev), pl_stream_probs(net, XA, idx, va, TTA_PL[v], dev))
        if net is not None and (args.instream_ft or args.instream_sched):   # T2-D509 / D510: in-stream fine-tunes of the trained net
            import torch
            Xte_i = torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev)
            p0_ = T.predict_proba(net, Xte_i).astype(np.float32)
            pv0_ = (T.predict_proba(net, torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev)) if len(va) else np.zeros((0, 3))).astype(np.float32)
            twin("is0", p0_, pv0_, scored=np.zeros(len(te), bool))
            seen_ = set(idx.iloc[tr]["subject"])
            st_ = idx.iloc[te]["stream"].to_numpy()
            od_ = idx.iloc[te]["order"].to_numpy()
            held_ = [u for u in pd.unique(idx.iloc[te]["subject"]) if u not in seen_]
            adm_ = np.ones(len(te), bool) if args.instream_all else idx.iloc[te]["subject"].isin(held_).to_numpy()   # T2-D510: every test stream
            streams_ = pd.unique(st_[adm_])
            sels_ = args.instream_sel.split(",")
            assert set(sels_) <= {"true", "all", "conf", "soft", "confR"}, sels_
            assert "confR" not in sels_ or not args.instream_ft, "T2-D511: confR is a schedule-only label form"
            CODE_ = {"true": "", "all": "pa", "conf": "pc", "soft": "ps", "confR": "pr"}
            tpost_ = {}
            if set(sels_) - {"true"}:                   # T2-D510: the teacher's posterior per global row
                zt_ = np.load(args.instream_pl.format(protocol=fname, seed=seed))
                tpost_ = {int(r_): p_ for r_, p_ in zip(zt_["rows"], zt_["post"].astype(np.float64))}
                assert {int(r_) for r_ in te[adm_]} <= set(tpost_), "teacher rows do not cover the adapted streams"
                tpostR_ = {}                            # T2-D511: per re-fit c, the prefix re-decoded posterior of windows 1 .. c
                if "confR" in sels_:
                    for c_ in [int(v) for v in args.instream_sched.split(",")]:
                        a_r = zt_[f"post_r{c_}"].astype(np.float64)
                        tpostR_[c_] = {int(r_): p_ for r_, p_ in zip(zt_["rows"], a_r) if np.isfinite(p_).all()}
            ft_sec_ = []

            SV_ = {"m": None, "n": 0, "rows": 0}               # T2-D594b: the current copy's label sets (None = plain CE)

            def labels_(sel_, a_, c_=None):
                """-> (the rows the copy trains on, its targets, (teacher-correct, n) or None)."""
                SV_["m"] = None
                if sel_ == "true":
                    return a_, torch.as_tensor(y[te][a_], dtype=torch.long, device=dev), None
                if sel_ == "confR":                     # T2-D511: re-decoded over the delivered prefix 1 .. c (c = the re-fit point)
                    assert c_ == len(a_), (c_, len(a_))
                    pp_ = np.stack([tpostR_[c_][int(r_)] for r_ in te[a_]])
                else:
                    pp_ = np.stack([tpost_[int(r_)] for r_ in te[a_]])     # causal: the teacher's window t used windows <= t
                if sel_ == "soft":
                    return a_, torch.as_tensor(pp_, dtype=torch.float32, device=dev), (int((pp_.argmax(1) == y[te][a_]).sum()), len(a_))
                ya_, cf_ = pp_.argmax(1), pp_.max(1)
                keep_ = np.ones(len(a_), bool)
                if sel_ in ("conf", "confR"):                      # the confident half of each predicted class (class-balanced; no class vanishes)
                    keep_[:] = False
                    for c_ in np.unique(ya_):
                        jj_ = np.flatnonzero(ya_ == c_)
                        keep_[jj_[np.argsort(-cf_[jj_], kind="stable")[:int(np.ceil(len(jj_) / 2))]]] = True
                ka_ = a_[keep_]
                if args.instream_setval > 0 and sel_ == "confR":   # T2-D594b: the low-margin MI / WORD rows learn the set
                    m_sv = setval_mask(pp_[keep_], ya_[keep_], args.instream_setval)
                    SV_["m"] = torch.as_tensor(m_sv, device=dev)
                    SV_["n"] += int((m_sv.sum(1) > 1).sum())
                    SV_["rows"] += len(ka_)
                return ka_, torch.as_tensor(ya_[keep_], dtype=torch.long, device=dev), (int((ya_[keep_] == y[te][ka_]).sum()), len(ka_))

            def fit_(ka_, tg_, sd_):
                t_ = time.time()
                m_ = T.finetune(net, Xte_i[ka_], tg_, T.peruser_for(recipe), sd_, **({"yset": SV_["m"]} if SV_["m"] is not None else {}))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ft_sec_.append((len(ka_), time.time() - t_))
                return m_

            def acc_line_(lab_, v_):
                if v_:
                    print(f"[t2p_bench] {tag} {lab_}: teacher label accuracy {sum(a for a, _ in v_) / max(1, sum(b for _, b in v_)):.3f} "
                          f"on {sum(b for _, b in v_)} windows", flush=True)

            def stream_rows_(s_):
                j_ = np.flatnonzero(st_ == s_)
                return j_[np.argsort(od_[j_], kind="stable")]

            for n_ in [int(v) for v in args.instream_ft.split(",") if v]:          # fixed N: fit on the first N, score the rest
                pns_ = {sel_: p0_.copy() for sel_ in sels_}
                acc_ = {sel_: [] for sel_ in sels_}
                sc_ = np.zeros(len(te), bool)
                for si_, s_ in enumerate(streams_):
                    j_ = stream_rows_(s_)
                    if len(j_) < n_ + 30:
                        continue
                    a_, b_ = j_[:n_], j_[n_:]
                    for sel_ in sels_:
                        ka_, tg_, la_ = labels_(sel_, a_)
                        m_s = fit_(ka_, tg_, seed * 100 + 70 + si_)
                        pns_[sel_][b_] = T.predict_proba(m_s, Xte_i[b_])
                        del m_s
                        if la_:
                            acc_[sel_].append(la_)
                    sc_[b_] = True
                for sel_ in sels_:
                    twin(f"is{n_}{CODE_[sel_]}", pns_[sel_], pv0_, scored=sc_)
                    acc_line_(f"is{n_}{CODE_[sel_]}", acc_[sel_])
            if args.instream_sched:                     # T2-D510: the ship schedule: re-fit from the base at each checkpoint on every window so far;
                cks_ = [int(v) for v in args.instream_sched.split(",")]     # each window is scored by the newest copy that existed before it
                for sel_ in sels_:
                    for rep_ in range(args.instream_reps if sel_ in args.instream_rep_sel.split(",") else 1):
                        pn_, sc_, pos_ = p0_.copy(), np.zeros(len(te), bool), np.full(len(te), -1)
                        acc_ = []
                        for si_, s_ in enumerate(streams_):
                            j_ = stream_rows_(s_)
                            pos_[j_] = np.arange(len(j_))
                            for k_, c_ in enumerate(cks_):
                                if len(j_) <= c_:
                                    break
                                e_ = cks_[k_ + 1] if k_ + 1 < len(cks_) else len(j_)
                                ka_, tg_, la_ = labels_(sel_, j_[:c_], c_)
                                m_s = fit_(ka_, tg_, seed * 100 + 70 + si_ + 1000 * rep_ + 100000 * (k_ + 1))
                                pn_[j_[c_:e_]] = T.predict_proba(m_s, Xte_i[j_[c_:e_]])
                                sc_[j_[c_:e_]] = True
                                del m_s
                                if la_:
                                    acc_.append(la_)
                        lab_ = f"isS{CODE_[sel_]}" + (f"r{rep_}" if rep_ else "")
                        twin(lab_, pn_, pv0_, scored=sc_, pos=pos_)
                        acc_line_(lab_, acc_)
                        if args.instream_setval > 0 and sel_ == "confR":   # T2-D594b: the set rows of this arm so far
                            print(f"[t2p_bench] {tag} {lab_}: T2-D594b set rows {SV_['n']} of {SV_['rows']} selected (threshold {args.instream_setval:g})", flush=True)
            if ft_sec_:
                twin("isT", p0_, pv0_, ft=np.array(ft_sec_, dtype=np.float64))
                f_ = np.array(ft_sec_)
                print(f"[t2p_bench] {tag} in-stream fine-tunes: {len(f_)}, {f_[:, 1].mean():.2f} s each (median windows {np.median(f_[:, 0]):.0f})", flush=True)
            del Xte_i
        if net is not None and args.peruser:
            import torch
            Xt = torch.as_tensor(np.ascontiguousarray(XA[tr]), device=dev)
            yt = torch.as_tensor(y[tr], dtype=torch.long, device=dev)
            Xte_t = torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev)
            Xva_t = torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev)
            put, puv, put_c, puv_c = [], [], [], []
            Lt = None
            if ltr_ is not None:                        # T2-D486: the copies crop their own users' long windows too
                assert recipe.pu_rcx == 0 and not recipe.pu_kl, "T2-D486: copies on their own windows only"
                Lt = torch.as_tensor(ltr_, device=dev)
            if recipe.pu_rcx > 0:                       # T2-D239: the others' windows mapped onto each copy's user (full maps of the training windows)
                ut_ = torch.as_tensor(utr_, dtype=torch.long, device=dev)
                rcb = T.rc_bank(Xt, ut_, len(tr_users), 1.0)
            for k, u in enumerate(tr_users):
                if recipe.pu_rcx > 0:
                    Xk, yk = T.copy_data(Xt, yt, ut_, k, rcb, recipe.pu_rcx, seed * 100 + k + 50)
                    print(f"[t2p_bench] {tag} copy {u}: {int((ut_ == k).sum())} own + {len(Xk) - int((ut_ == k).sum())} mapped windows (pu_rcx {recipe.pu_rcx:g})", flush=True)
                else:
                    Xk, yk = Xt[torch.as_tensor(utr_ == k, device=dev)], yt[torch.as_tensor(utr_ == k, device=dev)]
                x_oth = Xt[torch.as_tensor(utr_ != k, device=dev)] if recipe.pu_kl > 0 else None    # T2-D430: the strangers of copy k
                sid_k = None
                if recipe.sbal:                         # T2-D478: the copy's windows' stream ids (its own sessions; no mapped windows)
                    assert recipe.pu_rcx == 0, "T2-D478: sbal copies take their own windows only"
                    sid_k = T.stream_ids(utr_, idx.iloc[tr]["session"].astype(str).to_numpy())[utr_ == k]
                m_u = T.finetune(net, Xk, yk, T.peruser_for(recipe), seed * 100 + k, X_other=x_oth, sid=sid_k,
                                 Lx=None if Lt is None else Lt[torch.as_tensor(utr_ == k, device=dev)])
                put.append(T.predict_proba(m_u, Xte_t))
                if args.tta_vp and args.tta_copies:     # T2-D302 (c): the copy through the same K views
                    put_c.append(T.predict_proba_vp(m_u, Xte_t, args.tta_vp))
                    puv_c.append(T.predict_proba_vp(m_u, Xva_t, args.tta_vp) if len(va) else np.zeros((0, 3), np.float32))
                puv.append(T.predict_proba(m_u, Xva_t) if len(va) else np.zeros((0, 3), np.float32))
            twin("pu", r["prob_last"].astype(np.float32), (pv if pv is not None else np.zeros((0, 3))).astype(np.float32),
                 prob_users_test=np.stack(put).astype(np.float32), prob_users_val=np.stack(puv).astype(np.float32),
                 users=np.array(tr_users))
        if net is not None and args.tta_vp:            # backlog 71: the pooled net's TTA twin (the copies are trained unaugmented: unchanged)
            import torch
            assert recipe.vp and T.VP_BANK, "--tta-vp needs a vp recipe (its bank)"
            ptv = T.predict_proba_vp(net, torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev), args.tta_vp).astype(np.float32)
            pvv = (T.predict_proba_vp(net, torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev), args.tta_vp) if len(va)
                   else np.zeros((0, 3))).astype(np.float32)
            twin(f"tv{args.tta_vp}", ptv, pvv)
            if args.peruser:
                twin(f"tv{args.tta_vp}pu", ptv, pvv, prob_users_test=np.stack(put).astype(np.float32),
                     prob_users_val=np.stack(puv).astype(np.float32), users=np.array(tr_users))
            if args.tta_shift:                          # T2-D302 (a): + shift-only views
                pts = T.predict_proba_vp(net, torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev), args.tta_vp,
                                         shift_only=args.tta_shift).astype(np.float32)
                pvs = (T.predict_proba_vp(net, torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev), args.tta_vp, shift_only=args.tta_shift)
                       if len(va) else np.zeros((0, 3))).astype(np.float32)
                sfx = f"tv{args.tta_vp}s{args.tta_shift}"
                twin(sfx, pts, pvs)
                if args.peruser:
                    twin(f"{sfx}pu", pts, pvs, prob_users_test=np.stack(put).astype(np.float32),
                         prob_users_val=np.stack(puv).astype(np.float32), users=np.array(tr_users))
            if args.tta_copies:                         # T2-D302 (c): the pooled TTA + the copies through the same views
                assert args.peruser and put_c, "--tta-copies needs --peruser"
                twin(f"tv{args.tta_vp}c", ptv, pvv)
                twin(f"tv{args.tta_vp}cpu", ptv, pvv, prob_users_test=np.stack(put_c).astype(np.float32),
                     prob_users_val=np.stack(puv_c).astype(np.float32), users=np.array(tr_users))
        if args.test_mirror and not model.startswith("fm:") and model not in CLASSICAL:   # T2-D619 (diagnostic): the TEST windows mirrored
            import torch
            import t2p_fm as _fmm
            assert net is not None and not args.side and not args.bag, "--test-mirror: a pooled CNN net (no side head, no bag)"
            mr_ = _fmm.mirror_setup(1.0, list(meta["ch_names"]), 0, log=lambda m: None)
            pm_ = mr_["perm"].to(dev)
            Xt_ = torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev)
            p_id = T.predict_proba(net, Xt_)
            d_id = float(np.abs(p_id - r["prob_last"]).max())
            assert d_id <= 1e-4, f"T2-D619: the re-predicted unmirrored test != prob_last (max {d_id:.2e})"
            v0_ = np.zeros((0, 3), np.float32)
            twin("mr", T.predict_proba(net, Xt_[:, pm_]).astype(np.float32), v0_, mirror_pairs=mr_["pairs"], mirror_id_maxdiff=d_id)
            if args.tta_vp:
                twin(f"tv{args.tta_vp}mr", T.predict_proba_vp(net, Xt_[:, pm_], args.tta_vp).astype(np.float32), v0_, mirror_pairs=mr_["pairs"])
            print(f"[t2p_bench] {tag}: T2-D619 mirrored twins written ({mr_['pairs']} pairs; unmirrored re-predict max diff {d_id:.1e})", flush=True)
        if model.startswith("fm:") and args.fm_kd_dump:                  # backlog 69: the FM teacher's probabilities
            import torch
            import t2p_fm
            assert args.kd_pools and r.get("net_last") is not None, "--fm-kd-dump needs --kd-pools (and the fine-tuned net)"
            if "kd_sel" not in ext_cache:
                ext_cache["kd_sel"] = kd_select(args.kd_pools.split(","), args.kd_n)
            Xk, kk = kd_aligned(ext_cache["kd_sel"], align, args.shrink, list(meta["ch_names"]))
            fnet = r["net_last"].to(dev).eval()
            with torch.no_grad():
                p_tr = t2p_fm.predict_proba(fnet, torch.as_tensor(np.ascontiguousarray(XA[tr]), device=dev))
                p_ex = np.concatenate([t2p_fm.predict_proba(fnet, torch.as_tensor(Xk[i:i + 2048], device=dev)) for i in range(0, len(Xk), 2048)])
            fnet.cpu()
            kd_fp = out / f"kdteacher__{proto}__s{seed}.npz"
            np.savez_compressed(str(kd_fp) + ".tmp.npz", train_idx=tr, p_train=p_tr.astype(np.float32), kd_key=kk, p_ext=p_ex.astype(np.float32),
                                res=json.dumps(dict(tag=tag, pools=args.kd_pools, kd_n=args.kd_n, n_ext=int(len(kk)), align=align, rng=KD_RNG)))
            os.replace(str(kd_fp) + ".tmp.npz", kd_fp)
            print(f"[{tag}] kd teacher dump -> {kd_fp.name}: {len(tr)} training windows (mean max p {p_tr.max(1).mean():.3f}), "
                  f"{len(kk)} external windows (mean max p {p_ex.max(1).mean():.3f}; argmax shares {np.bincount(p_ex.argmax(1), minlength=3) / len(p_ex)})", flush=True)
        if model.startswith("fm:") and args.fm_peruser_heads and r.get("net_last") is not None:   # T2-D166
            import copy as _copy
            from dataclasses import replace as _replace
            import torch
            fnet = r["net_last"].to(dev).eval()

            def _feats(rows):
                out_ = []
                with torch.no_grad():
                    for i in range(0, len(rows), 128):
                        xb = torch.as_tensor(np.ascontiguousarray(XA[rows[i:i + 128]]), device=dev, dtype=torch.float32)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=xb.is_cuda):
                            out_.append(fnet.features(xb).float())
                return torch.cat(out_) if out_ else torch.zeros((0, fnet.out.in_features), device=dev)
            Ftr, Fte, Fva = _feats(tr), _feats(te), _feats(va)
            yt = torch.as_tensor(y[tr], dtype=torch.long, device=dev)
            rec_h = _replace(T.PERUSER, lr=1e-3, epochs=40)
            put, puv = [], []
            for k, u in enumerate(tr_users):
                m_u = torch.as_tensor(utr_ == k, device=dev)
                h_u = T.finetune(_copy.deepcopy(fnet.out).float(), Ftr[m_u], yt[m_u], rec_h, seed * 100 + k)
                with torch.no_grad():
                    put.append(torch.softmax(h_u(Fte), 1).cpu().numpy())
                    puv.append(torch.softmax(h_u(Fva), 1).cpu().numpy() if len(va) else np.zeros((0, 3), np.float32))
            twin("pu", r["prob_last"].astype(np.float32), (pv if pv is not None else np.zeros((0, 3))).astype(np.float32),
                 prob_users_test=np.stack(put).astype(np.float32), prob_users_val=np.stack(puv).astype(np.float32),
                 users=np.array(tr_users))
            del fnet, Ftr, Fte, Fva
            torch.cuda.empty_cache()
        if model.startswith("fm:") and args.fm_peruser_lora and r.get("net_last") is not None:   # T2-D189: per-user LoRA deltas
            import copy as _copy
            import math as _math
            import torch
            import torch.nn.functional as _F
            t1m = t2p_fm.t1
            base = r["net_last"].to(dev).eval()
            Xtr_t = torch.as_tensor(np.ascontiguousarray(XA[tr]), device=dev, dtype=torch.float32)
            Xte_t = torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev, dtype=torch.float32)
            Xva_t = torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev, dtype=torch.float32)
            yt_ = torch.as_tensor(y[tr], dtype=torch.long, device=dev)
            put, puv = [], []
            for k, u in enumerate(tr_users):
                torch.manual_seed(seed * 100 + k)
                g_ = torch.Generator().manual_seed(seed * 100 + k)
                net_k = _copy.deepcopy(base)
                t1m.inject_lora(net_k.enc, net_k.layer, 16, 32.0, 0.1, lambda m: None)
                for n_, p_ in net_k.named_parameters():
                    p_.requires_grad_(".parametrizations.weight.0." in n_)
                own, oth = np.flatnonzero(utr_ == k), np.flatnonzero(utr_ != k)
                params = [p_ for p_ in net_k.parameters() if p_.requires_grad]
                opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.0)
                total = 10 * int(_math.ceil(len(own) / 16))
                sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: 0.5 * (1 + _math.cos(_math.pi * min(s_, total) / total)))
                net_k.train()
                for _ep in range(10):
                    po = own[torch.randperm(len(own), generator=g_).numpy()]
                    pt = oth[torch.randint(len(oth), (len(own),), generator=g_).numpy()]
                    for i0 in range(0, len(po), 16):
                        b_ = torch.as_tensor(np.concatenate([po[i0:i0 + 16], pt[i0:i0 + 16]]), device=dev)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xtr_t.is_cuda):
                            loss = _F.cross_entropy(net_k(Xtr_t[b_]).float(), yt_[b_], label_smoothing=0.1)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        sched.step()
                net_k.eval()
                n_ad = 0
                for m_ in net_k.modules():
                    if isinstance(m_, t1m.LoRADelta):
                        m_.scale *= 0.5                      # the blend r 1/2, in weight space
                        n_ad += 1
                with torch.no_grad():
                    put.append(t2p_fm.predict_proba(net_k, Xte_t))
                    puv.append(t2p_fm.predict_proba(net_k, Xva_t) if len(va) else np.zeros((0, 3), np.float32))
                print(f"[t2p_bench] {tag} user {u}: LoRA delta on {n_ad} Linear, {len(own)} own windows, last loss {float(loss):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                del net_k, opt, params
                torch.cuda.empty_cache()
            twin("pu", r["prob_last"].astype(np.float32), (pv if pv is not None else np.zeros((0, 3))).astype(np.float32),
                 prob_users_test=np.stack(put).astype(np.float32), prob_users_val=np.stack(puv).astype(np.float32), users=np.array(tr_users))
            del base, Xtr_t, Xte_t, Xva_t
            torch.cuda.empty_cache()
        if model.startswith("fm:") and args.fm_soup > 1 and r.get("state_last") is not None:   # T2-D176: uniform soup
            import torch
            states_ = [r["state_last"]]
            for j_ in range(1, args.fm_soup):
                rj = t2p_fm.fit_eval_fm(fmname, data, seed + 1000 * j_, meta["ch_names"], **dict(fm_kw, return_state=True))
                states_.append(rj["state_last"])
                from ni26 import metrics as _M
                print(f"[t2p_bench] {tag} soup member {j_ + 1}/{args.fm_soup} (seed {seed + 1000 * j_}): last cellBA "
                      f"{_M.cell_balanced_accuracy(y[te], rj['prob_last'].argmax(1), cells[te])[0]:.4f} ({time.time() - t0:.0f}s)", flush=True)
                if args.fm_soup_members:                # T2-D493: this member's own predictions (the ensemble of the souped nets)
                    pvj = rj["prob_val_last"] if rj["prob_val_last"] is not None else np.zeros((0, 3), np.float32)
                    twin(f"soup{args.fm_soup}m{j_}", rj["prob_last"].astype(np.float32), pvj.astype(np.float32), member_seed=seed + 1000 * j_)
                del rj
            avg = {k_: (torch.stack([st[k_].float() for st in states_]).mean(0).to(v_.dtype) if v_.is_floating_point() else v_)
                   for k_, v_ in states_[0].items()}
            snet = r["net_last"].to(dev).eval()
            snet.load_state_dict(avg, strict=True)
            Xte_t = torch.as_tensor(np.ascontiguousarray(XA[te]), device=dev, dtype=torch.float32)
            pte_s = t2p_fm.predict_proba(snet, Xte_t)
            pva_s = t2p_fm.predict_proba(snet, torch.as_tensor(np.ascontiguousarray(XA[va]), device=dev, dtype=torch.float32)) \
                if len(va) else np.zeros((0, 3), np.float32)
            twin(f"soup{args.fm_soup}", pte_s.astype(np.float32), pva_s.astype(np.float32))
            del states_, avg, snet
            torch.cuda.empty_cache()
        if args.tta_bn and net is not None:
            for m in [float(v) for v in args.tta_bn.split(",")]:
                fpm = out / f"{tag.replace(f'__{rec}{xtag}__', f'__{rec}bn{m:g}{xtag}__')}.npz"
                if fpm.exists():
                    continue
                pte_m, pva_m = bn_stream_probs(net, XA, idx, te, m, dev), bn_stream_probs(net, XA, idx, va, m, dev)
                np.savez_compressed(str(fpm) + ".tmp.npz", test_idx=te, prob_last=pte_m, val_idx=va, prob_val_last=pva_m,
                                    prob_best=pte_m, res=json.dumps(dict(res, tta_bn=m, twin=fp.name)))
                os.replace(str(fpm) + ".tmp.npz", fpm)
                print(f"[t2p_bench] {tag} bn{m:g}: written ({time.time() - t0:.0f}s)", flush=True)
        from ni26 import metrics
        p = np.load(fp)
        cb = metrics.cell_balanced_accuracy(y[te], p["prob_best"].argmax(1), cells[te])[0]
        cl = metrics.cell_balanced_accuracy(y[te], p["prob_last"].argmax(1), cells[te])[0]
        print(f"[t2p_bench] {tag}: cellBA best {cb:.4f} last {cl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    print("[t2p_bench] DONE", flush=True)


if __name__ == "__main__":
    main()

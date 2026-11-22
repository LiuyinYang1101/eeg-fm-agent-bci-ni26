#!/usr/bin/env python
"""T2-D549 (template W19): the post-build check of a W19 package against the W18 package it was built from (OLD = W18 build N), on real public
streams, on the GPU path the worker takes. = t2p_w18_pkgcheck's checks (gate I reused verbatim) + the W19 config and Gate H:
  (0) the config is what D549 builds: W18's (fm_side on, imcoh off, six FM members, five heads, one head-less swapped member) + side.refit
      {"lam": 1} + ten side nets per family (OLD: five, no refit) + slots 2 / 4 = the D546b continued members (cont.decision T2-D546b);
  (a) finite; (b) NEW moves against OLD on every stream; (c) a re-run reproduces NEW (<= 1e-6); (d) ms / window NEW vs OLD;
  (I) the in-stream P1 refit recomputed (t2p_w18_pkgcheck.gate_I, unchanged);
  (H) NEW: the side-head refit recomputed independently on the gate streams (refits 60 / 100 / 150, both side families, every net): the rows /
      labels handed to the side refit == the P1 copies' own; the recorded side logits == each side net run on the stream's aligned inputs (loop,
      not the stacked path) <= 1e-4; the recorded side vectors == t2p_bench's side_raw + the causal z with side.pt's prior <= 1e-5; each
      re-fitted head == scripts/local/t2_d539_read.py fit_head (its def, compiled from the file's source) on those inputs <= 1e-6; refit
      errors 0 and kept-trained fallbacks 0 (D549 condition K: a fallback on the public streams stops the build);
  (HF) T2-D553 (cfg fm_side_refit): the FM side-head refit on the same streams: at each refit point the rows / labels are the P1 copies' own,
      each headed member's re-fitted head == the reader's fit_head on the recorded (log p, s) of those rows <= 1e-6, the window AT the refit
      point already uses the re-fitted heads (softmax(log p + W s + b) <= 1e-9), every window before the first refit point uses the trained
      heads, fits = headed members x refits, errors 0 and kept-trained fallbacks 0 (on the public streams too);
  (B) the in-stream ON vs OFF argmax-change fraction of NEW within [0.5 OLD - 0.005, 1.5 OLD + 0.005] (OLD = W18 bN, the same run);
  (S) D534's bound, now over every W19 change together: m = mean |CNN part (NEW, ON) - (OLD, ON)| <= m_adapt = mean |(OLD, ON) - (OLD, OFF)|.
  python tracks/t2_bci/scripts/t2p_w19_pkgcheck.py --old <unzipped W18 dir> --new <unzipped W19 dir> [--out res.json]
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2p_w18_pkgcheck as P18  # noqa: E402

REPO = HERE.parents[2]


def reader_fit_head():
    """scripts/local/t2_d539_read.py's fit_head, its def only (the module itself reads local-only paths)."""
    src = (REPO / "scripts/local/t2_d539_read.py").read_text()
    ns = {"np": np}
    exec("from scipy.optimize import minimize", ns)
    node = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "fit_head"]
    assert len(node) == 1
    exec(compile(ast.Module(node, []), "t2_d539_read(fit_head)", "exec"), ns)
    print(f"[w19 check] gate H reference: t2_d539_read.fit_head, sha256 {hashlib.sha256(ast.unparse(node[0]).encode()).hexdigest()[:12]}", flush=True)
    return ns["fit_head"]


def side_z_indep(xs, mu, sd, n0):
    """the side vectors of the DELIVERED windows: t2p_bench.side_raw 'dcmean', causal z with side.pt's prior (= t2p_repack's side-head check)."""
    Fs = B.side_raw(xs, "dcmean")
    xf = np.where(np.isfinite(Fs), Fs, mu)
    nn = n0 + np.arange(1, len(xs) + 1)[:, None]
    mm = (n0 * mu + np.cumsum(xf, 0)) / nn
    vv = (n0 * (sd ** 2 + mu ** 2) + np.cumsum(xf * xf, 0)) / nn - mm ** 2
    return ((xf - mm) / np.sqrt(np.maximum(vv, 1e-12))).astype(np.float32).astype(np.float64)


def gate_H(model, xs, dev, lam_expect=1.0):
    fit_head = reader_fit_head()
    mm = copy.deepcopy(model)
    mm.reset_state()
    assert mm.side_refit is not None and mm.side_refit["lam"] == lam_expect, mm.side_refit
    p1, caps = [], []
    f_fit, f_side = mm._instream_fit, mm._side_refit
    st0 = dict(mm.side_refit_stats)

    def _fit(rows, ya, dev_, f0=f_fit):
        p1.append((mm.is_n, np.asarray(rows).copy(), np.asarray(ya).copy()))
        return f0(rows, ya, dev_)

    def _side(rows, ya, dev_, f0=f_side):
        c = mm.is_n
        Lh = {f: [None if v is None else v.copy() for v in mm.sr_L[f]] for f in mm.side}
        Sh = [None if v is None else v.copy() for v in mm.sr_s]
        out_ = f0(rows, ya, dev_)
        caps.append(dict(n=c, rows=np.asarray(rows).copy(), ya=np.asarray(ya).copy(), L=Lh, S=Sh,
                         heads={f: (mm.side_hw_now[f][0].double().cpu().numpy(), mm.side_hw_now[f][1].double().cpu().numpy()) for f in mm.side}))
        return out_
    mm._instream_fit, mm._side_refit = _fit, _side
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
    want = [int(c) for c in mm.is_cfg["refits"] if c < len(xs)]
    assert [c_["n"] for c_ in caps] == want, ([c_["n"] for c_ in caps], want)
    zi = side_z_indep(xs, mm.side_mu, mm.side_sd, mm.side_n0)
    out = []
    for c_, (n1, r1, y1) in zip(caps, p1 if len(p1) == len(caps) else [(None, None, None)] * len(caps)):
        c = c_["n"]
        r = dict(refit=c, same_rows_labels_as_P1=bool(n1 == c and np.array_equal(r1, c_["rows"]) and np.array_equal(y1, c_["ya"])),
                 s_vs_indep=float(np.abs(np.stack([c_["S"][t] for t in range(c)]) - zi[:c]).max()), logits_vs_nets={}, head_vs_reader={})
        for f in mm.side:
            Z = torch.as_tensor(np.stack(mm.is_hist_z[f][:c]), dtype=torch.float32, device=dev)
            with torch.no_grad():
                Lr = np.stack([net(Z).float().cpu().numpy().astype(np.float64) for net, _ in mm.side[f]], 1)     # (c, n, K), the loop path
            Lc = np.stack([c_["L"][f][t] for t in range(c)])
            r["logits_vs_nets"][f] = float(np.abs(Lc - Lr).max())
            W0, b0 = mm.side_W0[f], mm.side_b0[f]
            sel = c_["rows"]
            S_ = np.stack([c_["S"][t] for t in sel])
            dmax = 0.0
            for i in range(W0.shape[0]):
                W, b, ok = fit_head(Lc[sel][:, i, :], S_, c_["ya"].astype(np.int64), W0[i], b0[i], lam_expect)
                assert ok, (f, i, c)
                dmax = max(dmax, float(np.abs(c_["heads"][f][0][i] - W).max()), float(np.abs(c_["heads"][f][1][i] - b).max()))
            r["head_vs_reader"][f] = dmax
        out.append(r)
        print(f"[w19 check] gate H refit {c}: {r}", flush=True)
    st1 = mm.side_refit_stats
    stats = {k: st1[k] - st0[k] for k in st0}
    ok = bool(out) and len(p1) == len(caps) and stats["errors"] == 0 and stats["kept_trained"] == 0 and all(
        r["same_rows_labels_as_P1"] and r["s_vs_indep"] <= 1e-5 and max(r["logits_vs_nets"].values()) <= 1e-4
        and max(r["head_vs_reader"].values()) <= 1e-6 for r in out)
    return ok, dict(refits=out, stats=stats)


def gate_HF(model, xs, dev, lam_expect=1.0):
    """T2-D553: the FM side-head refit recomputed on one stream (see the module doc, (HF))."""
    fit_head = reader_fit_head()
    mm = copy.deepcopy(model)
    mm.reset_state()
    assert mm.fm_refit is not None and mm.fm_refit["lam"] == lam_expect, mm.fm_refit
    p1, caps, apl = [], [], []
    f_fit, f_fr, f_ap = mm._instream_fit, mm._fm_refit, mm._fm_head_apply
    st0 = dict(mm.fm_refit_stats)
    nh = sum(h is not None for h in mm.fm_heads)

    def _fit(rows, ya, dev_, f0=f_fit):
        p1.append((mm.is_n, np.asarray(rows).copy(), np.asarray(ya).copy()))
        return f0(rows, ya, dev_)

    def _fr(c, f0=f_fr):
        hist = {k: (v[0].copy(), [None if a_ is None else a_.copy() for a_ in v[1]]) for k, v in mm.fr_hist.items()}
        out_ = f0(c)
        caps.append(dict(n=c, hist=hist, heads=None if mm.fm_heads_now is None else [None if h is None else (h[0].copy(), h[1].copy()) for h in mm.fm_heads_now]))
        return out_

    def _ap(ps, f0=f_ap):
        o_ = f0(ps)
        apl.append((int(getattr(mm, "is_n", 0)), [np.asarray(p, np.float64).copy() for p in ps], np.asarray(mm._side_now, np.float64).copy(),
                    [np.asarray(q, np.float64).copy() for q in o_]))
        return o_
    mm._instream_fit, mm._fm_refit, mm._fm_head_apply = _fit, _fr, _ap
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
    want = [int(c) for c in mm.is_cfg["refits"] if c < len(xs)]
    assert [c_["n"] for c_ in caps] == want, ([c_["n"] for c_ in caps], want)
    p1d = {n_: (r_, y_) for n_, r_, y_ in p1}

    def heads_out(ps, s_, heads):
        out_ = []
        for p, h in zip(ps, heads):
            if h is not None:
                lg = np.log(np.maximum(p, 1e-30)) + h[0] @ s_ + h[1]
                e = np.exp(lg - lg.max())
                p = e / e.sum()
            out_.append(p)
        return out_
    out = []
    for c_ in caps:
        c = c_["n"]
        rows, ya = p1d.get(c, (None, None))
        r = dict(refit=c, p1_present=rows is not None)
        dmax, dapp = 0.0, 0.0
        if rows is not None and c_["heads"] is not None:
            keep = np.array([int(t) in c_["hist"] for t in rows], bool)
            S_ = np.stack([c_["hist"][int(t)][0] for t in rows[keep]])
            for i, h in enumerate(mm.fm_heads):
                if h is None:
                    continue
                Ls = np.stack([c_["hist"][int(t)][1][i] for t in rows[keep]])
                W, b, ok_ = fit_head(Ls, S_, ya[keep].astype(np.int64), h[0], h[1], lam_expect)
                assert ok_, (i, c)
                dmax = max(dmax, float(np.abs(c_["heads"][i][0] - W).max()), float(np.abs(c_["heads"][i][1] - b).max()))
            at = [a_ for a_ in apl if a_[0] == c]
            if at:
                exp_ = heads_out(at[0][1], at[0][2], c_["heads"])
                dapp = max(float(np.abs(e_ - q_).max()) for e_, q_ in zip(exp_, at[0][3]))
            r.update(n_rows=int(keep.sum()), head_vs_reader=dmax, apply_at_refit=dapp)
        out.append(r)
        print(f"[w19 check] gate HF refit {c}: {r}", flush=True)
    first = min(want) if want else len(xs)
    pre = [a_ for a_ in apl if a_[0] < first]
    d_pre = max([max(float(np.abs(e_ - q_).max()) for e_, q_ in zip(heads_out(a_[1], a_[2], mm.fm_heads), a_[3])) for a_ in pre] or [0.0])
    st1 = mm.fm_refit_stats
    stats = {k: st1[k] - st0[k] for k in st0}
    ok = bool(out) and all(r["p1_present"] and r.get("head_vs_reader", 1.0) <= 1e-6 and r.get("apply_at_refit", 1.0) <= 1e-9 for r in out) \
        and d_pre <= 1e-12 and stats["fits"] == nh * len(want) and stats["errors"] == 0 and stats["kept_trained"] == 0
    return ok, dict(refits=out, before_first_refit_max_dp=d_pre, stats=stats, headed=nh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--streams", default="P04|2,P05|1,P09|1")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(P18.EXP), help="the PROTEUS export (sofia: data/ni26/t2_proteus/export_proteus_v1)")
    ap.add_argument("--gate-stream", default="P04|2", help="gate I's stream (>= 150 windows)")
    ap.add_argument("--h-streams", default="P04|2,P09|1", help="gate H's streams (>= 150 windows each)")
    ap.add_argument("--bs-streams", type=int, default=0, help="gates B / S on the first N public streams only (a smoke test; 0 = all)")
    a = ap.parse_args()
    os.environ["NI26_PARTS"] = "1"
    RP.kit.use_track("bci_decoding")
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    cfo = json.loads((Path(a.old) / "config.json").read_text())
    fm = cfg["fm_members"]
    sn = torch.load(Path(a.new) / cfg["side"]["file"], map_location="cpu", weights_only=True)
    so = torch.load(Path(a.old) / cfo["side"]["file"], map_location="cpu", weights_only=True)
    cfg0 = dict(fm_side=bool(cfg.get("fm_side")), imcoh=cfg.get("imcoh"), n_members=len(fm), heads=sum(bool(m.get("side")) for m in fm),
                headless=[m["file"] for m in fm if not m.get("side")], refit=(cfg["side"] or {}).get("refit"),
                side_nets={f: len(sn[f]) for f in cfg["side"]["families"]}, old_side_nets={f: len(so[f]) for f in cfo["side"]["families"]},
                cont=[(j, (m.get("cont") or {}).get("decision"), (m.get("cont") or {}).get("source", {}).get("seed")) for j, m in enumerate(fm) if m.get("cont")],
                fm_side_refit=cfg.get("fm_side_refit"))
    print(f"[w19 check] config: {cfg0}", flush=True)
    assert cfg0["fm_side"] and not cfg0["imcoh"] and cfg0["n_members"] == len(cfo["fm_members"]) == 6 and cfg0["heads"] == 5 and len(cfg0["headless"]) == 1
    assert cfg0["refit"] == {"lam": 1.0} and not (cfo["side"] or {}).get("refit"), (cfg0["refit"], cfo["side"])
    assert all(v == 10 for v in cfg0["side_nets"].values()) and all(v == 5 for v in cfg0["old_side_nets"].values()), (cfg0["side_nets"], cfg0["old_side_nets"])
    assert [j for j, d_, _ in cfg0["cont"]] == [2, 4] and all(d_ == "T2-D546b" for _, d_, _ in cfg0["cont"]), cfg0["cont"]
    assert cfo.get("fm_side") and cfo.get("instream") and cfo.get("dcfam"), "OLD must be W18"
    assert cfg0["fm_side_refit"] == {"lam": 1.0} and not cfo.get("fm_side_refit"), ("T2-D553: cfg fm_side_refit", cfg0["fm_side_refit"])
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    res = {"config": cfg0}
    for st in a.streams.split(","):
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        po, pn, pn2 = [], [], []
        t0 = time.time()
        RP.run_stream(old, xs, dev, po)
        t_old = time.time() - t0
        t0 = time.time()
        RP.run_stream(new, xs, dev, pn)
        t_new = time.time() - t0
        RP.run_stream(new, xs, dev, pn2)
        po, pn = np.stack(po), np.stack(pn)
        d = np.abs(po - pn).max(1)
        r = dict(n=len(rows), finite=bool(np.isfinite(pn).all()), moved_windows=int((d > 0).sum()), max_dp=float(d.max()),
                 labels_changed=int((po.argmax(1) != pn.argmax(1)).sum()), rerun_max_dp=float(np.abs(np.stack(pn2) - pn).max()),
                 ms_per_window_old=1e3 * t_old / len(rows), ms_per_window_new=1e3 * t_new / len(rows))
        res[st] = r
        print(f"[w19 check] {st}: {r}", flush=True)
        assert r["finite"] and r["moved_windows"] > 0, (st, r)
        assert r["rerun_max_dp"] <= 1e-6, (st, "a re-run of NEW differs", r["rerun_max_dp"])
    rows = idx.index[idx["stream"] == a.gate_stream].to_numpy()
    rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
    okI, resI = P18.gate_I(new, np.asarray(X[rows], np.float32), dev, cfg)
    res["gate_I"] = dict(ok=okI, stream=a.gate_stream, refits=resI)
    assert okI, ("gate I failed", resI)
    res["gate_H"] = {}
    for st in a.h_streams.split(","):
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        okH, resH = gate_H(new, np.asarray(X[rows], np.float32), dev)
        res["gate_H"][st] = dict(ok=okH, **resH)
        print(f"[w19 check] gate H {st}: {'PASS' if okH else 'FAIL'} {resH['stats']}", flush=True)
        assert okH, ("gate H failed", st, resH)
        okF, resF = gate_HF(new, np.asarray(X[rows], np.float32), dev)
        res.setdefault("gate_HF", {})[st] = dict(ok=okF, **resF)
        print(f"[w19 check] gate HF {st}: {'PASS' if okF else 'FAIL'} {resF['stats']}", flush=True)
        assert okF, ("gate HF (T2-D553) failed", st, resF)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    if a.bs_streams:
        streams = streams[:a.bs_streams]
    ch = {"old": 0, "new": 0}
    nwin, sw_sum, ad_sum, nprt = 0, 0.0, 0.0, 0
    st_run0 = dict(new.side_refit_stats)
    fr_run0 = dict(new.fm_refit_stats)
    pred_new = np.full(len(idx), -1, np.int64)           # D549 condition K2: NEW's argmax on every public window (in-stream + refit ON)
    for g in streams:
        xs = np.asarray(X[g["i"].to_numpy()], np.float32)
        on = {k: RP.run_parts(m_, xs, dev) for k, m_ in (("old", old), ("new", new))}
        pred_new[g["i"].to_numpy()] = np.stack(on["new"][0]).argmax(1)
        off = {k: RP.run_parts(m_, xs, dev, static=True) for k, m_ in (("old", old), ("new", new))}
        for k in ("old", "new"):
            ch[k] += int((np.stack(on[k][0]).argmax(1) != np.stack(off[k][0]).argmax(1)).sum())
        for t_ in range(len(xs)):
            for f in ("cnn2", "cnn3"):
                j = P18.IDX[f]
                if len(on["old"][1][t_]) > j and on["old"][1][t_][j] is not None:
                    sw_sum += float(np.abs(on["new"][1][t_][j] - on["old"][1][t_][j]).mean())
                    ad_sum += float(np.abs(on["old"][1][t_][j] - off["old"][1][t_][j]).mean())
                    nprt += 1
        nwin += len(xs)
    st_run = {k: new.side_refit_stats[k] - st_run0[k] for k in st_run0}
    fr_run = {k: new.fm_refit_stats[k] - fr_run0[k] for k in fr_run0}
    fo, fn = ch["old"] / nwin, ch["new"] / nwin
    gB = dict(windows=nwin, frac_change_old=fo, frac_change_new=fn, band=[0.5 * fo - 0.005, 1.5 * fo + 0.005])
    gB["ok"] = bool(gB["band"][0] <= fn <= gB["band"][1])
    gS = dict(m=sw_sum / max(nprt, 1), m_adapt=ad_sum / max(nprt, 1))
    gS["ok"] = bool(gS["m"] <= gS["m_adapt"])
    res["gate_B"], res["gate_S"], res["refit_stats_public"], res["fm_refit_stats_public"] = gB, gS, st_run, fr_run
    print(f"[w19 check] gate B: {gB}", flush=True)
    print(f"[w19 check] gate S: {gS}", flush=True)
    print(f"[w19 check] side refit over the public streams: {st_run}", flush=True)
    print(f"[w19 check] FM side refit (T2-D553) over the public streams: {fr_run}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not a.bs_streams:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert st_run["errors"] == 0 and st_run["kept_trained"] == 0, ("D549 condition K: a side-refit fallback on the public streams", st_run)
    assert fr_run["fits"] > 0 and fr_run["errors"] == 0 and fr_run["kept_trained"] == 0, ("T2-D553 / condition K: an FM-refit fallback on the public streams", fr_run)
    assert gB["ok"], ("gate B failed", gB)
    assert gS["ok"], ("gate S failed", gS)
    print("[w19 check] PASS: the D549 + D553 config; finite; moves against W18 on every stream; a re-run is reproducible; gates I, H, HF, B, S; "
          "0 refit fallbacks (side and FM)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-D560 (template W20 = W19 + cfg fm_proto): the post-build check of a W20 package against the W19 package it was built from (OLD = W19
b2), on real public streams, on the GPU path the worker takes. = t2p_w19_pkgcheck's checks with OLD = W19, plus gate HP:
  (0) the config: NEW == OLD + fm_proto {"w": 0.5, "lam_headless": 1.0}, every other key identical; OLD is W19 (fm_side_refit lam 1);
  (a) finite; (b) NEW moves against OLD on every stream; (c) a re-run reproduces NEW (<= 1e-6); (d) ms / window NEW vs OLD;
  (I) the in-stream P1 refit recomputed (t2p_w18_pkgcheck.gate_I, unchanged);
  (H) the side-head refit (t2p_w19_pkgcheck.gate_H, unchanged);
  (HF) the D553 FM side-head refit (t2p_w19_pkgcheck.gate_HF, unchanged) on a copy with fm_proto off: D560 builds on it, gate HP checks the rest;
  (HP) T2-D560 on the same streams (on the package's own path: W19 b2 runs the CUDA graph), recomputed independently from what the package delivered: at each refit point the rows / labels handed to
      the prototypes are the P1 copies' own (a subset of the P1 rows, >= 90 %, labels equal); per member the class prototypes = the means
      of the L2-normalised pooled features of THOSE windows as the hooks delivered them (recorded per window, not the template's history),
      L2-normalised; the head-less member's main head = scripts/local/t2_d539_read.py fit_head (its def) on (0, features, labels) from the
      net's own trained Linear head, pull lam_headless; every window from the refit point on == softmax(logits + w * 10 * cos(feature,
      prototypes)) with logits = log of W19's D553 output (the heads in use at that window) or the re-fitted head-less head <= 1e-6; every
      window before the first refit point == W19's formula <= 1e-12; (f) every window's recorded features through the member's own trained
      head reproduce the member probability of that forward (<= 0.02, bf16; stale graph buffers fail); the prototypes move the output (> 1e-4 somewhere); stats: refits =
      the refit points, head-less fits = head-less members x refits, errors 0, kept-trained 0;
  (B) the in-stream ON vs OFF argmax-change fraction of NEW within [0.5 OLD - 0.005, 1.5 OLD + 0.005] (OLD = W19 b2, the same run);
  (S) m = mean |CNN part (NEW, ON) - (OLD, ON)| <= m_adapt = mean |(OLD, ON) - (OLD, OFF)| (fm_proto leaves the CNN parts unchanged);
  public-stream stats: side refit, D553 FM refit and D560 proto all without errors / fallbacks; NEW's argmax on every public window -> .pred.npz (K2).
  python tracks/t2_bci/scripts/t2p_w20_pkgcheck.py --old <unzipped W19 dir> --new <unzipped W20 dir> [--out res.json]
"""
import argparse
import copy
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
import t2p_w19_pkgcheck as P19  # noqa: E402

PROTO = {"w": 0.5, "lam_headless": 1.0}


def _softmax(lg):
    e = np.exp(lg - lg.max())
    return e / e.sum()


def heads_out(ps, s_, heads):
    """W19's D553 FM head formula (t2p_w19_pkgcheck.gate_HF's), float64."""
    out_ = []
    for p, h in zip(ps, heads):
        p = np.asarray(p, np.float64)
        if h is not None:
            p = _softmax(np.log(np.maximum(p, 1e-30)) + h[0] @ s_ + h[1])
        out_.append(p)
    return out_


def gate_HP(model, xs, dev):
    """T2-D560: the prototypes + head-less refit recomputed on one stream (see the module doc, (HP))."""
    fit_head = P19.reader_fit_head()
    mm = copy.deepcopy(model)
    mm.reset_state()
    assert mm.fm_proto == dict(w=PROTO["w"], lam_hl=PROTO["lam_headless"]) and mm.fm_main is None, mm.fm_proto
    p1, caps, apl = [], [], []
    f_fit, f_pr, f_ap = mm._instream_fit, mm._fm_proto_refit, mm._fm_head_apply
    st0 = dict(mm.fm_proto_stats)
    nhl = sum(h is None for h in mm.fm_heads)
    trained_main = []                                    # the nets' own trained main heads (independent of the template's fm_main0)
    STRouted = type(mm)._fm_head_apply.__globals__["STRouted"]   # the package module's own class (RP.load does not register the module;
    # predict is wrapped by torch.inference_mode, so ITS __globals__ are torch's: an undecorated method's are the package's)
    for net in mm.fm_nets:
        head = (net.net if isinstance(net, STRouted) else net).out
        trained_main.append((head.weight.detach().double().cpu().numpy(), head.bias.detach().double().cpu().numpy()))

    def _fit(rows, ya, dev_, f0=f_fit):
        p1.append((mm.is_n, np.asarray(rows).copy(), np.asarray(ya).copy()))
        return f0(rows, ya, dev_)

    def _pr(c, rows, ys, f0=f_pr):
        out_ = f0(c, rows, ys)
        caps.append(dict(n=c, rows=np.asarray(rows).copy(), ys=np.asarray(ys).copy(),
                         now=None if mm.fm_proto_now is None else ([m_.copy() for m_ in mm.fm_proto_now[0]],
                                                                    [None if h is None else (h[0].copy(), h[1].copy()) for h in mm.fm_proto_now[1]])))
        return out_

    def _ap(ps, f0=f_ap):
        o_ = f0(ps)
        heads = mm.fm_heads if mm.fm_heads_now is None else mm.fm_heads_now
        apl.append(dict(t=int(getattr(mm, "is_n", 0)), ps=[np.asarray(p, np.float64).copy() for p in ps],
                        s=np.asarray(mm._side_now, np.float64).copy(), heads=[None if h is None else (h[0].copy(), h[1].copy()) for h in heads],
                        fe=[None if v is None else np.asarray(v, np.float64).copy() for v in mm._fm_feat],
                        out=[np.asarray(q, np.float64).copy() for q in o_]))
        return o_
    mm._instream_fit, mm._fm_proto_refit, mm._fm_head_apply = _fit, _pr, _ap
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
    want = [int(c) for c in mm.is_cfg["refits"] if c < len(xs)]
    assert [c_["n"] for c_ in caps] == want, ([c_["n"] for c_ in caps], want)
    byt = {a_["t"]: a_ for a_ in apl}
    assert sorted(byt) == list(range(len(xs))), "one FM head application per window"
    p1d = {n_: (r_, y_) for n_, r_, y_ in p1}
    w = PROTO["w"]
    out, moved = [], 0.0
    for k, c_ in enumerate(caps):
        c = c_["n"]
        e = want[k + 1] if k + 1 < len(want) else len(xs)
        rows, ya = p1d.get(c, (None, None))
        r = dict(refit=c, p1_present=rows is not None)
        if rows is not None:
            lab = {int(t): int(y) for t, y in zip(rows, ya)}
            rr = [int(t) for t in c_["rows"]]
            r.update(rows_subset=bool(all(t in lab for t in rr)), rows_frac=len(rr) / max(len(rows), 1),
                     labels_equal=bool(all(lab.get(t) == int(y) for t, y in zip(rr, c_["ys"]))))
            yy = np.asarray([lab[t] for t in rr], np.int64)
            mus, hl = [], []
            for i in range(len(mm.fm_heads)):
                Fe = np.stack([byt[t]["fe"][i] for t in rr])
                fz = Fe / (np.linalg.norm(Fe, axis=1, keepdims=True) + 1e-9)
                mu = np.stack([fz[yy == q].mean(0) if (yy == q).any() else np.zeros(fz.shape[1]) for q in range(3)])
                mus.append(mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9))
                if mm.fm_heads[i] is None:
                    W, b, ok_ = fit_head(np.zeros((len(yy), 3)), Fe, yy, trained_main[i][0], trained_main[i][1], PROTO["lam_headless"])
                    assert ok_, (i, c)
                    hl.append((W, b))
                else:
                    hl.append(None)
            dmu = max(float(np.abs(a_ - b_).max()) for a_, b_ in zip(mus, c_["now"][0]))
            dhl = max([max(float(np.abs(h[0] - g[0]).max()), float(np.abs(h[1] - g[1]).max()))
                       for h, g in zip(hl, c_["now"][1]) if h is not None] or [0.0])
            dapp = 0.0
            for t in range(c, e):
                a_ = byt[t]
                base = heads_out(a_["ps"], a_["s"], a_["heads"])
                for i in range(len(base)):
                    lg = hl[i][0] @ a_["fe"][i] + hl[i][1] if hl[i] is not None else np.log(np.maximum(base[i], 1e-30))
                    fz = a_["fe"][i] / (np.linalg.norm(a_["fe"][i]) + 1e-9)
                    exp_ = _softmax(lg + w * 10.0 * (mus[i] @ fz))
                    dapp = max(dapp, float(np.abs(exp_ - a_["out"][i]).max()))
                    moved = max(moved, float(np.abs(base[i] - a_["out"][i]).max()))
            r.update(n_rows=len(rr), proto_vs_indep=dmu, headless_vs_reader=dhl, apply_vs_indep=dapp)
        out.append(r)
        print(f"[w20 check] gate HP refit {c}: {r}", flush=True)
    first = min(want) if want else len(xs)
    d_pre = max([max(float(np.abs(e_ - q_).max()) for e_, q_ in zip(heads_out(a_["ps"], a_["s"], a_["heads"]), a_["out"]))
                 for t, a_ in byt.items() if t < first] or [0.0])
    # (f) the recorded features ARE this window's: each member's own trained head on them reproduces the member probability the same
    # forward delivered (bf16 autocast tolerance); stale buffers (a graph that does not replay the copy) fail here, not silently
    d_feat = max(float(np.abs(_softmax(trained_main[i][0] @ a_["fe"][i] + trained_main[i][1]) - a_["ps"][i]).max())
                 for a_ in byt.values() for i in range(len(a_["ps"])))
    st1 = mm.fm_proto_stats
    stats = {k: st1[k] - st0[k] for k in st0}
    ok = bool(out) and all(r["p1_present"] and r.get("rows_subset") and r.get("rows_frac", 0) >= 0.9 and r.get("labels_equal")
                           and r.get("proto_vs_indep", 1.0) <= 1e-6 and r.get("headless_vs_reader", 1.0) <= 1e-6 and r.get("apply_vs_indep", 1.0) <= 1e-6
                           for r in out) \
        and d_pre <= 1e-12 and d_feat <= 0.02 and moved > 1e-4 and stats["refits"] == len(want) and stats["hl_fits"] == nhl * len(want) \
        and stats["errors"] == 0 and stats["kept_trained"] == 0
    return ok, dict(refits=out, before_first_refit_max_dp=d_pre, features_vs_member_prob=d_feat, proto_moved_max_dp=moved, stats=stats,
                    headless=nhl, gpu_graph=bool(mm.gpu_graph))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--streams", default="P04|2,P05|1,P09|1")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(P18.EXP), help="the PROTEUS export (sofia: data/ni26/t2_proteus/export_proteus_v1)")
    ap.add_argument("--gate-stream", default="P04|2", help="gate I's stream (>= 150 windows)")
    ap.add_argument("--h-streams", default="P04|2,P09|1", help="gates H / HF / HP's streams (>= 150 windows each)")
    ap.add_argument("--bs-streams", type=int, default=0, help="gates B / S on the first N public streams only (a smoke test; 0 = all)")
    a = ap.parse_args()
    os.environ["NI26_PARTS"] = "1"
    RP.kit.use_track("bci_decoding")
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    cfo = json.loads((Path(a.old) / "config.json").read_text())
    rest = {k: v for k, v in cfg.items() if k != "fm_proto"}
    cfg0 = dict(fm_proto=cfg.get("fm_proto"), old_fm_proto=cfo.get("fm_proto"), rest_equal=rest == cfo, fm_side_refit=cfo.get("fm_side_refit"),
                n_members=len(cfg["fm_members"]), headless=[m["file"] for m in cfg["fm_members"] if not m.get("side")])
    print(f"[w20 check] config: {cfg0}", flush=True)
    assert cfg0["fm_proto"] == PROTO and cfg0["old_fm_proto"] is None and cfg0["rest_equal"], cfg0
    assert cfg0["fm_side_refit"] == {"lam": 1.0} and cfo.get("side", {}).get("refit") == {"lam": 1.0}, "OLD must be W19"
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    assert new.fm_proto is not None and old.__dict__.get("fm_proto") is None
    res = {"config": cfg0, "gpu_graph": bool(new.gpu_graph)}   # W19 b2 runs the CUDA-graph path: the hooks' buffers must be replayed (gate HP (f))
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
        print(f"[w20 check] {st}: {r}", flush=True)
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
        xs = np.asarray(X[rows], np.float32)
        okH, resH = P19.gate_H(new, xs, dev)
        res["gate_H"][st] = dict(ok=okH, **resH)
        print(f"[w20 check] gate H {st}: {'PASS' if okH else 'FAIL'} {resH['stats']}", flush=True)
        assert okH, ("gate H failed", st, resH)
        fp_ = new.fm_proto
        new.fm_proto = None                              # gate HF = D553 alone (its copy takes fm_proto off); restored right after
        try:
            okF, resF = P19.gate_HF(new, xs, dev)
        finally:
            new.fm_proto = fp_
        res.setdefault("gate_HF", {})[st] = dict(ok=okF, **resF)
        print(f"[w20 check] gate HF {st}: {'PASS' if okF else 'FAIL'} {resF['stats']}", flush=True)
        assert okF, ("gate HF (T2-D553) failed", st, resF)
        okP, resP = gate_HP(new, xs, dev)
        res.setdefault("gate_HP", {})[st] = dict(ok=okP, **resP)
        print(f"[w20 check] gate HP {st}: {'PASS' if okP else 'FAIL'} {resP['stats']} moved {resP['proto_moved_max_dp']:.3e}", flush=True)
        assert okP, ("gate HP (T2-D560) failed", st, resP)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    if a.bs_streams:
        streams = streams[:a.bs_streams]
    ch = {"old": 0, "new": 0}
    nwin, sw_sum, ad_sum, nprt = 0, 0.0, 0.0, 0
    st_run0, fr_run0, pr_run0 = dict(new.side_refit_stats), dict(new.fm_refit_stats), dict(new.fm_proto_stats)
    pred_new = np.full(len(idx), -1, np.int64)           # condition K2: NEW's argmax on every public window (in-stream + refits ON)
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
    pr_run = {k: new.fm_proto_stats[k] - pr_run0[k] for k in pr_run0}
    fo, fn = ch["old"] / nwin, ch["new"] / nwin
    gB = dict(windows=nwin, frac_change_old=fo, frac_change_new=fn, band=[0.5 * fo - 0.005, 1.5 * fo + 0.005])
    gB["ok"] = bool(gB["band"][0] <= fn <= gB["band"][1])
    gS = dict(m=sw_sum / max(nprt, 1), m_adapt=ad_sum / max(nprt, 1))
    gS["ok"] = bool(gS["m"] <= gS["m_adapt"])
    res["gate_B"], res["gate_S"] = gB, gS
    res["refit_stats_public"], res["fm_refit_stats_public"], res["fm_proto_stats_public"] = st_run, fr_run, pr_run
    print(f"[w20 check] gate B: {gB}", flush=True)
    print(f"[w20 check] gate S: {gS}", flush=True)
    print(f"[w20 check] public streams: side refit {st_run}; D553 FM refit {fr_run}; D560 proto {pr_run}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not a.bs_streams:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert st_run["errors"] == 0 and st_run["kept_trained"] == 0, ("condition K: a side-refit fallback on the public streams", st_run)
    assert fr_run["fits"] > 0 and fr_run["errors"] == 0 and fr_run["kept_trained"] == 0, ("condition K: a D553 FM-refit fallback", fr_run)
    assert pr_run["refits"] > 0 and pr_run["errors"] == 0 and pr_run["kept_trained"] == 0, ("condition K: a D560 proto fallback", pr_run)
    assert gB["ok"], ("gate B failed", gB)
    assert gS["ok"], ("gate S failed", gS)
    print("[w20 check] PASS: the D560 config (W19 + fm_proto only); finite; moves against W19 on every stream; a re-run is reproducible; "
          "gates I, H, HF, HP, B, S; 0 refit fallbacks (side, D553 FM, D560 proto)", flush=True)


if __name__ == "__main__":
    main()

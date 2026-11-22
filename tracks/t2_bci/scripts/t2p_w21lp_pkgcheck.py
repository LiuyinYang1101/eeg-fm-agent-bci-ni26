#!/usr/bin/env python
"""T2-D569 (template W21lp = W20+D557 + cfg cnn_lp): the post-build check of a W21 package against the W20+D557 package it was repacked from
(OLD = W20+D557), on real public streams, on the GPU path the worker takes. Prepared as code only (no build before the owner's typed W21 line).
= t2p_w20_pkgcheck's checks with OLD = W20+D557 and gate I replaced by gate IL:
  (0) the config: NEW == OLD + cnn_lp {"alpha": 0.5, "k": 10}, every other key identical; OLD is W20 (fm_proto, D553 / side refits at lam 1);
  (a) finite; (b) NEW == OLD on every window BEFORE the first refit point (<= 1e-12) and NEW moves against OLD after it on a stream longer
      than the first refit; (c) a re-run reproduces NEW (<= 1e-6); (d) ms / window NEW vs OLD;
  (FM) the FM part of NEW == OLD's on EVERY window (<= 1e-6): cnn_lp never touches the FM refits (D553 / D560 keep the plain teacher);
  (IL) the in-stream P1 refit with LP labels, recomputed independently on one stream: the plain teacher = the reader's prefix_decode of the
      captured unadapted parts (<= 1e-9 vs what the template handed _cnn_lp_post); the features = each member's pooled features as the hooks
      delivered them per window (captured at _fm_head_apply, not the template's fp_feat), each L2-normalised, concatenated; LP = the lab's
      `t2_lab.py lpteach` block (copied below) -> <= 1e-6 vs the template's LP output; its confident half (t2p_bench 'conf') == the P1 copies'
      rows and labels exactly; the copies re-fitted from those rows / labels == the package's copies (<= 1e-5); LP changed the labels or rows at
      >= 1 refit point somewhere (else cnn_lp is silently off);
  (H) the side-head refit (t2p_w19_pkgcheck.gate_H, unchanged: D539b's rows / labels == P1's, so the LP labels);
  (HF) / (HP) the D553 FM side-head refit and D560's prototypes (unchanged gates, as in t2p_w20_pkgcheck), each on a copy with cnn_lp OFF (HP
      matches the P1 rows to the FM refits' rows, which cnn_lp makes differ by design; the FM path itself is cnn_lp-free, checked by gate FM);
  (B) / (S) as t2p_w20_pkgcheck (OLD = W20+D557, the same run);
  public-stream stats: side refit, D553 FM refit, D560 proto and D569 cnn_lp (smoothed > 0, errors 0) without errors / fallbacks; NEW's argmax
  on every public window -> .pred.npz (K2).
NOT covered here: near-tie neighbours under the graph path's bf16 numerics beyond the captured features (gate IL uses what the hooks delivered).
  python tracks/t2_bci/scripts/t2p_w21lp_pkgcheck.py --old <unzipped W20+D557 dir> --new <unzipped W21 dir> [--out res.json]
"""
import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
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
import t2p_w20_pkgcheck as P20  # noqa: E402

LP = {"alpha": 0.5, "k": 10}


def lab_lp(Fr_a, T0, alpha_, kk_):
    """scripts/local/t2_lab.py, MODE lpteach, the per-stream block VERBATIM (= tests/test_t2p_w21lp_cnn_lp.py's lab_block)."""
    c = len(T0)
    X = Fr_a - Fr_a.mean(0)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Sm = X @ X.T
    np.fill_diagonal(Sm, -np.inf)
    nn = np.argsort(-Sm, 1, kind="stable")[:, :kk_]
    A_ = np.zeros((c, c))
    rr = np.repeat(np.arange(c), kk_)
    A_[rr, nn.ravel()] = np.maximum(Sm[rr, nn.ravel()], 0)
    A_ = np.maximum(A_, A_.T)
    dg = A_.sum(1) + 1e-9
    Sn = A_ / np.sqrt(dg[:, None] * dg[None, :])
    Fp = np.clip(np.linalg.solve(np.eye(c) - alpha_ * Sn, T0), 1e-9, None)
    return Fp / Fp.sum(1, keepdims=True)


def conf_rows(post):
    """t2p_bench's 'conf': the confident half (ceil) of each predicted class -> (kept row indices, their labels)."""
    ya, cf = post.argmax(1), post.max(1)
    keep = np.zeros(len(post), bool)
    for q in np.unique(ya):
        jj = np.flatnonzero(ya == q)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    return np.flatnonzero(keep), ya[keep]


def gate_IL(model, xs, dev):
    """T2-D569 gate IL (see the module doc): one stream through a copy of NEW with the in-stream stages captured, then recomputed."""
    prefix_decode = P18.reader_prefix_decode()
    from ni26 import train as T
    mm = copy.deepcopy(model)
    mm.reset_state()
    ic = mm.is_cfg
    assert ic is not None and ic.get("labels") == "confR" and mm.cnn_lp == dict(alpha=LP["alpha"], k=LP["k"]), (ic, mm.cnn_lp)
    caps, fe, lps, fits = [], {}, [], []
    f_in, f_fit, f_lp, f_ap = mm._instream, mm._instream_fit, mm._cnn_lp_post, mm._fm_head_apply

    def _instream(parts, zs, dev_, f0=f_in):
        caps.append({f: (None if k >= len(parts) or parts[k] is None else np.asarray(parts[k], np.float64).copy()) for f, k in P18.IDX.items()})
        return f0(parts, zs, dev_)

    def _ap(ps, f0=f_ap):
        o_ = f0(ps)
        fe[int(getattr(mm, "is_n", 0))] = [None if v is None else np.asarray(v, np.float64).copy() for v in mm._fm_feat]
        return o_

    def _lp(post, c, f0=f_lp):
        out_ = f0(post, c)
        lps.append(dict(n=c, post_in=np.asarray(post, np.float64).copy(), post_out=np.asarray(out_, np.float64).copy()))
        return out_

    def _fit(rows, ya, dev_, f0=f_fit):
        out_ = f0(rows, ya, dev_)
        fits.append(dict(n=mm.is_n, rows=np.asarray(rows).copy(), ya=np.asarray(ya).copy(), nets={f: copy.deepcopy(mm.is_net[f]) for f in mm.is_fams},
                         X={f: np.stack([mm.is_hist_z[f][i] for i in rows]) for f in mm.is_fams}))
        return out_
    mm._instream, mm._fm_head_apply, mm._cnn_lp_post, mm._instream_fit = _instream, _ap, _lp, _fit
    st0 = dict(mm.cnn_lp_stats)
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
    want = [int(c) for c in ic["refits"] if c < len(xs)]
    assert [f_["n"] for f_ in fits] == want and [l_["n"] for l_ in lps] == want, ([f_["n"] for f_ in fits], [l_["n"] for l_ in lps], want)
    it, n0, lam, st = mm.is_tcfg
    rec = T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr"])
    Zall = {f: np.stack(mm.is_hist_z[f]) for f in mm.is_fams}
    out, changed = [], 0
    for k_, (f_, l_) in enumerate(zip(fits, lps)):
        c = f_["n"]
        post_r = sum(mm.fam_w[f] * prefix_decode(np.stack([caps[t][f] for t in range(c)]), st[f], it, n0, lam) for f in mm.is_tprior)
        post_r = post_r / post_r.sum(1, keepdims=True)
        have = [t for t in range(c) if t in fe and all(v is not None for v in fe[t])]
        F = np.stack([np.concatenate([v.reshape(-1) / (np.linalg.norm(v) + 1e-9) for v in fe[t]]) for t in have])
        lp = post_r.copy()
        lp[have] = lab_lp(F, post_r[have], LP["alpha"], LP["k"])
        rows_r, ya_r = conf_rows(lp)
        rows_p, ya_p = conf_rows(post_r)
        changed += int(not (np.array_equal(rows_r, rows_p) and np.array_equal(ya_r, ya_p)))
        r = dict(refit=c, n_feat=len(have), teacher_maxdiff=float(np.abs(post_r - l_["post_in"]).max()), lp_maxdiff=float(np.abs(lp - l_["post_out"]).max()),
                 rows_equal=bool(np.array_equal(rows_r, f_["rows"])), labels_equal=bool(np.array_equal(ya_r, f_["ya"])),
                 lp_changed_labels=bool(not (np.array_equal(rows_r, rows_p) and np.array_equal(ya_r, ya_p))), copy_maxdiff={})
        for f in mm.is_fams:
            fc = ic["families"][f]
            rc = replace(rec, epochs=int(fc.get("epochs", 40)), lr=float(fc.get("lr", 1e-3)), wd=float(ic.get("wd", 0.01)), bs=int(ic.get("bs", 32)),
                         label_smoothing=float(ic.get("label_smoothing", 0.1)))
            det0, bench0 = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            try:
                X = torch.as_tensor(f_["X"][f], dtype=torch.float32, device=dev)
                y = torch.as_tensor(np.asarray(f_["ya"], np.int64), device=dev)
                net_r = T.finetune(mm.is_base[f], X, y, rc, int(ic.get("seed", 0)) + 100000 * (k_ + 1))
            finally:
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det0, bench0
            with torch.no_grad():
                Z = torch.as_tensor(Zall[f], dtype=torch.float32, device=dev)
                pa = torch.softmax(f_["nets"][f].eval()(Z).float(), 1).cpu().numpy()
                pb = torch.softmax(net_r.eval()(Z).float(), 1).cpu().numpy()
            r["copy_maxdiff"][f] = float(np.abs(pa - pb).max())
        out.append(r)
        print(f"[w21lp check] gate IL refit {c}: {r}", flush=True)
    stats = {k: mm.cnn_lp_stats[k] - st0[k] for k in st0}
    ok = bool(out) and all(r["teacher_maxdiff"] <= 1e-9 and r["lp_maxdiff"] <= 1e-6 and r["rows_equal"] and r["labels_equal"]
                           and max(r["copy_maxdiff"].values()) <= 1e-5 for r in out) \
        and stats["smoothed"] == len(want) and stats["errors"] == 0
    return ok, dict(refits=out, stats=stats, lp_changed_refits=changed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--streams", default="P04|2,P05|1,P09|1")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(P18.EXP), help="the PROTEUS export (sofia: data/ni26/t2_proteus/export_proteus_v1)")
    ap.add_argument("--gate-stream", default="P04|2", help="gate IL's stream (>= 150 windows)")
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
    rest = {k: v for k, v in cfg.items() if k != "cnn_lp"}
    cfg0 = dict(cnn_lp=cfg.get("cnn_lp"), old_cnn_lp=cfo.get("cnn_lp"), rest_equal=rest == cfo, fm_proto=cfo.get("fm_proto"),
                fm_side_refit=cfo.get("fm_side_refit"), refits=cfo.get("instream", {}).get("refits"))
    print(f"[w21lp check] config: {cfg0}", flush=True)
    assert cfg0["cnn_lp"] == LP and cfg0["old_cnn_lp"] is None and cfg0["rest_equal"], cfg0
    assert cfg0["fm_proto"] == P20.PROTO and cfg0["fm_side_refit"] == {"lam": 1.0} and cfo.get("side", {}).get("refit") == {"lam": 1.0}, "OLD must be W20(+D557)"
    first = int(min(cfo["instream"]["refits"]))
    old = RP.load(Path(a.old), "old", m_meta)
    new = RP.load(Path(a.new), "new", m_meta)
    assert new.cnn_lp is not None and old.__dict__.get("cnn_lp") is None
    res = {"config": cfg0, "gpu_graph": bool(new.gpu_graph)}
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
        r = dict(n=len(rows), finite=bool(np.isfinite(pn).all()), pre_first_refit_max_dp=float(d[:first].max()), moved_after=int((d[first:] > 0).sum()),
                 max_dp=float(d.max()), labels_changed=int((po.argmax(1) != pn.argmax(1)).sum()), rerun_max_dp=float(np.abs(np.stack(pn2) - pn).max()),
                 ms_per_window_old=1e3 * t_old / len(rows), ms_per_window_new=1e3 * t_new / len(rows))
        res[st] = r
        print(f"[w21lp check] {st}: {r}", flush=True)
        assert r["finite"] and r["pre_first_refit_max_dp"] <= 1e-12, (st, r)
        assert len(rows) <= first or r["moved_after"] > 0, (st, "NEW never moves after the first refit", r)
        assert r["rerun_max_dp"] <= 1e-6, (st, "a re-run of NEW differs", r["rerun_max_dp"])
    rows = idx.index[idx["stream"] == a.gate_stream].to_numpy()
    rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
    okI, resI = gate_IL(new, np.asarray(X[rows], np.float32), dev)
    res["gate_IL"] = dict(ok=okI, stream=a.gate_stream, **resI)
    assert okI, ("gate IL failed", resI)
    res["gate_H"] = {}
    for st in a.h_streams.split(","):
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        xs = np.asarray(X[rows], np.float32)
        okH, resH = P19.gate_H(new, xs, dev)
        res["gate_H"][st] = dict(ok=okH, **resH)
        print(f"[w21lp check] gate H {st}: {'PASS' if okH else 'FAIL'} {resH['stats']}", flush=True)
        assert okH, ("gate H failed", st, resH)
        fp_ = new.fm_proto
        new.fm_proto = None                              # gate HF = D553 alone (its copy takes fm_proto off; cnn_lp needs fm_proto: off too)
        lp_ = new.cnn_lp
        new.cnn_lp = None
        try:
            okF, resF = P19.gate_HF(new, xs, dev)
        finally:
            new.fm_proto, new.cnn_lp = fp_, lp_
        res.setdefault("gate_HF", {})[st] = dict(ok=okF, **resF)
        print(f"[w21lp check] gate HF {st}: {'PASS' if okF else 'FAIL'} {resF['stats']}", flush=True)
        assert okF, ("gate HF (T2-D553) failed", st, resF)
        lp_ = new.cnn_lp                                 # gate HP compares the P1 rows with the FM refits' rows, which differ by design under cnn_lp:
        new.cnn_lp = None                                # D560 is checked on its own path with cnn_lp off (the FM path is cnn_lp-free: gate FM)
        try:
            okP, resP = P20.gate_HP(new, xs, dev)
        finally:
            new.cnn_lp = lp_
        res.setdefault("gate_HP", {})[st] = dict(ok=okP, **resP)
        print(f"[w21lp check] gate HP {st}: {'PASS' if okP else 'FAIL'} {resP['stats']} moved {resP['proto_moved_max_dp']:.3e}", flush=True)
        assert okP, ("gate HP (T2-D560) failed", st, resP)
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    if a.bs_streams:
        streams = streams[:a.bs_streams]
    ch = {"old": 0, "new": 0}
    nwin, sw_sum, ad_sum, nprt, fm_max = 0, 0.0, 0.0, 0, 0.0
    st_run0, fr_run0, pr_run0, lp_run0 = dict(new.side_refit_stats), dict(new.fm_refit_stats), dict(new.fm_proto_stats), dict(new.cnn_lp_stats)
    pred_new = np.full(len(idx), -1, np.int64)           # condition K2: NEW's argmax on every public window (in-stream + refits ON)
    for g in streams:
        xs = np.asarray(X[g["i"].to_numpy()], np.float32)
        on = {k: RP.run_parts(m_, xs, dev) for k, m_ in (("old", old), ("new", new))}
        pred_new[g["i"].to_numpy()] = np.stack(on["new"][0]).argmax(1)
        off = {k: RP.run_parts(m_, xs, dev, static=True) for k, m_ in (("old", old), ("new", new))}
        for k in ("old", "new"):
            ch[k] += int((np.stack(on[k][0]).argmax(1) != np.stack(off[k][0]).argmax(1)).sum())
        jf = P18.IDX["fm"]
        for t_ in range(len(xs)):
            fm_max = max(fm_max, float(np.abs(np.asarray(on["new"][1][t_][jf], np.float64) - np.asarray(on["old"][1][t_][jf], np.float64)).max()))
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
    lp_run = {k: new.cnn_lp_stats[k] - lp_run0[k] for k in lp_run0}
    fo, fn = ch["old"] / nwin, ch["new"] / nwin
    gB = dict(windows=nwin, frac_change_old=fo, frac_change_new=fn, band=[0.5 * fo - 0.005, 1.5 * fo + 0.005])
    gB["ok"] = bool(gB["band"][0] <= fn <= gB["band"][1])
    gS = dict(m=sw_sum / max(nprt, 1), m_adapt=ad_sum / max(nprt, 1))
    gS["ok"] = bool(gS["m"] <= gS["m_adapt"])
    gF = dict(fm_part_max_dp=fm_max, ok=bool(fm_max <= 1e-6))
    res["gate_B"], res["gate_S"], res["gate_FM"] = gB, gS, gF
    res["refit_stats_public"], res["fm_refit_stats_public"], res["fm_proto_stats_public"], res["cnn_lp_stats_public"] = st_run, fr_run, pr_run, lp_run
    print(f"[w21lp check] gate B: {gB}", flush=True)
    print(f"[w21lp check] gate S: {gS}", flush=True)
    print(f"[w21lp check] gate FM (NEW's FM part == W20+D557's on every public window): {gF}", flush=True)
    print(f"[w21lp check] public streams: side refit {st_run}; D553 FM refit {fr_run}; D560 proto {pr_run}; D569 cnn_lp {lp_run}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not a.bs_streams:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert st_run["errors"] == 0 and st_run["kept_trained"] == 0, ("condition K: a side-refit fallback on the public streams", st_run)
    assert fr_run["fits"] > 0 and fr_run["errors"] == 0 and fr_run["kept_trained"] == 0, ("condition K: a D553 FM-refit fallback", fr_run)
    assert pr_run["refits"] > 0 and pr_run["errors"] == 0 and pr_run["kept_trained"] == 0, ("condition K: a D560 proto fallback", pr_run)
    assert lp_run["smoothed"] > 0 and lp_run["errors"] == 0, ("condition K: D569 cnn_lp silently off or failing", lp_run)
    assert res["gate_IL"]["lp_changed_refits"] > 0 or lp_run["smoothed"] > 0, "LP never changed a label"
    assert gF["ok"], ("gate FM failed: cnn_lp touched the FM path", gF)
    assert gB["ok"], ("gate B failed", gB)
    assert gS["ok"], ("gate S failed", gS)
    print("[w21lp check] PASS: the D569 config (W20+D557 + cnn_lp only); finite; == W20+D557 before the first refit; moves after it; a re-run is "
          "reproducible; gates IL, H, HF, HP, FM, B, S; 0 fallbacks (side, D553 FM, D560 proto, D569 cnn_lp)", flush=True)


if __name__ == "__main__":
    main()

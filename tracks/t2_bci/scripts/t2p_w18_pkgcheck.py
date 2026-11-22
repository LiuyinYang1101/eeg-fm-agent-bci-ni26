#!/usr/bin/env python
"""T2-D534 (template W18): the post-build check of a W18 package against the W17 package it was repacked from (OLD = W17 build N), on real
public streams, on the GPU path the worker takes:
  (0) the config is what the user's W18 line approved: cfg fm_side on, cfg imcoh absent / off, six FM members, exactly ONE head-less member (the
      swapped MI-expert member, carrying its merge source) and five with a window-mean side head;
  (a) every output finite; (b) NEW moves against OLD on every stream (the heads act from window 1, the swapped member always);
  (c) a second run of NEW on the same stream gives the same outputs (<= 1e-6);
  (d) the time per window of NEW vs OLD (the runtime the audit sees);
  T2-D534 conditions (registered before any build output):
  (I) the in-stream refit recomputed independently on one stream (refits 60 / 100 / 150, both adapted families): the template's teacher
      posteriors vs the reader's prefix decode (scripts/local/t2_d511_teacher.prefix_decode on the captured unadapted family parts) <= 1e-9; the
      selected rows / labels == an inline re-implementation of t2p_bench's 'conf'; each refitted copy vs ni26.train.finetune (same member, inputs,
      labels, seed, deterministic cuDNN): softmax outputs on the stream's windows <= 1e-5;
  (B) the fraction of public windows whose argmax changes between in-stream ON and OFF, for OLD and NEW (the band is judged across builds);
  (S) m_swap = mean |CNN part (NEW, ON) - (OLD, ON)| <= m_adapt = mean |CNN part (OLD, ON) - (OLD, OFF)| over the adapted families' parts.
The arithmetic itself is proven by t2p_w18_selftest.py and t2p_repack.py (the side-head checks: the members' plain mean == the head-free donor, the
heads == an independent recomputation; the independent recombination).
  python tracks/t2_bci/scripts/t2p_w18_pkgcheck.py --old <unzipped W17 dir> --new <unzipped W18 dir> [--streams P04|2,P05|1,P09|1]
"""
import argparse
import ast
import copy
import hashlib
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

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
IDX = {"cnn": 0, "cnn2": 1, "fm": 2, "cnn3": 3}


def reader_prefix_decode():
    """scripts/local/t2_d511_teacher.py's prefix_decode, compiled from that file's own source text WITHOUT running the module (importing it
    runs the whole D511 teacher build over local files: G2 run 1622536 died there on sofia). Same bytes as the reader's def; numpy only."""
    src = (HERE.parents[2] / "scripts/local/t2_d511_teacher.py").read_text()
    fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "prefix_decode"]
    assert len(fn) == 1, "t2_d511_teacher.prefix_decode not found exactly once"
    seg = ast.get_source_segment(src, fn[0])
    ns = {"np": np}
    exec(compile(seg, "t2_d511_teacher.py:prefix_decode", "exec"), ns)
    print(f"[w18 check] gate I reference: t2_d511_teacher.prefix_decode, sha256 {hashlib.sha256(seg.encode()).hexdigest()[:12]}", flush=True)
    return ns["prefix_decode"]


def gate_I(model, xs, dev, cfg):
    """T2-D534 gate I: run one stream through a copy of NEW with the in-stream stages captured, then recompute teacher, labels and copies."""
    prefix_decode = reader_prefix_decode()              # the reader's implementation (D511), its def only (G2 run 1: the import ran the script)
    from ni26 import train as T
    mm = copy.deepcopy(model)
    mm.reset_state()
    ic = mm.is_cfg
    assert ic is not None and ic.get("labels") == "confR", ic
    caps, fits = [], []
    f_in, f_fit = mm._instream, mm._instream_fit

    def _instream(parts, zs, dev_, f0=f_in):
        caps.append({f: (None if k >= len(parts) or parts[k] is None else np.asarray(parts[k], np.float64).copy()) for f, k in IDX.items()})
        return f0(parts, zs, dev_)

    def _fit(rows, ya, dev_, f0=f_fit):
        n = mm.is_n
        it, n0, lam, _ = mm.is_tcfg
        post_t = sum(mm.fam_w[f] * _tmpl_prefix(mm, cr) for f, cr in mm.is_tprior.items())
        post_t = post_t / post_t.sum(1, keepdims=True)
        out_ = f0(rows, ya, dev_)
        fits.append(dict(n=n, rows=np.asarray(rows).copy(), ya=np.asarray(ya).copy(), post_t=post_t,
                         nets={f: copy.deepcopy(mm.is_net[f]) for f in mm.is_fams},
                         X={f: np.stack([mm.is_hist_z[f][i] for i in rows]) for f in mm.is_fams}))
        return out_
    mm._instream, mm._instream_fit = _instream, _fit
    for x in xs:
        mm.predict(torch.as_tensor(x[None], device=dev))
    assert [f_["n"] for f_ in fits] == [int(c) for c in ic["refits"] if c < len(xs)], [f_["n"] for f_ in fits]
    it, n0, lam, st = mm.is_tcfg
    rec = T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr"])
    Zall = {f: np.stack(mm.is_hist_z[f]) for f in mm.is_fams}
    out = []
    for k_, f_ in enumerate(fits):
        c = f_["n"]
        post_r = sum(mm.fam_w[f] * prefix_decode(np.stack([caps[t][f] for t in range(c)]), st[f], it, n0, lam) for f in mm.is_tprior)
        post_r = post_r / post_r.sum(1, keepdims=True)
        ya_b, cf = post_r.argmax(1), post_r.max(1)               # t2p_bench's 'conf': the confident half (ceil) of each predicted class
        keep = np.zeros(c, bool)
        for q in np.unique(ya_b):
            jj = np.flatnonzero(ya_b == q)
            keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
        r = dict(refit=c, post_maxdiff=float(np.abs(post_r - f_["post_t"]).max()),
                 rows_equal=bool(np.array_equal(np.flatnonzero(keep), f_["rows"])), labels_equal=bool(np.array_equal(ya_b[keep], f_["ya"])), copy_maxdiff={})
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
        print(f"[w18 check] gate I refit {c}: {r}", flush=True)
    ok = bool(out) and all(r["post_maxdiff"] <= 1e-9 and r["rows_equal"] and r["labels_equal"] and max(r["copy_maxdiff"].values()) <= 1e-5 for r in out)
    return ok, out


def _tmpl_prefix(mm, cr):
    """the template's own _prefix_decode on its corrector's rows (what _instream_refit computed); RP.load does not register the module."""
    pdec = type(mm)._instream_refit.__globals__["_prefix_decode"]
    it, n0, lam, _ = mm.is_tcfg
    return pdec(cr.rows, it, lam, cr.k, n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--streams", default="P04|2,P05|1,P09|1")
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(EXP), help="the PROTEUS export (sofia: data/ni26/t2_proteus/export_proteus_v1)")
    ap.add_argument("--gate-stream", default="P04|2", help="T2-D534 gate I's stream (>= 150 windows: all three refits fire)")
    ap.add_argument("--bs-streams", type=int, default=0, help="gates B / S on the first N public streams only (a smoke test; 0 = all)")
    ap.add_argument("--no-config-check", action="store_true", help="a smoke test on a non-W18 package")
    a = ap.parse_args()
    os.environ["NI26_PARTS"] = "1"                  # the templates read it at import: last_parts for gates B / S (diagnostics only)
    RP.kit.use_track("bci_decoding")
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    cfg = json.loads((Path(a.new) / "config.json").read_text())
    cfo = json.loads((Path(a.old) / "config.json").read_text())
    fm = cfg["fm_members"]
    heads = [bool(m.get("side")) for m in fm]
    cfg0 = dict(fm_side=bool(cfg.get("fm_side")), imcoh=cfg.get("imcoh"), n_members=len(fm), heads=sum(heads),
                headless=[m["file"] for m in fm if not m.get("side")], headless_source=[m.get("source") for m in fm if not m.get("side")])
    print(f"[w18 check] config: {cfg0}", flush=True)
    if not a.no_config_check:
        assert cfg0["fm_side"] and not cfg0["imcoh"], "W18 check: cfg fm_side on and imcoh off"
        assert len(fm) == len(cfo["fm_members"]) == 6 and sum(heads) == 5, (len(fm), sum(heads))
        assert len(cfg0["headless"]) == 1 and (cfg0["headless_source"][0] or {}).get("zip"), "one head-less swapped member with its merge source"
        assert not cfo.get("fm_side") and cfo.get("side") and cfo.get("dcfam") and cfo.get("instream"), "OLD must be W17"
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
        print(f"[w18 check] {st}: {r}", flush=True)
        assert r["finite"] and r["moved_windows"] > 0, (st, r)
        assert r["rerun_max_dp"] <= 1e-6, (st, "a re-run of NEW differs", r["rerun_max_dp"])
    # T2-D534 gate I: the in-stream refit, recomputed independently on one stream
    rows = idx.index[idx["stream"] == a.gate_stream].to_numpy()
    rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
    okI, resI = gate_I(new, np.asarray(X[rows], np.float32), dev, cfg)
    res["gate_I"] = dict(ok=okI, stream=a.gate_stream, refits=resI)
    assert okI, ("gate I failed", resI)
    # T2-D534 gates B / S: in-stream ON vs OFF over the public streams, OLD and NEW
    streams = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)]
    if a.bs_streams:
        streams = streams[:a.bs_streams]
    ch = {"old": 0, "new": 0}
    nwin, sw_sum, ad_sum, nprt = 0, 0.0, 0.0, 0
    dpf, adn_sum = {"old": 0.0, "new": 0.0}, 0.0
    for g in streams:
        xs = np.asarray(X[g["i"].to_numpy()], np.float32)
        on = {k: RP.run_parts(m_, xs, dev) for k, m_ in (("old", old), ("new", new))}
        off = {k: RP.run_parts(m_, xs, dev, static=True) for k, m_ in (("old", old), ("new", new))}
        for k in ("old", "new"):
            ch[k] += int((np.stack(on[k][0]).argmax(1) != np.stack(off[k][0]).argmax(1)).sum())
            dpf[k] += float(np.abs(np.stack(on[k][0]) - np.stack(off[k][0])).mean(1).sum())   # continuous ON/OFF movement of the output
        for t_ in range(len(xs)):
            for f in ("cnn2", "cnn3"):
                j = IDX[f]
                if len(on["old"][1][t_]) > j and on["old"][1][t_][j] is not None:
                    sw_sum += float(np.abs(on["new"][1][t_][j] - on["old"][1][t_][j]).mean())
                    ad_sum += float(np.abs(on["old"][1][t_][j] - off["old"][1][t_][j]).mean())
                    adn_sum += float(np.abs(on["new"][1][t_][j] - off["new"][1][t_][j]).mean())
                    nprt += 1
        nwin += len(xs)
    gB = dict(windows=nwin, frac_change_old=ch["old"] / nwin, frac_change_new=ch["new"] / nwin,
              # added after build 1's check (the argmax fraction has no power on the in-sample public streams; diagnostics, judged only if adopted):
              mean_dp_final_old=dpf["old"] / nwin, mean_dp_final_new=dpf["new"] / nwin,
              m_adapt_old=ad_sum / max(nprt, 1), m_adapt_new=adn_sum / max(nprt, 1))
    gS = dict(m_swap=sw_sum / max(nprt, 1), m_adapt=ad_sum / max(nprt, 1))
    gS["ok"] = bool(gS["m_swap"] <= gS["m_adapt"])
    res["gate_B_inputs"], res["gate_S"] = gB, gS
    print(f"[w18 check] gate B inputs (the band is judged across builds): {gB}", flush=True)
    print(f"[w18 check] gate S: {gS}", flush=True)
    assert gS["ok"], ("gate S failed", gS)
    print("[w18 check] PASS: the approved config; finite; moves against W17 on every stream; a re-run is reproducible; gates I and S", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

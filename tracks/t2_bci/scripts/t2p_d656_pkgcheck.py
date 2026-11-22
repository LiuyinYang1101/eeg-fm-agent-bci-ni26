#!/usr/bin/env python
"""T2-D656 (registered 544853f8): the post-build check of a CONFIG-ONLY decode-lambda variant of W25-ALL6 (OLD = W25-ALL6 unzipped, NEW = the
repack with --set tta.lam=<x> --set-ref). Passes only if: (a) the configs differ in EXACTLY the dotted key tta.lam, NEW's value == --expect-lam;
(b) every other file is byte-equal and the file sets are equal; (c) on EVERY public PROTEUS stream, one window at a time from a fresh copy
(t2p_repack.run_stream = the kit's deepcopy + reset_state), NEW's probabilities are finite and a second NEW run is bitwise equal. Reported
(descriptive): windows whose label moved vs OLD, max |dp|, and the PAIRED ms / window of OLD and NEW on this GPU (the registered timing
fallback: within +-2 % -> NEW inherits W25-ALL6's H200 timing). Writes <out> (json) and <out stem>.pred.npz (NEW's predictions for K2).
    python t2p_d656_pkgcheck.py --old <dir> --new <dir> --export <export> --out <name>.pkgcheck.json [--expect-lam 1.0]
"""
import argparse
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


def flat(x, p=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{p}.{k}" if p else str(k)))
        return out
    return {p: x}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-lam", type=float, default=float(os.environ.get("D656_EXPECT_LAM", "1.0")))   # the pkgcheck slurm passes no flag: env
    a = ap.parse_args()
    RP.kit.use_track("bci_decoding")
    old, new = Path(a.old), Path(a.new)
    co, cn = (json.loads((d / "config.json").read_text()) for d in (old, new))
    fo, fn = flat(co), flat(cn)
    diff = sorted(k for k in set(fo) | set(fn) if fo.get(k, "<absent>") != fn.get(k, "<absent>"))
    files_o = sorted(str(p.relative_to(old)) for p in old.rglob("*") if p.is_file())
    files_n = sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file())
    neq = [f for f in files_o if f != "config.json" and f in files_n and sha(old / f) != sha(new / f)]
    res = dict(config_diff=diff, old_lam=fo.get("tta.lam"), new_lam=fn.get("tta.lam"), files_old=len(files_o), files_new=len(files_n),
               file_sets_equal=files_o == files_n, files_not_equal=neq)
    print(f"[d656 check] config diff {diff}: tta.lam {res['old_lam']} -> {res['new_lam']}; files {len(files_o)} / {len(files_n)}, not equal {neq}", flush=True)
    exp = os.environ.get("D656_EXPECT_SET", "")          # T2-D658: 'k1=v1;k2=v2' (dotted keys, JSON values); unset = D656's tta.lam only
    if exp:
        want = {kv.split("=", 1)[0]: json.loads(kv.split("=", 1)[1]) for kv in exp.split(";") if kv}
        res["expect_set"] = want
        assert diff == sorted(want) and all(abs(float(fn[k]) - float(v)) < 1e-12 for k, v in want.items()), ("D658 (a): the config diff must be exactly the expected keys", diff, {k: fn.get(k) for k in want})
        if os.environ.get("D658_SHARE_SCALE"):          # T2-D658 (Track3's pin 1): the implied family shares (main, w2, w3, fm) == OLD's x r (fm = 1 - r (1 - fm_old))
            num, den = (float(x) for x in os.environ["D658_SHARE_SCALE"].split("/")); r = num / den
            sh = lambda f: dict(main=1.0 - f["fm_weight"] - f["cnn2.weight"] - f["cnn3.weight"], w2=f["cnn2.weight"], w3=f["cnn3.weight"], fm=f["fm_weight"])   # noqa: E731
            so, sn = sh(fo), sh(fn)
            exp_sh = dict(main=so["main"] * r, w2=so["w2"] * r, w3=so["w3"] * r, fm=1.0 - r * (1.0 - so["fm"]))
            res["shares_old"], res["shares_new"], res["shares_expected"] = so, sn, exp_sh
            print(f"[d658 check] shares old {so} -> new {sn} (expected {exp_sh})", flush=True)
            assert all(abs(sn[k] - exp_sh[k]) < 1e-12 for k in sn), ("D658 (a'): the implied family shares != OLD's x r", sn, exp_sh)
    else:
        assert diff == ["tta.lam"] and abs(float(res["new_lam"]) - a.expect_lam) < 1e-12, ("D656 (a): the config diff must be exactly tta.lam", diff, res["new_lam"])
    assert res["file_sets_equal"] and not neq, ("D656 (b): files differ beyond config.json", neq)
    X, idx, meta = B.load_export(a.export)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    mo, mn = RP.load(old, "old", m_meta), RP.load(new, "new", m_meta)
    I, PR, per = [], [], {}
    t_old = t_new = 0.0
    for st, g in idx.groupby("stream", sort=False):
        g = g.sort_values("order")
        ii = g["i"].to_numpy()
        xs = np.asarray(X[ii], np.float32)
        po, pn, pn2 = [], [], []
        t0 = time.time(); lo, _ = RP.run_stream(mo, xs, dev, po); t_old += time.time() - t0
        t0 = time.time(); ln, fin = RP.run_stream(mn, xs, dev, pn); t_new += time.time() - t0
        ln2, _ = RP.run_stream(mn, xs, dev, pn2)
        po, pn, pn2 = np.stack(po), np.stack(pn), np.stack(pn2)
        r = dict(n=int(len(ii)), finite=bool(fin) and bool(np.isfinite(pn).all()), rerun_equal=bool(np.array_equal(pn, pn2)),
                 labels_moved=int((lo != ln).sum()), max_dp=float(np.abs(po - pn).max()))
        per[str(st)] = r
        I.append(ii)
        PR.append(ln)
        print(f"[d656 check] {st}: {r}", flush=True)
    n = sum(r["n"] for r in per.values())
    res.update(streams=len(per), windows=n, per_stream=per, all_finite=all(r["finite"] for r in per.values()),
               all_rerun_equal=all(r["rerun_equal"] for r in per.values()), labels_moved=sum(r["labels_moved"] for r in per.values()),
               ms_per_window_old=1e3 * t_old / n, ms_per_window_new=1e3 * t_new / n, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    res["paired_ms_ratio"] = res["ms_per_window_new"] / res["ms_per_window_old"]
    res["paired_timing_within_2pct"] = abs(res["paired_ms_ratio"] - 1.0) <= 0.02
    Path(a.out).write_text(json.dumps(res, indent=1))
    np.savez(Path(a.out).with_suffix("").as_posix() + ".pred.npz", i=np.concatenate(I), pred=np.concatenate(PR).astype(np.int64))
    print(f"[d656 check] {len(per)} streams, {n} windows: finite {res['all_finite']}, rerun-equal {res['all_rerun_equal']}, labels moved vs OLD "
          f"{res['labels_moved']}; PAIRED ms / window OLD {res['ms_per_window_old']:.2f} / NEW {res['ms_per_window_new']:.2f} (ratio {res['paired_ms_ratio']:.4f}) on {res['gpu']}", flush=True)
    assert len(per) == 14 and res["all_finite"] and res["all_rerun_equal"], "D656 (c): finite / rerun-equal on every public stream"
    print("[d656 check] PASS", flush=True)


if __name__ == "__main__":
    main()

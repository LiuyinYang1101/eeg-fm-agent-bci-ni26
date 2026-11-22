#!/usr/bin/env python
"""T2-D652 (W30 one-hop; the manager's amendment 2026-11-21 17:31): the post-build check of W30 = W25-ALL6 with its FM members swapped (OLD =
W25-ALL6 unzipped, NEW = the one-hop repack --fm-from the merged W30 donor). (= t2p_d656_pkgcheck.py with the FM-swap allowance.) Passes only if:
(a) the configs differ ONLY in fm_members (repo_commit allowed: the build snapshot); (b) outside the FM member / side files named in each config,
the file sets are equal and every file is byte-equal; (c) on EVERY public PROTEUS stream, one window at a time from a fresh copy
(t2p_repack.run_stream = the kit's deepcopy + reset_state), NEW's probabilities are finite and a second NEW run is bitwise equal. Reported
(descriptive): windows whose label moved vs OLD, max |dp|, and the PAIRED ms / window of OLD and NEW on this GPU (the registered timing
fallback: within +-2 % -> NEW inherits W25-ALL6's H200 timing: same template and member architecture). Writes <out> (json) and <out stem>.pred.npz (NEW's predictions for K2).
    python t2p_w30_pkgcheck.py --old <dir> --new <dir> --export <export> --out <name>.pkgcheck.json
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
    a = ap.parse_args()
    RP.kit.use_track("bci_decoding")
    old, new = Path(a.old), Path(a.new)
    co, cn = (json.loads((d / "config.json").read_text()) for d in (old, new))
    fo, fn = flat(co), flat(cn)
    diff = sorted(k for k in set(fo) | set(fn) if fo.get(k, "<absent>") != fn.get(k, "<absent>"))
    fm_files = lambda c: {m["file"] for m in c["fm_members"]} | {m["side"]["file"] for m in c["fm_members"] if m.get("side")}   # noqa: E731
    fmo, fmn = fm_files(co), fm_files(cn)
    files_o = sorted(str(p.relative_to(old)) for p in old.rglob("*") if p.is_file())
    files_n = sorted(str(p.relative_to(new)) for p in new.rglob("*") if p.is_file())
    rest_o, rest_n = sorted(set(files_o) - fmo - {"config.json"}), sorted(set(files_n) - fmn - {"config.json"})
    neq = [f for f in rest_o if f in rest_n and sha(old / f) != sha(new / f)]
    fm_changed = sorted(m["file"] for m in cn["fm_members"] if not ((old / m["file"]).exists() and sha(old / m["file"]) == sha(new / m["file"])))
    res = dict(config_diff=diff, files_old=len(files_o), files_new=len(files_n), nonfm_sets_equal=rest_o == rest_n, nonfm_not_equal=neq,
               nonfm_compared=len(rest_o), fm_members_changed=fm_changed, n_members=len(cn["fm_members"]))
    print(f"[w30 check] config diff {diff}; non-FM files {len(rest_o)} / {len(rest_n)}, not equal {neq}; FM members changed {fm_changed}", flush=True)
    assert set(diff) <= {"fm_members", "repo_commit"} and "fm_members" in diff, ("W30 (a): the config diff must be fm_members only", diff)
    assert res["nonfm_sets_equal"] and not neq, ("W30 (b): non-FM files differ", neq)
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
        print(f"[w30 check] {st}: {r}", flush=True)
    n = sum(r["n"] for r in per.values())
    res.update(streams=len(per), windows=n, per_stream=per, all_finite=all(r["finite"] for r in per.values()),
               all_rerun_equal=all(r["rerun_equal"] for r in per.values()), labels_moved=sum(r["labels_moved"] for r in per.values()),
               ms_per_window_old=1e3 * t_old / n, ms_per_window_new=1e3 * t_new / n, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu")
    res["paired_ms_ratio"] = res["ms_per_window_new"] / res["ms_per_window_old"]
    res["paired_timing_within_2pct"] = abs(res["paired_ms_ratio"] - 1.0) <= 0.02
    Path(a.out).write_text(json.dumps(res, indent=1))
    np.savez(Path(a.out).with_suffix("").as_posix() + ".pred.npz", i=np.concatenate(I), pred=np.concatenate(PR).astype(np.int64))
    print(f"[w30 check] {len(per)} streams, {n} windows: finite {res['all_finite']}, rerun-equal {res['all_rerun_equal']}, labels moved vs OLD "
          f"{res['labels_moved']}; PAIRED ms / window OLD {res['ms_per_window_old']:.2f} / NEW {res['ms_per_window_new']:.2f} (ratio {res['paired_ms_ratio']:.4f}) on {res['gpu']}", flush=True)
    assert len(per) == 14 and res["all_finite"] and res["all_rerun_equal"], "W30 (c): finite / rerun-equal on every public stream"
    print("[w30 check] PASS", flush=True)


if __name__ == "__main__":
    main()

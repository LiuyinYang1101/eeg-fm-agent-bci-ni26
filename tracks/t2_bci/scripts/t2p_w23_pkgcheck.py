#!/usr/bin/env python
"""T2-D634 (W23 = W21 + D632 P1, config-only; PREBUILD until item 166 is typed in T2's shell): the post-build check of a W23 package against
W21 (OLD), on every public stream, on the path the worker takes (t2p_repack.load / run_stream: a fresh deepcopy + reset_state per stream,
one window at a time). Both packages' StreamModel._user (the copy router) are wrapped at the class level to record each window's route k;
OLD's can instead REPLAY a given k sequence.
  (0) config: NEW == OLD except user_blend.{mean, scale, W, mu}, which equal tracks/t2_bci/sofia/w23_route_s1.json exactly; every other file
      of the package (submission.py, weights, FM members, ...) byte-identical to OLD's;
  (a) finite; (c) a re-run reproduces NEW (<= 1e-6);
  (K) OLD's routes == the lab route in W21's space (t2p_fpcalib fit_space on ALL export rows + stream_route_every) on every window (the inert
      check); NEW's == the lab route in the session-1 space (fit_space on every user's session-1 rows) on every window;
  (G) GOLDEN: OLD with its router REPLAYING NEW's recorded k == NEW on every window (<= 1e-6): the route index is the ONLY thing that changed;
  descriptive: per stream the windows whose route moved, and NEW vs OLD on the streams whose route never moved;
  NEW's argmax on every public window -> .pred.npz (K2).
  python tracks/t2_bci/scripts/t2p_w23_pkgcheck.py --old <unzipped W21 dir> --new <unzipped W23 dir> [--out res.json] [--streams N]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402
import t2p_route as RT  # noqa: E402

KEYS = ("mean", "scale", "W", "mu")
ROUTE_JSON = REPO / "tracks/t2_bci/sofia/w23_route_s1.json"
LOG = {"old": [], "new": []}
REPLAY = {"on": False, "q": []}


def wrap_router(model, tag):
    """wrap the package's StreamModel._user at the class level (each package has its own module, so its own class)."""
    cls = type(model)
    orig = cls._user

    def _user(self, xi):
        if tag == "old" and REPLAY["on"]:
            return REPLAY["q"].pop(0)
        k = orig(self, xi)
        LOG[tag].append(-1 if k is None else int(k))
        return k
    cls._user = _user


def run(model, xs, dev, tag):
    LOG[tag].clear()
    probs = []
    RP.run_stream(model, xs, dev, probs)
    return np.stack(probs), np.array(LOG[tag])


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--export", default=str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    ap.add_argument("--streams", type=int, default=0, help="the first N public streams only (a smoke test; 0 = all, and the .pred.npz for K2)")
    ap.add_argument("--max-windows", type=int, default=0, help="a local smoke only: the first N windows of each stream (no .pred.npz)")
    a = ap.parse_args()
    smoke = bool(a.streams or a.max_windows)
    RP.kit.use_track("bci_decoding")
    old_d, new_d = Path(a.old), Path(a.new)
    # (0) config + files
    cfo, cfn = (json.loads((d / "config.json").read_text()) for d in (old_d, new_d))
    doc = json.loads(ROUTE_JSON.read_text())
    rest_eq = {k: v for k, v in cfo.items() if k != "user_blend"} == {k: v for k, v in cfn.items() if k != "user_blend"}
    ub_eq = {k: v for k, v in cfo["user_blend"].items() if k not in KEYS} == {k: v for k, v in cfn["user_blend"].items() if k not in KEYS}
    vals = all(cfn["user_blend"][k] == doc[k] for k in KEYS) and doc["users"] == cfn["user_blend"]["users"]
    moved = all(cfn["user_blend"][k] != cfo["user_blend"][k] for k in KEYS)
    fo = sorted(p.relative_to(old_d).as_posix() for p in old_d.rglob("*") if p.is_file())
    fn = sorted(p.relative_to(new_d).as_posix() for p in new_d.rglob("*") if p.is_file())
    files_eq = fo == fn and all(sha(old_d / f) == sha(new_d / f) for f in fo if f != "config.json")
    cfg0 = dict(rest_equal=rest_eq, user_blend_rest_equal=ub_eq, values_eq_json=vals, four_keys_moved=moved, files=len(fo), files_equal=files_eq,
                route_json_sha256=sha(ROUTE_JSON))
    print(f"[w23 check] (0) {cfg0}", flush=True)
    assert rest_eq and ub_eq and vals and moved and files_eq, cfg0
    # the lab routes (independent of the package: t2p_route.fingerprint + sklearn refits, as t2p_final.py)
    X, idx, meta = B.load_export(a.export)
    subj, ses = idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy()
    Fp = RT.fingerprint(np.asarray(X, np.float64), float(meta["sfreq"]), list(meta["ch_names"]))
    projS, muS, usersS = FC.fit_space(Fp, subj)
    s1 = ses == "1"
    projP, muP, usersP = FC.fit_space(Fp[s1], subj[s1])
    assert list(usersS) == list(usersP) == list(cfo["user_blend"]["users"]), (usersS, usersP)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
    old = RP.load(old_d, "old", m_meta)
    new = RP.load(new_d, "new", m_meta)
    wrap_router(old, "old")
    wrap_router(new, "new")
    streams = list(dict.fromkeys(idx.sort_values(["stream", "order"])["stream"].tolist()))
    if a.streams:
        streams = streams[:a.streams]
    pred_new = np.full(len(idx), -1)
    worst = dict(rerun=0.0, golden=0.0, same_route_new_vs_old=0.0)
    bad = dict(old_route=0, new_route=0)
    finite, n_win, per = True, 0, {}
    for st in streams:
        rows = idx.index[idx["stream"] == st].to_numpy()
        rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
        if a.max_windows:
            rows = rows[:a.max_windows]
        xs = np.asarray(X[rows], np.float32)
        kS = FC.stream_route_every(projS(Fp[rows]), muS, 1)
        kP = FC.stream_route_every(projP(Fp[rows]), muP, 1)
        REPLAY["on"] = False
        po, ko = run(old, xs, dev, "old")
        pn, kn = run(new, xs, dev, "new")
        pn2, kn2 = run(new, xs, dev, "new")
        REPLAY["on"], REPLAY["q"] = True, [None if k < 0 else int(k) for k in kn]
        pg, _ = run(old, xs, dev, "old")
        assert not REPLAY["q"], (st, len(REPLAY["q"]))      # every recorded route consumed, one per window
        REPLAY["on"] = False
        finite &= bool(np.isfinite(pn).all())
        pred_new[rows] = pn.argmax(1)
        n_win += len(xs)
        bad["old_route"] += int((ko != kS).sum())
        bad["new_route"] += int((kn != kP).sum()) + int((kn2 != kn).sum())
        worst["rerun"] = max(worst["rerun"], float(np.abs(pn2 - pn).max()))
        worst["golden"] = max(worst["golden"], float(np.abs(pg - pn).max()))
        mv = int((kn != ko).sum())
        if mv == 0:
            worst["same_route_new_vs_old"] = max(worst["same_route_new_vs_old"], float(np.abs(pn - po).max()))
        per[st] = dict(windows=len(xs), route_moved=mv, argmax_moved=int((pn.argmax(1) != po.argmax(1)).sum()),
                       new_routes={cfo["user_blend"]["users"][k] if k >= 0 else "none": int(c) for k, c in zip(*np.unique(kn, return_counts=True))})
        print(f"[w23 check] {st}: {per[st]}; worst {worst}; bad {bad}", flush=True)
    res = dict(config=cfg0, streams=len(streams), windows=n_win, finite=finite, worst=worst, bad_routes=bad, per_stream=per,
               gates=dict(a=finite, c=worst["rerun"] <= 1e-6, K_old=bad["old_route"] == 0, K_new=bad["new_route"] == 0, G=worst["golden"] <= 1e-6,
                          same_route=worst["same_route_new_vs_old"] <= 1e-6))
    print(f"[w23 check] {json.dumps(res['gates'])}", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
        if not smoke:
            np.savez(a.out.replace(".json", "") + ".pred.npz", i=np.arange(len(idx)), pred=pred_new)   # read by t2p_w19_kitfull.py (K2)
    assert all(res["gates"].values()), res["gates"]
    print("[w23 check] PASS: config / files == W21 except the four route-space keys (== the committed JSON); finite; reproducible; OLD routes "
          "as the lab's W21 space, NEW as the lab's session-1 space, on every window; OLD replaying NEW's routes == NEW on every window", flush=True)


if __name__ == "__main__":
    assert os.environ.get("NI26_PARTS") != "1", "run on the worker's path (NI26_PARTS unset)"
    main()

#!/usr/bin/env python
"""T2-D634 (W23 = W21 + D632 P1, config-only): the golden test of the route-space change, CPU, on the real export and W21's real config.
  1. the inert check: t2p_final.py's route-space code on ALL rows reproduces W21's config user_blend mean / scale / W / mu (<= 1e-6);
  2. the committed tracks/t2_bci/sofia/w23_route_s1.json == the same code on every user's session-1 rows (<= 1e-12), same users / shapes;
  3. the launcher's four --set edits (t2p_repack.set_cfg, @file#key), applied to W21's config, change ONLY user_blend.{mean, scale, W, mu}
     (every other key of the whole config identical) and give exactly the JSON's values;
  4. the TEMPLATE's own router (t2p_submission_w21lp.StreamModel._user, fed window by window from a fresh state per stream, as predict
     does) on all 14 public streams: with W21's values == the lab route (t2p_fpcalib fit_space on all rows + stream_route_every) on every
     window; with the W23 values == the lab session-1 route (fit_space on session-1 rows) on every window. The decode downstream of the
     route index k is W21's code (template byte-identical; the post-build check proves OLD-with-NEW's-routes == NEW on sofia).
  python tests/test_t2_w23_route_s1.py  ->  "OK"
"""
import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "scripts/local"))
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402
import t2p_repack as RP  # noqa: E402
import t2_w23_route_s1 as W23  # noqa: E402

TEMPLATE = ROOT / "tracks/t2_bci/submission/t2p_submission_w21lp.py"
JSON_REL = "tracks/t2_bci/sofia/w23_route_s1.json"
SETS = [f"user_blend.{k}=@{JSON_REL}#{k}" for k in W23.KEYS]      # the launcher's edits, verbatim


class _R:
    """a bare object carrying exactly the attributes StreamModel._user reads."""


def template_routes(mod, ub, cfg, X, streams):
    out = {}
    for st, rows in streams.items():
        r = _R()
        r.cfg = cfg
        r.ub_mean, r.ub_scale = np.asarray(ub["mean"], np.float64), np.asarray(ub["scale"], np.float64)
        r.ub_W, r.ub_mu = np.asarray(ub["W"], np.float64), np.asarray(ub["mu"], np.float64)
        r.ub_sum, r.ub_n, r.ub_every, r.ub_t = None, 0, int(ub.get("every", 1)), 0       # = reset_state's
        ks = [mod.StreamModel._user(r, np.asarray(X[i], np.float32)) for i in rows]
        out[st] = np.array([-1 if k is None else k for k in ks])
    return out


def lab_routes(Fp, subj, fit_rows, streams):
    proj, mu, users = FC.fit_space(Fp[fit_rows], subj[fit_rows])
    return {st: FC.stream_route_every(proj(Fp[rows]), mu, 1) for st, rows in streams.items()}, users


def main():
    Fp, subj, ses = W23.fingerprints()
    X, idx, meta = B.load_export(str(W23.EXP))
    with zipfile.ZipFile(W23.W21_ZIP) as z:
        cfg21 = json.loads(z.read("config.json"))
    ub21 = cfg21["user_blend"]
    # 1. inert
    d_all = W23.maxd(W23.fit(Fp, subj), W23.w21_route_space())
    assert max(d_all.values()) <= 1e-6, d_all
    print(f"1. inert: the route-space code on ALL rows reproduces W21's config (max |d| {max(d_all.values()):.1e})")
    # 2. the JSON
    doc = json.loads((ROOT / JSON_REL).read_text())
    s1 = ses == "1"
    new = W23.fit(Fp[s1], subj[s1])
    d_js = W23.maxd(new, doc)
    assert max(d_js.values()) <= 1e-12 and doc["users"] == new["users"] == ub21["users"], d_js
    assert all(np.asarray(doc[k]).shape == np.asarray(ub21[k]).shape for k in W23.KEYS)
    print(f"2. {JSON_REL} == the session-1 refit (max |d| {max(d_js.values()):.1e}); users and shapes == W21's")
    # 3. the --set edits
    cfg23 = copy.deepcopy(cfg21)
    for kv in SETS:
        RP.set_cfg(cfg23, kv)
    rest21 = {k: v for k, v in cfg21.items() if k != "user_blend"}
    rest23 = {k: v for k, v in cfg23.items() if k != "user_blend"}
    ub_rest21 = {k: v for k, v in ub21.items() if k not in W23.KEYS}
    ub_rest23 = {k: v for k, v in cfg23["user_blend"].items() if k not in W23.KEYS}
    assert rest21 == rest23 and ub_rest21 == ub_rest23 and set(cfg23["user_blend"]) == set(ub21)
    assert all(cfg23["user_blend"][k] == doc[k] for k in W23.KEYS) and all(cfg23["user_blend"][k] != ub21[k] for k in W23.KEYS)
    print("3. the four --set edits change only user_blend.{mean, scale, W, mu} and carry the JSON's values exactly")
    # 4. the template's own router vs the lab routes, all 14 public streams
    RP.kit.use_track("bci_decoding")
    spec = importlib.util.spec_from_file_location("ni26_t2p_w21lp_golden", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert list(cfg21["ch_names"]) == list(meta["ch_names"]), "the export's channel order != the package's (predict's idx would not be identity)"
    tcfg = dict(sfreq=cfg21["sfreq"], ch_names=cfg21["ch_names"])
    streams = {}
    for st in dict.fromkeys(idx.sort_values(["stream", "order"])["stream"].tolist()):
        rows = idx.index[idx["stream"] == st].to_numpy()
        streams[st] = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
    assert len(streams) == 14
    n = sum(len(r) for r in streams.values())
    for name, ub, fit_rows in (("W21", ub21, np.ones(len(idx), bool)), ("W23", cfg23["user_blend"], s1)):
        tk = template_routes(mod, ub, tcfg, X, streams)
        lk, users = lab_routes(Fp, subj, fit_rows, streams)
        assert list(users) == list(ub["users"])
        bad = sum(int((tk[st] != lk[st]).sum()) for st in streams)
        shares = {}
        for st in streams:
            u0 = str(subj[streams[st][0]]).split("/")[-1]
            c = dict(zip(*np.unique(tk[st], return_counts=True)))
            shares[st] = f"{u0}->" + ",".join(f"{ub['users'][k]}:{v}" if k >= 0 else f"none:{v}" for k, v in c.items())
        print(f"4. {name}: the template's router == the lab route on {n - bad} / {n} windows; routes: " + "; ".join(shares.values()))
        assert bad == 0, (name, bad)
    print("OK")


if __name__ == "__main__":
    main()

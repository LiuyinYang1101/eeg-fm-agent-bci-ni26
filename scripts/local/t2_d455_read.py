#!/usr/bin/env python
"""T2-D455 read (written before any number). M7's shape on kit + seen1 with three of its five ST-L members (e100f0-2) replaced by fresh nets
(seeds 472-485; draw i uses seeds 472+i, 473+i, 474+i): C3 = three control restarts, MI3 = three MI-expert nets, MI2 = one control + two MI.
Typical = every user except P04; paired per draw with C3.  BAR: MI3 or MI2 vs C3 >= +0.003 with >= 9 / 12 and no non-P04 user < -0.005.
Also printed: the three-net votes alone.
    mem-guard python scripts/local/t2_d455_read.py
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
w = [M7[k] for k in ("e100f0", "e100f1", "e100f2")]
for e in ("ctl", "mi"):
    path = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{O}/sofia_s1/ni26-t2p-d455-{e}-{pr}-[0-9]*")
        assert len(ds) == 1, (e, pr, ds)
        fs = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472.npz")
        assert len(fs) == 1, fs
        path[pr] = fs[0].replace("__s472.npz", "__s{s}.npz")
    for k in range(3):
        M[f"{e}{k}"] = {"offset": 472 + k, **path}
base = {k: v for k, v in M7.items() if k not in ("e100f0", "e100f1", "e100f2")}
sets = {"C3": ("ctl0", "ctl1", "ctl2"), "MI3": ("mi0", "mi1", "mi2"), "MI2": ("ctl0", "mi1", "mi2")}
arms = {}
for n, ms in sets.items():
    arms[n] = {m: 1 / 3 for m in ms}
    arms["M7_" + n] = {**base, **dict(zip(ms, w))}
spec = {"members": M, "arms": arms}
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
R = {a: cell_rows(spec, a, 12, idx) for a in arms}
for pre in ("", "M7_"):
    for a in ("MI3", "MI2"):
        m = R[pre + a].merge(R[pre + "C3"][key + ["ba"]], on=key, suffixes=("", "_b"))
        t = m[m.subj != "P04"]
        d = (t.ba - t.ba_b).groupby(t.i).mean()
        u = (m.ba - m.ba_b).groupby(m.subj).mean()
        pp = (t.ba - t.ba_b).groupby(t.proto).mean()
        print(f"{pre or 'alone ':6s}{a} vs C3: typical {d.mean():+.4f} ({int((d > 0).sum()):2d}/12) level {t.ba.mean():.3f} vs {t.ba_b.mean():.3f}  "
              + " ".join(f"{s} {v:+.4f}" for s, v in u.items()) + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in pp.items()), flush=True)

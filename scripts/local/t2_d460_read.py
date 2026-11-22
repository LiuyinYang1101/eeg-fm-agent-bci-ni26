#!/usr/bin/env python
"""T2-D460 read (written before any number). Pretrained ST-EEGFormer base (b) / small (s), with (c) or without (n) CPT, seeds 486-499, as FM
members next to M7's stored members (outputs/t2_d408/spec_m7.json); the equal-count comparator nets are T2-D455's fresh ST-L controls (ctl0-2).
 (a) alone: three-net votes X3 vs L3 (three ST-L), typical level and paired difference; STOP rule: X3 more than 0.03 below L3 -> (b), (c) not read as candidates.
 (b) added: M7's five ST-L + three X nets vs + three ST-L, eight FM nets at equal weight, FM share unchanged.
 (c) replacing: two of M7's five ST-L (e100f0, e100f1) -> two X nets vs -> two fresh ST-L.
BAR for (b) and (c), each size, CPT form: typical >= +0.003 with >= 9 / 12 and no typical user < -0.005; P09 printed apart.
    mem-guard python scripts/local/t2_d460_read.py
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


def paths(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{O}/sofia_s1/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        fs = sorted(glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s*.npz"))
        s0 = min(int(f.rsplit("__s", 1)[1][:-4]) for f in fs)
        out[pr] = [f for f in fs if f.endswith(f"__s{s0}.npz")][0].replace(f"__s{s0}.npz", "__s{s}.npz")
    return out, s0


p, s0 = paths("ni26-t2p-d455-ctl-{pr}")
for k in range(3):
    M[f"L{k}"] = {"offset": s0 + k, **p}
TAG = "d463" if "--d463" in sys.argv else "d464" if "--d464" in sys.argv else "d460"                  # T2-D463: the same three shapes for the C2L300 kinds raw / ann
KINDS = [k for k in ({"d463": ("raw", "ann"), "d464": ("ar",)}.get(TAG, ("bc", "bn", "sc", "sn"))) if glob.glob(f"{O}/sofia_s1/ni26-t2p-{TAG}-{k}-kit-[0-9]*")]
# T2-D463 full-FT re-read (2026-11-13): --kinds raw,ft picks the kinds (ft = the local 5090 full fine-tune, seeds 500-504); --n N = draws
# (five seeds -> three-net votes over 3 draws); the comparator ST-L nets (D455 ctl) and every kind use the same N.
for a in sys.argv:
    if a.startswith("--kinds="):
        KINDS = [k for k in a.split("=", 1)[1].split(",") if glob.glob(f"{O}/sofia_s1/ni26-t2p-{TAG}-{k}-kit-[0-9]*")]
N = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--n=")), 12)
for kd in KINDS:
    p, s0 = paths(f"ni26-t2p-{TAG}-{kd}-{{pr}}")
    for k in range(3):
        M[f"{kd}{k}"] = {"offset": s0 + k, **p}
FM5 = [k for k in M7 if k.startswith("e100f") or k == "lp10"]
share = sum(M7[k] for k in FM5)
cnn = {k: v for k, v in M7.items() if k not in FM5}
arms = {}
for kd in ["L"] + KINDS:
    x = [f"{kd}{k}" for k in range(3)]
    arms[f"{kd}_alone3"] = {m: 1 / 3 for m in x}
    arms[f"{kd}_added"] = {**cnn, **{m: share / 8 for m in FM5 + x}}
    arms[f"{kd}_repl"] = {**{k: v for k, v in M7.items() if k not in ("e100f0", "e100f1")}, x[0]: M7["e100f0"], x[1]: M7["e100f1"]}
spec = {"members": M, "arms": arms}
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
R = {a: cell_rows(spec, a, N, idx) for a in arms}
for kd in KINDS:
    for shape in ("alone3", "added", "repl"):
        m = R[f"{kd}_{shape}"].merge(R[f"L_{shape}"][key + ["ba"]], on=key, suffixes=("", "_b"))
        t = m[m.subj != "P04"]
        d = (t.ba - t.ba_b).groupby(t.i).mean()
        u = (m.ba - m.ba_b).groupby(m.subj).mean()
        pp = (t.ba - t.ba_b).groupby(t.proto).mean()
        print(f"{kd} {shape:6s} vs ST-L: typical {d.mean():+.4f} ({int((d > 0).sum()):2d}/{N}) level {t.ba.mean():.3f} vs {t.ba_b.mean():.3f}  "
              + " ".join(f"{s} {v:+.4f}" for s, v in u.items()) + "  | " + " ".join(f"{s} {v:+.4f}" for s, v in pp.items()), flush=True)

#!/usr/bin/env python
"""T2-D470 read (written before any number): causal per-stream BatchNorm-statistics tracking, member-level kill test on today's recipe.
Per family (FBCNet, ShallowNet-EA, IFNet-EA; pooled nets, no copies / views), fresh seeds 540-551, kit + seen1: the plain net vs the SAME
net with BN tracking at momentum 0.01 / 0.05, each through the causal decoder; paired cell differences, typical users (non-P04).
The momentum is picked per family on the VAL cells, judged on the TEST cells. KILL BAR: a family with test typical >= +0.005, >= 9 / 12.
    mem-guard python scripts/local/t2_d470_read.py
"""
import glob
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

S = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1"
FAM = {"fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75", "sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr", "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vr"}
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
for f, stem in FAM.items():
    d = S / f"ni26-t2p-d470-{f}-5090/runs"
    M = {}
    for tag in ("", "bn0.01", "bn0.05"):
        st = stem + tag                                      # the bench writes <recipe>bn<m> twins
        M[tag or "plain"] = {"offset": 540, **{p: str(d / f"{p}__{st}__s{{s}}.npz") for p in ("kit", "seen1")}}
    assert all(glob.glob(m["kit"].replace("{s}", "540")) for m in M.values()), (f, {k: m["kit"] for k, m in M.items()})
    spec = {"members": M, "arms": {k: {k: 1.0} for k in M}}
    R = {a: cell_rows(spec, a, 12, idx) for a in M}
    res = {}
    for a in ("bn0.01", "bn0.05"):
        m = R[a].merge(R["plain"][key + ["ba"]], on=key, suffixes=("", "_b"))
        t = m[m.subj != "P04"]
        res[a] = {sp: (t[t.split == sp].ba - t[t.split == sp].ba_b).groupby(t[t.split == sp].i).mean() for sp in ("val", "test")}
        u = (m[m.split == "test"].ba - m[m.split == "test"].ba_b).groupby(m[m.split == "test"].subj).mean()
        print(f"{f:3s} {a}: val {res[a]['val'].mean():+.4f} ({int((res[a]['val'] > 0).sum())}/12)  test {res[a]['test'].mean():+.4f} ({int((res[a]['test'] > 0).sum())}/12)  "
              + " ".join(f"{s} {v:+.4f}" for s, v in u.items()), flush=True)
    pick = max(res, key=lambda a: res[a]["val"].mean())
    print(f"{f:3s} picked on val: {pick} -> TEST typical {res[pick]['test'].mean():+.4f} ({int((res[pick]['test'] > 0).sum())}/12); plain level {R['plain'][R['plain'].subj != 'P04'].ba.mean():.4f}", flush=True)

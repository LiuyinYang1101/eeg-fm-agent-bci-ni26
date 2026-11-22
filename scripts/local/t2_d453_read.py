#!/usr/bin/env python
"""T2-D453 read (written before any number). Stage 1: which Scherer2015 users help a PROTEUS ShallowNet when added to its training set?
Per arm (control, all nine, one user at a time; seeds 436-447): the member alone under causal SK, typical (non-P04) and per user, paired
with the control. Prints the ranking and the stage-2 sets: PLUS = every user with delta > 0, TOP3 = the three best.
Stage 2 (--stage 2, seeds 448-459): PLUS and TOP3 vs a fresh control, alone and in M7's shape (ShallowNet slot = the arm, both pooled-only).
    mem-guard python scripts/local/t2_d453_read.py [--stage 2]
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
stage = 2 if "--stage" in sys.argv else 1
off = 436 if stage == 1 else 448
ds = glob.glob(f"{O}/sofia_s1/ni26-t2p-d453-s{stage}-[0-9]*")
assert len(ds) == 1, ds
stems = sorted({Path(f).name.rsplit("__s", 1)[0].split("__", 1)[1] for f in glob.glob(f"{ds[0]}/runs/kit__shallow__*__s*.npz")})
J = json.load(open(O / "t2_d408/spec_m7.json"))
M, M7 = dict(J["members"]), J["arms"]["M7"]
arms = {}
for st in stems:
    tag = st.split("__x", 1)[1].replace("scherer2015_v1", "") if "__x" in st else "ctl"
    name = {"": "all9"}.get(tag, tag)
    M[name] = {"offset": off, **{p: f"{ds[0]}/runs/{p}__{st}__s{{s}}.npz" for p in ("kit", "seen1")}}
    arms[name] = {name: 1.0}
    arms["M7_" + name] = {(name if k == "sh_tv8c" else k): v for k, v in M7.items()}
spec = {"members": M, "arms": arms}
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]


def read(a, b):
    A, B = cell_rows(spec, a, 12, idx), cell_rows(spec, b, 12, idx)
    m = A.merge(B[key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    return d.mean(), int((d > 0).sum()), u, t.ba.mean()


rank = []
for a in [x for x in arms if not x.startswith("M7_") and x != "ctl"]:
    d, up, u, lvl = read(a, "ctl")
    rank.append((d, a))
    print(f"alone {a:10s} vs ctl: typical {d:+.4f} ({up:2d}/12) level {lvl:.3f}  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items()), flush=True)
    if stage == 2:
        d, up, u, _ = read("M7_" + a, "M7_ctl")
        print(f"   M7 shape          : typical {d:+.4f} ({up:2d}/12)              " + " ".join(f"{s} {v:+.4f}" for s, v in u.items()), flush=True)
if stage == 1:
    single = sorted([(d, a) for d, a in rank if a.startswith("u")], reverse=True)
    plus = [a[1:] for d, a in single if d > 0]
    print("ranking:", " ".join(f"{a}{d:+.4f}" for d, a in single))
    print("STAGE-2 SETS  PLUS:", "+".join(sorted(plus)) or "(none)", " TOP3:", "+".join(sorted(a[1:] for _, a in single[:3])))

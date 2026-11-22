#!/usr/bin/env python
"""T2 new-user swing reads (zero GPU), as pre-registered: T2-D323 (ST-L + causal EA), T2-D325 (sw2 in the tv8c route), T2-D326 (two
ST-L seeds, fresh-seed replicate). Each on the X4 package shape (fbc_tv8c 0.4, ShallowNet 0.3, ST-L 0.3) under SK10 lambda 0.95,
via t2_d320_rescreen.py; groups typical (P05 P08 P09 P10) / newT / P10 / P04. Bar (all three): typical >= +0.003 with >= 9 / 12 up
AND P04 >= -0.010.
  python t2_swings_read.py d323|d325|d326 [--runs DIR ...]   (DIRs = the fetched sofia job dirs; d325 also needs the blend output)
"""
import argparse
import glob
import json
import os
import subprocess
import sys

O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
EXP = os.path.expanduser("~/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1")
HERE = os.path.dirname(os.path.abspath(__file__))
RESCREEN = os.path.join(HERE, "..", "..", "tracks", "t2_bci", "scripts", "t2_d320_rescreen.py")
PY = sys.executable
D310 = json.load(open(f"{O}/t2_d310/spec.json"))["members"]
D274 = json.load(open(f"{O}/t2_d274/spec.json"))["members"]
REC = {"kit": "fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cptstlarge-d89b",
       "seen1": "fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75cptpt-d111-with"}


def find(dirs, proto, stem):
    """the {s}-pattern of a member's result files in the fetched job dirs (exactly one match per protocol)."""
    hits = sorted({f.rsplit("__s", 1)[0] for d in dirs for f in glob.glob(f"{d}/runs/{proto}__{stem}__s*.npz")})
    assert len(hits) == 1, (proto, stem, hits)
    return f"{hits[0]}__s{{s}}.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("row", choices=("d323", "d325", "d326", "d329", "d330"))
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--route", default="", help="d325: the t2p_fpcalib blend output dir of the sw2 tv8c member")
    a = ap.parse_args()
    C = {"fbc_tv8c": D310["fbc_tv8c"], "sh_tv8c": D310["sh_tv8c"]}
    if a.row in ("d323", "d329", "d330"):
        stem = (lambda p: f"fm-st_large-mean__ea__n00__{REC[p]}") if a.row == "d323" else (lambda p: "fm-st_large-mean__sz__n00__*")
        M = dict(C, stl_rcll=D274["stl_rcll"], stl_ea={"offset": 0, **{p: find(a.runs, p, stem(p)) for p in ("kit", "seen1")}})
        arms = {"X4": {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, "stl_rcll": 0.3}, "ea": {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, "stl_ea": 0.3},
                "ea_alone": {"stl_ea": 1.0}, "rcll_alone": {"stl_rcll": 1.0}}
        arm = "ea"
    elif a.row == "d326":
        M = dict(C, stl_rcll12=D274["stl_rcll12"],
                 stl_rcll24={"offset": 24, **{p: find(a.runs, p, f"fm-st_large-mean__sz__n00__{REC[p]}") for p in ("kit", "seen1")}})
        arms = {"X4r": {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, "stl_rcll12": 0.3},
                "k2r": {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, "stl_rcll12": 0.15, "stl_rcll24": 0.15}}
        arm = "k2r"
    else:
        r = "r1_pulx_vp2_e180_vrtv8c"
        M = dict(C, stl_rcll=D274["stl_rcll"],
                 sw2_tv8c={"offset": 156, **{p: f"{a.route}/{p}__shallow_w2__ea__n00__{r}fb2__s{{s}}.npz" for p in ("kit", "seen1")}})
        arms = {"X4": {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, "stl_rcll": 0.3}, "sw2": {"fbc_tv8c": 0.4, "sw2_tv8c": 0.3, "stl_rcll": 0.3}}
        arm = "sw2"
    out = f"{O}/t2_{a.row}"
    os.makedirs(out, exist_ok=True)
    spec = {"n": 12, "anchor": list(arms)[0], "members": M, "arms": arms}
    json.dump(spec, open(f"{out}/spec.json", "w"), indent=1)
    json.dump({f"{out}/spec.json": list(arms)}, open(f"{out}/sk.json", "w"))
    subprocess.run([PY, "-I", RESCREEN, "--export", EXP, "--out", out, "--jobs", "1", "--specs", f"{out}/spec.json",
                    "--sk-list", f"{out}/sk.json"], check=True, stdout=subprocess.DEVNULL)
    import pandas as pd
    d = pd.read_csv(f"{out}/rescreen_sk.csv")
    d = d[(d.status == "ok") & (d.dec == "sk")]
    print(d[["arm", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    r_ = d[d.arm == arm].iloc[0]
    ok = r_.typical >= 0.003 and r_.typical_up >= 9 and r_.P04 >= -0.010
    v = dict(row=a.row, arm=arm, typical=float(r_.typical), typical_up=int(r_.typical_up), P04=float(r_.P04), newT=float(r_.newT),
             P10=float(r_.P10), bar="PASS" if ok else "not met")
    json.dump(v, open(f"{out}/verdict.json", "w"), indent=1)
    print(json.dumps(v))


if __name__ == "__main__":
    main()

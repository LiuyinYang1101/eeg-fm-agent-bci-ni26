#!/usr/bin/env python
"""T2-FM-D7: THE BACKBONE-DIVERSITY ST-L SLOT (zero GPU; written and committed BEFORE any D6 number). M6's shape (T2-D378 S3f) with
its CNNs and their weights unchanged (sh_tv8c, ifn, fbc_tl15f) and its ST-L slot (0.35) rebuilt from plain-recipe members (the
stl_rcll line, seeds paired by draw index):
  anchor C3 = the slot from ONE backbone, three seed sets: CPT v1 r0 (seeds 0-11), r12 (12-23), r24 (24-35)   (seed diversity)
  arm    B3 = the slot from THREE backbones, one seed set each: CPT v1 r0, J-CPT A0 5k, ST-L + P1 15k (seeds 0-11) (backbone diversity)
  descriptive B4 = r0 + A0 5k + P1 5k + P1 15k at 0.35 / 4 (R1's full mix; member count differs from C3), and B3 / B4 vs S3f (M6 itself).
The mix is FIXED a priori (R1 idea 3: P1 15k is kept over 5k because the two P1 doses are the most alike pair); nothing is selected on
these seeds (the critic's caveat). Only three plain CPT v1 seed sets exist (r0 / r12 / r24), hence the 3-member primary.
BAR (B3 vs C3): typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0, the P08-only guard (LOPO -P08 >= +0.0015, P05 >= -0.010,
P09 >= -0.010); the new vs returning split and the sealed-like weighting (0.5 ret + 0.5 new) reported.
  mem-guard python scripts/local/t2fm_d7_slot_read.py
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2fm_p1_read as P  # noqa: E402  (members, guard, per-user / split parsers, the rescreen + CI paths)

O, S = P.O, P.S
OUT = f"{O}/t2fm_d7"


def jcpt_member(tag, step):
    out = {"offset": 0}
    for p in ("kit", "seen1"):
        dirs = glob.glob(f"{S}/ni26-t2p-jft-{tag}-s{step:05d}-{p}-*")
        hits = sorted({f.rsplit("__s", 1)[0] for d in dirs for f in glob.glob(f"{d}/runs/{p}__fm-st_large-mean__sz__n00__*__s*.npz")})
        assert len(hits) == 1 and len(glob.glob(f"{hits[0]}__s*.npz")) == 12, (tag, step, p, hits)
        out[p] = f"{hits[0]}__s{{s}}.npz"
    return out


def main():
    D378 = json.load(open(f"{O}/t2_d378/spec.json"))
    D379 = json.load(open(f"{O}/t2_d379/spec_a.json"))["members"]
    S3f = D378["arms"]["S3f"]
    cnn = {k: v for k, v in S3f.items() if k in ("sh_tv8c", "ifn", "fbc_tl15f")}
    assert abs(sum(cnn.values()) - 0.65) < 1e-9, cnn
    M = dict(D378["members"], r0=D379["r0"], r12=D379["r12"], r24=D379["r24"], a0_5k=jcpt_member("jA0", 5000))
    for s in (5000, 15000):
        m = P.member("plain", s, 0)
        assert m is not None, f"the ST-L + P1 plain {s} fine-tunes are not fetched yet"
        M[f"p1_{s // 1000}k"] = m
    for k in ("r0", "r12", "r24"):
        assert M[k]["offset"] == {"r0": 0, "r12": 12, "r24": 24}[k], (k, M[k]["offset"])
    slot = lambda names: dict(cnn, **{n: 0.35 / len(names) for n in names})  # noqa: E731
    arms = {"C3": slot(["r0", "r12", "r24"]), "B3": slot(["r0", "a0_5k", "p1_15k"]), "B4": slot(["r0", "a0_5k", "p1_5k", "p1_15k"])}
    specs = {"slot": ("C3", arms), "vsM6": ("S3f", {"S3f": dict(S3f), "B3": arms["B3"], "B4": arms["B4"], "C3": arms["C3"]})}
    os.makedirs(OUT, exist_ok=True)
    sk = {}
    for name, (anchor, a) in specs.items():
        for arm in a.values():
            assert abs(sum(arm.values()) - 1) < 1e-9, (name, arm)
        f = f"{OUT}/spec_{name}.json"
        json.dump({"n": 12, "anchor": anchor, "members": M, "arms": a}, open(f, "w"), indent=1)
        sk[f] = list(a)
    json.dump(sk, open(f"{OUT}/sk.json", "w"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([P.PY, "-I", P.RESCREEN, "--export", P.EXP, "--out", OUT, "--jobs", "2", "--specs", f"{OUT}/spec_*.json",
                    "--sk-list", f"{OUT}/sk.json"], check=True, stdout=subprocess.DEVNULL, env=env)
    import pandas as pd
    d = pd.read_csv(f"{OUT}/rescreen_sk.csv")
    d = d[(d.status == "ok") & (d.dec == "sk")].copy()
    d["spec"] = d["spec"].map(lambda p: Path(p).stem.replace("spec_", ""))
    print(d[["spec", "arm", "anchor", "typical", "typical_up", "P04", "newT", "P10", "typ_level"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    V = {}
    for spec, anchor, arm in (("slot", "C3", "B3"), ("slot", "C3", "B4"), ("vsM6", "S3f", "B3"), ("vsM6", "S3f", "B4")):
        r = d[(d.spec == spec) & (d.arm == arm)].iloc[0]
        lines = P.ci_full(f"{OUT}/spec_{spec}.json", anchor, arm, env)
        pu, lo = P.per_user(lines)
        g = P.guard(pu, lo)
        V[f"{arm}_vs_{anchor}"] = dict(typical=round(float(r.typical), 4), up=int(r.typical_up), P04=round(float(r.P04), 4),
                                       new=round(float(r.newT), 4), users=pu, lopo=lo, p08_guard="ok" if g else "FAILED",
                                       new_vs_returning=P.split(lines), ci=lines[-9:],
                                       bar=("PASS" if (P.pkg_bar(r) and g) else "not met") if (arm, anchor) == ("B3", "C3") else "descriptive")
    json.dump(V, open(f"{OUT}/verdict.json", "w"), indent=1)
    for k, v in V.items():
        print(f"{k}: typical {v['typical']:+.4f} ({v['up']} / 12), P04 {v['P04']:+.4f}, new {v['new']:+.4f}, P05 {v['users'].get('P05', float('nan')):+.4f}, "
              f"P09 {v['users'].get('P09', float('nan')):+.4f}, guard {v['p08_guard']}, {v['new_vs_returning']} -> {v['bar']}")


if __name__ == "__main__":
    main()

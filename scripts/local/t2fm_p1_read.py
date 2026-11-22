#!/usr/bin/env python
"""T2-FM-D6 read (zero GPU; written and committed BEFORE any ST-L + P1 number): the T2 fine-tunes from the ST-L + P1 checkpoints
(kit from t2fm-p1-kit, seen1 from t2fm-p1-seen1; fetched dirs ni26-t2p-p1ft-<recipe>-s<step>-<proto>-<job>), SK10 lambda 0.95 via
t2_d320_rescreen.py. Specs:
  x4     plain recipe (seeds 0-11): anchor X4 (fbc_tv8c 0.4 + sh_tv8c 0.3 + stl_rcll 0.3), arm X4_p1_<s>.  T2-D350 bar.
  alone  plain: the member alone vs stl_rcll alone (the dose curve).  TOP-2 bar (a): >= +0.05 typical.
  m6c    plain, today's stack with the matched control: anchor SW_rcll = M6 (T2-D378 S3f) with vp12 -> stl_rcll (same seeds 0-11),
         arm SW_p1_<s> = vp12 -> the P1 member.  Picks the step for the vp12 recipe: the best m6c typical (ties within 0.0005 -> shorter).
  m6     vp12 recipe at that step (seeds 12-23, paired with M6's vp12): anchor S3f = M6; SWv = vp12 -> the P1-vp12 member (runtime-
         neutral swap); ADD = M6 + the P1-vp12 member at 0.05 (T2-D377's add_frac).  PACKAGE BAR (T2-D386's): typical >= +0.003 with
         >= 9 / 12, P04 >= -0.010, new >= 0.  TOP-2 bar (b): SWv or ADD >= +0.012 typical.
+ the user-cluster CI (t2_cluster_ci.py) of the best x4 / m6c arm and of SWv / ADD, and the PER-USER read (T2's cross-track note,
  T2-D103 / D105: supervised pre-FT gained on P08 only, P05 / P09 fell): P08-ONLY GUARD = the typical without P08 (LOPO) >= +0.0015
  AND P05 >= -0.010 AND P09 >= -0.010; a package / top-2 pass needs the guard too (applied to SWv / ADD and to the best alone step).
+ DESCRIPTIVE (the manager, 2026-11-06 ≈ 23:0x; no bar changes): the NEW vs RETURNING split (t2_cluster_ci.py's 'returning (P10 + seen1
  P04)' / 'new (P05 P08 P09 + kit P04)') beside the per-user table, and a sealed-like weighting 0.5 x returning + 0.5 x new (sealed is
  ≈ 50 % returning vs 25 % in 'typical'): every external supervised / longer FT on T2 so far moved gain to returning / hard users.
  mem-guard python scripts/local/t2fm_p1_read.py [--steps 5000,10000,15000,30000] [--out DIR]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
S = f"{O}/sofia_s1"
EXP = os.path.expanduser("~/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1")
HERE = os.path.dirname(os.path.abspath(__file__))
RESCREEN = os.path.join(HERE, "..", "..", "tracks", "t2_bci", "scripts", "t2_d320_rescreen.py")
CI = os.path.join(HERE, "t2_cluster_ci.py")
PY = sys.executable
TOP2_ALONE, TOP2_PKG = 0.05, 0.012


def member(rec, step, offset):
    out = {"offset": offset}
    for p in ("kit", "seen1"):
        dirs = glob.glob(f"{S}/ni26-t2p-p1ft-{rec}-s{step:05d}-{p}-*")
        hits = sorted({f.rsplit("__s", 1)[0] for d in dirs for f in glob.glob(f"{d}/runs/{p}__fm-st_large-mean__sz__n00__*__s*.npz")})
        if len(hits) != 1 or len(glob.glob(f"{hits[0]}__s*.npz")) != 12:
            return None
        out[p] = f"{hits[0]}__s{{s}}.npz"
    return out


def per_user(lines):
    """{user: delta} and {left-out user: typical without it} from t2_cluster_ci.py's text."""
    import re
    pu, lo = {}, {}
    for ln in lines:
        m = re.match(r"\s*(P\d\d): ([+-]\d+\.\d+) \(", ln)
        if m:
            pu[m.group(1)] = float(m.group(2))
        if "LOPO" in ln:
            for u, v in re.findall(r"-(P\d\d) ([+-]\d+\.\d+)", ln):
                lo[u] = float(v)
    return pu, lo


def split(lines):
    """{'returning', 'new', 'P04', 'sealed_like'} from t2_cluster_ci.py's group line (descriptive)."""
    import re
    for ln in lines:
        m = re.search(r"returning \(.*?\): ([+-]\d+\.\d+); new \(.*?\): ([+-]\d+\.\d+); P04 alone: ([+-]\d+\.\d+)", ln)
        if m:
            r, n, p4 = (float(v) for v in m.groups())
            return dict(returning=r, new=n, P04=p4, sealed_like=round(0.5 * r + 0.5 * n, 4))
    return {}


def guard(pu, lo):
    return bool(lo.get("P08", -1) >= 0.0015 and pu.get("P05", -1) >= -0.010 and pu.get("P09", -1) >= -0.010)


def ci_full(spec_file, anchor, arm, env):
    c = subprocess.run([PY, CI, spec_file, anchor, arm], capture_output=True, text=True, env=env)
    return c.stdout.strip().splitlines() if c.returncode == 0 else [f"failed: {c.stderr[-300:]}"]


def pkg_bar(r):
    return bool(r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010 and r.newT >= 0)


def x4_bar(r):
    return bool(r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="5000,10000,15000,30000")
    ap.add_argument("--out", default=f"{O}/t2fm_p1")
    a = ap.parse_args()
    steps = [int(v) for v in a.steps.split(",") if v]
    D310 = json.load(open(f"{O}/t2_d310/spec.json"))["members"]
    D274 = json.load(open(f"{O}/t2_d274/spec.json"))["members"]
    D378 = json.load(open(f"{O}/t2_d378/spec.json"))
    S3f = D378["arms"]["S3f"]
    assert abs(S3f["vp12"] - 0.07) < 1e-9
    M = dict(D378["members"], fbc_tv8c=D310["fbc_tv8c"], sh_tv8c=D310["sh_tv8c"], stl_rcll=D274["stl_rcll"])
    have = [s for s in steps if member("plain", s, 0)]
    for s in have:
        M[f"p1_s{s}"] = member("plain", s, 0)
    print("plain members:", have)
    x4 = lambda m: {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, m: 0.3}  # noqa: E731

    def sw(m, o="vp12"):
        b = dict(S3f)
        b[m] = b.pop(o)
        return b

    def add(m, frac=0.05):
        tot = sum(S3f.values())
        b = {k: v * (tot - frac) / tot for k, v in S3f.items()}
        b[m] = frac
        return b
    specs = {"x4": ("X4", {"X4": x4("stl_rcll"), **{f"X4_p1_s{s}": x4(f"p1_s{s}") for s in have}}),
             "alone": ("alone_rcll", {"alone_rcll": {"stl_rcll": 1.0}, **{f"alone_p1_s{s}": {f"p1_s{s}": 1.0} for s in have}}),
             "m6c": ("SW_rcll", {"SW_rcll": sw("stl_rcll"), **{f"SW_p1_s{s}": sw(f"p1_s{s}") for s in have}})}
    vp = {s: member("vp12", s, 12) for s in steps}
    vp = {s: m for s, m in vp.items() if m}
    for s, m in vp.items():
        M[f"p1vp_s{s}"] = m
        specs[f"m6_s{s}"] = ("S3f", {"S3f": dict(S3f), f"SWv_s{s}": sw(f"p1vp_s{s}"), f"ADD_s{s}": add(f"p1vp_s{s}")})
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(f"{a.out}/spec_*.json"):
        os.remove(f)
    sk = {}
    for name, (anchor, arms) in specs.items():
        for arm in arms.values():
            assert abs(sum(arm.values()) - 1) < 1e-9, (name, arm)
        f = f"{a.out}/spec_{name}.json"
        json.dump({"n": 12, "anchor": anchor, "members": M, "arms": arms}, open(f, "w"), indent=1)
        sk[f] = list(arms)
    json.dump(sk, open(f"{a.out}/sk.json", "w"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    subprocess.run([PY, "-I", RESCREEN, "--export", EXP, "--out", a.out, "--jobs", "4", "--specs", f"{a.out}/spec_*.json",
                    "--sk-list", f"{a.out}/sk.json"], check=True, stdout=subprocess.DEVNULL, env=env)
    import pandas as pd
    d = pd.read_csv(f"{a.out}/rescreen_sk.csv")
    d = d[(d.status == "ok") & (d.dec == "sk")].copy()
    d["spec"] = d["spec"].map(lambda p: Path(p).stem.replace("spec_", ""))
    print(d[["spec", "arm", "anchor", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    row = lambda spec, arm: d[(d.spec == spec) & (d.arm == arm)].iloc[0]  # noqa: E731
    pick = lambda r: dict(typical=round(float(r.typical), 4), up=int(r.typical_up), P04=round(float(r.P04), 4),  # noqa: E731
                          new=round(float(r.newT), 4), P10=round(float(r.P10), 4), level=round(float(r.typ_level), 4))
    V = dict(plain_steps=have, vp12_steps=sorted(vp), x4={}, alone={}, m6c={}, m6={}, ci={})
    for s in have:
        V["x4"][s] = dict(pick(row("x4", f"X4_p1_s{s}")), bar="PASS" if x4_bar(row("x4", f"X4_p1_s{s}")) else "not met")
        V["alone"][s] = dict(pick(row("alone", f"alone_p1_s{s}")), top2_a="MET" if float(row("alone", f"alone_p1_s{s}").typical) >= TOP2_ALONE else "not met")
        V["m6c"][s] = pick(row("m6c", f"SW_p1_s{s}"))
    if have:
        best = max(have, key=lambda s: (round(V["m6c"][s]["typical"] / 0.0005), -s))
        V["vp12_step_rule"] = best
        print(f"the vp12 recipe goes to step {best} (best m6c typical; ties within 0.0005 -> shorter)")
        ba = max(have, key=lambda s: V["alone"][s]["typical"])
        for spec, anchor, arm in (("x4", "X4", f"X4_p1_s{max(have, key=lambda s: V['x4'][s]['typical'])}"),
                                  ("m6c", "SW_rcll", f"SW_p1_s{best}"), ("alone", "alone_rcll", f"alone_p1_s{ba}")):
            lines = ci_full(f"{a.out}/spec_{spec}.json", anchor, arm, env)
            V["ci"][f"{spec}:{arm}"] = lines[-9:]
            pu, lo = per_user(lines)
            V.setdefault("per_user", {})[f"{spec}:{arm}"] = dict(users=pu, lopo=lo, p08_guard="ok" if guard(pu, lo) else "FAILED",
                                                                  new_vs_returning=split(lines))
        V["alone"][ba]["top2_a_with_guard"] = "MET" if (V["alone"][ba]["top2_a"] == "MET"
                                                        and V["per_user"][f"alone:alone_p1_s{ba}"]["p08_guard"] == "ok") else "not met"
    for s in vp:
        for arm in (f"SWv_s{s}", f"ADD_s{s}"):
            r = row(f"m6_s{s}", arm)
            lines = ci_full(f"{a.out}/spec_m6_s{s}.json", "S3f", arm, env)
            V["ci"][f"m6:{arm}"] = lines[-9:]
            pu, lo = per_user(lines)
            g = guard(pu, lo)
            V["m6"][arm] = dict(pick(r), users=pu, lopo=lo, p08_guard="ok" if g else "FAILED", new_vs_returning=split(lines),
                                package_bar="PASS" if (pkg_bar(r) and g) else "not met",
                                top2_b="MET" if (float(r.typical) >= TOP2_PKG and g) else "not met")
    # THE SUMMARY (the manager, 2026-11-07: P05 / P09 prominent beside new / returning; every supervised external pretraining on T2 so far
    # cost NEW users, P05 most: D105, D396, D398 P05 -0.0306). Descriptive layout only; the bars are the pre-registered ones.
    rows = [(k, v) for k, v in V.get("per_user", {}).items()] + [(f"m6:{k}", v) for k, v in V["m6"].items()]
    summ = []
    for k, v in rows:
        pu, nr = v.get("users", {}), v.get("new_vs_returning", {})
        summ.append(f"{k:28s} P05 {pu.get('P05', float('nan')):+.4f}  P09 {pu.get('P09', float('nan')):+.4f}  (guard P05 / P09 >= -0.010; "
                    f"LOPO -P08 {v.get('lopo', {}).get('P08', float('nan')):+.4f} >= +0.0015 -> {v.get('p08_guard')})  new {nr.get('new', float('nan')):+.4f}  "
                    f"returning {nr.get('returning', float('nan')):+.4f}  sealed-like {nr.get('sealed_like', float('nan')):+.4f}")
    V["summary"] = summ
    json.dump(V, open(f"{a.out}/verdict.json", "w"), indent=1)
    print("== SUMMARY (P05 / P09 clause of the guard, new vs returning, sealed-like = 0.5 ret + 0.5 new)")
    print("\n".join(summ))
    print(json.dumps({k: v for k, v in V.items() if k not in ("ci", "summary")}, indent=1))
    for k, v in V["ci"].items():
        print(f"== CI {k}")
        print("\n".join(v) if isinstance(v, list) else v)


if __name__ == "__main__":
    main()

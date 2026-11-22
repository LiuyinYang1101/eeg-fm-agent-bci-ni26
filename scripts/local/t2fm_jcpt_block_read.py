#!/usr/bin/env python
"""T2-FM-D1 / D4 / D5: THE J-CPT BLOCK READ (zero GPU; written and committed BEFORE any J-CPT fine-tune number was looked at).
Every fetched J-CPT fine-tune set (the shipped ST-L line from ck_<tag>_s<step>, seeds 0-11 paired with stl_rcll = CPT v1, kit + seen1;
fetched dirs ni26-t2p-jft-<tag>-s<step>-<proto>-<job>) is read under SK10 lambda 0.95 (t2_d320_rescreen.py) in five specs:
  x4     (THE PRE-REGISTERED T2-D350 READ) anchor X4 = fbc_tv8c 0.4 + sh_tv8c 0.3 + stl_rcll 0.3; arm X4_<tag>_<s> = stl_rcll -> the member.
         Bar: typical >= +0.003 with >= 9 / 12 AND P04 >= -0.010.
  alone  the member-alone dose curve: anchor alone_rcll (stl_rcll 1.0); arms alone_<tag>_<s> (descriptive).
  m6     today's stack (the manager's ask 2026-11-06 16:0x): anchor SW_rcll = M6's shape (T2-D378's S3f) with its vp12 member (0.07)
         -> stl_rcll (the plain CPT-v1 member on the same seeds 0-11: the matched control); arm SW_<tag>_<s> = vp12 -> the J-CPT member.
         The arm minus the anchor isolates the CPT change (same recipe, same seeds, same weight; one of the slot's five members).
         Reported with T2-D386's swap bar (typical >= +0.003 with >= 9 / 12, P04 >= -0.010, new >= 0), descriptive; M6 itself (S3f) too.
  vsA0   T2-D354: X4_jA_<s> vs X4_jA0_<s> at the matched step (A's cross-track pool vs task-only), for every step both exist.
  expo   the descriptive matched-T2-exposure pairs (R1_T2FM.md): jAR4 20k vs jA0 5k (exact), jA 5k vs jA0 2k (0.83x), jA 20k vs jA0 5k (1.33x).
  vsA    T2-FM-D4 / D5: X4_jAR4_<s> / X4_jAR2_<s> vs X4_jA_<s> at the matched step; a replay win = the x4 bar AND >= +0.002 with >= 8 / 12 over A.
+ the participant-clustered CI (t2_cluster_ci.py) of the best x4 and m6 arm per tag; + per-source token exposure from the fetched cpt_summaries.
  mem-guard python scripts/local/t2fm_jcpt_block_read.py [--tags jA,jA0,jAR4,jAR2] [--out DIR]
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
STEPS = (2000, 5000, 10000, 20000)
EXPO = [("jAR4", 20000, "jA0", 5000), ("jA", 5000, "jA0", 2000), ("jA", 20000, "jA0", 5000)]


def member(tag, step):
    """{offset 0, kit, seen1} of the fine-tunes from ck_<tag>_s<step>, or None if not (completely) fetched."""
    out = {"offset": 0}
    for p in ("kit", "seen1"):
        dirs = glob.glob(f"{S}/ni26-t2p-jft-{tag}-s{step:05d}-{p}-*")
        hits = sorted({f.rsplit("__s", 1)[0] for d in dirs for f in glob.glob(f"{d}/runs/{p}__fm-st_large-mean__sz__n00__*__s*.npz")})
        if len(hits) != 1 or len(glob.glob(f"{hits[0]}__s*.npz")) != 12:
            return None
        out[p] = f"{hits[0]}__s{{s}}.npz"
    return out


def exposure(tag, step):
    f = Path(f"{S}/ni26-t2p-jcpt-{tag[1:]}/ck_{tag}_s{step:05d}/cpt_summary.json")
    if not f.exists():
        return None
    sm = json.loads(f.read_text())
    acc = int(sm["args"].get("accum", 1))
    tok = {p["name"]: p["p_draw"] * p["micro_batch"] * len(p["ch"]) * (int(p.get("n_out", 512)) // 16) * acc * step for p in sm["pools"]}
    tot = sum(tok.values()) or 1.0
    grp = {}
    for p in sm["pools"]:
        grp[p.get("group") or "?"] = grp.get(p.get("group") or "?", 0.0) + tok[p["name"]] / tot
    return dict(by_group={k: round(v, 4) for k, v in sorted(grp.items())},
                by_pool={k: round(v / tot, 4) for k, v in sorted(tok.items(), key=lambda kv: -kv[1])})


def bar_x4(r):
    return bool(r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010)


def bar_sw(r):
    return bool(bar_x4(r) and r.newT >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="jA,jA0,jAR4,jAR2")
    ap.add_argument("--out", default=f"{O}/t2fm_jcpt_block")
    a = ap.parse_args()
    tags = [t for t in a.tags.split(",") if t]
    D310 = json.load(open(f"{O}/t2_d310/spec.json"))["members"]
    D274 = json.load(open(f"{O}/t2_d274/spec.json"))["members"]
    D378 = json.load(open(f"{O}/t2_d378/spec.json"))
    S3f = D378["arms"]["S3f"]
    assert "vp12" in S3f and abs(S3f["vp12"] - 0.07) < 1e-9, S3f
    M = dict(D378["members"], fbc_tv8c=D310["fbc_tv8c"], sh_tv8c=D310["sh_tv8c"], stl_rcll=D274["stl_rcll"])
    have = {}
    for t in tags:
        for s in STEPS:
            m = member(t, s)
            if m:
                M[f"{t}_s{s}"] = m
                have.setdefault(t, []).append(s)
    print("members read:", {t: v for t, v in have.items()})
    x4 = lambda m: {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, m: 0.3}  # noqa: E731

    def sw(m):
        b = dict(S3f)
        b[m] = b.pop("vp12")
        return b
    keys = [f"{t}_s{s}" for t, ss in have.items() for s in ss]
    specs = {"x4": ("X4", {"X4": x4("stl_rcll"), **{f"X4_{k}": x4(k) for k in keys}}),
             "alone": ("alone_rcll", {"alone_rcll": {"stl_rcll": 1.0}, **{f"alone_{k}": {k: 1.0} for k in keys}}),
             "m6": ("SW_rcll", {"SW_rcll": sw("stl_rcll"), "S3f": dict(S3f), **{f"SW_{k}": sw(k) for k in keys}})}
    for s in STEPS:
        if "jA" in have and "jA0" in have and s in have["jA"] and s in have["jA0"]:
            specs[f"vsA0_s{s}"] = (f"X4_jA0_s{s}", {f"X4_jA0_s{s}": x4(f"jA0_s{s}"), f"X4_jA_s{s}": x4(f"jA_s{s}")})
        rep = [t for t in ("jAR4", "jAR2") if t in have and s in have[t]]
        if "jA" in have and s in have["jA"] and rep:
            specs[f"vsA_s{s}"] = (f"X4_jA_s{s}", {f"X4_jA_s{s}": x4(f"jA_s{s}"), **{f"X4_{t}_s{s}": x4(f"{t}_s{s}") for t in rep}})
    for t1, s1, t0, s0 in EXPO:
        if s1 in have.get(t1, []) and s0 in have.get(t0, []):
            specs[f"expo_{t1}{s1}_vs_{t0}{s0}"] = (f"X4_{t0}_s{s0}", {f"X4_{t0}_s{s0}": x4(f"{t0}_s{s0}"), f"X4_{t1}_s{s1}": x4(f"{t1}_s{s1}")})
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
    cols = ["spec", "arm", "anchor", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]
    print(d[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    V = dict(have=have, x4={}, alone={}, m6={}, vsA0={}, expo={}, vsA={}, ci={}, exposure={})
    row = lambda spec, arm: d[(d.spec == spec) & (d.arm == arm)].iloc[0]  # noqa: E731
    pick = lambda r: dict(typical=round(float(r.typical), 4), up=int(r.typical_up), P04=round(float(r.P04), 4),  # noqa: E731
                          new=round(float(r.newT), 4), P10=round(float(r.P10), 4), level=round(float(r.typ_level), 4))
    for k in keys:
        V["x4"][k] = dict(pick(row("x4", f"X4_{k}")), bar="PASS" if bar_x4(row("x4", f"X4_{k}")) else "not met")
        V["alone"][k] = pick(row("alone", f"alone_{k}"))
        V["m6"][k] = dict(pick(row("m6", f"SW_{k}")), swap_bar="meets" if bar_sw(row("m6", f"SW_{k}")) else "not met")
    V["m6"]["S3f_vs_SW_rcll"] = pick(row("m6", "S3f"))
    for name in specs:
        if name.startswith(("vsA0", "vsA_", "expo")):
            grp = "vsA0" if name.startswith("vsA0") else "vsA" if name.startswith("vsA_") else "expo"
            for arm in specs[name][1]:
                if arm != specs[name][0]:
                    r = row(name, arm)
                    V[grp][f"{name}:{arm}"] = pick(r)
                    if grp == "vsA":
                        rx = row("x4", arm)
                        V[grp][f"{name}:{arm}"]["replay_win"] = bool(bar_x4(rx) and r.typical >= 0.002 and r.typical_up >= 8)
    for t, ss in have.items():
        bx = max(ss, key=lambda s: V["x4"][f"{t}_s{s}"]["typical"])
        bm = max(ss, key=lambda s: V["m6"][f"{t}_s{s}"]["typical"])
        for spec, anchor, arm in (("x4", "X4", f"X4_{t}_s{bx}"), ("m6", "SW_rcll", f"SW_{t}_s{bm}")):
            c = subprocess.run([PY, CI, f"{a.out}/spec_{spec}.json", anchor, arm], capture_output=True, text=True, env=env)
            V["ci"][f"{spec}:{arm}"] = c.stdout.strip().splitlines()[-9:] if c.returncode == 0 else f"failed: {c.stderr[-300:]}"
        for s in ss:
            e = exposure(t, s)
            if e:
                V["exposure"][f"{t}_s{s}"] = e
    json.dump(V, open(f"{a.out}/verdict.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in V.items() if k not in ("ci", "exposure")}, indent=1))
    for k, v in V["ci"].items():
        print(f"== CI {k}")
        print("\n".join(v) if isinstance(v, list) else v)
    for k, v in V["exposure"].items():
        print(f"== tokens {k}: {v['by_group']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-FM-D1 / D3 / D4 / D5 reads (zero GPU), as pre-registered in docs/DECISIONS_T2FM.md: the T2 fine-tunes from a CPT run's
checkpoints (seeds 0-11, kit + seen1, the shipped ST-L line) read on the one-member X4 shape (fbc_tv8c 0.4, sh_tv8c 0.3, ST-L 0.3)
under SK10 lambda 0.95 via t2_d320_rescreen.py, each checkpoint's member in place of stl_rcll (the shipped 5k CPT v1, seeds 0-11).
  bar (T2-D350's, per checkpoint): typical >= +0.003 with >= 9 / 12 up AND P04 >= -0.010
  --vs TAG0: also the matched-step contrast (T2-D354's): X4 with TAG's member vs X4 with TAG0's member at the same step
  --fm st_base: the member is an ST-base (T2-D356's reads); the stem's model token follows
  T2-FM-D2: --fm st_base --prefix ni26-t2p-fmft --insurance: (a) the x4 bar; (b) X4-with-member vs cnn_only, P04 >= -0.0118 (amended);
  + the member alone vs stl_rcll alone (the 'alone' spec)
Run dirs = the fetched sofia fine-tune job dirs (ni26-t2p-jft-<tag>-s<step>-<proto>-<job>, or --runs given explicitly).
  python scripts/local/t2fm_jcpt_read.py jA [--vs jA0] [--steps 2000,5000,10000,20000] [--runs DIR ...] [--fm st_large]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
EXP = os.path.expanduser("~/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1")
HERE = os.path.dirname(os.path.abspath(__file__))
RESCREEN = os.path.join(HERE, "..", "..", "tracks", "t2_bci", "scripts", "t2_d320_rescreen.py")
CI = os.path.join(HERE, "t2_cluster_ci.py")
PY = sys.executable


def member(runs, tag, step, fm, prefix="ni26-t2p-jft"):
    """{offset 0, kit: pattern, seen1: pattern} of the fine-tunes from checkpoint ck_<tag>_s<step> (exactly one stem per protocol)."""
    out = {"offset": 0}
    for p in ("kit", "seen1"):
        dirs = runs or glob.glob(f"{O}/sofia_s1/{prefix}-{tag}-s{step:05d}-{p}-*")
        hits = sorted({f.rsplit("__s", 1)[0] for d in dirs for f in glob.glob(f"{d}/runs/{p}__fm-{fm}-mean__sz__n00__*__s*.npz")})
        if len(hits) != 1:
            raise SystemExit(f"{tag} s{step} {p}: {len(hits)} member stems in {dirs}: {hits[:3]}")
        n = len(glob.glob(f"{hits[0]}__s*.npz"))
        if n != 12:
            raise SystemExit(f"{tag} s{step} {p}: {n} seeds, need 12 (0-11)")
        out[p] = f"{hits[0]}__s{{s}}.npz"
    return out


def exposure(tag, run, steps):
    """T2-D354: per-source token exposure of a CPT checkpoint from its cpt_summary.json (fetched alone:
    `s1 fetch ni26 <run>/ck_<tag>_s<step>/cpt_summary.json`): p_draw x micro-batch x channels x patches x accum x steps."""
    out = {}
    for s in steps:
        f = Path(f"{O}/sofia_s1/{run}/ck_{tag}_s{s:05d}/cpt_summary.json")
        if not f.exists():
            continue
        sm = json.loads(f.read_text())
        acc = int(sm["args"].get("accum", 1))
        tok = {p["name"]: p["p_draw"] * p["micro_batch"] * len(p["ch"]) * (int(p.get("n_out", 512)) // 16) * acc * s for p in sm["pools"]}
        tot = sum(tok.values()) or 1.0
        grp = {}
        for p in sm["pools"]:
            grp[p.get("group") or "?"] = grp.get(p.get("group") or "?", 0.0) + tok[p["name"]] / tot
        out[s] = dict(tokens_total=tot, by_group={k: round(v, 4) for k, v in grp.items()},
                      by_pool={k: round(v / tot, 4) for k, v in sorted(tok.items(), key=lambda kv: -kv[1])})
    return out


def rescreen(out, spec, sk):
    os.makedirs(out, exist_ok=True)
    json.dump(spec, open(f"{out}/spec.json", "w"), indent=1)
    json.dump({f"{out}/spec.json": sk}, open(f"{out}/sk.json", "w"))
    subprocess.run([PY, "-I", RESCREEN, "--export", EXP, "--out", out, "--jobs", "1", "--specs", f"{out}/spec.json",
                    "--sk-list", f"{out}/sk.json"], check=True, stdout=subprocess.DEVNULL)
    import pandas as pd
    d = pd.read_csv(f"{out}/rescreen_sk.csv")
    return d[(d.status == "ok") & (d.dec == "sk")]


def bar(r):
    return bool(r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.010)


def bar_b(r):
    """T2-FM-D2 bar (b), amended 2026-11-06 before any ST-base number: P04 >= the shipped member's own P04 vs cnn_only (-0.0118)."""
    return bool(r.typical >= 0.003 and r.typical_up >= 9 and r.P04 >= -0.0118)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--vs", default="")
    ap.add_argument("--steps", default="2000,5000,10000,20000")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--fm", default="st_large")
    ap.add_argument("--name", default="")
    ap.add_argument("--prefix", default="ni26-t2p-jft", help="job-name prefix of the fetched dirs (ST-base: ni26-t2p-fmft)")
    ap.add_argument("--insurance", action="store_true", help="T2-FM-D2 bar (b): X4 with the member vs the CNN-only pair (fbc_tv8c 4/7 + sh_tv8c 3/7)")
    ap.add_argument("--run", default="", help="the CPT run dir under sofia_s1 for the token-exposure table (default ni26-t2p-jcpt-<tag without j>)")
    a = ap.parse_args()
    D310 = json.load(open(f"{O}/t2_d310/spec.json"))["members"]
    D274 = json.load(open(f"{O}/t2_d274/spec.json"))["members"]
    steps = [int(v) for v in a.steps.split(",") if v]
    C = {"fbc_tv8c": D310["fbc_tv8c"], "sh_tv8c": D310["sh_tv8c"], "stl_rcll": D274["stl_rcll"]}
    M = dict(C, **{f"{a.tag}_s{s}": member(a.runs, a.tag, s, a.fm, a.prefix) for s in steps})
    if a.vs:
        M.update({f"{a.vs}_s{s}": member([], a.vs, s, a.fm, a.prefix) for s in steps})
    x4 = lambda m: {"fbc_tv8c": 0.4, "sh_tv8c": 0.3, m: 0.3}  # noqa: E731
    arms = {"X4": x4("stl_rcll"), **{f"{a.tag}_s{s}": x4(f"{a.tag}_s{s}") for s in steps},
            **{f"alone_{a.tag}_s{s}": {f"{a.tag}_s{s}": 1.0} for s in steps}, "alone_rcll": {"stl_rcll": 1.0}}
    out = f"{O}/t2fm_{a.name or a.tag}"
    d = rescreen(out, {"n": 12, "anchor": "X4", "members": M, "arms": arms}, list(arms))
    cols = ["arm", "typical", "typical_up", "P04", "P04_up", "P10", "newT", "typ_level"]
    print(d[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    V = dict(tag=a.tag, fm=a.fm, out=out, steps={})
    for s in steps:
        r = d[d.arm == f"{a.tag}_s{s}"].iloc[0]
        V["steps"][s] = dict(typical=float(r.typical), typical_up=int(r.typical_up), P04=float(r.P04), newT=float(r.newT),
                             P10=float(r.P10), level=float(r.typ_level), bar="PASS" if bar(r) else "not met")
    best = max(steps, key=lambda s: V["steps"][s]["typical"])
    V["best_step"] = best
    ci = subprocess.run([PY, CI, f"{out}/spec.json", "X4", f"{a.tag}_s{best}"], capture_output=True, text=True)
    V["best_cluster_ci"] = ci.stdout.strip().splitlines()[-12:] if ci.returncode == 0 else f"failed: {ci.stderr[-300:]}"
    if a.vs:                                          # T2-D354: the matched-step contrast, paired by seed
        spec = {"n": 12, "anchor": f"X4_{a.vs}_s{steps[0]}", "members": M, "arms": {}}
        V["vs"] = {}
        for s in steps:
            spec_s = dict(spec, anchor=f"X4_{a.vs}_s{s}", arms={f"X4_{a.vs}_s{s}": x4(f"{a.vs}_s{s}"), f"X4_{a.tag}_s{s}": x4(f"{a.tag}_s{s}")})
            e = rescreen(f"{out}/vs_{a.vs}_s{s}", spec_s, list(spec_s["arms"]))
            r = e[e.arm == f"X4_{a.tag}_s{s}"].iloc[0]
            V["vs"][s] = dict(typical=float(r.typical), typical_up=int(r.typical_up), P04=float(r.P04), newT=float(r.newT))
            print(f"{a.tag} - {a.vs} at step {s}: typical {r.typical:+.4f} ({int(r.typical_up)}/12), P04 {r.P04:+.4f}, newT {r.newT:+.4f}")
    if a.insurance:                                   # T2-FM-D2 (b): does the member beat having NO FM member at all?
        V["insurance"] = {}
        for s in steps:
            spec_s = {"n": 12, "anchor": "cnn_only", "members": M,
                      "arms": {"cnn_only": {"fbc_tv8c": 4 / 7, "sh_tv8c": 3 / 7}, f"X4_{a.tag}_s{s}": x4(f"{a.tag}_s{s}"), "X4": x4("stl_rcll")}}
            e = rescreen(f"{out}/insurance_s{s}", spec_s, list(spec_s["arms"]))
            for arm in (f"X4_{a.tag}_s{s}", "X4"):
                r = e[e.arm == arm].iloc[0]
                V["insurance"].setdefault(s, {})[arm] = dict(typical=float(r.typical), typical_up=int(r.typical_up), P04=float(r.P04),
                                                             new=float(r.newT), bar_b="PASS" if bar_b(r) else "not met")
            print(f"insurance s{s} vs cnn_only: {V['insurance'][s]}")
    spec_al = {"n": 12, "anchor": "alone_rcll", "members": M,          # the member alone vs the shipped CPT-v1 ST-L alone
               "arms": {"alone_rcll": {"stl_rcll": 1.0}, **{f"alone_{a.tag}_s{s}": {f"{a.tag}_s{s}": 1.0} for s in steps}}}
    e = rescreen(f"{out}/alone", spec_al, list(spec_al["arms"]))
    V["alone_vs_rcll"] = {}
    for s in steps:
        r = e[e.arm == f"alone_{a.tag}_s{s}"].iloc[0]
        V["alone_vs_rcll"][s] = dict(typical=float(r.typical), typical_up=int(r.typical_up), P04=float(r.P04), new=float(r.newT),
                                     P10=float(r.P10), level=float(r.typ_level), anchor_level=float(r.anc_typ_level))
        print(f"alone {a.tag} s{s} vs alone_rcll: {V['alone_vs_rcll'][s]}")
    V["token_exposure"] = exposure(a.tag, a.run or f"ni26-t2p-jcpt-{a.tag[1:]}", steps)
    for s, e in V["token_exposure"].items():
        print(f"{a.tag} s{s} token exposure by group {e['by_group']}; top pools {dict(list(e['by_pool'].items())[:5])}")
    json.dump(V, open(f"{out}/verdict.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in V.items() if k != "best_cluster_ci"}, indent=1))


if __name__ == "__main__":
    main()

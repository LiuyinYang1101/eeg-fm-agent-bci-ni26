#!/usr/bin/env python
"""T2-D282 (zero GPU; pre-registered in docs/DECISIONS_T2.md): the TTT panel's Step 0 (lead/PANEL_2026-10-27_TTT_CONCLUSION.md).
T2-D125's archived ST-L outputs (job 1459249, restored from the NAS; kit + seen1, seeds 0-5): plain vs test-time-MAE-adapted, beside the
other members of T2-D253's spec at index i (fbc_rc75 204+i, sh_vr 168+i, mor i).
  (1) hard / easy (T2-D252's definitions): hard = the windows the ST-L-free blend {fbc 0.5, sh 0.375, mor 0.125} (prior-corrected) gets
      wrong; delta = adapted - plain ST-L window accuracy (each prior-corrected) on them. Kit test (P04 + P09) = the gate; seen1's mix
      windows (seen1 val + P04 / P10 seen) descriptive (D125's seen1 files are leaky: the CPT pool held seen1 test windows).
  (2) package shape (t2p_pkgshape, n 6): c_mae vs c_plain at w4321.
Gate of Step 1: the kit-test hard delta (seeds 0-5 mean) > 0.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d282_step0.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from t2p_report import causal_prior_correction  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
RUNS = O / "sofia_s1/ni26-t2p-d125-maetta-1459249/runs"
STEM = "{p}__fm-st_large-mean__sz__n00__fmE30b64le0.0001lh0.001wd0.05ll1ema0dp0mx0.4cgcptstlarge-d89b{t}__s{{s}}.npz"
N = 6
OFFS = {"fbc_rc75": 204, "sh_vr": 168, "mor": 0}
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "mor": 0.1}


def main():
    out = O / "t2_d282"
    out.mkdir(exist_ok=True)
    s253 = json.load(open(O / "t2_d253/spec.json"))
    mem = {m: s253["members"][m] for m in OFFS}
    assert {m: d["offset"] for m, d in mem.items()} == OFFS
    for k, t in (("st_plain", ""), ("st_mae", "mae1e-05")):
        mem[k] = dict(offset=0, **{p: str(RUNS / STEM.format(p=p, t=t)) for p in ("kit", "seen1")})
    arms = {"c_plain": dict(W, st_plain=0.2), "c_mae": dict(W, st_mae=0.2), "a_plain": {"st_plain": 1.0}, "a_mae": {"st_mae": 1.0},
            "c_noST": {"fbc_rc75": 0.5, "sh_vr": 0.375, "mor": 0.125}}
    sp = dict(n=N, anchor="c_plain", bar="", members=mem, arms=arms)
    (out / "spec.json").write_text(json.dumps(sp, indent=1) + "\n")
    with open(out / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(out / "spec.json"),
                             "--out", str(out / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc} (see {out / 'read.txt'})"
    r = json.load(open(out / "read.json"))["arms"]
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    y = idx["y"].to_numpy().astype(int)
    strm = (subj + "|" + idx["session"].to_numpy()).astype(object)
    ordr = idx["order"].to_numpy()
    res = {}
    for proto in ("kit", "seen1"):
        acc = dict(hard=[], easy=[], n_hard=[], parts={})
        for i in range(N):
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=int(d["offset"]) + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (m, proto, i)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            mix = np.zeros(len(idx), bool)
            if proto == "kit":
                mix[te] = True
            else:
                mix[va] = True
                mix[te[np.isin(subj[te], ["P04", "P10"])]] = True

            def pred(p):
                q = np.zeros_like(p)
                q[ok] = causal_prior_correction(p[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
                return q.argmax(1)
            cnn = pred(sum(w * P[m] for m, w in arms["c_noST"].items()))
            hard = mix & (cnn != y)
            ez = mix & ~hard
            pm, pa = pred(P["st_plain"]), pred(P["st_mae"])
            acc["hard"].append(float((pa[hard] == y[hard]).mean() - (pm[hard] == y[hard]).mean()))
            acc["easy"].append(float((pa[ez] == y[ez]).mean() - (pm[ez] == y[ez]).mean()))
            acc["n_hard"].append(int(hard.sum()))
            for u in sorted(set(subj[mix])):
                h = hard & (subj == u)
                acc["parts"].setdefault(u, []).append(float((pa[h] == y[h]).mean() - (pm[h] == y[h]).mean()))
        res[proto] = dict(hard=float(np.mean(acc["hard"])), hard_up=int(sum(v > 0 for v in acc["hard"])), hard_per_seed=acc["hard"],
                          easy=float(np.mean(acc["easy"])), n_hard=float(np.mean(acc["n_hard"])),
                          parts={u: float(np.mean(v)) for u, v in acc["parts"].items()})
        t = res[proto]
        print(f"{proto} ({'GATE: kit test' if proto == 'kit' else 'seen1 mix windows, descriptive, leaky'}): hard delta {t['hard']:+.4f} "
              f"({t['hard_up']} / {N} seeds up; {t['n_hard']:.0f} hard windows / seed), easy {t['easy']:+.4f}; per participant (hard) "
              + ", ".join(f"{u} {v:+.4f}" for u, v in t["parts"].items()))
    c, a = r["c_mae"], r["a_mae"]
    print(f"package c_mae vs c_plain (n {N}): mix {c['mix']['d']:+.4f} ({c['mix']['up']} / {N}), kit test {c['kit_test']['d']:+.4f} ({c['kit_test']['up']} / {N}), "
          f"kit val {c['kit_val']['d']:+.4f}, seen {c['seen']['d']:+.4f}, new {c['new']['d']:+.4f}; per participant "
          + ", ".join(f"{k} {v:+.4f}" for k, v in c["participants"].items() if k.startswith("kit:")))
    print(f"member alone a_mae vs c_plain is not a pair; member a_mae vs a_plain: kit test {a['level']['kit_test'] - r['a_plain']['level']['kit_test']:+.4f}, "
          f"kit val {a['level']['kit_val'] - r['a_plain']['level']['kit_val']:+.4f}")
    gate = res["kit"]["hard"] > 0
    print("STEP 1 GATE (kit-test hard delta > 0):", "MET -> Step 1 waits for the user's paste" if gate else "NOT MET -> Step 1 not run; TTT closes on T2")
    (out / "step0.json").write_text(json.dumps(dict(hard_easy=res, package=dict(c_mae=c), gate=gate), indent=1) + "\n")


if __name__ == "__main__":
    main()

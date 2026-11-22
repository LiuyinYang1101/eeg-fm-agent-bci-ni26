#!/usr/bin/env python
"""T2-D254 (zero compute), as pre-registered in docs/DECISIONS_T2.md: a CNN-confidence gate on the package blend.  q = the CNN pair's blend
(FBCNet 0.4 : ShallowNet 0.3, renormalised, before the prior correction); a window with max q < 0.5 takes the low-confidence weights
{FBCNet 2/7, ShallowNet 1.5/7, ST-L 0.4, MOR 0.1}, every other window keeps w4321 (t2p_pkgshape `arm_gate`).  Arm g50 vs the anchor
c_rcll = {FBCNet 0.4, ShallowNet 0.3, ST-L 0.2, MOR 0.1} on T2-D250's 24 replicates (T2-D247's CNN seeds 204 / 168, ST-L 0; T2-D247r's
192 / 156, ST-L 12).  Identity controls (asserted, not deciding): tau 0 = c_rcll, tau 1.01 = the plain low weights, row for row.
T2-D209's leave-one-participant-out rule (one arm):
  delta_p = g50 - c_rcll on p's cells in its mix components (P04: mean of kit test and seen1 seen; P09: kit test; P05 / P08: seen1 val;
            P10: seen1 seen), averaged over the 24 replicates;
  fold p takes g50 iff mean_{q != p} delta_q > 0 (else the anchor, score 0); held-out score_p = delta_p;
  candidate iff mean_p score_p > 0 AND >= 4 / 5 folds have score_p > 0.
  mem-guard python scripts/local/t2_d254_gate.py
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
COLS = {"P04": [("kit", "P04"), ("seen1", "P04")], "P09": [("kit", "P09")], "P05": [("seen1", "P05")], "P08": [("seen1", "P08")], "P10": [("seen1", "P10")]}
SETS = {"a": ("t2_d247", {"fbc_rc75": 204, "sh_vr": 168, "mor": 0, "stl_rcll": 0}),
        "b": ("t2_d247r", {"fbc_rc75": 192, "sh_vr": 156, "mor": 0, "stl_rcll": 12})}
W4321 = {"fbc_rc75": 0.4, "sh_vr": 0.3, "stl_rcll": 0.2, "mor": 0.1}
LOW = {"fbc_rc75": 2 / 7, "sh_vr": 1.5 / 7, "stl_rcll": 0.4, "mor": 0.1}
ON = {"fbc_rc75": 0.4, "sh_vr": 0.3}
TAU = 0.5
METRICS = ("test", "val", "seen", "p09", "unseen")


def spec_for(src, offs):
    s = json.load(open(O / src / "spec.json"))
    mem = {m: s["members"][m] for m in offs}
    assert {m: d["offset"] for m, d in mem.items()} == offs, (src, {m: d["offset"] for m, d in mem.items()})
    arms = {"c_rcll": W4321, "g50": W4321, "ctl_t0": W4321, "ctl_t101": W4321, "ctl_low": LOW}
    gate = {"g50": dict(on=ON, tau=TAU, low=LOW), "ctl_t0": dict(on=ON, tau=0.0, low=LOW), "ctl_t101": dict(on=ON, tau=1.01, low=LOW)}
    return dict(n=12, anchor="c_rcll", bar="d141", members=mem, arms=arms, arm_gate=gate)


def read(out, k):
    sp = out / f"spec_{k}.json"
    sp.write_text(json.dumps(spec_for(*SETS[k]), indent=1) + "\n")
    print(f"spec_{k} ({SETS[k][0]}'s seeds) sha256 {hashlib.sha256(sp.read_bytes()).hexdigest()[:12]}")
    with open(out / f"read_{k}.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(sp),
                             "--out", str(out / f"read_{k}.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc} (see {out / f'read_{k}.txt'})"
    return json.load(open(out / f"read_{k}.json"))


def main():
    out = O / "t2_d254"
    out.mkdir(exist_ok=True)
    reads = {k: read(out, k) for k in SETS}
    rows = {(x["proto"], k, x["i"], x["arm"]): x for k, r in reads.items() for x in r["rows"]}
    reps = [(k, i) for k in SETS for i in range(12)]
    for ctl, ref in (("ctl_t0", "c_rcll"), ("ctl_t101", "ctl_low")):      # identity controls
        for (pr, k, i, a), x in rows.items():
            if a == ctl:
                y_ = rows[(pr, k, i, ref)]
                keys = [c for c in x if c in METRICS or c.startswith("P")]
                assert all(x[c] == y_[c] or (x[c] != x[c] and y_[c] != y_[c]) for c in keys), (ctl, pr, k, i)
    print("identity controls: tau 0 = c_rcll, tau 1.01 = the plain low weights (every row)")
    gated = {pr: float(np.mean([rows[(pr, k, i, "g50")]["gated"] for k, i in reps])) for pr in ("kit", "seen1")}
    dv = {p: [float(np.mean([rows[(pr, k, i, "g50")][c] - rows[(pr, k, i, "c_rcll")][c] for pr, c in cs])) for k, i in reps] for p, cs in COLS.items()}
    delta = {p: float(np.mean(v)) for p, v in dv.items()}
    up = {p: int(sum(x > 0 for x in v)) for p, v in dv.items()}
    pkg = {k: {c: reads[k]["arms"]["g50"][c]["d"] for c in ("mix", "new", "seen", "kit_test")} | {"mix_up": reads[k]["arms"]["g50"]["mix"]["up"]} for k in SETS}
    low_pkg = {k: {c: reads[k]["arms"]["ctl_low"][c]["d"] for c in ("mix", "new", "kit_test")} for k in SETS}
    folds = {}
    for p in COLS:
        other = float(np.mean([delta[q] for q in COLS if q != p]))
        folds[p] = dict(other=other, pick="g50" if other > 0 else "c_rcll", score=delta[p] if other > 0 else 0.0)
    print(f"gated share of windows (max q < {TAU}): kit {gated['kit']:.3f}, seen1 {gated['seen1']:.3f}")
    print("g50 vs c_rcll: per participant " + ", ".join(f"{p} {delta[p]:+.4f} ({up[p]} / 24 up)" for p in COLS)
          + f"; 5-participant mean {np.mean(list(delta.values())):+.4f}")
    print("package (descriptive): " + "; ".join(f"set {k}: mix {v['mix']:+.4f} ({v['mix_up']} / 12 up), new {v['new']:+.4f}, seen {v['seen']:+.4f}, "
                                                 f"kit test {v['kit_test']:+.4f}" for k, v in pkg.items()))
    print("the plain low weights everywhere (control, descriptive): " + "; ".join(f"set {k}: mix {v['mix']:+.4f}, new {v['new']:+.4f}, kit test {v['kit_test']:+.4f}"
                                                                         for k, v in low_pkg.items()))
    for p, f in folds.items():
        print(f"fold {p}: other-four mean {f['other']:+.4f} -> {f['pick']}, held-out {f['score']:+.4f}")
    m, n = float(np.mean([f["score"] for f in folds.values()])), sum(f["score"] > 0 for f in folds.values())
    ok = m > 0 and n >= 4
    print(f"held-out mean {m:+.4f}, folds > 0: {n} / 5")
    print("VERDICT:", "CANDIDATE g50 for the user (a template change: t2p_submission gains the gate; its own row, repack, D140 bars; only on the user's OK)"
          if ok else "NOT A CANDIDATE (w4321 everywhere; the gate line closes)")
    (out / "verdict.json").write_text(json.dumps(dict(delta=delta, up=up, gated=gated, pkg=pkg, low_pkg=low_pkg, folds=folds, mean=m, folds_pos=n,
                                                      candidate=ok), indent=1) + "\n")


if __name__ == "__main__":
    main()

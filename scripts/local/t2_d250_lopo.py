#!/usr/bin/env python
"""T2-D250 (zero compute), as pre-registered in docs/DECISIONS_T2.md: does the adopted ST-L (T2-D247r: rc 0.75 + LLRD 0.75) earn a larger
package weight than w4321's 0.2?  Arms vs the anchor c_rcll = {FBCNet 0.4, ShallowNet 0.3, ST-L 0.2, MOR 0.1}:
  w4231 = {0.4, 0.2, 0.3, 0.1} (ShallowNet -> ST-L), w3331 = {0.3, 0.3, 0.3, 0.1} (FBCNet -> ST-L),
on the 24 replicates of T2-D247 (CNN seeds 204 / 168, ST-L 0) and T2-D247r (192 / 156, ST-L 12): two specs written from theirs, one
t2p_pkgshape read each.  T2-D209's leave-one-participant-out rule, nested over the two arms:
  delta_p(arm) = arm - c_rcll on p's cells in its mix components (P04: mean of kit test and seen1 seen; P09: kit test; P05 / P08: seen1 val;
                 P10: seen1 seen), averaged over the 24 replicates;
  fold p: pick the arm with the larger mean_{q != p} delta_q iff that mean > 0 (else the anchor); held-out score_p = delta_p(pick), else 0;
  candidate iff mean_p score_p > 0 AND >= 4 / 5 folds have score_p > 0; the arm = the larger 5-participant mean delta.
  mem-guard python scripts/local/t2_d250_lopo.py
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
ARMS = {"c_rcll": (0.4, 0.3, 0.2, 0.1), "w4231": (0.4, 0.2, 0.3, 0.1), "w3331": (0.3, 0.3, 0.3, 0.1)}
SETS = {"a": ("t2_d247", {"fbc_rc75": 204, "sh_vr": 168, "mor": 0, "stl_rcll": 0}),
        "b": ("t2_d247r", {"fbc_rc75": 192, "sh_vr": 156, "mor": 0, "stl_rcll": 12})}
NAMES = ("fbc_rc75", "sh_vr", "stl_rcll", "mor")


def spec_for(src, offs):
    s = json.load(open(O / src / "spec.json"))
    mem = {m: s["members"][m] for m in offs}
    assert {m: d["offset"] for m, d in mem.items()} == offs, (src, {m: d["offset"] for m, d in mem.items()})
    return dict(n=12, anchor="c_rcll", bar="d141", members=mem, arms={a: dict(zip(NAMES, w)) for a, w in ARMS.items()})


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
    out = O / "t2_d250"
    out.mkdir(exist_ok=True)
    reads = {k: read(out, k) for k in SETS}
    rows = {(pr, k, i, arm): x for k, r in reads.items() for x in r["rows"] for pr, i, arm in [(x["proto"], x["i"], x["arm"])]}
    reps = [(k, i) for k in SETS for i in range(12)]
    res = {}
    for arm in ("w4231", "w3331"):
        dv = {p: [float(np.mean([rows[(pr, k, i, arm)][c] - rows[(pr, k, i, "c_rcll")][c] for pr, c in cs])) for k, i in reps] for p, cs in COLS.items()}
        res[arm] = dict(delta={p: float(np.mean(v)) for p, v in dv.items()}, up={p: int(sum(x > 0 for x in v)) for p, v in dv.items()},
                        pkg={k: {c: reads[k]["arms"][arm][c]["d"] for c in ("mix", "new", "seen", "kit_test")} for k in SETS})
    folds = {}
    for p in COLS:
        other = {a: float(np.mean([res[a]["delta"][q] for q in COLS if q != p])) for a in res}
        pick = max(other, key=other.get)
        pick = pick if other[pick] > 0 else "c_rcll"
        folds[p] = dict(other=other, pick=pick, score=(res[pick]["delta"][p] if pick != "c_rcll" else 0.0))
    for a, r in res.items():
        print(f"{a} vs c_rcll: per participant " + ", ".join(f"{p} {r['delta'][p]:+.4f} ({r['up'][p]} / 24 up)" for p in COLS)
              + f"; 5-participant mean {np.mean(list(r['delta'].values())):+.4f}; package (descriptive) "
              + "; ".join(f"set {k}: mix {v['mix']:+.4f}, new {v['new']:+.4f}, seen {v['seen']:+.4f}, kit test {v['kit_test']:+.4f}" for k, v in r["pkg"].items()))
    for p, f in folds.items():
        print(f"fold {p}: other-four means " + ", ".join(f"{a} {v:+.4f}" for a, v in f["other"].items()) + f" -> pick {f['pick']}, held-out {f['score']:+.4f}")
    m, k = float(np.mean([f["score"] for f in folds.values()])), sum(f["score"] > 0 for f in folds.values())
    best = max(res, key=lambda a: np.mean(list(res[a]["delta"].values())))
    ok = m > 0 and k >= 4
    print(f"held-out mean {m:+.4f}, folds > 0: {k} / 5")
    print("VERDICT:", f"CANDIDATE {best} for the user (upload 5 re-weighted in a config-only repack, only on the user's OK)" if ok
          else "NOT A CANDIDATE (w4321 stays with the adopted ST-L)")
    (out / "verdict.json").write_text(json.dumps(dict(arms=res, folds=folds, mean=m, folds_pos=k, best=best, candidate=ok), indent=1) + "\n")


if __name__ == "__main__":
    main()

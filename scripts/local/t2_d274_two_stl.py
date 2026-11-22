#!/usr/bin/env python
"""T2-D274 (zero compute), as pre-registered in docs/DECISIONS_T2.md (the panel's T2 item 2, user-approved 2026-10-26): two ST-L members in the
ST-L slot of the package, at 0.1 + 0.1 instead of one at 0.2.  On T2-D253's members (fbc_rc75 204-215, sh_vr 168-179, stl_rcll 0-11, mor 0-11;
12 replicates, seeds i = 0..11):
  anchor c_rcll = {FBCNet 0.4, ShallowNet 0.3, stl_rcll(i) 0.2, MOR 0.1};
  arm A  = {FBCNet 0.4, ShallowNet 0.3, stl_rcll(i) 0.1, stl_rcll(i + 12) 0.1, MOR 0.1}  (the same recipe twice: T2-D247r's seeds 12-23);
  arm B  = {FBCNet 0.4, ShallowNet 0.3, stl_rcll(i) 0.1, stl_lp10(i) 0.1, MOR 0.1}       (T2-D267's LP-FT lp10 runs; T2-D271's diversity).
Per arm: bar d141 (mix d >= +0.005 with >= 9 / 12 up AND kit test d >= -0.005) AND new d >= 0 AND T2-D209's leave-one-participant-out rule
(delta_p = arm - c_rcll on p's mix cells: P04 mean of kit test and seen1 seen, P09 kit test, P05 / P08 seen1 val, P10 seen1 seen, averaged
over the 12 replicates; fold p takes the arm iff the other four's mean delta > 0, else the anchor (score 0); pass iff the held-out mean > 0
AND >= 4 / 5 folds > 0).  Both pass -> the higher mix d.  Member specs' sha256 asserted (t2_d253 / t2_d247r / t2_d267 as recorded).
  mem-guard python scripts/local/t2_d274_two_stl.py
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
SHA = {"t2_d267": "ccfa010a31d4"}
BASE = {"fbc_rc75": 0.4, "sh_vr": 0.3, "mor": 0.1}
ARMS = {"c_rcll": dict(BASE, stl_rcll=0.2), "A": dict(BASE, stl_rcll=0.1, stl_rcll12=0.1), "B": dict(BASE, stl_rcll=0.1, stl_lp10=0.1)}


def load(row):
    p = O / row / "spec.json"
    h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    assert row not in SHA or h == SHA[row], (row, h)
    return json.load(open(p))["members"], h


def main():
    out = O / "t2_d274"
    out.mkdir(exist_ok=True)
    m253, h253 = load("t2_d253")
    m247r, h247r = load("t2_d247r")
    m267, h267 = load("t2_d267")
    offs = {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0}
    mem = {m: m253[m] for m in offs}
    assert {m: d["offset"] for m, d in mem.items()} == offs
    assert m247r["stl_rcll"]["offset"] == 12 and m267["stl_lp10"]["offset"] == 0
    assert m267["stl_rcll"] == mem["stl_rcll"], "t2_d267's stl_rcll = t2_d253's"
    mem["stl_rcll12"] = m247r["stl_rcll"]
    mem["stl_lp10"] = m267["stl_lp10"]
    for p in ("kit", "seen1"):                                    # the two ST-L files of each arm are different runs
        for i in range(12):
            a = np.load(mem["stl_rcll"][p].format(s=i))
            for k, s in (("stl_rcll12", i + 12), ("stl_lp10", i)):
                b = np.load(mem[k][p].format(s=s))
                assert np.array_equal(a["test_idx"], b["test_idx"]) and float(np.abs(a["prob_last"] - b["prob_last"]).max()) > 1e-4, (p, i, k)
    spec = dict(n=12, anchor="c_rcll", bar="d141", members=mem, arms=ARMS)
    sp = out / "spec.json"
    sp.write_text(json.dumps(spec, indent=1) + "\n")
    print(f"member specs: t2_d253 {h253}, t2_d247r {h247r}, t2_d267 {h267}; spec {sp} sha256 {hashlib.sha256(sp.read_bytes()).hexdigest()[:12]}")
    with open(out / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(sp),
                             "--out", str(out / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc} (see {out / 'read.txt'})"
    r = json.load(open(out / "read.json"))
    rows = {(x["proto"], x["i"], x["arm"]): x for x in r["rows"]}
    res = {}
    for a in ("A", "B"):
        c = r["arms"][a]
        dv = {p: [float(np.mean([rows[(pr, i, a)][q] - rows[(pr, i, "c_rcll")][q] for pr, q in cs])) for i in range(12)] for p, cs in COLS.items()}
        delta = {p: float(np.mean(v)) for p, v in dv.items()}
        folds = {}
        for p in COLS:
            other = float(np.mean([delta[q] for q in COLS if q != p]))
            folds[p] = dict(other=other, score=delta[p] if other > 0 else 0.0)
        lm, ln = float(np.mean([f["score"] for f in folds.values()])), sum(f["score"] > 0 for f in folds.values())
        met = c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005
        ok = met and c["new"]["d"] >= 0 and lm > 0 and ln >= 4
        res[a] = dict(pkg={q: c[q]["d"] for q in ("mix", "kit_test", "seen", "new", "kit_val", "seen1_val", "unseen")}, mix_up=c["mix"]["up"],
                      delta=delta, folds=folds, lopo_mean=lm, lopo_pos=ln, d141=met, passed=ok)
        print(f"arm {a} vs c_rcll: mix {c['mix']['d']:+.4f} ({c['mix']['up']} / 12), kit test {c['kit_test']['d']:+.4f}, seen {c['seen']['d']:+.4f}, "
              f"new {c['new']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}, seen1 val {c['seen1_val']['d']:+.4f}, unseen {c['unseen']['d']:+.4f}; "
              f"bar d141 {'met' if met else 'not met'}")
        print(f"  LOPO: " + ", ".join(f"{p} {delta[p]:+.4f} (others {folds[p]['other']:+.4f} -> {folds[p]['score']:+.4f})" for p in COLS)
              + f"; held-out mean {lm:+.4f}, {ln} / 5 folds > 0 -> {'PASS' if ok else 'no'}")
    win = [a for a in ("A", "B") if res[a]["passed"]]
    best = max(win, key=lambda a: res[a]["pkg"]["mix"]) if win else None
    print("VERDICT:", f"PASS arm {best} -> the runtime profile (bar <= 20 ms / window), then one ZIP for upload 5's line by Nov 4" if best
          else "NOT A CANDIDATE (one ST-L member stays)")
    (out / "verdict.json").write_text(json.dumps(dict(res=res, best=best), indent=1) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-D252 (zero compute, descriptive), as pre-registered in docs/DECISIONS_T2.md: why do ST-L member gains mostly vanish in the package?
Every ST-L variant on disk on ST-L seeds 0-11 with the same CNN seed set (fbc_rc75 of T2-D231, 204-215; sh_vr of T2-D215r, 168-179; mor 0-11),
each vs the D112a ST-L (stl), seed-paired:
  member  = the variant alone vs stl alone, package mix delta (t2p_pkgshape);
  package = c_<v> vs c_stl at w4321 (FBCNet 0.4, ShallowNet 0.3, ST-L 0.2, MOR 0.1), package mix delta (t2p_pkgshape);
  hard / easy = the member's window-accuracy delta (v - stl, both prior-corrected as in the package) on the mix's windows (kit test, seen1 seen,
         seen1 val) that the ST-L-free blend {FBCNet 0.5, ShallowNet 0.375, MOR 0.125} (prior-corrected, same index) gets wrong / right;
  flips   = the package's net correct flips (c_v vs c_stl) per 1000 mix windows.
Across the variants: Pearson r of package with member, hard and easy.  The pre-registered reading: direction "complementary" iff
r(package, hard) > 0.5 AND r(package, hard) > r(package, member); else no direction.
  mem-guard python scripts/local/t2_d252_diag.py
"""
import hashlib
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
# variant -> (spec it is read from, member name there); every one at offset 0 beside fbc_rc75 204 / sh_vr 168 / mor 0
SRC = {"stl": ("t2_d247", "stl"), "rc75": ("t2_d235", "stl_rc75"), "rc10": ("t2_d238", "stl_rc10"), "dp10": ("t2_d241", "stl_dp10"),
       "ll75": ("t2_d241", "stl_ll75"), "e60": ("t2_d242", "stl_e60"), "ll85": ("t2_d243", "stl_ll85"), "ll65": ("t2_d243", "stl_ll65"),
       "rcll": ("t2_d247", "stl_rcll"), "rcll_mx8": ("t2_d245", "stl_mx8")}
CNN = {"fbc_rc75": 204, "sh_vr": 168, "mor": 0}
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "mor": 0.1}


def spec():
    mem = {}
    for v, (d, m) in SRC.items():
        s = json.load(open(O / d / "spec.json"))
        assert all(s["members"][c]["offset"] == o for c, o in CNN.items()), d
        for c in CNN:
            if c in mem:
                assert mem[c]["kit"] == s["members"][c]["kit"] and mem[c]["seen1"] == s["members"][c]["seen1"], (d, c)
            mem[c] = s["members"][c]
        assert s["members"][m]["offset"] == 0, (d, m)
        mem[f"st_{v}"] = s["members"][m]
    arms = {f"c_{v}": dict(W, **{f"st_{v}": 0.2}) for v in SRC}
    arms.update({f"a_{v}": {f"st_{v}": 1.0} for v in SRC})
    arms["c_noST"] = {"fbc_rc75": 0.5, "sh_vr": 0.375, "mor": 0.125}
    return dict(n=12, anchor="c_stl", bar="d141", members=mem, arms=arms)


def main():
    out = O / "t2_d252"
    out.mkdir(exist_ok=True)
    sp = spec()
    (out / "spec.json").write_text(json.dumps(sp, indent=1) + "\n")
    print(f"spec sha256 {hashlib.sha256((out / 'spec.json').read_bytes()).hexdigest()[:12]}; variants {list(SRC)}")
    with open(out / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(out / "spec.json"),
                             "--out", str(out / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc}"
    r = json.load(open(out / "read.json"))["arms"]
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    y = idx["y"].to_numpy().astype(int)
    strm = (subj + "|" + idx["session"].to_numpy()).astype(object)
    ordr = idx["order"].to_numpy()
    mem = sp["members"]
    acc = {v: dict(hard=[], easy=[], flips=[]) for v in SRC if v != "stl"}
    for proto in ("kit", "seen1"):
        for i in range(12):
            P, te = {}, None
            for m, d in mem.items():
                z = np.load(d[proto].format(s=int(d["offset"]) + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m] = Pm
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            if proto == "kit":
                mix = np.zeros(len(idx), bool); mix[te] = True                                   # kit test = P04 + P09
            else:
                mix = np.zeros(len(idx), bool); mix[va] = True                                   # seen1 val = P05 + P08
                mix[te[np.isin(subj[te], ["P04", "P10"])]] = True                                # + seen1 seen = P04 + P10

            def pred(p):
                q = np.zeros_like(p)
                q[ok] = causal_prior_correction(p[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
                return q.argmax(1)
            cnn = pred(sum(w * P[m] for m, w in sp["arms"]["c_noST"].items()))
            hard = mix & (cnn != y)
            base_m, base_c = pred(P["st_stl"]), pred(sum(w * P[m] for m, w in sp["arms"]["c_stl"].items()))
            for v in acc:
                vm, vc = pred(P[f"st_{v}"]), pred(sum(w * P[m] for m, w in sp["arms"][f"c_{v}"].items()))
                acc[v]["hard"].append(float((vm[hard] == y[hard]).mean() - (base_m[hard] == y[hard]).mean()))
                ez = mix & ~hard
                acc[v]["easy"].append(float((vm[ez] == y[ez]).mean() - (base_m[ez] == y[ez]).mean()))
                acc[v]["flips"].append(1000 * float(((vc[mix] == y[mix]).astype(int) - (base_c[mix] == y[mix]).astype(int)).sum()) / mix.sum())
    tab = {}
    for v in acc:
        tab[v] = dict(member=r[f"a_{v}"]["mix"]["d"] - r["a_stl"]["mix"]["d"], package=r[f"c_{v}"]["mix"]["d"],
                      hard=float(np.mean(acc[v]["hard"])), easy=float(np.mean(acc[v]["easy"])), flips=float(np.mean(acc[v]["flips"])))
    print(f"{'variant':10s} {'member':>8s} {'package':>8s} {'hard':>8s} {'easy':>8s} {'flips/1k':>8s}")
    for v, t in tab.items():
        print(f"{v:10s} {t['member']:+8.4f} {t['package']:+8.4f} {t['hard']:+8.4f} {t['easy']:+8.4f} {t['flips']:+8.2f}")
    pk = np.array([t["package"] for t in tab.values()])
    cor = {k: float(np.corrcoef(pk, [t[k] for t in tab.values()])[0, 1]) for k in ("member", "hard", "easy")}
    print("Pearson r with the package delta across the variants: " + ", ".join(f"{k} {c:+.2f}" for k, c in cor.items()))
    comp = cor["hard"] > 0.5 and cor["hard"] > cor["member"]
    print("READING:", "direction COMPLEMENTARY (the package gain tracks the member's gain on the CNN-wrong windows)" if comp else "no direction")
    (out / "diag.json").write_text(json.dumps(dict(table=tab, r=cor, complementary=comp), indent=1) + "\n")


if __name__ == "__main__":
    main()

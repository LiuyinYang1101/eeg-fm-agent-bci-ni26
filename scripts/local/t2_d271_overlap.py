#!/usr/bin/env python
"""T2-D271 (zero compute; ranking only), as pre-registered in docs/DECISIONS_T2.md: do the ST-L head lr x 3 (T2-D266, stl_lh3) and LP-FT's
probe 10 (T2-D267, stl_lp10) fix the same windows? On the mix's windows (kit test, seen1 seen P04 + P10, seen1 val; T2-D255's set), per seed-
protocol pair (seeds 0-11 x kit / seen1), each member's prior-corrected argmax as in the package (t2p_report.causal_prior_correction);
F_x = the windows arm x gets right and stl_rcll gets wrong. J(x) = |F_x & F_lp10| / |F_x | F_lp10| (Jaccard), averaged over the 24 pairs, for
x = lh3 (the question), lp5 (T2-D263, the same mechanism: the 'redundant' reference) and ri (T2-D264, a member-level null: the 'noise'
reference). Rule: if J(lp5) <= J(ri) the read is uninformative; else J(lh3) <= (J(lp5) + J(ri)) / 2 = complementary, otherwise redundant.
Each arm's files come from its own row's spec (outputs/t2_d26x/spec.json, its sha256 asserted against the read's record).
  mem-guard python scripts/local/t2_d271_overlap.py
"""
import hashlib
import json
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
ARMS = {"lp10": ("t2_d267", "stl_lp10", "ccfa010a31d4"), "lh3": ("t2_d266", "stl_lh3", "5f45c7952635"),
        "lp5": ("t2_d263", "stl_lp", "a008625c3e44"), "ri": ("t2_d264", "stl_ri", "90c6bc78c68d")}


def main():
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    y = idx["y"].to_numpy().astype(int)
    strm = (subj + "|" + idx["session"].to_numpy()).astype(object)
    ordr = idx["order"].to_numpy()
    paths, base = {}, None
    for a, (row, arm, sha) in ARMS.items():
        sp_path = O / row / "spec.json"
        h = hashlib.sha256(sp_path.read_bytes()).hexdigest()[:12]
        assert h == sha, (row, h)
        mem = json.load(open(sp_path))["members"]
        assert mem[arm]["offset"] == 0 and mem["stl_rcll"]["offset"] == 0
        paths[a] = mem[arm]
        base = base or mem["stl_rcll"]
        assert mem["stl_rcll"] == base, "every row's stl_rcll = the same files"
    print("specs:", ", ".join(f"{r} {s}" for r, _, s in ARMS.values()))

    def pred(z, ok, te, va):
        P = np.full((len(idx), 3), np.nan)
        P[te], P[va] = z["prob_last"], z["prob_val_last"]
        q = np.zeros_like(P)
        q[ok] = causal_prior_correction(P[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
        return q.argmax(1)

    J = {a: [] for a in ("lh3", "lp5", "ri")}
    nfix = {a: [] for a in ARMS}
    for proto in ("kit", "seen1"):
        for i in range(12):
            zb = np.load(base[proto].format(s=i))
            te, va = zb["test_idx"], zb["val_idx"]
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            mix = np.zeros(len(idx), bool)
            if proto == "kit":
                mix[te] = True
            else:
                mix[va] = True
                mix[te[np.isin(subj[te], ["P04", "P10"])]] = True
            b = pred(zb, ok, te, va)
            F = {}
            for a, d in paths.items():
                z = np.load(d[proto].format(s=i))
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va)
                v = pred(z, ok, te, va)
                F[a] = mix & (v == y) & (b != y)
                nfix[a].append((int(F[a].sum()), int((mix & (v != y) & (b == y)).sum())))
            for a in J:
                u = (F[a] | F["lp10"]).sum()
                J[a].append(float((F[a] & F["lp10"]).sum() / u) if u else 0.0)
    m = {a: float(np.mean(v)) for a, v in J.items()}
    for a in ARMS:
        f, br = np.array(nfix[a]).mean(0)
        print(f"{a:5s}: fixes {f:.1f} / breaks {br:.1f} windows per pair (vs stl_rcll, mix windows)" + (f"; J with lp10 {m[a]:.3f}" if a in m else ""))
    if m["lp5"] <= m["ri"]:
        verdict = "UNINFORMATIVE (J(lp5) <= J(ri))"
    else:
        mid = (m["lp5"] + m["ri"]) / 2
        verdict = f"{'COMPLEMENTARY' if m['lh3'] <= mid else 'REDUNDANT'} (J(lh3) {m['lh3']:.3f} vs midpoint {mid:.3f})"
    print("T2-D271:", verdict)
    out = O / "t2_d271"
    out.mkdir(exist_ok=True)
    (out / "overlap.json").write_text(json.dumps(dict(J=J, mean=m, fixes_breaks=nfix, verdict=verdict), indent=1) + "\n")


if __name__ == "__main__":
    main()

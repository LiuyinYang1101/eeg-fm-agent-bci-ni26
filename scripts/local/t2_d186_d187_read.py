#!/usr/bin/env python
"""T2-D186 / D187 (zero compute; pre-registered in docs/DECISIONS_T2.md; panel T2 task 1, user GO 2026-10-22).
D186: the oracle per-interface prior. t2p_pkgshape on T2-D164's cA members with two arms on the same files: cA (the causal prior per stream)
      and cA_cell (the prior per stream x the TRUE interface, via the spec's arm_prior_key). Pass iff mix delta >= +0.004; kill iff < +0.002.
D187: the between-user IAF spread. Per window IAF = column 0 of t2_d124b/fp_proteus.npy; per user the median; the between-user sd of the
      medians (within-user sd beside). Kill the IAF-warp idea iff the between-user sd < 0.5 Hz.
  python scripts/local/t2_d186_d187_read.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
D = O / "t2_d186"


def d186():
    spec = json.load(open(O / "t2_d164/spec.json"))
    w = spec["arms"]["cA"]
    out = {"n": 12, "anchor": "cA", "bar": "", "members": {k: spec["members"][k] for k in w},
           "arms": {"cA": w, "cA_cell": dict(w)}, "arm_prior_key": {"cA_cell": "cell"}}
    (D / "spec.json").write_text(json.dumps(out, indent=2) + "\n")
    with open(D / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(D / "spec.json"),
                             "--out", str(D / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    a = json.load(open(D / "read.json"))["arms"]["cA_cell"]
    mix = a["mix"]["d"]
    verdict = "PASS (>= +0.004)" if mix >= 0.004 else ("KILL (< +0.002)" if mix < 0.002 else "between (no follow-up)")
    print(f"D186 rc {rc}: cA_cell vs cA mix {mix:+.4f} ({a['mix']['up']} / 12 up), seen {a['seen']['d']:+.4f}, new {a['new']['d']:+.4f}, "
          f"kit test {a['kit_test']['d']:+.4f} -> {verdict}")


def d187():
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    fp = np.load(O / "t2_d124b/fp_proteus.npy")
    assert len(fp) == len(idx), (fp.shape, len(idx))
    iaf = fp[:, 0].astype(np.float64)
    ok = np.isfinite(iaf)
    users = sorted(set(subj))
    med = {u: float(np.median(iaf[ok & (subj == u)])) for u in users}
    within = {u: float(np.std(iaf[ok & (subj == u)])) for u in users}
    between = float(np.std(list(med.values()), ddof=1))
    res = dict(per_user_median=med, per_user_within_sd=within, between_sd=between, mean_within_sd=float(np.mean(list(within.values()))),
               verdict="KILL the IAF warp (< 0.5 Hz)" if between < 0.5 else "keep (>= 0.5 Hz)")
    (D / "d187_iaf.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"D187: per-user median IAF {med}; between-user sd {between:.3f} Hz (mean within-user sd {res['mean_within_sd']:.3f}) -> {res['verdict']}")


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    d186()
    d187()

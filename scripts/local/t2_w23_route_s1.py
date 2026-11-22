#!/usr/bin/env python
"""T2-D634 (W23 = W21 + D632 P1, the owner's exception; PREBUILD until item 166 is typed in T2's shell): the session-1 route space for the
config-only repack of W21. The code is t2p_final.py:318-328's (t2p_route.fingerprint of the export in float64; StandardScaler; LDA eigen /
shrinkage auto on the subject labels; W = scalings_[:, :_max_components]; mu = each user's mean of sc.transform(Fp) @ W), run on:
  (1) ALL rows (= W21's --train all fit): must reproduce W21's config user_blend mean / scale / W / mu (the inert check, max |d| <= 1e-6);
  (2) every user's SESSION-1 rows -> tracks/t2_bci/sofia/w23_route_s1.json {mean, scale, W, mu, users, n_rows, source} (the --set values).
  mem-guard python scripts/local/t2_w23_route_s1.py [--check]      (--check: recompute and compare with the committed JSON, write nothing)
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402
import t2p_route as RT  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
W21_ZIP = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1/ni26-t2p-w21lp-t2p_fin2_w21lp-1633265/t2p_fin2_w21lp.zip"
JSON_OUT = ROOT / "tracks/t2_bci/sofia/w23_route_s1.json"
KEYS = ("mean", "scale", "W", "mu")


def fit(Fp, subj):
    """t2p_final.py:322-328, verbatim in substance."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    users = sorted(set(subj))
    sc_ = StandardScaler().fit(Fp)
    lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(sc_.transform(Fp), subj)
    assert list(lda.classes_) == users
    W = lda.scalings_[:, :lda._max_components]
    Ztr = sc_.transform(Fp) @ W
    mu = np.stack([Ztr[subj == u].mean(0) for u in users])
    return dict(mean=sc_.mean_, scale=sc_.scale_, W=W, mu=mu, users=users)


def fingerprints():
    X, idx, meta = B.load_export(str(EXP))
    Fp = RT.fingerprint(np.asarray(X, np.float64), float(meta["sfreq"]), list(meta["ch_names"]))
    return Fp, idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy()


def w21_route_space():
    with zipfile.ZipFile(W21_ZIP) as z:
        ub = json.loads(z.read("config.json"))["user_blend"]
    return {k: np.asarray(ub[k], np.float64) for k in KEYS} | dict(users=list(ub["users"]))


def maxd(a, b):
    return {k: float(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)).max()) for k in KEYS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    Fp, subj, ses = fingerprints()
    w21 = w21_route_space()
    allr = fit(Fp, subj)
    d_all = maxd(allr, w21)
    print(f"(1) ALL rows vs W21's config route space: max |d| {d_all}; users {allr['users'] == w21['users']}", flush=True)
    assert allr["users"] == w21["users"] and max(d_all.values()) <= 1e-6, d_all
    s1 = ses == "1"
    assert set(subj[s1]) == set(subj), "a user has no session-1 rows"
    new = fit(Fp[s1], subj[s1])
    assert new["users"] == w21["users"] and all(np.asarray(new[k]).shape == np.asarray(w21[k]).shape for k in KEYS)
    rows = {u: int((subj[s1] == u).sum()) for u in new["users"]}
    print(f"(2) session-1 rows per user: {rows}", flush=True)
    print(f"    session-1 space vs W21's: max |d| {maxd(new, w21)}; |mu| from the origin: "
          + ", ".join(f"{u} {v:.2f}" for u, v in zip(new["users"], np.linalg.norm(new["mu"], axis=1))), flush=True)
    doc = {k: np.asarray(new[k], np.float64).tolist() for k in KEYS} | dict(
        users=new["users"], n_rows=rows, source=dict(code="t2p_final.py:322-328 on session-1 rows (T2-D634)", w21_zip=W21_ZIP.name,
                                                     inert_all_rows_maxd=d_all))
    if a.check:
        old = json.loads(JSON_OUT.read_text())
        d = maxd(doc, old)
        print(f"--check: recomputed vs the committed JSON: max |d| {d}", flush=True)
        assert max(d.values()) <= 1e-12 and old["users"] == doc["users"], d
        print("OK")
        return
    JSON_OUT.write_text(json.dumps(doc))
    print(f"wrote {JSON_OUT} (sha256 {hashlib.sha256(JSON_OUT.read_bytes()).hexdigest()})")


if __name__ == "__main__":
    main()

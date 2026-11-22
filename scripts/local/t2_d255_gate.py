#!/usr/bin/env python
"""T2-D255's launch condition (zero compute), as queued in docs/DECISIONS_T2.md backlog row 19 (2026-10-25 19:15, before any D253 number):
T2-D255 (gamma 2) runs iff T2-D253 is a candidate or a near miss = mix delta > 0 AND sw1's hard-window accuracy delta > its easy-window delta,
T2-D252's definition (t2_d252_diag.py), here with T2-D253's anchor as the base: the member's window-accuracy delta (stl_sw1 - stl_rcll, both
prior-corrected as in the package) on the mix's windows (kit test, seen1 seen, seen1 val) that the ST-L-free blend {FBCNet 0.5, ShallowNet 0.375,
MOR 0.125} gets wrong (hard) / right (easy), seeds 0-11, T2-D253's spec (outputs/t2_d253/spec.json; run t2_d253_read.sh first).
  mem-guard python scripts/local/t2_d255_gate.py <D253 package mix delta>
T2-D255's descriptive split of its own arm (same windows, base stl_rcll): ... t2_d255_gate.py <D255 mix delta> t2_d255 stl_sw2
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
NOST = {"fbc_rc75": 0.5, "sh_vr": 0.375, "mor": 0.125}


def main():
    mix_d = float(sys.argv[1])
    row, arm = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("t2_d253", "stl_sw1")
    sp_path = O / row / "spec.json"
    sp = json.load(open(sp_path))
    mem = sp["members"]
    assert {m: d["offset"] for m, d in mem.items()} == {"fbc_rc75": 204, "sh_vr": 168, "stl_rcll": 0, "mor": 0, arm: 0}
    print(f"spec {sp_path} sha256 {hashlib.sha256(sp_path.read_bytes()).hexdigest()[:12]}")
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    y = idx["y"].to_numpy().astype(int)
    strm = (subj + "|" + idx["session"].to_numpy()).astype(object)
    ordr = idx["order"].to_numpy()
    hard_d, easy_d = [], []
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
            mix = np.zeros(len(idx), bool)
            if proto == "kit":
                mix[te] = True                                                   # kit test = P04 + P09
            else:
                mix[va] = True                                                   # seen1 val = P05 + P08
                mix[te[np.isin(subj[te], ["P04", "P10"])]] = True                # + seen1 seen = P04 + P10

            def pred(p):
                q = np.zeros_like(p)
                q[ok] = causal_prior_correction(p[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
                return q.argmax(1)
            hard = mix & (pred(sum(w * P[m] for m, w in NOST.items())) != y)
            easy = mix & ~hard
            b, v = pred(P["stl_rcll"]), pred(P[arm])
            hard_d.append(float((v[hard] == y[hard]).mean() - (b[hard] == y[hard]).mean()))
            easy_d.append(float((v[easy] == y[easy]).mean() - (b[easy] == y[easy]).mean()))
    h, e = float(np.mean(hard_d)), float(np.mean(easy_d))
    go = mix_d > 0 and h > e
    print(f"{arm} vs rcll (24 seed-protocol pairs): hard-window accuracy delta {h:+.4f}, easy {e:+.4f}; package mix delta {mix_d:+.4f}")
    if arm == "stl_sw1":
        print("T2-D255:", "RUNS (mix delta > 0 AND hard > easy)" if go else "CLOSES UNRUN (condition not met)")
    (O / row / f"{arm}_split.json" if arm != "stl_sw1" else O / "t2_d253/d255_gate.json").write_text(json.dumps(dict(mix=mix_d, hard=h, easy=e, hard_pairs=hard_d, easy_pairs=easy_d, runs=go), indent=1) + "\n")


if __name__ == "__main__":
    main()

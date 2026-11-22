#!/usr/bin/env python
"""T2-D638 W24' builder GOLDEN check: t2p_final's pooled nets on fold ss8_P01 / seed 880 (tracks/t2_bci/sofia/t2p_w24_golden.slurm, fetched)
vs D630's pooled files (outputs/t2_d630/{B,C}/cnn_<fam>/runs/ss8_P01__<model>__<align>__n00__<recipe>[__ubw]__s880.npz, prob_last on P01's rows).
B = t2p_final == t2p_bench at all; C = the --user-balance path == D630's C net. Bars (Track3, from identity (ii)): ShallowNet BITWISE; FBCNet
(spatial conv groups = 13) and IFNet max |dp| <= 1e-3 with 100 % argmax; FBCNet's own run-to-run drift (fbc_C vs fbc_C_rep) reported.
  python scripts/local/t2_w24_golden_check.py <fetched golden job dir>
"""
import sys
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75"), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr"),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr")}


def main():
    G = Path(sys.argv[1])
    ok, lines = True, []
    for fam, (model, al, rec) in FAM.items():
        for arm, xt in (("B", ""), ("C", "__ubw")):
            ref = np.load(O / "t2_d630" / arm / f"cnn_{fam}" / "runs" / f"ss8_P01__{model}__{al}__n00__{rec}{xt}__s880.npz")
            g = np.load(G / f"golden_{fam}_{arm}.npz")
            assert str(g["model"]) == model and str(g["recipe"]) == rec and bool(g["user_balance"]) == (arm == "C") and str(g["train_protocol"]) == "ss8_P01"
            te = ref["test_idx"]
            pg = g["prob"][0].astype(np.float64)[np.searchsorted(g["rows"], te)]
            assert np.array_equal(g["rows"][np.searchsorted(g["rows"], te)], te)
            pr = ref["prob_last"].astype(np.float64)
            d, agree = float(np.abs(pg - pr).max()), float(np.mean(pg.argmax(1) == pr.argmax(1)))
            bar = (d == 0.0) if fam == "sh" else (d <= 1e-3 and agree == 1.0)
            ok &= bar
            lines.append(f"{fam} {arm}: t2p_final vs D630 on {len(te)} P01 windows: max |dp| {d:.2e}, argmax agree {agree:.4f} -> {'PASS' if bar else 'FAIL'}")
    a, b = np.load(G / "golden_fbc_C.npz")["prob"][0].astype(np.float64), np.load(G / "golden_fbc_C_rep.npz")["prob"][0].astype(np.float64)
    lines.append(f"FBCNet run-to-run (fbc_C vs fbc_C_rep, all windows): max |dp| {np.abs(a - b).max():.2e}, argmax agree {np.mean(a.argmax(1) == b.argmax(1)):.4f}")
    print("\n".join(lines))
    print("GOLDEN " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

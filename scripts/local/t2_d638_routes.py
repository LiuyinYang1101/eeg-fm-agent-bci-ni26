#!/usr/bin/env python
"""T2-D638 prep (no metric computed): the routed CNN files of D630's B arm (ship composition) under W23's copy router, in t2p_fpcalib blend's
file format, beside a recomputation of W21's router that must reproduce D630's B route files (max |dp| <= 1e-6, else STOP).
  W21 space = t2p_fpcalib fit_space on the fold's training fingerprints (= D630's route files); W23 space = the same on the fold's SESSION-1
  training rows (T2-D632 P1 / T2-D634); nearest user centroid of the stream's running-mean fingerprint, every window; blend r = fbc 1/3,
  sh / ifn 0.5 (the shipped r).
  -> outputs/t2_d638/route_<b|c><w21|w23>/route_<fam>/<fold>__<model>__<align>__n00__<rec>tv8cfb<tag>__s<seed>.npz  (keys as fpcalib blend writes)
  python scripts/local/t2_d638_routes.py     (OMP / MKL / OPENBLAS threads capped by the caller)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FC  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
RB, OUT = O / "t2_d630" / "B", O / "t2_d638"
SEEDS = (880, 881, 882)
FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75", "3", 1 / 3), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5),
       "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr", "2", 0.5)}
FOLDS = [f"ss8_{u}" for u in ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")] + ["rt1_P04", "rt1_P10", "rt2_P04", "rt2_P10"]


def main():
    X, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    del X
    subj, ses, stream = idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy(), idx["stream"].astype(str).to_numpy()
    Fp = np.load(O / "t2_d124b" / "fp_proteus.npy")
    check, n_out = {"B": 0.0, "C": 0.0}, 0
    for fold in FOLDS:
        (_, trm, _, tem), = B.folds(idx, fold)
        tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
        tr1 = tr[ses[tr] == "1"]
        spaces = {"w21": FC.fit_space(Fp[tr], subj[tr]), "w23": FC.fit_space(Fp[tr1], subj[tr1])}
        assert spaces["w21"][2] == spaces["w23"][2], (fold, "a training user has no session-1 rows")
        ks = {}
        for arm, (proj, mu, users) in spaces.items():
            k = np.full(len(te), -1)
            pos = {int(r): j for j, r in enumerate(te)}
            for g in B.stream_groups(idx, te):                    # kit order inside each stream (= fpcalib blend)
                k[[pos[int(r)] for r in g]] = FC.stream_route_every(proj(Fp[g]), mu, 1)
            ks[arm] = k
        for nets, xt in (("B", ""), ("C", "__ubw")):             # T2-D638's 2 x 2: the nets (D630's B / C) x the router (W21 / W23 space)
            RA = O / "t2_d630" / nets
            for fam, (model, al, rec, tag, r) in FAM.items():
                for s in SEEDS:
                    pu = np.load(RA / f"cnn_{fam}" / "runs" / f"{fold}__{model}__{al}__n00__{rec}tv8cpu{xt}__s{s}.npz")
                    assert np.array_equal(pu["test_idx"], te) and list(pu["users"]) == spaces["w21"][2], (nets, fold, fam, s)
                    sh, Pu = pu["prob_last"].astype(np.float64), pu["prob_users_test"].astype(np.float64)
                    name = f"{fold}__{model}__{al}__n00__{rec}tv8cfb{tag}__s{s}.npz"                     # written WITHOUT the __ubw tag
                    ref = np.load(RA / f"route_{fam}" / f"{fold}__{model}__{al}__n00__{rec}tv8cfb{tag}{xt}__s{s}.npz")
                    for arm, k in ks.items():
                        p = FC.blend_route(sh, Pu, k, r)
                        if arm == "w21":                       # the identity: W21's router == D630's own route files of these nets
                            check[nets] = max(check[nets], float(np.abs(p - ref["prob_last"].astype(np.float64)).max()))
                        d = OUT / f"route_{nets.lower()}{arm}" / f"route_{fam}"
                        d.mkdir(parents=True, exist_ok=True)
                        res = dict(json.loads(str(pu["res"])), routed=f"blend r={r} ({'W21' if arm == 'w21' else 'W23 session-1'} space; nets {nets}; T2-D638)")
                        np.savez_compressed(d / name, test_idx=te, prob_last=p.astype(np.float32), val_idx=ref["val_idx"],
                                            prob_val_last=ref["prob_val_last"], prob_best=p.astype(np.float32), res=json.dumps(res))
                        n_out += 1
        moved = int((ks["w21"] != ks["w23"]).sum())
        print(f"[d638 routes] {fold}: {len(te)} test windows, route moved on {moved}", flush=True)
    if max(check.values()) > 1e-6:
        raise SystemExit(f"STOP: the W21 recomputation does not reproduce D630's route files (max |dp| {check})")
    print(f"[d638 routes] W21 recomputation == D630's B / C route files: max |dp| {check}; {n_out} files written under {OUT}/route_<b|c><w21|w23>", flush=True)


if __name__ == "__main__":
    main()

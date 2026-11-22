#!/usr/bin/env python
"""T2 fit audit, CAPACITY PROBES for the CNN families of M7 (lead/FIT_AUDIT_2026-11-09/BRIEF.md §4; pre-registered labels there): one seed (0), the kit fold
(train = P01 P02 P03 P06 P07 P10, val = P05 P08), each family's own alignment, per epoch the CLEAN train BA / CE in eval mode (ni26.train.FIT_AUDIT) + val cell BA.
  --mode shipped : the family's shipped recipe as is (aug / vp / vr / rc / LS / wd as M7 trains it), 180 epochs (its schedule)
  --mode off     : regularisers OFF = r1 with no aug / vp / vr / rc / mixup, wd 0, LS 0, dropout 0, no early stop; 3x the shipped epochs (540)
  --width 2      : (only if 'off' plateaus < 0.95) the family's width x2
Writes outputs/neural_interface_challenge/lead/FIT_AUDIT_2026-11-09/t2_probe_<family>_<mode>[_w2].json (the curves).
    gpu-submit ... python scripts/local/t2_fit_probe.py --family fbc|sh|ifn --mode shipped|off [--width 2]
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tracks" / "t2_bci" / "scripts"))
import ni26.train as T  # noqa: E402
from t2p_bench import aligned_all, folds, load_export, vp_bank  # noqa: E402

FAM = {"fbc": ("fbcnet_bboth", "none", "r1_pulxx_vp2_e180_vr_rc75"), "sh": ("shallow", "ea", "r1_pulx_vp2_e180_vr"), "ifn": ("ifnet", "ea", "r1_pulx_vp2_e180_vr")}
WIDTH = {"fbcnet_bboth": "m", "shallow": "n_filters_time", "ifnet": "F"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAM))
    ap.add_argument("--mode", required=True, choices=["shipped", "off"])
    ap.add_argument("--width", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
    X, idx, meta = load_export(E)
    (_, trm, vam, tem), = folds(idx, "kit")
    model, align, rname = FAM[a.family]
    scale = float(np.median(np.asarray(X[np.flatnonzero(trm)[::7]]).std(-1)))
    XA = aligned_all(X, idx, align, trm, 0.0, 0.05, scale)
    tr, va, te = np.flatnonzero(trm), np.flatnonzero(vam), np.flatnonzero(tem)
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    users = sorted(idx.iloc[tr]["subject"].unique())
    data = dict(Xtr=XA[tr], ytr=y[tr], Xva=XA[va], yva=y[va], cva=cells[va], Xte=XA[te], yte=y[te], cte=cells[te],
                utr=idx.iloc[tr]["subject"].map({u: k for k, u in enumerate(users)}).to_numpy(), str=idx.iloc[tr]["session"].astype(str).to_numpy())
    if a.mode == "shipped":
        recipe = T.RECIPES[rname]
        if recipe.vp:
            T.VP_BANK.update({k: v.to("cuda") for k, v in vp_bank(list(meta["ch_names"]), K=recipe.vp_k, sigma=0.005 * recipe.vp, gain=recipe.vp_g,
                                                                  shift=recipe.vp_s, vref=recipe.vp_r, cap=recipe.vp_cap).items()})
        T.FIT_AUDIT.update(clean=True)
    else:
        recipe = replace(T.RECIPES["r1"], name="fit_off", epochs=3 * T.RECIPES[rname].epochs, wd=0.0, label_smoothing=0.0)
        assert not (recipe.vp or recipe.rc or recipe.mixup or recipe.tilt or recipe.gain_jit or recipe.tshift or recipe.chdrop or recipe.ema)
        T.FIT_AUDIT.update(clean=True, dropout0=True)
    if a.width != 1:
        kw = WIDTH[model]
        name, extra = T.MODEL_SPECS[model]
        base = extra.get(kw)
        assert base is not None, f"set the default {kw} of {model} in MODEL_SPECS before a width probe"
        T.MODEL_SPECS[model] = (name, dict(extra, **{kw: base * a.width}))
    r = T.fit_eval(model, data, recipe, a.seed, device="cuda", sfreq=float(meta["sfreq"]), log_every=10 ** 6)
    c = r["curves"]
    n = len(c); k = max(1, int(round(0.2 * n)))
    ba = np.array([e["clean_train_ba"] for e in c]); ce = np.array([e["clean_train_ce"] for e in c]); va_ = np.array([e["val_cell_bacc"] for e in c])
    summ = dict(family=a.family, model=model, align=align, mode=a.mode, width=a.width, recipe=recipe.name, epochs=n, n_params=r["n_params"],
                clean_ba_last=float(ba[-1]), clean_ce_last=float(ce[-1]), clean_ba_max=float(ba.max()), d_ba_last20=float(ba[-1] - ba[-k - 1]),
                d_ce_last20=float(ce[-1] - ce[-k - 1]), val_best=float(va_.max()), val_best_ep=int(va_.argmax()) + 1, val_last=float(va_[-1]),
                clean_ba_at_val_best=float(ba[int(va_.argmax())]))
    out = Path.home() / "workspace/outputs/neural_interface_challenge/lead/FIT_AUDIT_2026-11-09" / f"t2_probe_{a.family}_{a.mode}{'' if a.width == 1 else f'_w{a.width}'}.json"
    json.dump(dict(summary=summ, curves=c), open(out, "w"), indent=1)
    print("== " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()

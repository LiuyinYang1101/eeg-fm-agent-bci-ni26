#!/usr/bin/env python
"""T2 fit audit, ZERO-GPU part (lead/FIT_AUDIT_2026-11-09/BRIEF.md; the user's ask relayed 2026-11-09 22:4x): the per-epoch curves already saved in the
bench result files of M7's member recipes (the pooled nets; 12 seeds x kit / seen1). These curves are the TRAINING-TIME numbers: CNN train_loss /
train_acc on AUGMENTED batches (vp / vr / rc) with label smoothing 0.1 (a loss floor > 0 by construction), ST-L train_loss under mixup 0.4 + LS 0.1
(no train acc); val_cell_bacc per epoch on the kit val (P05 P08). Not clean train fit (that is the probe's job). Per family / protocol (median over seeds):
last-epoch train loss / acc, the change of train acc (CNN) or loss (ST-L) over the last 20 % of epochs, best val (epoch) and last val.
    mem-guard python scripts/local/t2_fit_audit_logs.py
"""
import glob
import json
from pathlib import Path

import numpy as np

S = Path.home() / "workspace/outputs/neural_interface_challenge/sofia_s1"
FAM = {
    "FBCNet (fbc_tv8c pooled)": "ni26-t2p-d296r-fbc-{p}-5090/runs/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75__s*.npz",
    "ShallowNet-EA (sh_tv8c pooled)": "ni26-t2p-d296r-sh-5090/runs/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr__s*.npz",
    "IFNet-EA (ifn pooled)": "ni26-t2p-d345r-ifnet-s168-5090/runs/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vr__s*.npz",
    "ST-L LP-FT lp10 (30 ep)": "ni26-t2p-d345r-lp10-{p}-*/runs/{p}__fm-st_large-mean__*__s*.npz",
    "ST-L 100 ep (e100f0-f3)": "ni26-t2p-d379-e100-{p}-*/runs/{p}__fm-st_large-mean__*__s*.npz",
}
out = {}
for fam, pat in FAM.items():
    for p in ("kit", "seen1"):
        fs = sorted(glob.glob(str(S / pat.format(p=p))))
        rows = []
        for f in fs:
            r = json.loads(str(np.load(f)["res"]))
            c = r.get("curves", [])
            if not c:
                continue
            n = len(c)
            k = max(1, int(round(0.2 * n)))
            tl = np.array([e["train_loss"] for e in c])
            ta = np.array([e.get("train_acc", np.nan) for e in c])
            va = np.array([e.get("val_cell_bacc", np.nan) for e in c])
            rows.append(dict(n=n, tl=tl[-1], ta=ta[-1], d_ta=ta[-1] - ta[-k - 1] if np.isfinite(ta).all() else np.nan, d_tl=tl[-1] - tl[-k - 1],
                             vbest=np.nanmax(va), ebest=int(np.nanargmax(va)) + 1, vlast=va[-1], d_va=va[-1] - va[-k - 1]))
        if not rows:
            print(f"{fam} / {p}: no curves ({len(fs)} files)")
            continue
        med = {k_: float(np.nanmedian([r_[k_] for r_ in rows])) for k_ in rows[0]}
        out[f"{fam} | {p}"] = dict(seeds=len(rows), **med)
        print(f"{fam:32s} {p:5s} seeds {len(rows):2d} ep {int(med['n']):3d} | last train loss {med['tl']:.3f} acc {med['ta']:.3f} "
              f"| last-20% d(acc) {med['d_ta']:+.3f} d(loss) {med['d_tl']:+.3f} | val best {med['vbest']:.3f} @ep {int(med['ebest'])} last {med['vlast']:.3f} "
              f"(last-20% d {med['d_va']:+.3f})")
json.dump(out, open(Path.home() / "workspace/outputs/neural_interface_challenge/lead/FIT_AUDIT_2026-11-09/t2_logs.json", "w"), indent=1)

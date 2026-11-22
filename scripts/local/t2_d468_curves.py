#!/usr/bin/env python
"""T2-D468: add 'curves_trainval' to the recipe page's T2.json (mean ± sd over seeds of the clean per-epoch training BA and the held-out
val BA; same metric for both: cell-averaged balanced accuracy, raw argmax, no augmentation / mixup / views).
    mem-guard python scripts/local/t2_d468_curves.py
"""
import glob
import json
from pathlib import Path

import numpy as np

O = Path.home() / "workspace/outputs/neural_interface_challenge"
p = O / "lead/recipe_explainer_2026-11-12/T2.json"
T = json.loads(p.read_text())
out, missing = [], []
for size, name in (("st_large", "ST-EEGFormer-large + CPT"), ("st_base", "ST-EEGFormer-base + CPT"), ("st_small", "ST-EEGFormer-small + CPT")):
    fs = sorted(glob.glob(f"{O}/sofia_s1/ni26-t2p-d468-*/runs/kit__fm-{size}-*__s60[0-2].npz"))
    if not fs:
        missing.append(name)
        continue
    tr, va, te = [], [], []
    for f in fs:
        c = json.loads(str(np.load(f, allow_pickle=True)["res"]))["curves"]
        c = json.loads(c) if isinstance(c, str) else c
        tr.append([x["train_cell_bacc_clean"] for x in c])
        va.append([x["val_cell_bacc"] for x in c])
        te.append([x.get("test_cell_bacc") for x in c])
    tr, va = np.array(tr), np.array(va)
    tep = [(e + 1, round(float(np.mean([t[e] for t in te])), 4)) for e in range(len(te[0])) if te[0][e] is not None]
    out.append(dict(size=name, x="epoch", n_seeds=len(fs), protocol="kit (train P01-P03, P06, P07, P10; val P05, P08; test P04, P09)",
                    epoch=list(range(1, tr.shape[1] + 1)),
                    train_bal_acc_clean=np.round(tr.mean(0), 4).tolist(), train_bal_acc_clean_sd=np.round(tr.std(0), 4).tolist(),
                    val_bal_acc=np.round(va.mean(0), 4).tolist(), val_bal_acc_sd=np.round(va.std(0), 4).tolist(),
                    test_bal_acc_every5=tep,
                    what="Same metric on both curves: cell-averaged balanced accuracy of one network, raw argmax. Training curve = a clean extra pass over the training windows after each epoch (no mixup, augmentation or views); training itself used mixup 0.4 and re-colouring.",
                    source=f"T2-D468; {Path(fs[0]).parent.parent.name}"))
T["curves_trainval"] = out
T["curves_trainval_note"] = ("Training accuracy is not logged by the foundation-model fine-tune by default (it logs loss, validation each epoch and test every 5 epochs); "
                             "these curves come from a dedicated diagnostic run with an extra clean pass." + (f" Missing (re-running): {', '.join(missing)}." if missing else ""))
p.write_text(json.dumps(T, indent=1))
print("curves_trainval:", [o["size"] for o in out], "missing:", missing)

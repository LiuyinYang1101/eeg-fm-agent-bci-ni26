#!/usr/bin/env python
"""T2-D196 diagnosis (CPU, zero compute): why the vp2 kit twin replays 0.5301 vs the non-vp kit twin's 0.5677 (b012b1a1). Per package, each
CNN family's POOLED nets (seed-averaged softmax, no copies, no prior) scored on the kit test rows (cell-averaged balanced accuracy), per seed and
pooled; plus per test user.
  python scripts/local/t2_d196_diag.py <pkg dir> [<pkg dir> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
from ni26 import metrics, train as T  # noqa: E402

E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    X = np.load(E / "X.npy", mmap_mode="r")
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    subj = idx["subject"].to_numpy()
    te = np.flatnonzero((idx["split"] == "test").to_numpy())
    trm = idx["split"].isin(["train", "val"]).to_numpy()
    for pkg in sys.argv[1:]:
        cfg = json.load(open(Path(pkg) / "config.json"))
        out = {}
        for fam, wf, model, al in (("fbcnet", "weights.pt", cfg["model"], cfg["align"]), ("shallow", "weights2.pt", cfg["cnn2"]["model"], cfg["cnn2"]["align"])):
            XA = B.aligned_all(X, idx, al["mode"], trm, float(al.get("n0", 0.0)), float(al.get("shrink", 0.05)), float(al["scale"]))
            Xt = torch.as_tensor(np.ascontiguousarray(XA[te]))
            P = []
            for sd in torch.load(Path(pkg) / wf, map_location="cpu", weights_only=False):
                net = T.build_model(model, X.shape[1], 3, X.shape[2], float(cfg["sfreq"]))
                net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
                P.append(T.predict_proba(net.eval(), Xt))
            P = np.stack(P)
            per = [metrics.cell_balanced_accuracy(y[te], p.argmax(1), cells[te])[0] for p in P]
            pool = metrics.cell_balanced_accuracy(y[te], P.mean(0).argmax(1), cells[te])[0]
            users = {u: float(metrics.cell_balanced_accuracy(y[te][m], P.mean(0)[m].argmax(1), cells[te][m])[0])
                     for u in sorted(set(subj[te])) for m in [subj[te] == u]}
            out[fam] = dict(per_seed=[round(float(v), 4) for v in per], pooled=round(float(pool), 4), users={k: round(v, 4) for k, v in users.items()})
        print(Path(pkg).name, cfg.get("recipe"), json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

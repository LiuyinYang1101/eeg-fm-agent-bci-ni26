#!/usr/bin/env python
"""T2-D196 diagnosis (local GPU, through gpu-submit): which member carries the vp2 kit twin's replay drop (0.5301 vs b012b1a1's 0.5677)?
Builds four package dirs of symlinks under outputs/t2_d196/swap/ = {CNN files (config.json, weights*.pt) of D167 | D196} x {ST-L fm_0.pt of
D167 | D196}; the MOR fm_1.pt is byte-identical in both; submission.py = the D196 template (clean differ 0 vs D167's on its own package).
Each is scored on the kit test streams through the template as the kit feeds it (B = 1, kit order, fresh deepcopy + reset_state per stream):
cell-averaged balanced accuracy of the argmax. The unswapped pairs must reproduce 0.5677 / 0.5301.
  gpu-submit python scripts/local/t2_d196_swap.py        (GPU)
  DEV=cpu mem-guard python scripts/local/t2_d196_swap.py (CPU: the eager path; the unswapped pairs check the numerics)
"""
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit, metrics  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
PK = {"d167": O / "t2_d196/diag_d167", "d196": O / "sofia_s1/ni26-t2p-repack-d196-kit-1531390/t2p_d196_kit_r"}


def build(cnn, st):
    d = O / f"t2_d196/swap/cnn_{cnn}__st_{st}"
    d.mkdir(parents=True, exist_ok=True)
    src = {"config.json": PK[cnn], "weights.pt": PK[cnn], "weights2.pt": PK[cnn], "weights_pu.pt": PK[cnn], "weights2_pu.pt": PK[cnn],
           "fm_0.pt": PK[st], "fm_1.pt": PK["d196"], "submission.py": PK["d196"]}
    for f, s in src.items():
        if (d / f).is_symlink() or (d / f).exists():
            (d / f).unlink()
        os.symlink(s / f, d / f)
    return d


def main():
    dev = os.environ.get("DEV", "cuda")
    kit.use_track("bci_decoding")
    meta = json.loads((E / "meta.json").read_text())
    X = np.load(E / "X.npy", mmap_mode="r")
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    te = idx[idx["split"] == "test"]
    groups = [g.sort_values("order")["i"].to_numpy() for _, g in te.groupby("stream", sort=False)]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    res = {}
    for cnn in ("d167", "d196"):
        for st in ("d167", "d196"):
            d = build(cnn, st)
            spec = importlib.util.spec_from_file_location(f"t2_swap_{cnn}_{st}", d / "submission.py")
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            m = mod.Solver.__new__(mod.Solver).load_model(dict(submission_dir=d, device=dev, ch_names=list(meta["ch_names"]), sfreq=meta["sfreq"],
                                                               n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3))
            rows, pred = [], []
            with torch.inference_mode():
                for g in groups:
                    mm = copy.deepcopy(m); mm.reset_state()
                    for r in g:
                        pred.append(int(mm.predict(torch.as_tensor(np.asarray(X[int(r)], np.float32)[None], device=dev))[0]))
                        rows.append(int(r))
            rows = np.asarray(rows)
            ba = metrics.cell_balanced_accuracy(y[rows], np.asarray(pred), cells[rows])[0]
            res[f"cnn_{cnn}__st_{st}"] = round(float(ba), 4)
            print(f"CNN {cnn} + ST-L {st}: kit test cellBA {ba:.4f} ({len(rows)} windows)", flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
    (O / f"t2_d196/swap/read_{dev}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()

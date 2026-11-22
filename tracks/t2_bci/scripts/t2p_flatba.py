#!/usr/bin/env python
"""Track 2: balanced accuracy of packaged T2 models when one channel is FLAT (all-zero) for a whole stream, each channel in
turn, vs clean (the lead's robustness item for T2-D116: agreement with clean is not accuracy). Streamed like Codabench
(deepcopy + reset_state per stream, B = 1). Scored on the kit TEST streams (P04 + P09; held out for --train kit packages).

  python t2p_flatba.py --zips A.zip,B.zip --labels a,b --export <export_proteus_v1> --out O.json [--channels Fz,C3,...]
"""

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import kit, metrics  # noqa: E402
from t2p_repack import load, run_stream  # noqa: E402


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default="Fz,FC3,C3,Cz,C4,CP4,Pz,PO7")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    E = Path(args.export)
    meta_e = json.loads((E / "meta.json").read_text())
    names = list(meta_e["ch_names"])
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    y = idx["y"].to_numpy().astype(int)
    cells = idx["cell"].to_numpy()
    streams = [g.sort_values("order") for _, g in idx[idx["split"] == "test"].groupby("stream", sort=False)]
    kit.use_track("bci_decoding")
    meta = dict(device=args.device, ch_names=names, sfreq=meta_e["sfreq"], n_chans=len(names), n_times=int(X.shape[-1]), n_classes=3)
    conds = ["clean"] + args.channels.split(",")
    res = {}
    tmp = Path(tempfile.mkdtemp(prefix="t2p_flatba_"))
    for zp, lab in zip(args.zips.split(","), args.labels.split(",")):
        d = tmp / lab
        with zipfile.ZipFile(zp) as z:
            z.extractall(d)
        m = load(d, lab, meta)
        res[lab] = {}
        for c in conds:
            rows, preds = [], []
            for g in streams:
                xs = np.asarray(X[g["i"].to_numpy()], np.float32).copy()
                if c != "clean":
                    xs[:, names.index(c)] = 0.0
                p, _ = run_stream(m, xs, args.device)
                rows.append(g["i"].to_numpy())
                preds.append(p)
            r, p = np.concatenate(rows), np.concatenate(preds)
            res[lab][c] = dict(cellba=float(metrics.cell_balanced_accuracy(y[r], p, cells[r])[0]),
                               pooled=float(metrics.pooled_balanced_accuracy(y[r], p)))
            print(f"[flatba] {lab:8s} {c:6s} cell-BA {res[lab][c]['cellba']:.4f} pooled {res[lab][c]['pooled']:.4f}", flush=True)
    labs = args.labels.split(",")
    summ = {lab: dict(clean=res[lab]["clean"]["cellba"],
                      flat_mean=float(np.mean([res[lab][c]["cellba"] for c in conds[1:]])),
                      flat_min=float(np.min([res[lab][c]["cellba"] for c in conds[1:]]))) for lab in labs}
    for lab in labs:
        s = summ[lab]
        s["drop_mean"], s["drop_worst"] = s["clean"] - s["flat_mean"], s["clean"] - s["flat_min"]
        print(f"[flatba] {lab}: clean {s['clean']:.4f}, one flat channel mean {s['flat_mean']:.4f} (drop {s['drop_mean']:+.4f}), "
              f"worst {s['flat_min']:.4f} (drop {s['drop_worst']:+.4f})", flush=True)
    Path(args.out).write_text(json.dumps(dict(zips=args.zips, labels=labs, channels=conds[1:], results=res, summary=summ), indent=1))


if __name__ == "__main__":
    main()

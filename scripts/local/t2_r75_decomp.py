#!/usr/bin/env python
"""Row 75: decompose the twin's kit-test replay vs T2-W7 (0.5799 -> 0.5722) into MOR removal and the CNN test-time averaging, on THIS GPU, with
the repack's own load / run_stream over every kit-test stream (cell BA as t2p_repack computes kit_test_cellba). Package dirs: W7 (MOR, tv off)
and the twin's config variants (cnn_tta_vp k 1 = MOR-free tv off; copies false = tv8; as built = tv8c), made by symlinks + an edited config.
  gpu-submit python scripts/local/t2_r75_decomp.py --export <export_proteus_v1> --w7 <W7 dir> --twin <twin dir> --work <scratch dir>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_repack as R  # noqa: E402
from ni26 import kit, metrics  # noqa: E402


def variant(twin, work, name, edit):
    d = Path(work) / name
    d.mkdir(parents=True, exist_ok=True)
    for f in Path(twin).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f)
    c = json.loads((Path(twin) / "config.json").read_text())
    edit(c)
    (d / "config.json").write_text(json.dumps(c, indent=1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--w7", required=True)
    ap.add_argument("--twin", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    kit.use_track("bci_decoding")
    meta = dict(device="cuda", ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]),
                n_times=int(X.shape[-1]), n_classes=3)
    test = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False) if g["split"].iloc[0] == "test"]
    y, cells = idx["y"].to_numpy().astype(int), idx["cell"].to_numpy()
    pk = {}
    dirs = {"W7 (MOR, tv off)": Path(a.w7),
            "MOR-free, tv off": variant(a.twin, a.work, "nomor_tv0", lambda c: c.pop("cnn_tta_vp")),
            "MOR-free, tv8": variant(a.twin, a.work, "nomor_tv8", lambda c: c["cnn_tta_vp"].update(copies=False)),
            "MOR-free, tv8c (the twin)": Path(a.twin)}
    for name, d in dirs.items():
        m = R.load(d, d.name, meta)
        p = np.full(len(idx), -1)
        for g in test:
            pr, _ = R.run_stream(m, np.asarray(X[g["i"].to_numpy()], np.float32), "cuda")
            p[g["i"].to_numpy()] = pr
        te = p >= 0
        pk[name] = p
        ba = metrics.cell_balanced_accuracy(y[te], p[te], cells[te])[0]
        print(f"{name:28s}: kit-test cell BA {ba:.4f}" + (f"; argmax differs from W7 on {int((p[te] != pk['W7 (MOR, tv off)'][te]).sum())} / {int(te.sum())}"
              if name != "W7 (MOR, tv off)" else ""), flush=True)
        del m


if __name__ == "__main__":
    main()

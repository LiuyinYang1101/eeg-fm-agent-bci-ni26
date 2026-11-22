#!/usr/bin/env python
"""Row 75: per-window predict time of package dirs on THIS GPU (the repack's own load / run_stream, one kit-test stream, after a warm-up
stream), to calibrate the local 5090 against sofia's H200 (T2-W7: 13.11 ms / window there) before the twin's sofia run.
  gpu-submit python scripts/local/t2_r75_time.py --export <export_proteus_v1> DIR [DIR ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts")); sys.path.insert(0, str(ROOT / "src"))
import t2p_repack as R  # noqa: E402
from ni26 import kit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    E = Path(a.export)
    meta_e = json.loads((E / "meta.json").read_text())
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    X = np.load(E / "X.npy", mmap_mode="r")
    kit.use_track("bci_decoding")
    meta = dict(device="cuda", ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]),
                n_times=int(X.shape[-1]), n_classes=3)
    test = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False) if g["split"].iloc[0] == "test"]
    warm, timed = np.asarray(X[test[0]["i"].to_numpy()[:40]], np.float32), np.asarray(X[test[1]["i"].to_numpy()], np.float32)
    for d in a.dirs:
        m = R.load(Path(d), Path(d).name, meta)
        R.run_stream(m, warm, "cuda")
        t0 = time.time()
        R.run_stream(m, timed, "cuda")
        print(f"{Path(d).name}: {1e3 * (time.time() - t0) / len(timed):.2f} ms / window ({len(timed)} windows)", flush=True)
        del m


if __name__ == "__main__":
    main()

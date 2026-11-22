#!/usr/bin/env python
"""T2-D484 (the CAR-view CPT): t2p_cpt --car on tiny synthetic pools, CPU, st_small:
  1. Pool.draw with car on = ni26.stream.car_windows of the plain draw, bitwise (a flat channel stays out of the mean); car off = the plain draw;
     a longer (wide) window is re-referenced before its 4 s crop (the crop of the CAR'd window);
  2. the CLI: --car runs 2 steps, every pool's summary entry carries car true, status ok, and t2p_fm.use_cpt accepts the checkpoint;
     --car with --store-split is refused.
  python tests/test_t2p_cpt_car.py  ->  "OK"
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.stream import car_windows  # noqa: E402
PROTEUS_CH = ("C2 CP2 P2 Pz CPz Cz C1 CP1 P1 P3 PO7 P5 TP7 CP5 CP3 C3 PO8 P6 P4 TP8 CP6 CP4 C6 C4 FC6 FC4 FC2 F6 F4 F2 AF4 AF3 F5 F3 F1 Fz "
              "FC5 FC3 FC1 FCz C5").split()


def export(d, X, idx, ch, fs):
    d.mkdir(parents=True)
    np.save(d / "X.npy", X.astype(np.float32))
    idx.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps(dict(sfreq=fs, n_times=X.shape[-1], ch_names=ch)))


def main():
    import t2p_cpt
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        rows = [(f"Dreyer2026Proteus/P{u:02d}", str(s)) for u in range(1, 11) for s in ((1, 2, 3) if u in (4, 10) else (1,)) for _ in range(4)]
        ip = pd.DataFrame(rows, columns=["subject", "session"])
        ip["split"] = np.where(ip.subject.str.endswith(("P04", "P09")), "test", np.where(ip.subject.str.endswith(("P05", "P08")), "val", "train"))
        X = (rng.normal(size=(len(ip), 41, 480)) * 10 + 3).astype(np.float32)
        X[5, 7] = 2.5                                                    # a flat channel in window 5
        export(T / "export_proteus_v1", X, ip, PROTEUS_CH, 120.0)
        Xw = (rng.normal(size=(30, 20, 720)) * 5).astype(np.float32)   # a wide (6 s) pool: re-referenced before the crop
        export(T / "wide", Xw, pd.DataFrame(dict(subject=[f"s{k}" for k in range(30)])), PROTEUS_CH[:20], 120.0)
        p = t2p_cpt.Pool(T / "export_proteus_v1", np.arange(len(ip)), "proteus_all")
        sel = np.array([9, 5, 2, 30])
        plain = p.draw(sel, np.random.default_rng(1))
        assert np.array_equal(plain, X[np.sort(sel)])
        p.car = True
        assert np.array_equal(p.draw(sel, np.random.default_rng(1)), car_windows(X[np.sort(sel)])), "car draw = car_windows, bitwise"
        c5 = p.draw(np.array([5]), np.random.default_rng(1))[0]
        good = np.setdiff1d(np.arange(41), [7])
        assert np.abs(c5[good].mean(0)).max() < 1e-4 and np.allclose(c5[7], 2.5 - X[5, good].mean(0), atol=1e-4), "the flat channel stays out of the mean"
        pw = t2p_cpt.Pool(T / "wide", np.arange(30), "extra_unlabelled")
        pw.car = True
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        got = pw.draw(np.array([3, 4]), r1)
        o = r2.integers(0, 720 - 480 + 1, size=2)
        ref = np.stack([xi[:, oi:oi + 480] for xi, oi in zip(car_windows(Xw[[3, 4]]), o)])
        assert np.array_equal(got, ref), "a wide window: CAR first, then the crop"
        out = T / "out"
        cmd = [sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_cpt.py"), "--fm", "st_small", "--proteus", str(T / "export_proteus_v1"),
               "--proteus-pool", "all", "--extra", str(T / "wide"), "--out", str(out), "--steps", "2", "--batch", "4", "--accum", "1",
               "--eval-every", "100", "--car"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        sm = json.loads((out / "cpt_summary.json").read_text())
        assert sm["status"] == "ok" and sm["args"]["car"] is True, sm["status"]
        assert [q["car"] for q in sm["pools"]] == [True, True], sm["pools"]
        import t2p_fm
        t2p_fm.use_cpt("st_small", out / "checkpoint-cpt.pth", lambda m: None)
        r = subprocess.run([sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_cpt.py"), "--fm", "st_small", "--out", str(T / "o2"),
                            "--store-split", "x.parquet", "--car"], capture_output=True, text=True)
        assert r.returncode != 0 and "--car: PROTEUS / export pools only" in (r.stdout + r.stderr), r.stderr[-800:]
    print("OK")


if __name__ == "__main__":
    main()

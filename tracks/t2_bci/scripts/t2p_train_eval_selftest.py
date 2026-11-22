#!/usr/bin/env python
"""Self-test (T2-D468): fit_eval_fm(train_eval=True) adds row 'train_cell_bacc_clean' each epoch and changes nothing else
(probabilities and every other curve value bitwise equal to train_eval=False). CPU, st_small, rc + mixup + LLRD, 2 epochs.
  python t2p_train_eval_selftest.py -> "OK"
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402
import t2p_fm  # noqa: E402
from t2p_aug_selftest import EXPORT  # noqa: E402


def main():
    _, idx, meta = B.load_export(EXPORT)
    X = np.load(EXPORT / "X.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    tr = np.sort(np.concatenate([rng.choice(np.flatnonzero((idx["subject"] == u).to_numpy()), 12, replace=False) for u in ("P01", "P02")]))
    te = rng.choice(np.flatnonzero((idx["subject"] == "P04").to_numpy()), 8, replace=False)
    y = idx["y"].to_numpy().astype(int)
    Xa = np.asarray(X[np.concatenate([tr, te])], np.float32)
    Xa = Xa / np.median(Xa[:len(tr)].std(-1))
    cells = idx["cell"].to_numpy()
    data = dict(Xtr=Xa[:len(tr)], ytr=y[tr], Xva=Xa[:0], yva=y[:0], cva=cells[:0], Xte=Xa[len(tr):], yte=y[te], cte=cells[te],
                utr=(idx.loc[tr, "subject"] == "P02").to_numpy().astype(int), str=idx.loc[tr, "session"].astype(str).to_numpy(), ctr=cells[tr])
    kw = dict(epochs=2, bs=8, llrd=0.75, mixup=0.4, rc=0.75, calib="gain", device="cpu", log=lambda m: None)
    a = t2p_fm.fit_eval_fm("st_small", data, 0, meta["ch_names"], **kw)
    b = t2p_fm.fit_eval_fm("st_small", data, 0, meta["ch_names"], train_eval=True, **kw)
    assert np.array_equal(a["prob_last"], b["prob_last"])
    for ra, rb in zip(a["curves"], b["curves"]):
        assert "train_cell_bacc_clean" in rb and "train_cell_bacc_clean" not in ra
        assert {k: v for k, v in rb.items() if k != "train_cell_bacc_clean"} == ra, (ra, rb)
        assert 0.0 <= rb["train_cell_bacc_clean"] <= 1.0
    print("train_eval: probabilities and curves unchanged; clean train BA per epoch", [round(r["train_cell_bacc_clean"], 3) for r in b["curves"]])
    print("OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Track1's T2 experiment XDAY: t2p_bench (read-only, imported) with four extra protocols on the combined export
(t1seat_t2_xday_export.py). Everything else is t2p_bench's own code and arguments.
  xd   train = every PROTEUS window + Scherer DAY 1 of all nine users; test = Scherer DAY 2 (nine returning users, cross-day); no val
  xs   as xd, but only the first 80 day-1 windows of each Scherer user train; test = their day-1 windows 80+ (read 4's selection split)
  xh0 / xh1 / xh2   as xd, but Scherer users (S1 S4 S7) / (S2 S5 S8) / (S3 S6 S9) are held out of training; test = Scherer DAY 2 of all nine
  xl0 / xl1 / xl2   as xh, test = Scherer DAY 2 of all nine + DAY 1 of the three held-out users (read 5: strangers; run with copies)
  kx / kxh   C19 (any combined export): the kit split with Scherer's windows (all / every second) added to training
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402

HOLD = {"xs": (), "xd": (), "xh0": ("S1", "S4", "S7"), "xh1": ("S2", "S5", "S8"), "xh2": ("S3", "S6", "S9"),
        "xl0": ("S1", "S4", "S7"), "xl1": ("S2", "S5", "S8"), "xl2": ("S3", "S6", "S9")}
_folds = B.folds


def folds(idx, protocol):
    if protocol in ("kx", "kxh"):   # C19: the kit split + Scherer (both days, nine users) in training; kxh = every second Scherer window
        sub = idx["subject"].astype(str).to_numpy()
        sch = np.char.startswith(sub.astype(str), "S")
        (_, trn, va, te), = _folds(idx, "kit")
        assert sch.sum() == 2130 and not (trn | va | te)[sch].any()
        add = sch if protocol == "kx" else sch & (idx["order"].to_numpy() % 2 == 0)
        return [(protocol, trn | add, va, te)]
    if protocol not in HOLD:
        return _folds(idx, protocol)
    sub, ses = idx["subject"].astype(str).to_numpy(), idx["session"].astype(str).to_numpy()
    sch = np.char.startswith(sub.astype(str), "S")
    assert sch.sum() == 2130 and len(set(sub[sch])) == 9, "not the combined export"
    trn = ~sch | (sch & (ses == "1") & ~np.isin(sub, HOLD[protocol]))
    te = sch & (ses == "2")
    if protocol == "xs":       # read 4: the selection split INSIDE day 1 (one run per day: no leave-run-out): windows 0-79 train, 80+ scored
        late = idx["order"].to_numpy() >= 80
        trn, te = trn & ~(sch & late), sch & (ses == "1") & late
    if protocol.startswith("xl"):   # read 5: the held-out users' DAY 1 is scored too (a stranger's first stream)
        te = te | (sch & (ses == "1") & np.isin(sub, HOLD[protocol]))
    assert not (trn & te).any()
    return [(protocol, trn, np.zeros(len(idx), bool), te)]


B.folds = folds
if __name__ == "__main__":
    B.main()

#!/usr/bin/env python
"""Self-test (T2-D277): t2p_cpt.pool_keep and --extra-proteus-pool.

  1. the refactor: for the named PROTEUS pools (seen1, seen1_out, all), pool_keep on export_proteus_v1's index selects exactly the rows of
     the inline rule it replaced (T2-D111 iv), so every earlier CPT's PROTEUS pool is unchanged;
  2. the seen1 rule on session-crop-style rows (subject 'Dreyer2026Proteus/Pxx', the export's session strings): P01-P03 / P05-P08 every
     session, P04 / P10 session 1 only, P09 never; a seen1 test stream in a 'seen1' pool raises;
  3. is_proteus_extra: only Dreyer2026Proteus/<id> pools take the rule (job 1537179 applied it to Dreyer2023 too and kept 0 windows);
     --extra-proteus-pool defaults to '' (off: T2-D129's kit CPT path, --extra-exclude-subjects only, unchanged).
  python t2p_cpt_pool_selftest.py -> "OK"
"""

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_cpt as C  # noqa: E402

EXPORT = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    idx = pd.read_csv(EXPORT / "index.csv", dtype={"subject": str, "session": str})
    sub, ses = idx["subject"].str.split("/").str[-1], idx["session"]
    for pool in ("seen1", "seen1_out", "all"):
        old = sub.isin(C.PROTEUS_POOLS[pool]) | (sub.isin(["P04", "P10"]) & (ses == "1") & (pool == "seen1"))
        assert np.array_equal(C.pool_keep(sub, ses, pool), old.to_numpy()), pool
    print("refactor: pool_keep = the T2-D111 (iv) rule on export_proteus_v1 for seen1 / seen1_out / all")
    ie = pd.DataFrame(dict(subject="Dreyer2026Proteus/" + sub, session=ses))     # session-crop-style rows
    k = C.pool_keep(ie["subject"].str.split("/").str[-1].to_numpy(), ie["session"].to_numpy(), "seen1")
    kept = set(zip(sub[k], ses[k]))
    for u in ("P01", "P02", "P03", "P05", "P06", "P07", "P08"):
        assert {(u, s) for s in ses[sub == u].unique()} <= kept, u
    assert {(u, "1") for u in ("P04", "P10")} <= kept and not any(u == "P09" or (u in ("P04", "P10") and s != "1") for u, s in kept), kept
    try:
        C.pool_keep(np.array(["P04", "P09"]), np.array(["2", "1"]), "all")
        bad = pd.Series(["P04"]), pd.Series(["2"])
        C.PROTEUS_POOLS["_t"] = ["P04"]
        C.pool_keep(*bad, "_t")
        raise SystemExit("a seen1 test stream passed")
    except AssertionError:
        pass
    finally:
        C.PROTEUS_POOLS.pop("_t", None)
    print(f"seen1 rule on session-style rows: kept {sorted(kept)}; a seen1 test stream raises")
    dr = pd.DataFrame(dict(subject=["Dreyer2023/1", "Dreyer2023/2"], session=["0", "1"]))         # 1537179's failure: a non-PROTEUS pool
    assert C.is_proteus_extra(ie) and not C.is_proteus_extra(dr) and not C.is_proteus_extra(pd.DataFrame(dict(x=[1])))
    try:
        C.is_proteus_extra(pd.concat([ie.head(2), dr]))
        raise SystemExit("a mixed pool passed")
    except AssertionError:
        pass
    print("is_proteus_extra: the session crops yes; Dreyer2023-style subject / session pools no (untouched); a mixed pool raises")
    assert "\"--extra-proteus-pool\", default=\"\"" in inspect.getsource(C), "the default is off"
    print("--extra-proteus-pool defaults to '' (off)")
    print("OK")


if __name__ == "__main__":
    main()

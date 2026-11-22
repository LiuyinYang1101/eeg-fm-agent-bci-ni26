#!/usr/bin/env python
"""T2-D651 (Track3): t2p_bench.folds' pseudo-returning protocol rtl / rtl_<u>, CPU, on the REAL export index:
  1. 'rtl' = exactly the 8 single-session users' folds rtl_<u> (sorted), each = (name, train, val, test) with val empty;
  2. per fold: test = u's LAST TWO runs (by run number) of its one session, every other row trains (u's earlier runs in training: u is SEEN);
     train and test disjoint; test non-empty; no other user's row in test;
  3. every pre-existing protocol (ss8, rt, their single folds, sd46, lc9, kit) is BITWISE unchanged vs t2p_bench.py at the pinned REF_COMMIT;
  4. rtl_<u> for a multi-session user (P04 / P10) is refused.
  python tests/test_t2_d651_rtl.py  ->  "OK"
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402

REF_COMMIT = "58c41a6d"                     # the last t2p_bench.py before T2-D651 (no rtl)
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    idx = pd.read_csv(EXP / "index.csv", low_memory=False)
    sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    ses, run = idx["session"].astype(str).to_numpy(), idx["run"].to_numpy()
    single = sorted(u for u in np.unique(sub) if len(set(ses[sub == u])) == 1)
    assert len(single) == 8, single
    F = B.folds(idx, "rtl")
    assert [f[0] for f in F] == [f"rtl_{u}" for u in single], [f[0] for f in F]
    for name, trn, va, te in F:
        u = name.split("_", 1)[1]
        runs = np.unique(run[sub == u])
        want = (sub == u) & np.isin(run, runs[-2:])
        assert np.array_equal(te, want) and np.array_equal(trn, ~want) and not va.any(), name
        assert te.any() and not (trn & te).any() and set(sub[te]) == {u}, name
        early = (sub == u) & ~np.isin(run, runs[-2:])
        assert early.any() and trn[early].all(), (name, "u's early runs must train (u is seen)")
    print(f"1-2. rtl = {len(F)} folds; per fold test = the user's last two runs "
          f"({', '.join(f'{f[0]} {int(f[3].sum())}' for f in F)} windows), the early runs train", flush=True)
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_bench_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_bench.py"],
                                      capture_output=True, text=True, check=True).stdout)
        assert "rtl_" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_bench_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_bench_ref"] = OLD
        spec.loader.exec_module(OLD)
        protos = ["ss8", "rt", "ss8_P01", "rt1_P04", "rt2_P10", "sd46", "lc9", "kit"]
        for p in protos:
            a, b = B.folds(idx, p), OLD.folds(idx, p)
            assert [f[0] for f in a] == [f[0] for f in b], p
            for fa, fb in zip(a, b):
                assert all(np.array_equal(x, y) for x, y in zip(fa[1:], fb[1:])), (p, fa[0])
    finally:
        ref.unlink(missing_ok=True)
    print(f"3. {protos}: folds bitwise == t2p_bench.py at {REF_COMMIT}", flush=True)
    for x in ("P04", "P10"):
        try:
            B.folds(idx, f"rtl_{x}")
            raise SystemExit(f"FAIL: rtl_{x} (a multi-session user) was accepted")
        except AssertionError:
            pass
    print("4. rtl_P04 / rtl_P10 refused (multi-session users)")
    print("OK")


if __name__ == "__main__":
    main()

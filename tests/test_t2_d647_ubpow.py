#!/usr/bin/env python
"""T2-D647 (W28 = ALL6-HALF): --user-balance-pow p (weights = n_sessions^-p, self-normalised per batch):
  1. t2p_bench.user_balance_weights: p = 1 == the D630 / D642 expression 1 / n BITWISE (float32); p = 0.5 == n^-0.5;
  2. t2p_final: the flag defaults to 1.0 and p = 1 keeps the literal 1 / n path; user_balance_pow is recorded only when p != 1;
  3. the bench tag: __ubw at p = 1 (unchanged), __ubwp05 at p = 0.5;
  4. t2p_repack's training-only regex catches a template read of user_balance_pow; t2p_merge_donors' SKIP lists it.
  python tests/test_t2_d647_ubpow.py  ->  "OK"
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402


def main():
    sub = np.array(["P04"] * 6 + ["P01"] * 3 + ["P10"] * 4)
    ses = np.array(["1", "1", "2", "2", "3", "3"] + ["1"] * 3 + ["1", "1", "2", "2"])
    ns = {"P04": 3, "P01": 1, "P10": 2}
    w1, wu1 = B.user_balance_weights(sub, ses, 1.0)
    old = np.array([1.0 / ns[u] for u in sub], np.float32)            # the D630 / D642 expression verbatim
    assert w1.dtype == np.float32 and np.array_equal(w1, old), (w1, old)
    w5, wu5 = B.user_balance_weights(sub, ses, 0.5)
    assert np.allclose(w5, [ns[u] ** -0.5 for u in sub]) and abs(wu5["P04"] - 3 ** -0.5) < 1e-12
    print(f"1. p 1 == 1 / n bitwise {wu1}; p 0.5 == n^-0.5 {({k: round(v, 4) for k, v in wu5.items()})}")
    fsrc = (ROOT / "tracks/t2_bci/scripts/t2p_final.py").read_text()
    assert '"--user-balance-pow", type=float, default=1.0' in fsrc and "(1.0 / n if pw_ == 1.0 else float(n) ** (-pw_))" in fsrc
    assert '**({"user_balance_pow": float(args.user_balance_pow)} if wub and args.user_balance_pow != 1.0 else {})' in fsrc
    print("2. t2p_final: default 1.0, the literal 1 / n path at p 1, the member key only when p != 1")
    bsrc = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    line = [ln for ln in bsrc.splitlines() if "T2-D647 (__ubwp05)" in ln][0]
    tag = lambda p: "__ubw" + (f"p{p:g}".replace(".", "") if p != 1.0 else "")   # noqa: E731  (the bench line's expression)
    assert 'f"p{args.user_balance_pow:g}".replace(".", "")' in line and tag(1.0) == "__ubw" and tag(0.5) == "__ubwp05"
    print("3. bench tag: __ubw (p 1, unchanged) / __ubwp05 (p 0.5)")
    rsrc = (ROOT / "tracks/t2_bci/scripts/t2p_repack.py").read_text()
    rx = re.search(r"assert not re\.search\(r'(.+?)', src\)", rsrc).group(1)
    assert re.search(rx, 'x = m["user_balance_pow"]') and re.search(rx, 'm.get("user_balance_pow")'), "the repack regex misses user_balance_pow"
    msrc = (ROOT / "tracks/t2_bci/scripts/t2p_merge_donors.py").read_text()
    assert '"user_balance", "user_balance_pow")' in msrc
    print("4. repack's training-only regex catches user_balance_pow reads; merge_donors' SKIP lists it")
    print("OK")


if __name__ == "__main__":
    main()

"""T2-D410: the lcK learning-curve folds — K training users (session 1 only), balanced cyclic design, test = the other users + sessions
2-3 of a trained P04 / P10, no val."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402


def _idx():
    rows = []
    for k in range(1, 11):
        u = f"Dreyer2026Proteus/P{k:02d}"
        for s in (("1", "2", "3") if k in (4, 10) else ("1",)):
            rows += [dict(subject=u, session=s, split="train")] * 5
    return pd.DataFrame(rows)


def test_lc_folds():
    idx = _idx()
    sub = idx.subject.str[-3:].to_numpy(); ses = idx.session.to_numpy()
    for proto, K in (("lc3", 3), ("lc5", 5), ("lc7", 7), ("lc9", 9)):
        F = B.folds(idx, proto)
        assert len(F) == 10
        trained = Counter()
        for name, tr, va, te in F:
            assert not va.any() and not (tr & te).any()
            tu = set(sub[tr]); assert len(tu) == K, (name, tu)
            trained.update(tu)
            assert (ses[tr] == "1").all()                                       # session 1 only
            for u in set(sub) - tu:                                             # every window of an untrained user is test (new)
                assert te[sub == u].all()
            for u in tu:                                                        # a trained user's later sessions are test (returning)
                k = sub == u
                assert (te[k] == (ses[k] != "1")).all()
            assert (tr | te).all()                                              # every window is used
        assert set(trained.values()) == {K}, (proto, trained)                    # balanced: each user trains in exactly K folds
    assert len({tuple(sorted(set(sub[tr]))) for _, tr, _, _ in B.folds(idx, "lc9")}) == 10   # lc9 = the 10 leave-one-user-out folds

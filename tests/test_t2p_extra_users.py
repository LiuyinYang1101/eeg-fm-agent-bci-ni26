"""T2-D453: --extra-users keeps only the listed external users' labelled windows; '' leaves the corpus unchanged."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks" / "t2_bci" / "scripts"))
import t2p_bench as B  # noqa: E402


def _ext():
    idx = pd.DataFrame(dict(subject=[f"S/{u}" for u in (1, 1, 2, 2, 3, 3)], y=[0, -1, 1, 2, 0, 1]))
    return dict(name="x", X=np.zeros((6, 2, 4)), idx=idx, keep=np.flatnonzero(idx.y.to_numpy() >= 0), names=["a", "b"], sha="")


def test_empty_is_identity():
    e = _ext()
    assert B.extra_users(e, "") is e


def test_filter_keeps_listed_users_labelled_rows():
    assert B.extra_users(_ext(), "1+3")["keep"].tolist() == [0, 4, 5]


def test_unknown_user_refused():
    try:
        B.extra_users(_ext(), "9")
    except AssertionError:
        return
    raise AssertionError("an unknown user must be refused")

"""T2-D437: bag_rows keeps every user, resamples with repeats, and is deterministic per seed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks" / "t2_bci" / "scripts"))
from t2p_bench import bag_rows, boost_mult, boost_rows, expert_combine, expert_mask  # noqa: E402


def _toy():
    utr = np.repeat(np.arange(6), 300)
    grp = np.tile(np.repeat(np.arange(5), 60), 6).astype(str)
    return utr, grp


def test_user_weight_bags_keep_every_user():
    utr, grp = _toy()
    for mode in ("uw1", "uw4"):
        shares = []
        for seed in range(20):
            r = bag_rows(mode, utr, grp, seed)
            assert np.array_equal(r, bag_rows(mode, utr, grp, seed))
            c = np.bincount(utr[r], minlength=6) / len(r)
            assert c.min() > 0.02 and abs(len(r) - len(utr)) <= 6
            shares.append(c)
        assert np.std(shares) > 0.01
    assert np.std([np.bincount(utr[bag_rows("uw1", utr, grp, s)], minlength=6) for s in range(20)]) > \
        np.std([np.bincount(utr[bag_rows("uw4", utr, grp, s)], minlength=6) for s in range(20)])


def test_run_bags_resample_whole_runs_within_user():
    utr, grp = _toy()
    r = bag_rows("runs", utr, grp, 3)
    assert len(r) == len(utr) and len(np.unique(r)) < len(r)
    for u in range(6):
        rows = r[utr[r] == u]
        assert len(rows) == 300
        for g in np.unique(grp[rows]):
            assert (grp[rows] == g).sum() % 60 == 0


def test_boost_rows_follow_capped_multipliers():
    utr, _ = _toy()
    m = boost_mult([0.2, 0.2, 0.3, 0.4, 0.5, 0.9], 5.0)
    assert m[0] == 1.0 and m[-1] == 3.0 and np.all(np.diff(m) >= 0)
    c = np.mean([np.bincount(utr[boost_rows(utr, m, s)], minlength=6) for s in range(50)], 0)
    assert np.allclose(c / c.sum(), m / m.sum(), atol=0.01)
    assert len(boost_rows(utr, m, 0)) == len(utr)


def test_expert_combine_and_mask():
    S = np.array([[0.9, 0.1, 0.2], [0.2, 0.3, 0.8]])
    P = expert_combine(S, "ovr")
    assert np.allclose(P.sum(1), 1) and P.argmax(1).tolist() == [0, 2]
    # pairs (mi-sub, mi-word, sub-word); S[:, k] = p(second class): row 0 says mi beats both, row 1 says word beats both
    Q = expert_combine(np.array([[0.1, 0.1, 0.5], [0.5, 0.9, 0.9]]), "pair")
    assert np.allclose(Q.sum(1), 1) and Q.argmax(1).tolist() == [0, 2]
    assert np.allclose(expert_combine(np.full((1, 3), 0.5), "pair"), 1 / 3)
    ch = ["C3", "Fz", "F5", "TP8", "Pz"] + [c for c in ("C4", "C1", "C2", "Cz", "CP3", "CP4", "FC3", "FC4", "CP1", "CP2", "CPz")]
    assert expert_mask(ch, 0).ravel().tolist()[:5] == [1, 0, 0, 0, 0]

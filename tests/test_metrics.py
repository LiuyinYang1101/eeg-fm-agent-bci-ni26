#!/usr/bin/env python
"""src/ni26/metrics.py against the OFFICIAL kit scorer (review 2026-10-16 T2 P1b).

The kit's ``benchmark_utils/metrics.py`` (the commit pinned in ``ni26.kit.KIT_SHA``) is loaded from the kit clone by
file path; its ``group_scores(balanced_accuracy, …)`` is exactly what ``tracks/bci_decoding/objective.py`` ranks on.

  python tests/test_metrics.py            -> "### ALL PASS", exit 0
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from ni26 import kit, metrics  # noqa: E402


def kit_metrics():
    root = kit.kit_root()
    sha = kit.kit_sha(root)
    assert sha == kit.KIT_SHA, f"kit clone {root} is at {sha}, not the pinned {kit.KIT_SHA} (run scripts/get_kit.sh)"
    spec = importlib.util.spec_from_file_location("kit_metrics", root / "benchmark_utils" / "metrics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_missing_command_cell(K):
    # a sealed-like cell without command 2, where the model still predicts 2 once
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0, 0, 2, 1, 0])
    # union {0, 1, 2}: recalls 2/3, 1/2, 0 -> 7/18; sklearn would drop class 2 -> 7/12
    want = (2 / 3 + 1 / 2 + 0) / 3
    assert abs(metrics.balanced_accuracy(y, p) - want) < 1e-12
    assert abs(K.balanced_accuracy(y, p) - want) < 1e-12
    from sklearn.metrics import balanced_accuracy_score
    assert abs(balanced_accuracy_score(y, p) - (2 / 3 + 1 / 2) / 2) < 1e-12      # the old (sklearn) rule differs


def test_agrees_with_sklearn_when_predictions_in_targets(K):
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(0)
    for _ in range(200):
        y = rng.integers(0, 3, 40)
        y[:3] = [0, 1, 2]
        p = rng.integers(0, 3, 40)
        assert abs(metrics.balanced_accuracy(y, p) - balanced_accuracy_score(y, p)) < 1e-12


def test_cells_match_kit(K):
    rng = np.random.default_rng(1)
    for trial in range(300):
        n = int(rng.integers(5, 120))
        y = rng.integers(0, 3, n)
        p = rng.integers(0, 3, n)
        if trial % 3 == 0:                      # cells that lack a command entirely, or predict only one
            y[y == 2] = 0
        if trial % 5 == 0:
            p[:] = rng.integers(0, 3)
        subj = rng.choice(["P04", "P09", "P10"], n)
        sess = rng.choice(["1", "2", "3"], n)
        ctx = rng.choice(["acq", "online"], n)
        cells = [f"{a}|{b}|{c}" for a, b, c in zip(subj, sess, ctx)]
        ours, per = metrics.cell_balanced_accuracy(y, p, cells)
        theirs = K.group_scores(K.balanced_accuracy, y, p, np.stack([subj, sess, ctx], 1))
        assert len(per) == len(theirs)
        assert abs(ours - float(theirs.mean())) < 1e-12, (trial, ours, float(theirs.mean()))
        assert abs(metrics.pooled_balanced_accuracy(y, p) - K.balanced_accuracy(y, p)) < 1e-12
        sc = metrics.score_all(y, p, subj, sess, ctx)
        assert abs(sc["cell_bacc"] - float(theirs.mean())) < 1e-12


def main():
    K = kit_metrics()
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t(K)
        print(f"PASS {t.__name__}", flush=True)
    print(f"### ALL PASS ({len(tests)} tests, kit {kit.kit_sha()})")


if __name__ == "__main__":
    main()

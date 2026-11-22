"""Track-2 scoring, replicating the official definitions (kit 442edcd ``benchmark_utils/metrics.py`` +
``tracks/bci_decoding/objective.py``).

* Ranking (warm-up and sealed): balanced accuracy computed inside every subject × session × context cell, then
  averaged over cells with equal weight (``balanced_accuracy_cell_mean``). Also reported: the pooled balanced
  accuracy over all windows (``balanced_accuracy``) and accuracy.
* Balanced accuracy = the kit's rule (torchmetrics macro accuracy): the classes are union(y_true, y_pred), and a
  class predicted but absent from the targets counts with recall 0. scikit-learn's ``balanced_accuracy_score``
  leaves such a class out; the two agree whenever every predicted class occurs in the targets (every public cell
  holds all three commands, so every number before the 2026-10-16 switch is unchanged; a sealed cell missing a
  command would not be). ``tests/test_metrics.py`` checks this module against the kit's own function.

Model selection uses ``cell_balanced_accuracy`` (LESSONS A1/A2: select on the ranking statistic itself).
"""

import numpy as np
from sklearn.metrics import accuracy_score


def balanced_accuracy(y_true, y_pred) -> float:
    """The kit's ``benchmark_utils.metrics.balanced_accuracy`` (442edcd), same semantics: mean recall over
    union(y_true, y_pred); a predicted class absent from the targets has recall 0."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.union1d(y_true, y_pred)
    if not len(classes):
        return float("nan")
    recalls = [np.mean(y_pred[y_true == c] == c) if np.any(y_true == c) else 0.0 for c in classes]
    return float(np.mean(recalls))


def pooled_balanced_accuracy(y_true, y_pred) -> float:
    return balanced_accuracy(y_true, y_pred)


def cell_balanced_accuracy(y_true, y_pred, cells) -> tuple[float, dict]:
    """Mean over cells of the within-cell balanced accuracy. ``cells``: one hashable id per window."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cells = np.asarray([str(c) for c in cells])
    per = {}
    for c in np.unique(cells):
        m = cells == c
        per[c] = balanced_accuracy(y_true[m], y_pred[m])
    return float(np.mean(list(per.values()))), per


def score_all(y_true, y_pred, subject, session, context=None) -> dict:
    """Every number we report for one evaluation set."""
    context = np.zeros(len(y_true), int) if context is None else np.asarray(context)
    cells = [f"{a}|{b}|{c}" for a, b, c in zip(subject, session, context)]
    cba, per = cell_balanced_accuracy(y_true, y_pred, cells)
    subj_cells = {}
    for k, v in per.items():
        subj_cells.setdefault(k.split("|")[0], []).append(v)
    return dict(
        cell_bacc=cba,
        pooled_bacc=pooled_balanced_accuracy(y_true, y_pred),
        accuracy=float(accuracy_score(y_true, y_pred)),
        n_windows=int(len(y_true)), n_cells=len(per),
        per_subject=({s: float(np.mean(v)) for s, v in subj_cells.items()}),
    )

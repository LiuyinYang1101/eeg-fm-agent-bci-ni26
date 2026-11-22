"""Read a window cache written by ``scripts/t2_export_windows.py`` and build evaluation protocols.

Protocols (all score with ``ni26.metrics.score_all``; cells = subject × session [× context]):

* ``kit``      — the kit's own split column (warm-up: Dreyer train/val -> test subjects 61-81). Cross-subject.
* ``xsession`` — the SEALED mirror: a group of held-out subjects' LATER sessions/runs is the test set and the
                 training set is everything else, *including those subjects' own early (calibration) sessions*.
                 One population model per fold; predict() never sees the subject id.
* ``within``   — per-subject model trained only on s's early sessions (reference: needs the subject id at
                 test time, so it is not deployable as-is; it bounds what calibration alone buys).

"Early" vs "later" comes from an ordering key (session, then run); ``n_calib`` early units per subject are
calibration. For single-session studies (Dreyer) the unit is the run.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class WindowCache:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.index = pd.read_csv(self.root / "index.csv", dtype={"subject": str, "session": str, "run": str})
        self.X = np.load(self.root / "X.npy", mmap_mode="r")
        assert len(self.X) == len(self.index), "X / index length mismatch"
        self.ch_names = list(self.meta.get("ch_names") or [])

    def labels(self, code_map: dict | None = None):
        """Integer labels. ``code_map`` {code: class} keeps only listed codes (others -> -1)."""
        if code_map is None:
            return self.index["y_kit"].to_numpy()
        return self.index["code"].astype(str).map({str(k): v for k, v in code_map.items()}).fillna(-1).astype(int).to_numpy()

    def channel_idx(self, groups: dict, pick: list[str]) -> np.ndarray:
        """Indices of the channels in the requested groups (``groups``: name -> list of channel names)."""
        names = [c for g in pick for c in groups[g]]
        return np.array([self.ch_names.index(c) for c in names])

    def units(self, unit: str = "auto") -> pd.Series:
        """Ordered calibration/test unit per window: 'session' if a subject has >1 session, else 'run'."""
        idx = self.index
        if unit == "auto":
            unit = "session" if idx.groupby("subject")["session"].nunique().max() > 1 else "run"
        return idx[unit].astype(str), unit


def order_key(values):
    """Sort session/run labels numerically when possible ('0', '1', '10' / '0train', '1test')."""
    def k(v):
        digits = "".join(ch for ch in str(v) if ch.isdigit())
        return (int(digits) if digits else 1 << 30, str(v))
    return sorted(set(values), key=k)


def later_mask(cache: WindowCache, n_calib: int, unit: str = "auto") -> np.ndarray:
    """True for windows in a subject's units after its first ``n_calib`` (the 'hidden later sessions')."""
    u, _ = cache.units(unit)
    subj = cache.index["subject"].to_numpy()
    out = np.zeros(len(subj), bool)
    for s in np.unique(subj):
        ms = subj == s
        order = order_key(u[ms])
        out[ms] = u[ms].isin(set(order[n_calib:])).to_numpy()
    return out


def xsession_folds(cache: WindowCache, y: np.ndarray, n_calib: int, unit: str = "auto", n_folds: int = 5,
                   subjects=None, seed: int = 0):
    """Sealed mirror. Subjects are split into ``n_folds`` groups; fold g tests the LATER units of group g and
    trains ONE model on everything else — other subjects' full data plus group g's own calibration units —
    exactly as the sealed phase serves its 10 evaluation users with one model. Yields (fold, subjects, train, test)."""
    subj = cache.index["subject"].to_numpy()
    later = later_mask(cache, n_calib, unit)
    valid = y >= 0
    pool = list(subjects or order_key(np.unique(subj)))
    rng = np.random.default_rng(seed)
    perm = [pool[i] for i in rng.permutation(len(pool))]
    for g in range(n_folds):
        group = set(perm[g::n_folds])
        ing = np.isin(subj, list(group))
        test = np.where(ing & later & valid)[0]
        train = np.where(~(ing & later) & valid)[0]
        if len(test):
            yield g, sorted(group), train, test


def within_folds(cache: WindowCache, y: np.ndarray, n_calib: int, unit: str = "auto", subjects=None):
    u, unit = cache.units(unit)
    subj = cache.index["subject"].to_numpy()
    valid = y >= 0
    for s in (subjects or order_key(np.unique(subj))):
        ms = subj == s
        order = order_key(u[ms])
        if len(order) <= n_calib:
            continue
        early, later = set(order[:n_calib]), set(order[n_calib:])
        yield s, np.where(ms & u.isin(early).to_numpy() & valid)[0], np.where(ms & u.isin(later).to_numpy() & valid)[0]

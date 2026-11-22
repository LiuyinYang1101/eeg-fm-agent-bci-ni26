"""Evaluation protocols (fold construction) — ONE implementation shared by the bench, the tier runner, the
leaderboard and the verifier. Folds are materialised as user lists and stored in each run's folds.json, and the
protocol_id hashes that membership, so a change in this algorithm can never silently re-split old results.

Kinds (the sealed task is cross-SESSION: scored users' calibration units always stay in training):
  xsession      sealed mirror. Users are split into n_folds groups (rng(split_seed) permutation, round-robin);
                fold g scores group g's LATER units; trains on everything else incl. group g's calibration units.
                Inner validation = the later units of n_val_subjects other users (rng(1000+g)), removed from training.
  xsession_all  fast screen: ONE fold — every user's later units scored; training = every user's calibration units.
  xsubject      measurement only: like xsession, but the scored group's calibration units are removed too.
  kit           the kit's split column (warm-up, cross-subject): train / val / test.
fold_algo "v1" = the algorithm below (identical to scripts/t2_bench.py of 2026-10-05).
"""

import numpy as np

from .windows import later_mask, order_key

FOLD_ALGO = "v1"


def excluded_mask(cache, exclude_units, unit="auto"):
    """True for windows in excluded 'subject|unit' recordings (data-audit blocklist, e.g. duplicated recordings)."""
    if not exclude_units:
        return np.zeros(len(cache.index), bool)
    subj = cache.index["subject"].astype(str).to_numpy()
    units, _ = cache.units(unit)
    keys = np.array([f"{a}|{b}" for a, b in zip(subj, units.to_numpy())])
    return np.isin(keys, list(exclude_units))


def make_folds(cache, y, kind, n_calib=1, unit="auto", n_folds=5, split_seed=0, n_val_subjects=1,
               exclude_units=None):
    """-> list of dict(fold, tr, va, te, group, val_subjects); indices into the cache.
    Excluded recordings are dropped from train/val/test alike (they never change unit ordering)."""
    subj = cache.index["subject"].astype(str).to_numpy()
    valid = (y >= 0) & ~excluded_mask(cache, exclude_units, unit)
    if kind == "kit":
        sp = cache.index["split"].to_numpy()
        return [dict(fold=0, tr=np.where((sp == "train") & valid)[0], va=np.where((sp == "val") & valid)[0],
                     te=np.where((sp == "test") & valid)[0], group="kit-test", val_subjects="kit-val")]
    later = later_mask(cache, n_calib, unit)
    if kind == "xsession_all":
        return [dict(fold=0, tr=np.where(~later & valid)[0], va=np.zeros(0, int), te=np.where(later & valid)[0],
                     group="all", val_subjects=[])]
    pool = order_key(np.unique(subj))
    perm = [pool[i] for i in np.random.default_rng(split_seed).permutation(len(pool))]
    out = []
    for g in range(n_folds):
        group = perm[g::n_folds]
        others = [s for s in perm if s not in group]
        vsub = [others[i] for i in np.random.default_rng(1000 + g).permutation(len(others))[:n_val_subjects]]
        ing, inv = np.isin(subj, group), np.isin(subj, vsub)
        te = np.where(ing & later & valid)[0]
        va = np.where(inv & later & valid)[0]
        drop = (ing & later) | (inv & later)
        if kind == "xsubject":
            drop |= ing
        tr = np.where(~drop & valid)[0]
        out.append(dict(fold=g, tr=tr, va=va, te=te, group=list(group), val_subjects=list(vsub)))
    return out


def check_folds(cache, folds, n_calib, unit="auto", kind="xsession"):
    """Verifier: re-derive unit membership from index.csv and assert the sealed-mirror invariants."""
    subj = cache.index["subject"].astype(str).to_numpy()
    units, _ = cache.units(unit)
    units = units.to_numpy()
    later = later_mask(cache, n_calib, unit)
    problems = []
    for f in folds:
        scored = {(subj[i], units[i]) for i in f["te"]}
        if any((subj[i], units[i]) in scored for i in f["tr"]):
            problems.append(f"fold {f['fold']}: a scored unit is in training")
        if any((subj[i], units[i]) in scored for i in f["va"]):
            problems.append(f"fold {f['fold']}: a scored unit is in validation")
        if kind in ("xsession", "xsession_all") and not later[f["te"]].all():
            problems.append(f"fold {f['fold']}: a calibration window is scored")
        if kind == "xsession":
            g = set(f["group"])
            if not any(subj[i] in g for i in f["tr"]):
                problems.append(f"fold {f['fold']}: scored users' calibration units missing from training")
    return problems

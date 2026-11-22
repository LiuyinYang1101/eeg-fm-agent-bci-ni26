"""Experiment identity: protocol × method -> spec_id (WORKFLOW_DESIGN §2).

* A PROTOCOL fixes the data and the split: cache name + its x_sha256, label map, kind, unit, n_calib, n_folds,
  split_seed, n_val_subjects, screen folds, fold_algo, and the resolved fold membership (users per fold).
* A METHOD is everything between the cache and the predicted label: family, model, full recipe fields, resolved
  channel names, time crop, preprocessing, checkpoint sha, fusion inputs.
* ids = sha256 of canonical JSON (sorted keys, no whitespace)[:10]. spec_id = sha256(protocol_id + method_id)[:10].
Two results are comparable iff protocol_id and (seed, fold) match. YAML is the human form; the hash is the identity.
"""

import hashlib
import json
import os
from dataclasses import asdict, fields
from pathlib import Path

import yaml

from .train import RECIPES, Recipe

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
DATA_ROOTS = [Path(os.path.expanduser("~/workspace/datasets/neural_interface_challenge/t2_bci/windows")),
              Path(os.path.expanduser("~/workspace/datasets/neural_interface_challenge/t2_bci/windows_long")),
              Path(os.environ.get("S1_ROOT", "/nonexistent")) / "data/ni26/t2_bci/windows_local"]
STORE = Path(os.environ.get("NI26_STORE", os.path.expanduser("~/workspace/outputs/neural_interface_challenge/t2/runs")))


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def h(obj, n=10) -> str:
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:n]


def load_yaml(kind, name):
    p = CONFIGS / kind / f"{name}.yaml"
    return yaml.safe_load(p.read_text()), p


def resolve_cache(name, x_sha256=None):
    """Find a cache by name in the known roots; refuse if its sha differs from the protocol's (B4/E11)."""
    for root in DATA_ROOTS:
        d = root / name
        if (d / "meta.json").exists():
            sha = json.loads((d / "meta.json").read_text()).get("x_sha256")
            if x_sha256 and sha != x_sha256:
                raise ValueError(f"cache {d} has x_sha256 {sha[:12]} != protocol {x_sha256[:12]}")
            return d
    raise FileNotFoundError(f"cache {name} not found in {DATA_ROOTS}")


def resolve_recipe(r):
    if isinstance(r, str):
        return asdict(RECIPES[r])
    base = asdict(RECIPES[r.get("base", "r1")]) if isinstance(r, dict) else {}
    base.update({k: v for k, v in r.items() if k != "base"})
    valid = {f.name for f in fields(Recipe)}
    bad = set(base) - valid
    if bad:
        raise ValueError(f"unknown recipe fields {bad}")
    return base


def resolve_channels(spec, ch_names):
    """'all' | list of names | '@eeg+@eog' -> explicit ordered channel names."""
    groups = {"@eog": [c for c in ch_names if c.upper().startswith("EOG")],
              "@emg": [c for c in ch_names if c.upper().startswith("EMG")]}
    groups["@eeg"] = [c for c in ch_names if c not in groups["@eog"] + groups["@emg"]]
    if spec in (None, "all"):
        return list(ch_names)
    if isinstance(spec, str):
        return [c for tok in spec.split("+") for c in (groups[tok] if tok in groups else [tok])]
    return list(spec)


def protocol_record(p, fold_membership):
    keys = ["name", "cache", "x_sha256", "code_map", "kind", "unit", "n_calib", "n_folds", "split_seed",
            "n_val_subjects", "screen_folds", "fold_algo"]
    rec = {k: p.get(k) for k in keys}
    rec["folds"] = fold_membership
    return rec


def method_record(m, ch_names):
    rec = dict(name=m["name"], family=m.get("family", "cnn"), model=m["model"], recipe=resolve_recipe(m.get("recipe", "r1")),
               channels=resolve_channels(m.get("channels", "all"), ch_names), crop=m.get("crop"),
               preprocessing=m.get("preprocessing"), checkpoint_sha=m.get("checkpoint_sha"),
               fusion_inputs=m.get("fusion_inputs"), control=m.get("control"))
    if m.get("fusion_weights") is not None:      # optional keys enter the record ONLY when set (ids stay stable)
        rec["fusion_weights"] = m["fusion_weights"]
    return rec


NON_IDENTITY = {"name", "screen_folds"}   # labels / tier-subset metadata: never change a cell, so never change an id


def ids(prot_rec, meth_rec):
    pid = h({k: v for k, v in prot_rec.items() if k not in NON_IDENTITY})
    mid = h({k: v for k, v in meth_rec.items() if k not in NON_IDENTITY})
    return pid, mid, h(pid + mid)


def run_dir(prot_rec, meth_rec):
    pid, mid, sid = ids(prot_rec, meth_rec)
    return STORE / f"{prot_rec['name']}__{meth_rec['name']}__{sid}"


def provenance():
    import platform
    import subprocess
    from importlib import metadata as im
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *a], text=True).strip()
        except Exception:
            return None
    from . import kit
    vers = {}
    for p in ("torch", "braindecode", "numpy", "scipy", "scikit-learn", "neuralbench", "mne"):
        try:
            vers[p] = im.version(p)
        except Exception:
            vers[p] = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = None
    return dict(git_sha=git("rev-parse", "HEAD"), git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                kit_sha=kit.kit_sha() if True else None, versions=vers, host=platform.node(), gpu=gpu)

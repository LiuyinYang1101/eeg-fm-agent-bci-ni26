"""Locate the official competition kit and import its track code.

The kit (github.com/neural-interfaces26/2026-competition) is cloned at a pinned commit by
``scripts/get_kit.sh``: locally under ``third_party/2026-competition``, on sofia under
``$S1_ROOT/code/ni26_kit/current``. ``NI26_KIT`` overrides both.

Everything that must match the Codabench scorer bit-for-bit (window extraction, scaling, splits)
goes through the kit's own ``benchmark_utils`` rather than a re-implementation.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KIT_SHA = "442edcd8c920d8c099a7e60a2ded2a4b8d5f1f32"   # was 2067faf until 2026-10-16 (review T2 P1c)


def kit_root() -> Path:
    cands = [os.environ.get("NI26_KIT")]
    if os.environ.get("S1_ROOT"):
        cands.append(os.path.join(os.environ["S1_ROOT"], "code", "ni26_kit", "current"))
    cands.append(str(REPO / "third_party" / "2026-competition"))
    for c in cands:
        if c and (Path(c) / "benchmark_utils").is_dir():
            return Path(c).resolve()
    raise FileNotFoundError(f"competition kit not found (tried {cands}); run scripts/get_kit.sh or set NI26_KIT")


def kit_sha(root: Path | None = None) -> str:
    root = root or kit_root()
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return root.name  # sofia clones are named by their sha


def ensure_neuralbench_config() -> None:
    """Without ~/.neuralbench/config.json, neuralbench falls back to /tmp/neuralbench (auto-deleted; workspace
    policy forbids it). Point NEURALBENCH_CONFIG at a config living next to the data, with local execution."""
    import json
    if os.environ.get("NEURALBENCH_CONFIG"):
        return
    home = Path(os.environ.get("BENCHOPT_DATA_HOME") or REPO / "data" / "benchopt")
    base = home / "neuralbench"
    cfg = dict(USER=os.environ.get("USER", "user"), ENTITY_NAME=os.environ.get("USER", "user"),
               PROJECT_NAME="neuralbench", CACHE_DIR=str(base / "cache"), SAVE_DIR=str(base / "save"),
               DATA_DIR=str(base / "data"), WANDB_HOST="", SLURM_PARTITION="", SLURM_CONSTRAINT="",
               N_CPUS=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), CLUSTER=None)
    for k in ("CACHE_DIR", "SAVE_DIR", "DATA_DIR"):
        Path(cfg[k]).mkdir(parents=True, exist_ok=True)
    path = base / "config.json"
    path.write_text(json.dumps(cfg, indent=1))
    os.environ["NEURALBENCH_CONFIG"] = str(path)


def use_track(track: str = "bci_decoding") -> Path:
    """Put ``tracks/<track>`` on sys.path so ``benchmark_utils`` / ``datasets`` import as the kit does."""
    ensure_neuralbench_config()
    path = kit_root() / "tracks" / track
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path

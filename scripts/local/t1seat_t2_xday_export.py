#!/usr/bin/env python
"""Track1's T2 experiment (the manager's ask 2026-11-15 00:0x; lead/panel_2026-11-14_outliers/SEAT_T1_on_T2.md, section XDAY): the COMBINED
export for the cross-day returning panel = every PROTEUS window (export_proteus_v1, unchanged, first) + Scherer2015's three-command
windows (the local windows export scherer2015_cue3s4: 30 channels, 120 Hz, 4 s from the trigger + 3 s; codes 4 / 2 / 1 -> MI / SUB / WORD =
t2p_bench.THREE_COMMAND) mapped onto PROTEUS's 41 channels by t2p_bench.interp_matrix (spherical spline, as T2-D219) and multiplied by ONE
constant so that the median window x channel std equals PROTEUS's (FBCNet runs align 'none' = one global scale).
Scherer users are S1 .. S9, session '1' = day 1, '2' = day 2, interface 'X', split 'ext'; only the three-command windows are kept (a
sealed-like stream holds nothing else); order = acquisition order inside each user x day.
    mem-guard python scripts/local/t1seat_t2_xday_export.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402

D = Path.home() / "workspace/datasets/neural_interface_challenge"
OUT = Path.home() / "workspace/outputs/neural_interface_challenge/t1seat_t2/export_combo_v1"
CODE = {4: 0, 2: 1, 1: 2}

XP, ip, mp = B.load_export(D / "t2_proteus/export_proteus_v1")
ip = pd.read_csv(D / "t2_proteus/export_proteus_v1/index.csv", dtype={"session": str})
S = D / "t2_bci/windows/scherer2015_cue3s4"
ms = json.loads((S / "meta.json").read_text())
assert ms["sfreq"] == mp["sfreq"] == 120.0 and ms["n_times"] == mp["n_times"] == 480
isx = pd.read_csv(S / "index.csv")
XS = np.load(S / "X.npy", mmap_mode="r")
assert len(isx) == len(XS) and (isx.i.to_numpy() == np.arange(len(isx))).all()
keep = np.flatnonzero(isx.code.isin(list(CODE)).to_numpy())
g = isx.iloc[keep].copy()
W = B.interp_matrix(list(ms["ch_names"]), list(mp["ch_names"]))
xs = np.einsum("dc,nct->ndt", W, np.asarray(XS[keep], np.float64))
sp = float(np.median(np.asarray(XP[::5], np.float64).std(-1)))
ss = float(np.median(xs[::5].std(-1)))
xs *= sp / ss
u = "S" + g.subject.str.split("/").str[-1]
ses = (g.session.astype(int) + 1).astype(str)
assert sorted(ses.unique()) == ["1", "2"] and u.nunique() == 9
new = pd.DataFrame(dict(split="ext", subject="Scherer2015/" + u.to_numpy(), session=ses.to_numpy(), interface="X", run=g.run.to_numpy() + 1,
                        onset_s=g.onset_s.to_numpy(), y=g.code.map(CODE).to_numpy()))
new["stream"] = new.subject + "|" + new.session
new["cell"] = new.stream + "|X"
new["order"] = new.groupby("stream").onset_s.rank(method="first").astype(int) - 1
cols = ["split", "subject", "session", "interface", "run", "onset_s", "y", "stream", "cell", "order"]
idx = pd.concat([ip[cols], new[cols]], ignore_index=True)
idx.insert(0, "i", np.arange(len(idx)))
OUT.mkdir(parents=True, exist_ok=True)
X = np.lib.format.open_memmap(OUT / "X.npy", mode="w+", dtype=np.float32, shape=(len(idx), 41, 480))
X[:len(ip)] = np.asarray(XP)
X[len(ip):] = xs.astype(np.float32)
X.flush()
del X
sha = hashlib.sha256((OUT / "X.npy").read_bytes()).hexdigest()
idx.to_csv(OUT / "index.csv", index=False)
meta = dict(study="combo: Dreyer2026Proteus + Scherer2015Individually (three commands)", sfreq=120.0, n_chans=41, n_times=480,
            ch_names=mp["ch_names"], n_windows=int(len(idx)), n_proteus=int(len(ip)), n_scherer=int(len(new)), scherer_scale=sp / ss,
            proteus_sha=mp["x_sha256"], scherer_sha=ms["x_sha256"], x_sha256=sha,
            note="Scherer: 30 -> 41 channels by spherical spline; window = trigger + 3 s .. + 7 s (PROTEUS: cue + 1 s .. + 5 s)")
(OUT / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"combo export: {len(ip)} PROTEUS + {len(new)} Scherer windows, scale x{sp / ss:.4f}, sha {sha[:16]} -> {OUT}")
print(new.groupby(["subject", "session"]).y.value_counts().unstack().to_string())

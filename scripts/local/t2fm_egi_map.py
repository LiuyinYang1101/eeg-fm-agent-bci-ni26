#!/usr/bin/env python
"""T2-FM-D2 (T2-D356 store pool): EGI HydroCel electrode -> ST-EEGFormer channel name, written ONCE to
tracks/t2_bci/fm/egi_to_st.json (committed; the sofia job needs no mne).

Rule (fixed before any number): MNE's standard montages, each placed in the head frame by its own fiducials
(info.set_montage); unit direction vectors from the head origin (the EGI nets sit on a smaller sphere than
standard_1005, 0.109 / 0.102 vs 0.114 m, so raw distances mix in a scale offset); an EGI electrode takes an ST name
iff they are MUTUAL nearest neighbours in angle and the angle is <= 6 deg (≈ 1.2 cm on an adult head; the ST grid's
nearest-neighbour spacing is ≈ 10.7 deg). ST's synonyms at a shared position (T3/T4/T5/T6 = T7/T8/P7/P8, A1/A2 =
M1/M2) are left out of the target set so they cannot split a match. The EGI reference electrode (Cz / VREF, the
online reference: flat) is never mapped.

  mem-guard python scripts/local/t2fm_egi_map.py
"""
import hashlib
import json
import warnings
from pathlib import Path

import mne
import numpy as np

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "tracks/t1_image/fm/steeg_channels.json"
OUT = REPO / "tracks/t2_bci/fm/egi_to_st.json"
SYN = {"T3", "T4", "T5", "T6", "A1", "A2"}
THR_DEG = 6.0
NETS = ("GSN-HydroCel-129", "GSN-HydroCel-257")


def head_dirs(name):
    mon = mne.channels.make_standard_montage(name)
    info = mne.create_info(mon.ch_names, 256.0, "eeg")
    info.set_montage(mon)
    pos = {c["ch_name"]: np.asarray(c["loc"][:3], float) for c in info["chs"]}
    return {k: v / np.linalg.norm(v) for k, v in pos.items()}


def main():
    warnings.filterwarnings("ignore")
    table = json.loads(TABLE.read_text())
    low = {k.lower(): k for k in table}
    std = {low[n.lower()]: v for n, v in head_dirs("standard_1005").items() if n.lower() in low and low[n.lower()] not in SYN}
    sn = sorted(std)
    S = np.array([std[s] for s in sn])
    out = dict(_doc=__doc__.split("\n\n")[1].replace("\n", " "), _threshold_deg=THR_DEG, _mne=mne.__version__,
               _steeg_channels_sha256=hashlib.sha256(TABLE.read_bytes()).hexdigest())
    for net in NETS:
        egi = {k: v for k, v in head_dirs(net).items() if k.upper().startswith("E") and k[1:].isdigit()}
        en = sorted(egi, key=lambda e: int(e[1:]))
        E = np.array([egi[e] for e in en])
        ang = np.degrees(np.arccos(np.clip(E @ S.T, -1.0, 1.0)))
        m = {}
        for j, s in enumerate(sn):
            i = int(ang[:, j].argmin())
            if ang[i, j] <= THR_DEG and int(ang[i].argmin()) == j:
                m[en[i]] = s
        out[net] = dict(sorted(m.items(), key=lambda kv: int(kv[0][1:])))
        out[f"_{net}_angle_deg"] = {e: round(float(ang[en.index(e), sn.index(s)]), 2) for e, s in out[net].items()}
        print(f"{net}: {len(m)} of {len(en)} electrodes mapped; median angle "
              f"{np.median(list(out[f'_{net}_angle_deg'].values())):.2f} deg")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT} (sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]})")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-FM-D4 / D5 (J-CPT replay arms): ST-EEGFormer's own pretraining corpus (local copy) -> t2p export pools (X.npy / index.csv /
meta.json) for t2p_cpt --jcpt-spec, ONE source per call, ONLY after its licence is recorded as IN in docs/DECISIONS_T2FM.md (T2-FM-D3).

What enters: exactly ST-EEGFormer's pretraining split, `<source>/pretrain_data_indices.pkl` (ST's own pretraining loader,
STEEGFormer/pretrain/util/eeg_dataset.py: 5-s segments, 2 s for ssvep, at 128 Hz, at the indexed (recording, position)), cut into
NON-overlapping windows that each lie inside one indexed segment: 4 s (512 samples) for the continuous sources (large_im, p300; a
greedy left-to-right pick over each recording's sorted indexed positions), the whole 3-s trial (384 samples) for ssvep (a trial
enters iff every 2-s segment start 0..128 of it is indexed). Channel names = ST's own map for that source (STEEGFormer
pretrain/senloc_file/sen_chan_idx.pkl -> indices into ST's table, steeg_channels.json). Values: ST's own pretraining clip to
+-500 (its loader's step before the per-window z, which t2p_cpt applies), float32. The pkl files are read with a restricted
unpickler (numpy / pandas reconstruction only).
  mem-guard python scripts/local/t2fm_replay_export.py large_im|p300|ssvep [--out DIR] [--limit-recordings N]
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = Path("/media/USER/Data/eeg_hpc_datasets/EEG_ITransformer_preprocessed_data")
SENLOC = Path.home() / "workspace/projects/STEEGFormer/pretrain/senloc_file/sen_chan_idx.pkl"
OUT = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/replay"
FS, CLIP = 128, 500.0
SAFE = {"dict", "list", "tuple", "set", "frozenset", "slice", "range", "complex", "object", "int", "float", "str", "bytes", "bool"}
NAMES = {"large_im": "Kaya2018", "p300": "Won2022", "ssvep": "BETA"}       # provenance per T2-FM-D3 (prefix of the subject ids)


class Restricted(pickle.Unpickler):
    def find_class(self, m, n):
        root = m.split(".")[0]
        if root == "numpy" or root == "pandas" or (m == "builtins" and n in SAFE) \
                or (m in ("copyreg", "_codecs", "collections") and n in ("_reconstructor", "encode", "OrderedDict")):
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f"refused to unpickle {m}.{n}")


def load(p):
    with open(p, "rb") as fh:
        return Restricted(fh).load()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def greedy(pos, win):
    """non-overlapping window starts from sorted indexed positions (each window inside the segment that starts there)."""
    out, last = [], -win
    for p in sorted(pos):
        if p >= last + win:
            out.append(p)
            last = p
    return out


def plan(source, ix, limit):
    """[(file, key, starts, subject, session, run)] windows to cut, + the window length."""
    ix = ix.assign(Position=ix["Position"].astype(int))
    rows = []
    if source == "large_im":
        for rec, g in ix.groupby("Recording", sort=True):
            rows.append((f"data/{rec}.pkl", ("eeg", None), greedy(g["Position"], 512), str(g["Subject"].iloc[0]), rec, ""))
        return rows[:limit] if limit else rows, 512
    if source == "p300":
        for (rec, typ, run), g in ix.groupby(["Recording", "Type", "Run"], sort=True):
            rows.append((f"data/{rec}.pkl", (typ, int(run)), greedy(g["Position"], 512), str(g["Subject"].iloc[0]), rec, f"{typ}_{run}"))
        return rows[:limit] if limit else rows, 512
    if source == "ssvep":                         # whole 3-s trials whose 2-s segment starts 0..128 (hop 16) are all indexed
        need = set(range(0, 129, 16))
        for (sub, blk, tri), g in ix.groupby(["Subject", "Block", "Trial"], sort=True):
            if need <= set(g["Position"]):
                rows.append((f"data/{sub}.pkl", ("eeg", (int(blk), int(tri))), [0], str(sub), str(sub), f"b{blk}_t{tri}"))
        return rows[:limit * 160] if limit else rows, 384
    raise SystemExit(f"unknown source {source}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=sorted(NAMES))
    ap.add_argument("--out", default="")
    ap.add_argument("--limit-recordings", type=int, default=0, help="tests only")
    ap.add_argument("--licence", default="", help="the licence recorded in T2-FM-D3 (written to meta.json; required unless --limit)")
    a = ap.parse_args()
    if not a.limit_recordings and not a.licence:
        raise SystemExit("--licence: the T2-FM-D3 licence string (a source enters only after its licence is verified)")
    t0 = time.time()
    table = json.loads((REPO / "tracks/t1_image/fm/steeg_channels.json").read_text())
    inv = {v: k for k, v in table.items()}
    ch = [inv[int(i)] for i in load(SENLOC)[a.source]]
    ix_path = SRC / a.source / "pretrain_data_indices.pkl"
    rows, T = plan(a.source, load(ix_path), a.limit_recordings)
    n = sum(len(r[2]) for r in rows)
    out = Path(a.out) if a.out else OUT / f"{a.source}_v1"
    out.mkdir(parents=True, exist_ok=False)
    X = np.lib.format.open_memmap(out / "X.npy.tmp", mode="w+", dtype=np.float32, shape=(n, len(ch), T))
    meta_rows, k, cache = [], 0, {}
    for f, (key, sub), starts, subj, ses, run in rows:
        if f not in cache:
            cache.clear()
            cache[f] = load(SRC / a.source / f)
            d = cache[f]
            if "fs" in d:
                assert int(d["fs"]) == FS, (f, d["fs"])
            if "channel" in d:                    # p300 stores its names: they must be ST's map, in order
                assert [c.lower() for c in d["channel"]] == [c.lower() for c in ch], (f, d["channel"])
        arr = cache[f][key]
        if a.source == "ssvep":
            sig = arr[..., sub[0], sub[1]]
        elif sub is None or arr.ndim == 2:            # p300 'rsvp' is one run, (32, T): its index says Run 0
            assert sub in (None, 0), (f, key, sub, arr.shape)
            sig = arr
        else:
            sig = arr[..., sub]
        assert sig.shape[0] == len(ch), (f, sig.shape, len(ch))
        for s in starts:
            w = sig[:, s:s + T]
            if w.shape[1] < T:
                continue
            X[k] = np.clip(w, -CLIP, CLIP)
            meta_rows.append(dict(i=k, subject=f"{NAMES[a.source]}/{subj}", session=ses, run=run, position=int(s), split="train"))
            k += 1
    X.flush()
    del X
    if k < n:                                     # windows past a recording's end were skipped: rewrite at the true length
        Y = np.load(out / "X.npy.tmp", mmap_mode="r")[:k]
        np.save(out / "X.npy", Y)
        del Y
        (out / "X.npy.tmp").unlink()
    else:
        (out / "X.npy.tmp").rename(out / "X.npy")
    pd.DataFrame(meta_rows).to_csv(out / "index.csv", index=False)
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(sfreq=float(FS), n_times=T, ch_names=ch, n_windows=k, source=a.source, dataset=NAMES[a.source], licence=a.licence,
                src=str(SRC / a.source), pretrain_index=str(ix_path), pretrain_index_sha256=sha256(ix_path),
                senloc=str(SENLOC), senloc_sha256=sha256(SENLOC), clip=CLIP, window=dict(samples=T, s=T / FS),
                subjects=sorted({r["subject"] for r in meta_rows}), hours=round(k * T / FS / 3600, 3),
                converter="scripts/local/t2fm_replay_export.py", converter_commit=commit, sha256=sha256(out / "X.npy"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{a.source}: {k} windows x {len(ch)} ch x {T} ({meta['hours']} h, {len(meta['subjects'])} subjects) -> {out} "
          f"({(out / 'X.npy').stat().st_size / 2**30:.2f} GiB, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

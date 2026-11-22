#!/usr/bin/env python
"""Track 2 (FM program §0d item 7, T2-D129): CONTINUOUS-SESSION crops of every PROTEUS recording (baselines eyes open /
closed, acquisition and online runs, inter-trial periods) through the kit's preprocessing, for unlabelled CPT.

Preprocessing = the kit's EegExtractor config (export meta data_cfg.neuro): the 41 EEG channels of the contract export
(same names and order), notch 50 / 60 Hz, band-pass 0.1-75 Hz at the recorded rate, resample to 120 Hz, x 1e6 (V -> uV),
no scaler / clamp. Each recording is tiled into --crop-s windows every --hop-s. Validation: for every cue window of the
contract export, the crop of the same 4 s cut from this preprocessed signal is compared with the exported window
(median relative error and correlation reported; MNE's filters vs neuralset's may differ in edge handling).

  python t2p_session_export.py --bids <.../nm000290> --names-from <export_proteus_v1> --out <dir> [--crop-s 4 --hop-s 2]
Output: X.npy float16 (N, 41, 480), index.csv (i, subject, session, task, run, t0_s, stream, order, split), meta.json.
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


def norm(n):
    s = str(n).strip().upper()
    s = s[4:] if s.startswith("EEG ") else s
    return s.split("-")[0].strip()


def main():
    import mne
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", required=True)
    ap.add_argument("--names-from", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-s", type=float, default=4.0)
    ap.add_argument("--hop-s", type=float, default=2.0)
    ap.add_argument("--validate-only", type=int, default=0, help="check N recordings at window offsets 0 and +window_start; write nothing")
    args = ap.parse_args()
    t_start = time.time()
    E = Path(args.names_from)
    meta_e = json.loads((E / "meta.json").read_text())
    names = list(meta_e["ch_names"])
    neuro = meta_e["data_cfg"]["neuro"]
    fs, (lo, hi), notch, gain = float(neuro["frequency"]), neuro["filter"], neuro["notch_filter"], float(neuro["scale_factor"])
    idx_e = pd.read_csv(E / "index.csv", dtype=str)
    split_of = {s.split("/")[-1]: sp for s, sp in zip(idx_e["subject"], idx_e["split"])}
    Xe = np.load(E / "X.npy", mmap_mode="r")
    files = sorted(Path(args.bids).glob("sub-*/ses-*/eeg/*_eeg.edf"))
    crop, hop = int(round(args.crop_s * fs)), int(round(args.hop_s * fs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunks, rows, val = [], [], []
    if args.validate_only:                  # does onset_s mark the cue or the window start? (median corr per offset)
        res = {}
        for f in [f for f in files if "Acquisition" in f.name or "Online" in f.name][:args.validate_only]:
            m = re.match(r"sub-(\w+)_ses-(\w+)_task-(\w+)_run-(\d+)_eeg\.edf", f.name)
            sub, ses, task, run = m.group(1), m.group(2), m.group(3), m.group(4)
            raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
            lookup = {norm(c): c for c in raw.ch_names}
            raw.pick([lookup[norm(c)] for c in names])
            raw.notch_filter(notch, verbose="ERROR")
            raw.filter(lo, hi, verbose="ERROR")
            raw.resample(fs, verbose="ERROR")
            x = raw.get_data() * gain
            ee = idx_e[(idx_e["subject"].str.endswith(sub)) & (idx_e["session"] == ses) & (idx_e["task"] == task) &
                       (idx_e["run"].astype(str) == str(int(run)))]
            for off in (0.0, float(meta_e["window_start"]), -float(meta_e["window_start"])):
                cs = []
                for _, r in ee.head(20).iterrows():
                    a = int(round((float(r["onset_s"]) + off) * fs))
                    if 0 <= a and a + crop <= x.shape[1]:
                        cs.append(float(np.corrcoef(x[:, a:a + crop].ravel(), np.asarray(Xe[int(r["i"])], np.float64).ravel())[0, 1]))
                res.setdefault(off, []).extend(cs)
            print(f"[validate] {f.name}: " + ", ".join(f"offset {o:+.1f}s median corr {np.median(v):.3f}" for o, v in res.items()), flush=True)
        print(json.dumps({str(o): dict(n=len(v), median_corr=float(np.median(v))) for o, v in res.items()}), flush=True)
        return
    for f in files:
        m = re.match(r"sub-(\w+)_ses-(\w+)_task-(\w+)_run-(\d+)_eeg\.edf", f.name)
        sub, ses, task, run = m.group(1), m.group(2), m.group(3), m.group(4)
        raw = mne.io.read_raw_edf(f, preload=True, verbose="ERROR")
        lookup = {norm(c): c for c in raw.ch_names}
        missing = [c for c in names if norm(c) not in lookup]
        if missing:
            print(f"[session_export] {f.name}: missing channels {missing}: skipped", flush=True)
            continue
        raw.pick([lookup[norm(c)] for c in names])
        raw.notch_filter(notch, verbose="ERROR")
        raw.filter(lo, hi, verbose="ERROR")
        raw.resample(fs, verbose="ERROR")
        x = raw.get_data() * gain                                       # (41, T) uV at 120 Hz
        n = max(0, (x.shape[1] - crop) // hop + 1)
        if n == 0:
            continue
        starts = np.arange(n) * hop
        chunks.append(np.stack([x[:, s:s + crop] for s in starts]).astype(np.float16))
        for k, s in enumerate(starts):
            rows.append(dict(subject=f"Dreyer2026Proteus/{sub}", session=ses, task=task, run=int(run), t0_s=s / fs,
                             stream=f"Dreyer2026Proteus/{sub}|{ses}", order=k, split=split_of.get(sub, "unknown")))
        # validation against the contract export: its cue windows of this recording, cut from this signal
        ee = idx_e[(idx_e["subject"].str.endswith(sub)) & (idx_e["session"] == ses) & (idx_e["task"] == task) &
                   (idx_e["run"].astype(str) == str(int(run)))]
        for _, r in ee.head(20).iterrows():
            a = int(round(float(r["onset_s"]) * fs))      # the export's onset_s IS the window start (validated: corr 1.000)
            if 0 <= a and a + crop <= x.shape[1]:
                w = x[:, a:a + crop]
                ref = np.asarray(Xe[int(r["i"])], np.float64)
                if ref.shape == w.shape:
                    val.append(dict(rel=float(np.median(np.abs(w - ref)) / (np.median(np.abs(ref)) + 1e-9)),
                                    corr=float(np.corrcoef(w.ravel(), ref.ravel())[0, 1])))
        print(f"[session_export] {f.name}: {x.shape[1] / fs:.0f} s -> {n} crops ({time.time() - t_start:.0f}s)", flush=True)
    X = np.concatenate(chunks)
    np.save(out / "X.npy", X)
    idx = pd.DataFrame(rows)
    idx.insert(0, "i", np.arange(len(idx)))
    idx.to_csv(out / "index.csv", index=False)
    v = pd.DataFrame(val)
    meta = dict(study="Dreyer2026Proteus", kind="continuous-session crops (unlabelled; FM program T2-D129)", sfreq=fs,
                ch_names=names, n_times=crop, crop_s=args.crop_s, hop_s=args.hop_s, n_windows=int(len(X)),
                n_recordings=len(chunks), preprocessing=dict(notch=notch, filter=[lo, hi], frequency=fs, scale_factor=gain,
                                                             implementation="mne notch_filter / filter / resample (defaults)"),
                split_counts=idx["split"].value_counts().to_dict(), tasks=idx["task"].value_counts().to_dict(),
                validation=dict(n=int(len(v)), median_rel_err=float(v["rel"].median()) if len(v) else None,
                                median_corr=float(v["corr"].median()) if len(v) else None,
                                min_corr=float(v["corr"].min()) if len(v) else None),
                seconds=round(time.time() - t_start))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_windows", "n_recordings", "split_counts", "tasks", "validation")}, indent=1), flush=True)


if __name__ == "__main__":
    main()

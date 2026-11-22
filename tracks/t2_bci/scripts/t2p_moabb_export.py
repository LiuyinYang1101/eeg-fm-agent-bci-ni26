#!/usr/bin/env python
"""Track 2 CPT pools: a public MOABB corpus -> UNLABELLED contract windows (120 Hz, 4 s, float16), raw deleted as it goes.

Built for continued MAE pretraining (t2p_cpt.py --extra): the corpora are far too large for the kit's neuralset pipeline,
whose prepare() caches every preprocessed recording (Stieger2021 alone ≈ 400 GB raw on NEMAR nm000339). Per subject, in a
worker process: MOABB downloads the subject into a private MNE_DATA dir, every run is preprocessed exactly as the
PROTEUS contract (neuralset EegExtractor, kit 442edcd: pick EEG -> notch 50 / 60 Hz -> band-pass 0.1-75 Hz at the
recorded rate -> resample 120 Hz -> x1e6 = uV; the notch is applied at 50 and 60 Hz only: their harmonics >= 100 Hz lie
in the band-pass stop band), one window [cue + start, cue + start + duration] per MOABB event is cut, the shard is saved
and the subject's raw files are deleted. The merge writes the CPT store; labels are kept in index.csv but CPT ignores them.

Output (--out): X.npy float16 (N, C, T) memmap, index.csv (i, subject, session, run, onset_s, code, event), meta.json
(sfreq 120, ch_names, dataset, window, filters, per-subject counts, sha256 of X.npy). Channels = the corpus's EEG
channels in the first subject's order; a subject whose EEG set differs is mapped by name (missing -> skipped subject).

  python t2p_moabb_export.py --dataset Stieger2021 --out <dir> --raw-root <node-local tmp> --workers 6 [--subjects 1,2]
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FS_OUT, NOTCH, BAND, SCALE = 120.0, (50.0, 60.0), (0.1, 75.0), 1e6


def _dataset(name):
    import moabb.datasets as D
    kw = {"fix_bads": False} if name == "Stieger2021" else {}   # as neuralfetch's Stieger2021Continuous
    if name.startswith("Shin2017"):
        kw["accept"] = True   # MOABB's licence flag: Shin2017A/B are released under GNU GPL v3 (public, redistributable)
    return getattr(D, name)(**kw)


def export_subject(name, subject, raw_root, shard_dir, start, duration, threads):
    """worker: download + preprocess + window one subject; returns (subject, n, ch_names, seconds, error)."""
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
    t0 = time.time()
    mdir = Path(raw_root) / f"sub-{subject}"
    shard = Path(shard_dir) / f"sub-{subject}.npz"
    if shard.exists():
        z = np.load(shard, allow_pickle=False)
        return subject, int(len(z["X"])), [str(c) for c in z["ch"]], 0.0, None
    try:
        mdir.mkdir(parents=True, exist_ok=True)
        # private MNE config home + data dir per worker: MOABB's get_dataset_path writes MNE_DATASETS_<X>_PATH into the
        # MNE config on first use; the real ~/.mne (shared by every project of the user) is never touched
        os.environ["_MNE_FAKE_HOME_DIR"] = str(mdir / "home")
        (mdir / "home").mkdir(exist_ok=True)
        os.environ["MNE_DATA"] = str(mdir)
        import mne
        mne.set_log_level("ERROR")
        ds = _dataset(name)
        os.environ[f"MNE_DATASETS_{ds.code.upper()}_PATH"] = str(mdir)
        inv = {v: k for k, v in ds.event_id.items()}
        data = ds.get_data(subjects=[subject])[subject]
        Xs, rows, ch = [], [], None
        n_win = int(round(duration * FS_OUT))
        for sess, runs in data.items():
            for run, raw in runs.items():
                ev = mne.find_events(raw, shortest_event=0, verbose=False) if "stim" in raw.get_channel_types() \
                    else mne.events_from_annotations(raw, verbose=False)[0]
                sf = float(raw.info["sfreq"])
                raw = raw.copy().pick("eeg").load_data()
                freqs = [f for f in NOTCH if f < sf / 2]
                if freqs:
                    raw.notch_filter(freqs, phase="zero", filter_length="auto", n_jobs=1, verbose=False)
                raw.filter(BAND[0], BAND[1] if BAND[1] < sf / 2 else None, n_jobs=1, verbose=False)
                raw.resample(FS_OUT, n_jobs=1, verbose=False)
                x = raw.get_data()
                if ch is None:
                    ch = list(raw.ch_names)
                elif list(raw.ch_names) != ch:
                    pos = {c: i for i, c in enumerate(raw.ch_names)}
                    if not all(c in pos for c in ch):
                        continue
                    x = x[[pos[c] for c in ch]]
                for onset, _, code in ev:
                    i0 = int(round((onset / sf + start) * FS_OUT))
                    if i0 < 0 or i0 + n_win > x.shape[1]:
                        continue
                    w = x[:, i0:i0 + n_win] * SCALE
                    if not np.isfinite(w).all():
                        continue
                    Xs.append(np.clip(w, -65000, 65000).astype(np.float16))
                    rows.append((str(sess), str(run), round(onset / sf, 3), int(code), inv.get(int(code), "")))
        X = np.stack(Xs) if Xs else np.zeros((0, len(ch or []), n_win), np.float16)
        tmp = shard.with_suffix(".tmp.npz")
        np.savez(tmp, X=X, ch=np.array(ch or []), sess=np.array([r[0] for r in rows]), run=np.array([r[1] for r in rows]),
                 onset=np.array([r[2] for r in rows]), code=np.array([r[3] for r in rows]),
                 event=np.array([r[4] for r in rows]))
        tmp.rename(shard)
        return subject, int(len(X)), ch or [], time.time() - t0, None
    except Exception as e:  # one bad subject never kills the corpus
        return subject, 0, [], time.time() - t0, f"{e!r}\n{traceback.format_exc()[-1500:]}"
    finally:
        shutil.rmtree(mdir, ignore_errors=True)       # keep only the window stage


def merge(out, shard_dir, subjects, args, log):
    shards = [(s, Path(shard_dir) / f"sub-{s}.npz") for s in subjects if (Path(shard_dir) / f"sub-{s}.npz").exists()]
    heads = [(s, np.load(p, allow_pickle=False)) for s, p in shards]
    ch = next([str(c) for c in z["ch"]] for _, z in heads if len(z["X"]))
    keep = [(s, z) for s, z in heads if len(z["X"]) and [str(c) for c in z["ch"]] == ch]
    dropped = [s for s, z in heads if len(z["X"]) and [str(c) for c in z["ch"]] != ch]
    N = sum(len(z["X"]) for _, z in keep)
    T = int(round(args.duration * FS_OUT))
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float16, shape=(N, len(ch), T))
    import pandas as pd
    rows, i, counts = [], 0, {}
    for s, z in keep:
        n = len(z["X"])
        X[i:i + n] = z["X"]
        rows.append(pd.DataFrame(dict(i=np.arange(i, i + n), subject=str(s), session=z["sess"], run=z["run"],
                                      onset_s=z["onset"], code=z["code"], event=z["event"])))
        counts[str(s)] = n
        i += n
    X.flush()
    pd.concat(rows).to_csv(out / "index.csv", index=False)
    h = hashlib.sha256()
    with open(out / "X.npy", "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    import moabb
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), dataset=args.dataset, moabb=moabb.__version__,
                sfreq=FS_OUT, n_chans=len(ch), n_times=T, ch_names=ch, n_windows=int(N), n_subjects=len(keep),
                window=dict(start=args.start, duration=args.duration, anchor="MOABB event onset"),
                preprocessing=dict(picks="eeg", notch=list(NOTCH), band=list(BAND), resample=FS_OUT, scale=SCALE,
                                   dtype="float16", note="contract pipeline (neuralset EegExtractor, kit 442edcd); "
                                   "notch at 50/60 Hz only (harmonics lie in the band-pass stop band)"),
                per_subject=counts, dropped_channel_mismatch=dropped, x_sha256=h.hexdigest(), purpose="CPT pool (labels unused)")
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"merged {N} windows x {len(ch)} ch from {len(keep)} subjects ({len(dropped)} dropped: channel set) -> {out}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="MOABB class name, e.g. Stieger2021, Lee2019_MI, Cho2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw-root", required=True, help="scratch for per-subject raw downloads (deleted per subject)")
    ap.add_argument("--subjects", default="", help="comma list (default: all)")
    ap.add_argument("--start", type=float, default=1.0, help="window start re event (s); 1.0 = the PROTEUS contract")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[t2p_moabb_export {time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    subjects = [int(s) for s in args.subjects.split(",")] if args.subjects else list(_dataset(args.dataset).subject_list)
    log(f"{args.dataset}: {len(subjects)} subjects, {args.workers} workers x {args.threads} threads")
    t0, n_all, errs = time.time(), 0, {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(export_subject, args.dataset, s, args.raw_root, shard_dir, args.start, args.duration, args.threads)
                for s in subjects]
        for k, f in enumerate(as_completed(futs)):
            s, n, ch, sec, err = f.result()
            n_all += n
            if err:
                errs[str(s)] = err
                log(f"!! subject {s}: {err.splitlines()[0][:300]}")
            else:
                log(f"subject {s}: {n} windows x {len(ch)} ch ({sec:.0f}s) [{k + 1}/{len(subjects)}, total {n_all}, "
                    f"{(time.time() - t0) / 60:.0f} min]")
    (out / "errors.json").write_text(json.dumps(errs, indent=1))
    meta = merge(out, shard_dir, subjects, args, log)
    meta["errors"] = sorted(errs)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    shutil.rmtree(shard_dir)
    log(f"done in {(time.time() - t0) / 60:.0f} min: {meta['n_windows']} windows, {len(errs)} failed subjects")
    sys.exit(0 if meta["n_windows"] else 1)


if __name__ == "__main__":
    main()

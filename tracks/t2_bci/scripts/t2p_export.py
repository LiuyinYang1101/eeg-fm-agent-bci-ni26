#!/usr/bin/env python
"""Track 2 (PROTEUS, kit >= 442edcd): export every window through the OFFICIAL neuralbench pipeline into a memmap store.

Why: since 2026-10-13 Codabench streams PROTEUS windows produced by the kit's ``benchmark_utils.nb_task`` on
neuralbench's ``eeg/_motor_imagery_stream`` task (neuroai 0.4.0): 41 EEG channels, notch 50/60 Hz, band-pass
0.1-75 Hz at the recorded rate, resampled to 120 Hz, x1e6 (V -> uV), NO scaler / clamp, one 4 s window from 1 s
after each cue -> X (1, 41, 480). Training on anything else is a train/test mismatch, so this script runs that
exact pipeline once and stores every window with the identities the sealed cells care about (subject, session,
interface = context, run) plus a data audit.

``--start/--duration`` override the window (research variants, e.g. a wide 0-8 s export for crop augmentation);
the default is the contract window and the output is then the sealed-contract export.

Output (``--out``):
  X.npy        float32 (N, 41, T) memmap -- the windows exactly as predict() receives them (default window)
  index.csv    i, split (kit), subject, session, task, interface, run, rec, onset_s, y, k (window # in its run),
               stream (subject|session), cell (subject|session|interface), order (kit acquisition order within
               the stream), sha1, trig_*
  meta.json    sfreq, ch_names, kit sha, versions, data config, split counts, sha256(X.npy)
  audit.json   label <-> trigger-code map, per-run class counts, split membership, amplitude stats, duplicates,
               constant-predictor scores through the KIT metric

Usage (CPU; zen5_vis on sofia; NI26_KIT must point at a kit >= 442edcd, env ni26b):
  BENCHOPT_DATA_HOME=... NI26_KIT=... python tracks/t2_bci/scripts/t2p_export.py --out <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from ni26 import kit  # noqa: E402

TASK = "_motor_imagery_stream"
INTERFACES = {"AcquisitionBH": "BrainHero", "OnlineRawBH": "BrainHero",
              "AcquisitionGraz": "Graz", "OnlineRawGraz": "Graz"}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def trigger_fields(seg):
    """All scalar fields of a segment's trigger event (robust to the event class)."""
    trig = seg.trigger
    d = {}
    if hasattr(trig, "to_dict"):
        try:
            d.update(trig.to_dict())
        except Exception:
            pass
    for src in (getattr(trig, "__dict__", {}), getattr(trig, "extra", None) or {}):
        for k, v in dict(src).items():
            d.setdefault(k, v)
    out = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None:
            out[k] = v.item() if isinstance(v, (np.integer, np.floating)) else v
    return out


def copy_bids_sidecars(study_dir: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    found = []
    for name in ("participants.tsv", "participants.json", "dataset_description.json", "README", "README.md",
                 "CHANGES", "LICENSE", "task-*_events.json", "task-*_eeg.json"):
        for p in list(study_dir.glob(name)) + list(study_dir.glob(f"*/{name}"))[:3]:
            if p.is_file() and p.stat().st_size < 5_000_000:
                shutil.copy2(p, dest / p.name)
                found.append(str(p.relative_to(study_dir)))
    subs = sorted(s for s in study_dir.rglob("sub-*") if s.is_dir())
    if subs:
        s0 = subs[0]
        listing = sorted(str(p.relative_to(study_dir)) + (f"  {p.stat().st_size}" if p.is_file() else "/")
                         for p in s0.rglob("*"))
        (dest / "subject0_listing.txt").write_text("\n".join(listing))
        for p in list(s0.rglob("*.tsv"))[:40]:
            if p.stat().st_size < 2_000_000:
                shutil.copy2(p, dest / p.name)
                found.append(str(p.relative_to(study_dir)))
        for p in list(s0.rglob("*.json"))[:10]:
            if p.stat().st_size < 200_000:
                shutil.copy2(p, dest / p.name)
                found.append(str(p.relative_to(study_dir)))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None, help="default: $BENCHOPT_DATA_HOME/neural_compet")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--start", type=float, default=None, help="window start re cue (s); default = task (1.0)")
    ap.add_argument("--duration", type=float, default=None, help="window length (s); default = task (4.0)")
    ap.add_argument("--max-windows", type=int, default=0, help="debug: stop after N windows per split")
    ap.add_argument("--study", default=None, help="EXTERNAL corpus through the same stream pipeline: override "
                    "study.source.name (e.g. Scherer2015Individually = BNCI2015_004; its marker is the TRIAL start, the "
                    "cue +3 s, so the contract window is --start 4.0)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="extra data-config override (repeatable), e.g. channel_positions.layout_or_montage_name=standard_1005 "
                         "(Dreyer2023 records no electrode positions; neuralbench's own dreyer2023 overlay sets exactly this)")
    ap.add_argument("--label-map", default=None, help="trigger code -> PROTEUS class, e.g. '4:0,2:1,1:2' (Scherer2015 "
                    "codes 1..5 = WORD, SUB, NAV, HAND, FEET; MOABB 1.7.2's names are wrong, hub page); unmapped -> y=-1")
    args = ap.parse_args()

    kit.use_track("bci_decoding")
    from benchmark_utils import nb_task  # the kit's own loader code
    from benchmark_utils.metrics import balanced_accuracy, group_scores
    from neuralbench.data import Data

    data_dir = Path(args.data_dir or Path(os.environ["BENCHOPT_DATA_HOME"]) / "neural_compet")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[t2p_export] kit {kit.kit_root()} sha {kit.kit_sha()}", flush=True)
    if not args.no_download and not args.study:
        nb_task.download_study("eeg", TASK, data_dir)
    print(f"[t2p_export] download/check {time.time() - t0:.0f}s", flush=True)

    overrides = {"batch_size": 64, "seed": 42, "num_workers": 0, "pin_memory": False, "persistent_workers": False}
    if args.start is not None:
        overrides["start"] = args.start
    if args.duration is not None:
        overrides["duration"] = args.duration
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = v
    if args.study:
        overrides["study.source.name"] = args.study
        if not args.no_download:   # download_study builds its config without our override: mirror it
            import neuralset as ns
            st = dict(nb_task._task_data_config("eeg", TASK, None, data_dir, overrides)["study"]["source"])
            st["path"] = Path(st["path"]) / st["name"]
            ns.Study(**st).download()
    lmap = {str(a): int(b) for a, b in (kv.split(":") for kv in args.label_map.split(","))} if args.label_map else None
    cfg = nb_task._task_data_config("eeg", TASK, None, data_dir, overrides)
    contract = args.start is None and args.duration is None
    print(f"[t2p_export] window start={cfg.get('start')} duration={cfg.get('duration')} contract={contract} "
          f"neuro={ {k: cfg['neuro'][k] for k in ('frequency', 'filter', 'notch_filter', 'scaler', 'clamp', 'scale_factor') if k in cfg['neuro']} }",
          flush=True)
    loaders = Data(**cfg).prepare()
    print(f"[t2p_export] prepare {time.time() - t0:.0f}s splits={ {k: len(v.dataset) for k, v in loaders.items()} }",
          flush=True)
    study_dir = data_dir / cfg["study"]["source"]["name"]
    try:
        bids = copy_bids_sidecars(study_dir, out / "bids")
    except Exception as e:  # sidecars are a convenience; never fail the export on them
        bids = [f"!! {e!r}"]

    first = next(v.dataset for v in loaders.values() if len(v.dataset))
    x0 = np.asarray(first[0].data["neuro"])[0]
    C, T = x0.shape
    ch_names = nb_task._channel_names(first)
    neuro = first.extractors["neuro"]
    sfreq = nb_task._window_sfreq(neuro.frequency, T, cfg.get("duration"))
    sizes = {k: (len(v.dataset) if not args.max_windows else min(len(v.dataset), args.max_windows))
             for k, v in loaders.items()}
    n_total = sum(sizes.values())
    X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=(n_total, C, T))

    rows, i = [], 0
    trig_keys = None
    for split, ld in loaders.items():
        ds = ld.dataset
        for j in range(sizes[split]):
            b = ds[j]
            x = np.asarray(b.data["neuro"], dtype=np.float32)[0]
            X[i] = x
            seg = b.segments[0]
            tf = trigger_fields(seg)
            if trig_keys is None:
                trig_keys = sorted(tf)
                print(f"[t2p_export] trigger fields of window 0: {tf}", flush=True)
            y = int(np.asarray(b.data["target"]).reshape(-1).argmax())  # one-hot (3,) -> class, as the kit does
            y_kit = y
            if lmap is not None:
                y = lmap.get(str(tf.get("code", "")), -1)
            task = str(tf.get("task", ""))
            rows.append(dict(
                i=i, split=split, subject=str(tf.get("subject", "")), session=str(tf.get("session", "")),
                task=task, interface=INTERFACES.get(task, "ext" if args.study else "?"),
                run=int(tf["run"]) if str(tf.get("run", "")).lstrip("-").isdigit() else -1,
                timeline=str(seg.timeline), onset_s=float(seg.start), y=y, y_kit=y_kit,
                sha1=hashlib.sha1(x.tobytes()).hexdigest(),
                **{f"trig_{k}": v for k, v in tf.items() if k not in ("subject", "session", "task", "run")},
            ))
            i += 1
            if i % 2000 == 0:
                print(f"[t2p_export] {i}/{n_total} windows {time.time() - t0:.0f}s", flush=True)
    X.flush()

    import pandas as pd
    idx = pd.DataFrame(rows)
    recs = {t: r for r, t in enumerate(sorted(idx["timeline"].unique()))}
    idx.insert(8, "rec", idx["timeline"].map(recs).astype(int))
    idx["stream"] = idx["subject"] + "|" + idx["session"]
    idx["cell"] = idx["stream"] + "|" + idx["interface"]
    # kit acquisition order within a stream: runs by BIDS run number, windows by onset
    idx = idx.sort_values(["stream", "run", "rec", "onset_s"], kind="stable")
    idx["order"] = idx.groupby("stream").cumcount()
    idx["k"] = idx.groupby("rec").cumcount()
    idx = idx.sort_values("i").reset_index(drop=True)
    idx.to_csv(out / "index.csv", index=False)

    # ---------------- audit ----------------
    Xm = np.load(out / "X.npy", mmap_mode="r")
    audit = {"window": dict(start=cfg.get("start"), duration=cfg.get("duration"), contract=contract)}
    audit["n_windows"] = int(len(idx))
    audit["n_recordings"] = int(idx["rec"].nunique())
    audit["n_subjects"] = int(idx["subject"].nunique())
    audit["n_streams"] = int(idx["stream"].nunique())
    audit["n_cells"] = int(idx["cell"].nunique())
    audit["split_windows"] = idx["split"].value_counts().to_dict()
    audit["split_subjects"] = {s: sorted(g["subject"].unique().tolist()) for s, g in idx.groupby("split")}
    audit["sessions_per_subject"] = idx.groupby("subject")["session"].unique().apply(sorted).apply(list).to_dict()
    audit["class_counts"] = idx["y"].value_counts().sort_index().to_dict()
    # label <-> trigger code / description (must read 0 = MI, 1 = subtraction, 2 = word per the FAQ)
    lab = {}
    for c in [c for c in idx.columns if c in ("trig_code", "trig_description", "trig_value", "trig_trial_type")]:
        lab[c] = {int(k): sorted(map(str, v.unique().tolist()))[:10] for k, v in idx.groupby("y")[c]}
    audit["label_map"] = lab
    per_run = idx.groupby(["subject", "session", "run", "task"]).agg(
        n=("i", "size"), split=("split", "first"), n_y0=("y", lambda s: int((s == 0).sum())),
        n_y1=("y", lambda s: int((s == 1).sum())), n_y2=("y", lambda s: int((s == 2).sum())),
        onset_min=("onset_s", "min"), onset_max=("onset_s", "max"))
    audit["per_run"] = per_run.reset_index().to_dict("records")
    # trial order: P(next label == this label) within a run (LESSONS: non-random order makes pre-cue decodable)
    rep = []
    for _, g in idx.sort_values(["rec", "onset_s"]).groupby("rec"):
        yv = g["y"].to_numpy()
        if len(yv) > 1:
            rep.append(float((yv[1:] == yv[:-1]).mean()))
    audit["p_repeat_within_run_mean"] = float(np.mean(rep)) if rep else None
    # inter-window gaps (s) within a run: overlap check
    gaps = idx.sort_values(["rec", "onset_s"]).groupby("rec")["onset_s"].diff().dropna()
    audit["onset_gap_s_percentiles"] = {q: float(np.percentile(gaps, q)) for q in (0, 5, 50, 95, 100)} if len(gaps) else {}
    q = {c: [] for c in range(C)}
    n_nan = n_flat = n_big = 0
    for s in range(0, len(idx), 1000):
        xb = np.asarray(Xm[s:s + 1000], dtype=np.float64)
        n_nan += int(np.isnan(xb).any(axis=(1, 2)).sum())
        sd = xb.std(axis=-1)
        n_flat += int((sd < 0.1).any(axis=1).sum())
        n_big += int((np.abs(xb).max(axis=-1) > 500).any(axis=1).sum())
        for c in range(C):
            q[c].append(sd[:, c])
    audit["amplitude"] = dict(
        windows_with_nan=n_nan, windows_with_flat_channel_sd_lt_0p1uV=n_flat, windows_with_abs_gt_500uV=n_big,
        channel_sd_uV_percentiles={(ch_names[c] if ch_names else str(c)):
                                   {p: float(np.percentile(np.concatenate(q[c]), p)) for p in (1, 50, 99)}
                                   for c in range(C)})
    # per-session scale (median channel sd): the drift a per-session normaliser must absorb
    sess_sd = {}
    for st, g in idx.groupby("stream"):
        xb = np.asarray(Xm[g["i"].to_numpy()[:200]], dtype=np.float64)
        sess_sd[st] = float(np.median(xb.std(axis=-1)))
    audit["stream_median_sd_uV"] = sess_sd
    dup = idx[idx.duplicated("sha1", keep=False)]
    audit["duplicates"] = dict(
        windows_sharing_bytes=int(len(dup)),
        cross_recording_groups=int(dup.groupby("sha1")["rec"].nunique().gt(1).sum()) if len(dup) else 0,
        examples=dup.head(10)[["i", "rec", "subject", "session", "run", "onset_s", "y"]].to_dict("records")
        if len(dup) else [])
    # constant predictors through the KIT metric helpers, cells = stream x interface (kit objective)
    consts = {}
    for split in ("val", "test"):
        sub = idx[(idx["split"] == split) & (idx["y"] >= 0)]
        if len(sub):
            cells = np.stack([sub["stream"].astype("category").cat.codes.to_numpy(),
                              sub["interface"].astype("category").cat.codes.to_numpy()], axis=1)
            consts[split] = {str(c): float(group_scores(balanced_accuracy, sub["y"].to_numpy(),
                                                        np.full(len(sub), c), cells).mean()) for c in (0, 1, 2)}
    audit["constant_cell_bacc"] = consts
    audit["bids_sidecars"] = bids
    (out / "audit.json").write_text(json.dumps(audit, indent=1, default=str))

    import importlib.metadata as im
    vers = {}
    for p in ("neuralset", "neuralfetch", "neuraltrain", "neuralbench", "exca", "mne", "numpy", "scipy",
              "scikit-learn", "torch", "nemar-py", "moabb"):
        try:
            vers[p] = im.version(p)
        except Exception:
            vers[p] = None
    meta = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=os.uname().nodename, task=f"eeg/{TASK}",
        study=cfg["study"]["source"]["name"], sfreq=sfreq, n_chans=int(C), n_times=int(T), ch_names=ch_names,
        window_start=cfg.get("start"), window_duration=cfg.get("duration"), contract=contract,
        label_map=args.label_map, overrides=args.set,
        n_windows=int(n_total), split_counts=idx["split"].value_counts().to_dict(),
        n_subjects=int(idx["subject"].nunique()), n_recordings=int(idx["rec"].nunique()),
        trigger_fields=trig_keys, kit_sha=kit.kit_sha(), versions=vers,
        data_cfg=json.loads(json.dumps(cfg, default=str)),
        x_sha256=sha256_file(out / "X.npy"), seconds=round(time.time() - t0, 1),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: audit[k] for k in ("window", "n_windows", "n_recordings", "n_subjects", "n_streams",
                                            "n_cells", "split_subjects", "sessions_per_subject", "class_counts",
                                            "label_map", "p_repeat_within_run_mean", "onset_gap_s_percentiles",
                                            "duplicates", "constant_cell_bacc")}, indent=1, default=str), flush=True)
    print(f"[t2p_export] DONE {n_total} windows ({C}x{T} @ {sfreq} Hz) -> {out}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

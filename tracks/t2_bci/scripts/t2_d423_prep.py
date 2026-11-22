#!/usr/bin/env python
"""T2-D423 Stage A data (pre-registered in docs/DECISIONS_T2.md): OpenNeuro ds003029 1.0.7 (CC0, "Epilepsy-iEEG-Multicenter-Dataset") -> one float32 array
per run at the codec's 512 Hz, in µV, holding the channels that channels.tsv marks type ECOG / SEEG and status good. Polyphase resampling
(scipy.signal.resample_poly with a rational ratio; relative rate error printed, < 1e-4). Output: <out>/runs/<run>.npy (C, T) + <out>/manifest.json,
and (--pack, for the sofia push: sofia's file count is near its cap) <out>/packed/{flat.npy, manifest.json}: every run's (C, T) block raveled into ONE
float32 array, with its offset / n_ch / T in the manifest.
    mem-guard python tracks/t2_bci/scripts/t2_d423_prep.py --bids <ds003029 dir> --out <dir>
"""
import argparse
import json
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

SR = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); (out / "runs").mkdir(parents=True, exist_ok=True)
    man = []
    for vh in sorted(Path(a.bids).glob("sub-*/ses-*/ieeg/*_ieeg.vhdr")):
        run = vh.name[:-len("_ieeg.vhdr")]
        ch = pd.read_csv(vh.with_name(run + "_channels.tsv"), sep="\t")
        keep = ch[ch.type.str.upper().isin(["ECOG", "SEEG"]) & (ch.status.astype(str).str.lower() == "good")].name.tolist()
        if not keep:
            print(f"[prep] {run}: no good ECOG / SEEG channel, skipped", flush=True); continue
        raw = mne.io.read_raw_brainvision(vh, preload=False, verbose="error")
        keep = [c for c in keep if c in raw.ch_names]
        x = raw.get_data(picks=keep).astype(np.float64) * 1e6          # V -> µV
        fs = float(raw.info["sfreq"])
        fr = Fraction(SR / fs).limit_denominator(2000)
        y = resample_poly(x, fr.numerator, fr.denominator, axis=1).astype(np.float32)
        err = abs(fs * fr.numerator / fr.denominator - SR) / SR
        assert err < 1e-4, (run, fs, fr, err)
        ok = np.isfinite(y).all(1)
        y = y[ok]
        np.save(out / "runs" / f"{run}.npy", y)
        sd = np.median(y.std(1)) if len(y) else float("nan")
        man.append(dict(run=run, sub=run.split("_")[0], n_ch=int(len(y)), T=int(y.shape[1]), seconds=round(y.shape[1] / SR, 2), sfreq_in=fs, ratio=f"{fr.numerator}/{fr.denominator}",
                        rate_err=err, median_std_uV=float(sd), dropped_nonfinite=int((~ok).sum())))
        print(f"[prep] {run}: {len(y)} ch, {y.shape[1] / SR:.0f} s, fs {fs:.2f} -> {SR} ({fr}), median std {sd:.1f} µV", flush=True)
    json.dump(dict(source="OpenNeuro ds003029 1.0.7 (CC0)", sr=SR, runs=man), open(out / "manifest.json", "w"), indent=1)
    tot = sum(m["n_ch"] * m["seconds"] for m in man) / 3600
    print(f"[prep] DONE {len(man)} runs, {len({m['sub'] for m in man})} subjects, {tot:.1f} channel-hours", flush=True)
    if a.pack:
        pk = out / "packed"; pk.mkdir(exist_ok=True)
        off, rows = 0, []
        for m in man:
            rows.append(dict(m, offset=off)); off += m["n_ch"] * m["T"]
        flat = np.lib.format.open_memmap(pk / "flat.npy", mode="w+", dtype=np.float32, shape=(off,))
        for r in rows:
            y = np.load(out / "runs" / f"{r['run']}.npy")
            assert y.shape == (r["n_ch"], r["T"]), (r["run"], y.shape, r["T"])
            flat[r["offset"]:r["offset"] + y.size] = y.ravel()
        flat.flush(); del flat
        json.dump(dict(source="OpenNeuro ds003029 1.0.7 (CC0)", sr=SR, packed=True, runs=rows), open(pk / "manifest.json", "w"), indent=1)
        print(f"[prep] packed {len(rows)} runs -> {pk}/flat.npy ({off * 4 / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    main()

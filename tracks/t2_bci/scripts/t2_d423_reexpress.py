#!/usr/bin/env python
"""T2-D423 (pre-registered in docs/DECISIONS_T2.md): re-express the PROTEUS export through the frozen codec prior, and the Stage A FIDELITY GATE.
  every window, every channel: 120 Hz -> 512 Hz (resample_poly 64 / 15: 480 -> 2048 samples = exactly one 4 s codec chunk), codec (eval:
  encoder -> RVQ nearest codes -> decoder), -> 120 Hz (15 / 64). Writes <out>/X.npy (same shape and dtype) + index.csv + meta.json (x_sha256 updated,
  provenance added) = an export the bench reads with --export.
  GATE: per window-channel, the Pearson correlation of the 8-30 Hz band-passed (4th-order Butterworth, zero phase) original vs re-expressed signal;
  median < 0.5 -> KILLED. Also reported: the same for the resampling round trip alone (no codec) and the PRD.
    gpu-submit python tracks/t2_bci/scripts/t2_d423_reexpress.py --export <plain export> --ckpt codec_last.pt --out <dir>
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebrain as R  # noqa: E402

KILL = 0.5


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model, quant = R.build_codec(ck["nq"])
    model.load_state_dict(ck["model"]); quant.load_state_dict(ck["quantizer"])
    return model.to(dev).eval(), quant.to(dev).eval(), ck


def band_corr(a, b, sos):
    fa, fb = sosfiltfilt(sos, a, axis=-1), sosfiltfilt(sos, b, axis=-1)
    fa -= fa.mean(-1, keepdims=True); fb -= fb.mean(-1, keepdims=True)
    return (fa * fb).sum(-1) / np.sqrt((fa ** 2).sum(-1) * (fb ** 2).sum(-1) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="first N windows only (a code check; writes no export)")
    a = ap.parse_args()
    E, out = Path(a.export), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((E / "meta.json").read_text())
    fs = float(meta["sfreq"]); assert fs == 120.0, fs
    X = np.load(E / "X.npy", mmap_mode="r")
    X = np.asarray(X[:a.limit] if a.limit else X).astype(np.float64)   # (n, C, 480) µV
    n, C, T = X.shape; assert T == 480, T
    up = resample_poly(X, 64, 15, axis=-1)                           # (n, C, 2048)
    assert up.shape[-1] == R.SR * R.SEG_S, up.shape
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, quant, ck = load(a.ckpt, dev)
    flat = up.reshape(-1, up.shape[-1]).astype(np.float32)
    rec = np.empty_like(flat)
    for i in range(0, len(flat), a.batch):
        rec[i:i + a.batch] = R.reexpress_eval(model, quant, torch.from_numpy(flat[i:i + a.batch]).to(dev)).cpu().numpy()
    Y = resample_poly(rec.reshape(up.shape).astype(np.float64), 15, 64, axis=-1)[..., :T]
    RT = resample_poly(up, 15, 64, axis=-1)[..., :T]                 # the round trip without the codec
    sos = butter(4, [8, 30], btype="bandpass", fs=fs, output="sos")
    c_codec, c_rt = band_corr(X, Y, sos), band_corr(X, RT, sos)
    prd = 100 * np.sqrt(((X - Y) ** 2).sum(-1) / np.maximum((X ** 2).sum(-1), 1e-6))
    st = dict(step=int(ck["step"]), nq=int(ck["nq"]), median_corr_8_30=float(np.median(c_codec)), p10_corr_8_30=float(np.percentile(c_codec, 10)),
              median_corr_8_30_roundtrip_only=float(np.median(c_rt)), median_prd=float(np.median(prd)),
              std_ratio_median=float(np.median(Y.std(-1) / np.maximum(X.std(-1), 1e-9))), killed=bool(np.median(c_codec) < KILL))
    print(f"[D423 A] {json.dumps(st)}", flush=True)
    if a.limit:
        return
    Xo = Y.astype(np.float32)
    np.save(out / "X.npy", Xo)
    shutil.copy(E / "index.csv", out / "index.csv")
    m2 = dict(meta, x_sha256=hashlib.sha256(Xo.tobytes()).hexdigest(),
              d423=dict(source_export=str(E), source_x_sha256=meta.get("x_sha256"), ckpt=str(a.ckpt), fidelity=st))
    (out / "meta.json").write_text(json.dumps(m2, indent=1))
    json.dump(st, open(out / "fidelity.json", "w"), indent=1)
    print(f"== STAGE A GATE (pre-registered, median 8-30 Hz corr < {KILL} kills): {'KILLED' if st['killed'] else 'PASS -> Stage B'} "
          f"(median {st['median_corr_8_30']:.3f}; round trip alone {st['median_corr_8_30_roundtrip_only']:.3f})", flush=True)


if __name__ == "__main__":
    main()

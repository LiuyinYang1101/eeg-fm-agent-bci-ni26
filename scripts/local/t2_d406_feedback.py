#!/usr/bin/env python
"""T2-D406 (pre-registered 2026-11-07 08:06, before any code or number): do PROTEUS windows carry a FEEDBACK-EVOKED class signal?
Window = 1-5 s after the cue; feedback from 3 s = window time 2 s. Acquisition runs show sham feedback, Online runs the online decoder's.
 (i)  M6's shape (outputs/t2_endphase/spec.json arm M6, causal SKλ, 12 draws): held-out balanced accuracy BY TASK, new vs returning windows.
 (ii) a phase-locked ERP classifier: windows band-passed 0.5-8 Hz (zero-phase), decimated to 20 Hz; PRE half (0-2 s) vs POST half (2-4 s);
      41 channels, per-stream per-channel z-scored (non-causal: a diagnostic); shrinkage LDA on the kit TRAINING users, scored on the kit
      held-out users (val + test); balanced accuracy overall and by run type (Acquisition / Online).
Reading: signal = (ii) post BA >= 0.45 AND post - pre >= +0.05 in BOTH run types.
  python t2_d406_feedback.py --export E [--out JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def part_i(d):
    spec = json.load(open(O / "t2_endphase/spec.json")); M, w = spec["members"], spec["arms"]["M6"]
    strm, ordr, y = d.stream.to_numpy(), d.order.to_numpy(), d.y.to_numpy()
    subj, task = d.subject.str[-3:].to_numpy(), d.task.to_numpy()
    rows = []
    for i in range(spec["n"]):
        for proto in ("kit", "seen1"):
            for split in ("test", "val"):
                ix, mix = None, 0.0
                for m, wm in w.items():
                    z = np.load(M[m][proto].format(s=M[m].get("offset", 0) + i))
                    ix_, p = (z["test_idx"], z["prob_last"]) if split == "test" else (z["val_idx"], z["prob_val_last"])
                    ix, mix = ix_, mix + wm * p.astype(np.float64)
                pr = causal_sinkhorn(mix, (strm[ix], ordr[ix]), 10, 3, 0.95).argmax(1)
                ret = (subj[ix] == "P10") | ((subj[ix] == "P04") & (proto == "seen1"))
                rows.append(pd.DataFrame(dict(draw=i, task=task[ix], grp=np.where(ret, "returning", "new"), y=y[ix], pred=pr)))
    D = pd.concat(rows)
    out = {}
    for (g, t), x in D.groupby(["grp", "task"]):
        out[f"{g}|{t}"] = dict(ba=bal(x.y.to_numpy(), x.pred.to_numpy()), n=int(len(x) // spec["n"]))
    return out


def part_ii(X, idx, d):
    sos = butter(4, [0.5, 8.0], btype="band", fs=120.0, output="sos")
    F = sosfiltfilt(sos, np.asarray(X, np.float64), axis=-1)[..., ::6]            # 20 Hz, 80 samples
    strm = d.stream.to_numpy()
    for s in np.unique(strm):                                                       # per-stream per-channel z (non-causal diagnostic)
        k = strm == s
        mu, sd = F[k].mean((0, 2), keepdims=True), F[k].std((0, 2), keepdims=True) + 1e-9
        F[k] = (F[k] - mu) / sd
    (_, trm, vam, tem), = B.folds(idx, "kit")
    tr, ho = np.flatnonzero(trm), np.flatnonzero(vam | tem)
    y = d.y.to_numpy(); acq = d.task.str.startswith("Acquisition").to_numpy()
    out = {}
    for nm, sl in (("pre", slice(0, 40)), ("post", slice(40, 80)), ("full", slice(0, 80))):
        Z = F[..., sl].reshape(len(F), -1)
        clf = LDA(solver="lsqr", shrinkage="auto").fit(Z[tr], y[tr])
        pr = clf.predict(Z[ho])
        out[nm] = dict(all=bal(y[ho], pr), acquisition=bal(y[ho][acq[ho]], pr[acq[ho]]), online=bal(y[ho][~acq[ho]], pr[~acq[ho]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default=str(O / "t2_d406/feedback.json"))
    a = ap.parse_args()
    X, idx, meta = B.load_export(a.export)
    d = pd.read_csv(f"{a.export}/index.csv", dtype={"subject": str, "session": str})
    assert len(d) == len(X) and float(meta["sfreq"]) == 120.0
    r1 = part_i(d)
    print("[D406] (i) M6 held-out BA by task (SKλ, 12 draws pooled):")
    for k, v in sorted(r1.items()):
        print(f"   {k:32s} {v['ba']:.4f}  (n/draw {v['n']})")
    r2 = part_ii(X, idx, d)
    print("[D406] (ii) phase-locked 0.5-8 Hz LDA, kit train → held-out users (chance 0.333):")
    for k, v in r2.items():
        print(f"   {k:5s} all {v['all']:.4f} · acquisition {v['acquisition']:.4f} · online {v['online']:.4f}")
    sig = all(r2["post"][t] >= 0.45 and r2["post"][t] - r2["pre"][t] >= 0.05 for t in ("acquisition", "online"))
    print(f"[D406] feedback-lever signal (post >= 0.45 and post - pre >= +0.05 in both run types): {'YES' if sig else 'NO'}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(i=r1, ii=r2, signal=sig), indent=1))


if __name__ == "__main__":
    main()

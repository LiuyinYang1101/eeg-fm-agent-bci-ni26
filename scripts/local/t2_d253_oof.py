#!/usr/bin/env python
"""T2-D253 stage-1 read (local CPU, zero compute), as pre-registered in docs/DECISIONS_T2.md, after `s1 fetch ni26` of ni26-t2p-d253-oof-<jobid>:
the out-of-fold CNN outputs on the ST-L's training windows (folds kitoof_<user> / seen1oof_<user>; pooled FBCNet + ShallowNet, seeds 0-2).
Per protocol (kit, seen1): the OOF blend (t2p_bench.oof_pcnn: FBCNet 4/7 + ShallowNet 3/7, seed means) window accuracy (raw argmax) on the
protocol's training rows, overall and per user; mean p_CNN(y); the gamma-1 loss weights (mean-normalised). Stage-1 stop iff the window
accuracy < 0.40 in either protocol. Writes outputs/t2_d253/oof.json.
  mem-guard python -I scripts/local/t2_d253_oof.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_bench as B  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    d = glob.glob(str(O / "sofia_s1/ni26-t2p-d253-oof-[0-9]*/runs"))
    assert len(d) == 1, d
    n = len(glob.glob(d[0] + "/*oof_*.npz"))
    assert n == 78, f"{n} / 78 OOF result files"
    _, idx, _ = B.load_export(E)
    y = idx["y"].to_numpy().astype(int)
    sub = idx["subject"].to_numpy()
    out, stop = dict(dir=d[0], n_files=n), False
    for proto in ("kit", "seen1"):
        rows = np.flatnonzero(B.folds(idx, proto)[0][1])
        P = B.oof_pcnn(d[0], proto, y, rows, full=True)
        ok = P.argmax(1) == y[rows]
        p = P[np.arange(len(rows)), y[rows]]
        w = 1.0 + (1.0 - p)
        w = w / w.mean()
        per = {u: float(ok[sub[rows] == u].mean()) for u in sorted(set(sub[rows]))}
        out[proto] = dict(n=int(len(rows)), acc=float(ok.mean()), per_user=per, p_y=float(p.mean()),
                          w_min=float(w.min()), w_max=float(w.max()), w_wrong=float(w[~ok].mean()), w_right=float(w[ok].mean()))
        print(f"{proto}: {len(rows)} training windows, OOF blend window accuracy {ok.mean():.4f} (per user "
              + ", ".join(f"{u} {a:.3f}" for u, a in per.items()) + f"); mean p(y) {p.mean():.3f}; gamma-1 weights {w.min():.2f}-{w.max():.2f}, "
              f"mean {w[~ok].mean():.2f} on OOF-wrong / {w[ok].mean():.2f} on OOF-right windows")
        stop |= ok.mean() < 0.40
    out["stop"] = bool(stop)
    print("STAGE 1:", "STOP (OOF window accuracy < 0.40: the weights are near noise)" if stop else "PASS -> stage 2 (the sw1 ST-L) on the user's GO call")
    (O / "t2_d253").mkdir(exist_ok=True)
    (O / "t2_d253/oof.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-D569 W21: a LOCAL smoke of the post-build check's new parts before the sofia check (1633266) runs. NOT a gate, never a package.
OLD = the W20+D557 upload ZIP unzipped (sha 096a428d…); NEW = the same files with submission.py = OLD's (ST-EEGFormer-inlined) template
+ the W20 -> W21lp diff (61 added lines) and config cnn_lp {alpha 0.5, k 10}. On one stream (P04|2, >= 150 windows), on the 5090:
  gate IL (tracks/t2_bci/scripts/t2p_w21lp_pkgcheck.gate_IL), the before-first-refit equality, the FM-part equality, and the stats.
    gpu-submit ... python scripts/local/t2_w21lp_smoke.py --old <dir> --new <dir>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2p_w18_pkgcheck as P18  # noqa: E402
import t2p_w21lp_pkgcheck as P21  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--old", required=True)
ap.add_argument("--new", required=True)
ap.add_argument("--stream", default="P04|2")
a = ap.parse_args()
os.environ["NI26_PARTS"] = "1"
RP.kit.use_track("bci_decoding")
X, idx, meta = B.load_export(str(P18.EXP))
dev = "cuda" if torch.cuda.is_available() else "cpu"
m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device=dev)
old = RP.load(Path(a.old), "old", m_meta)
new = RP.load(Path(a.new), "new", m_meta)
assert new.cnn_lp == dict(alpha=0.5, k=10) and old.__dict__.get("cnn_lp") is None, (new.cnn_lp, old.__dict__.get("cnn_lp"))
rows = idx.index[idx["stream"] == a.stream].to_numpy()
rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
xs = np.asarray(X[rows], np.float32)
po, pn = [], []
RP.run_stream(old, xs, dev, po)
RP.run_stream(new, xs, dev, pn)
po, pn = np.stack(po), np.stack(pn)
d = np.abs(po - pn).max(1)
first = int(min(new.is_cfg["refits"]))
print(f"[smoke] {a.stream}: {len(xs)} windows; before the first refit max |dp| {d[:first].max():.2e}; after: {int((d[first:] > 0).sum())} moved, "
      f"max {d.max():.3e}, argmax changed {int((po.argmax(1) != pn.argmax(1)).sum())}", flush=True)
on = {k: RP.run_parts(m_, xs, dev) for k, m_ in (("old", old), ("new", new))}
jf = P18.IDX["fm"]
fm = max(float(np.abs(np.asarray(on["new"][1][t][jf], np.float64) - np.asarray(on["old"][1][t][jf], np.float64)).max()) for t in range(len(xs)))
print(f"[smoke] FM part NEW vs OLD on every window: max |dp| {fm:.2e}", flush=True)
ok, res = P21.gate_IL(new, xs, dev)
print(f"[smoke] gate IL: {'PASS' if ok else 'FAIL'}; stats {res['stats']}; LP changed rows / labels at {res['lp_changed_refits']} refit(s)", flush=True)
assert d[:first].max() <= 1e-12 and fm <= 1e-6 and ok, "smoke FAILED"
print("[smoke] OK", flush=True)

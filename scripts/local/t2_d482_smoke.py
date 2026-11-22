#!/usr/bin/env python
"""T2-D482 (W14, PREPARED, NOT APPROVED) local CPU smoke, no ZIP: M7-W13's gated package unpacked once and run, one window at a time
(t2p_repack.run_stream), on the first N windows of the first public stream through
  w13      : its own submission.py (W13);
  w14      : W14's template, the same config (no member carries a ref) -> must equal w13 BITWISE (probabilities and predictions);
  w14car   : W14 with fm_members[4] marked ref "car" (its weights unchanged) -> must DIFFER (the flag reaches the predict path).
Writes outputs/t2_d482/smoke/smoke.json; the unpacked directories are deleted at the end.
    OMP_NUM_THREADS=8 mem-guard python scripts/local/t2_d482_smoke.py [N]
"""
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_repack as RP  # noqa: E402
from ni26 import kit  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
ZIP = O / "sofia_s1/ni26-t2p-repack-t2p_ms_M7_w13-1615509/t2p_ms_M7_w13.zip"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
W = O / "t2_d482/smoke"
if W.exists():
    shutil.rmtree(W)
base = W / "w13"
base.mkdir(parents=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(base)
src13 = (base / "submission.py").read_text()
m_ = re.search(r"# ---- inlined by t2p_final\.py: src/ni26/fm/steeg_pt\.py \(sha256 [0-9a-f]{12}\)\n.*?# ---- end inlined steeg_pt\.py", src13, re.S)
assert m_, "the package's inlined steeg_pt block"
tpl = (ROOT / "tracks/t2_bci/submission/t2p_submission_w14.py").read_text()
marker = next(ln for ln in tpl.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
src14 = tpl.replace(marker, m_.group(0))
cfg = json.loads((base / "config.json").read_text())
dirs = {"w13": base}
for name, edit in (("w14", None), ("w14car", 4)):
    d = W / name
    d.mkdir()
    for f in base.iterdir():
        if f.name not in ("submission.py", "config.json"):
            os.symlink(f.resolve(), d / f.name)
    (d / "submission.py").write_text(src14)
    c = json.loads(json.dumps(cfg))
    if edit is not None:
        c["fm_members"][edit]["ref"] = "car"
    (d / "config.json").write_text(json.dumps(c))
    dirs[name] = d
meta_e = json.loads((EXP / "meta.json").read_text())
idx = pd.read_csv(EXP / "index.csv")
X = np.load(EXP / "X.npy", mmap_mode="r")
kit.use_track("bci_decoding")
meta = dict(device="cpu", ch_names=list(meta_e["ch_names"]), sfreq=meta_e["sfreq"], n_chans=len(meta_e["ch_names"]), n_times=int(X.shape[-1]), n_classes=3)
g = [g.sort_values("order") for _, g in idx.groupby("stream", sort=False)][0]
xs = np.asarray(X[g["i"].to_numpy()[:N]], np.float32)
res = {}
for name, d in dirs.items():
    m = RP.load(d, name, meta)
    pr = []
    a, _ = RP.run_stream(m, xs, "cpu", pr)
    res[name] = (a, np.stack(pr))
    print(f"{name}: {N} windows done", flush=True)
    del m
same = bool(np.array_equal(res["w13"][1], res["w14"][1]) and np.array_equal(res["w13"][0], res["w14"][0]))
dcar = float(np.abs(res["w14car"][1] - res["w13"][1]).max())
out = dict(windows=N, stream=str(g["stream"].iloc[0]), w14_equals_w13_bitwise=same, w14car_max_abs_dp=dcar,
           w14car_pred_differ=int((res["w14car"][0] != res["w13"][0]).sum()))
(W.parent / "smoke.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out), flush=True)
shutil.rmtree(W)
assert same and dcar > 0, out
print("SMOKE OK")

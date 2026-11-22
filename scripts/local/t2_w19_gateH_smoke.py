#!/usr/bin/env python
"""T2-D549 / T2-D553 smoke of the W19 post-build check's GATES H and HF before any real W19 build exists (descriptive; never a gate verdict): a W19-LIKE package =
W18 build 2's files + the W19 template (steeg_pt inlined exactly as t2p_repack does) + cfg side.refit {"lam": 1} (side nets S5, the old FM
members), then t2p_w19_pkgcheck.gate_H on stream P04|2 on this machine's CPU. It exercises the refit INSIDE a real package (no exception, 0
fallbacks) and the gate's own code. The package dir is deleted at the end.
    mem-guard env OMP_NUM_THREADS=8 python scripts/local/t2_w19_gateH_smoke.py <W18 b2 zip> [stream]
"""
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(REPO / "src"))
os.environ["NI26_PARTS"] = "1"
import t2p_repack as RP  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2p_w19_pkgcheck as P19  # noqa: E402

ZIP = sys.argv[1]
STREAM = sys.argv[2] if len(sys.argv) > 2 else "P04|2"
D = Path.home() / "workspace/outputs/neural_interface_challenge/t2_w19_gateH_smoke_pkg"
if D.exists():
    shutil.rmtree(D)
D.mkdir(parents=True)
with zipfile.ZipFile(ZIP) as z:
    z.extractall(D)
src = (REPO / "tracks/t2_bci/submission/t2p_submission_w19.py").read_text()
st_src = RP.STEEG_PT.read_text()
sha = hashlib.sha256(st_src.encode()).hexdigest()[:12]
old_sha = re.search(r"inlined by t2p_final\.py: src/ni26/fm/steeg_pt\.py \(sha256 ([0-9a-f]{12})\)", (D / "submission.py").read_text())
assert old_sha and old_sha.group(1) == sha, (old_sha, sha)
marker = next(ln for ln in src.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
src = src.replace(marker, f"# ---- inlined by t2p_final.py: src/ni26/fm/steeg_pt.py (sha256 {sha})\n" + st_src + "# ---- end inlined steeg_pt.py")
(D / "submission.py").write_text(src)
cfg = json.loads((D / "config.json").read_text())
cfg["side"]["refit"] = {"lam": 1.0}
cfg["fm_side_refit"] = {"lam": 1.0}                      # T2-D553: gate HF on the same package
(D / "config.json").write_text(json.dumps(cfg, indent=1))
RP.kit.use_track("bci_decoding")
X, idx, meta = B.load_export(str(P19.P18.EXP))
m_meta = dict(ch_names=list(meta["ch_names"]), sfreq=float(meta["sfreq"]), n_chans=len(meta["ch_names"]), n_times=int(X.shape[-1]), n_classes=3, device="cpu")
model = RP.load(D, "w19smoke", m_meta)
rows = idx.index[idx["stream"] == STREAM].to_numpy()
rows = rows[np.argsort(idx["order"].to_numpy()[rows], kind="stable")]
ok, res = P19.gate_H(model, np.asarray(X[rows], np.float32), "cpu")
print(f"gate H smoke on {STREAM} ({len(rows)} windows): {'PASS' if ok else 'FAIL'}; stats {res['stats']}", flush=True)
for r in res["refits"]:
    print(f"  refit {r['refit']}: {r}", flush=True)
okF, resF = P19.gate_HF(model, np.asarray(X[rows], np.float32), "cpu")
print(f"gate HF (T2-D553) smoke on {STREAM}: {'PASS' if okF else 'FAIL'}; stats {resF['stats']}; headed {resF['headed']}; "
      f"before the first refit max |dp| {resF['before_first_refit_max_dp']:.1e}", flush=True)
for r in resF["refits"]:
    print(f"  FM refit {r['refit']}: {r}", flush=True)
shutil.rmtree(D)
sys.exit(0 if (ok and okF) else 3)

#!/usr/bin/env python
"""T2-D525 runtime PREPARATION (a scratch TIMING zip, never a ship candidate, never uploaded): W17 build 1 (t2p_fin1_w17.zip) with the W18
template and the imcoh switch on (cfg imcoh = configs/t2_d525_imcoh.json, imcoh.npz from t2_d525_final.py); every other file byte for byte.
Timed on one H200 with t2p_time_zip.slurm against W17 build 1's 54.6 / 53.9 / 53.9 ms (1619661). steeg_pt.py is inlined exactly as
t2p_repack does (its sha must equal the one the W17 package inlined).
    python scripts/local/t2_w18_timezip.py
"""
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
O = Path.home() / "workspace/outputs/neural_interface_challenge"
BASE = O / "sofia_s1/ni26-t2p-repack-t2p_fin1_w17-1619616/t2p_fin1_w17.zip"
OUT = O / "t2_w18_time/t2p_w18ic_timeonly"
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
with zipfile.ZipFile(BASE) as z:
    names = z.namelist()
    z.extractall(OUT)
old_src = (OUT / "submission.py").read_text()
st_src = (REPO / "src/ni26/fm/steeg_pt.py").read_text()
sha = hashlib.sha256(st_src.encode()).hexdigest()[:12]
m = re.search(r"inlined by t2p_final\.py: src/ni26/fm/steeg_pt\.py \(sha256 ([0-9a-f]{12})\)", old_src)
assert m and m.group(1) == sha, (m, sha)
src = (REPO / "tracks/t2_bci/submission/t2p_submission_w18.py").read_text()
marker = next(ln for ln in src.splitlines() if ln.startswith("# @@INLINE_STEEG_PT@@"))
src = src.replace(marker, f"# ---- inlined by t2p_final.py: src/ni26/fm/steeg_pt.py (sha256 {sha})\n" + st_src + "# ---- end inlined steeg_pt.py")
(OUT / "submission.py").write_text(src)
cfg = json.loads((OUT / "config.json").read_text())
assert not cfg.get("imcoh") and not cfg.get("fm_side"), "the base is W17"
cfg.update(json.loads((REPO / "tracks/t2_bci/submission/configs/t2_d525_imcoh.json").read_text()))
(OUT / "config.json").write_text(json.dumps(cfg, indent=1))
shutil.copyfile(O / "t2_d525_final/imcoh.npz", OUT / "imcoh.npz")
zp = OUT.parent / "t2p_w18ic_timeonly.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in names + ["imcoh.npz"]:
        z.write(OUT / f, f)
print(f"[w18 timezip] {zp} ({zp.stat().st_size / 1e6:.1f} MB, sha256 {hashlib.sha256(zp.read_bytes()).hexdigest()[:16]}); "
      f"files {len(names) + 1}; cfg imcoh {cfg['imcoh']}")

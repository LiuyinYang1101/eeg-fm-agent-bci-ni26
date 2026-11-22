#!/usr/bin/env python
"""T2-D642 (W25, user-balanced FM members): t2p_fm.fit_eval_fm's data['wtr'] (train.py's 'wtr' semantics: per batch sum w ce / sum w), CPU:
  1. wce: norm False == the pre-D641 expression bitwise; norm True == sum w ce / sum w; a global rescale of w (x 7.3) cancels (<= 1e-6);
  2. fit_eval_fm WITHOUT wtr == the pre-D641 t2p_fm.py (pinned commit 67fae8e0) BITWISE (prob_last), a random-init ST-base, 1 layer, 1 epoch;
  3. wtr all ones == unweighted (<= 1e-5); 1/3 weights on some rows move the output; wtr + sw refused; wtr with mixup runs (both terms
     self-normalised).
  python tests/test_t2_d642_fm_wtr.py  ->  "OK"
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

REF_COMMIT = "67fae8e0"
CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    # 1. wce
    g = torch.Generator().manual_seed(642)
    lo, y = torch.randn(64, 3, generator=g), torch.randint(0, 3, (64,), generator=g)
    w = torch.where(torch.rand(64, generator=g) < 0.4, torch.tensor(1 / 3), torch.tensor(1.0))
    old = (F.cross_entropy(lo, y, label_smoothing=0.1, reduction="none") * w).mean()
    assert torch.equal(FM.wce(lo, y, 0.1, w), old) and torch.equal(FM.wce(lo, y, 0.1, w, norm=False), old)
    ce = F.cross_entropy(lo, y, label_smoothing=0.1, reduction="none")
    n1, n2 = FM.wce(lo, y, 0.1, w, norm=True), FM.wce(lo, y, 0.1, 7.3 * w, norm=True)
    assert abs(float(n1 - (ce * w).sum() / w.sum())) < 1e-7 and abs(float(n1 - n2)) < 1e-6
    assert torch.equal(FM.wce(lo, y, 0.1, None, norm=True), F.cross_entropy(lo, y, label_smoothing=0.1))
    print(f"1. wce: norm False == the D253 mean(w ce) bitwise; norm True == sum w ce / sum w ({float(n1):.6f}); x7.3 rescale {abs(float(n1 - n2)):.1e}")
    # 2. absent wtr == the pre-D641 t2p_fm bitwise
    rng = np.random.default_rng(642)
    nn, nt = 48, 12
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64))
    kw = dict(epochs=1, bs=24, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.1)
    # the reference must sit at t2p_fm.py's depth (it derives REPO from its own path): a unique temp name in the scripts dir, removed after
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_fm_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_fm.py"], capture_output=True, text=True, check=True).stdout)
        assert "T2-D642" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_fm_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_fm_ref"] = OLD
        spec.loader.exec_module(OLD)
        r_new = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
        r_old = OLD.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    finally:
        ref.unlink(missing_ok=True)
    assert np.array_equal(r_new["prob_last"], r_old["prob_last"]), float(np.abs(r_new["prob_last"] - r_old["prob_last"]).max())
    print(f"2. no wtr == the pre-D641 t2p_fm.py ({REF_COMMIT}) bitwise (prob_last, random-init ST-base, 1 layer, 1 epoch)")
    # 3. the weighted path
    r1 = FM.fit_eval_fm("rand_st_base", dict(d, wtr=np.ones(nn, np.float32)), 0, CH, log=lambda m: None, **kw)
    d1 = float(np.abs(r1["prob_last"] - r_new["prob_last"]).max())
    assert d1 < 1e-5, d1
    wv = np.ones(nn, np.float32); wv[::3] = 1 / 3
    r3 = FM.fit_eval_fm("rand_st_base", dict(d, wtr=wv), 0, CH, log=lambda m: None, **kw)
    d3 = float(np.abs(r3["prob_last"] - r_new["prob_last"]).max())
    assert d3 > 1e-4, d3
    try:
        FM.fit_eval_fm("rand_st_base", dict(d, wtr=wv), 0, CH, log=lambda m: None, sw=wv, **kw)
        raise SystemExit("wtr + sw was accepted")
    except AssertionError:
        pass
    rm = FM.fit_eval_fm("rand_st_base", dict(d, wtr=wv), 0, CH, log=lambda m: None, mixup=0.2, **kw)
    assert np.isfinite(rm["prob_last"]).all()
    print(f"3. wtr = 1 vs unweighted max |dp| {d1:.1e}; 1/3 weights move the output ({d3:.2e}); wtr + sw refused; wtr + mixup runs")
    print("OK")


if __name__ == "__main__":
    main()

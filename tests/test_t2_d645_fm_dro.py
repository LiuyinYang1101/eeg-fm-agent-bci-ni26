#!/usr/bin/env python
"""T2-D645 (FM-DRO, Track3's registration 2026-11-19 21:03:43: online group DRO over the training users in fit_eval_fm), CPU, a random-init
ST-base, 1 layer, 1 epoch:
  1. dro_eta 0 == the pre-D645 t2p_fm.py (pinned commit a0135b47) BITWISE (prob_last);
  2. dro_eta > 0 runs (plain and mixup): q starts uniform, stays a distribution, moves; the log names eta and the final q; the output differs;
  3. refused with wtr / sw / micro > 1;
  4. t2p_bench: --fm-dro absent leaves the tag unchanged; on, the tag gains 'dro<eta>'.
  python tests/test_t2_d645_fm_dro.py  ->  "OK"
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

REF_COMMIT = "a0135b47"
CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(645)
    nn_, nt = 48, 12
    yy, yt = (np.arange(nn_) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn_, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn_) % 4).astype(np.int64) * 3 + 2)   # 4 users, non-contiguous ids
    kw = dict(epochs=1, bs=24, lr_enc=1e-4, lr_head=3e-2, wd=0.05, llrd=0.75, device="cpu", layer=1, calib="gain", ls=0.1)
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_fm_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_fm.py"], capture_output=True, text=True, check=True).stdout)
        assert "T2-D645" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_fm_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_fm_ref"] = OLD
        spec.loader.exec_module(OLD)
        r_off = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
        r_old = OLD.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    finally:
        ref.unlink(missing_ok=True)
    assert np.array_equal(r_off["prob_last"], r_old["prob_last"]), float(np.abs(r_off["prob_last"] - r_old["prob_last"]).max())
    assert "dro_q" not in r_off
    print(f"1. dro_eta 0 == the pre-D645 t2p_fm.py ({REF_COMMIT}) bitwise (prob_last); no dro_q in the result")
    for mx in (0.0, 0.4):
        logs = []
        r = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, dro_eta=0.5, mixup=mx, **kw)
        q = np.asarray(r["dro_q"])
        assert q.shape == (4,) and abs(q.sum() - 1) < 1e-9 and (q > 0).all() and q.max() - q.min() > 1e-6, q
        assert any(m.startswith("T2-D645 FM-DRO: eta 0.5, q uniform over 4 training users") for m in logs)
        assert any(m.startswith("T2-D645 FM-DRO final q:") for m in logs)
        dp = float(np.abs(r["prob_last"] - r_off["prob_last"]).max())
        assert mx > 0 or dp > 0, dp
        print(f"2. dro_eta 0.5, mixup {mx:g}: final q {np.round(q, 4).tolist()} (sums to 1, moved); output vs off max |d| {dp:.2e}")
    for bad, why in ((dict(d=dict(d, wtr=np.ones(nn_, np.float32))), "wtr"), (dict(sw=np.ones(nn_, np.float32)), "sw"), (dict(micro=2), "micro")):
        dd = bad.pop("d", d)
        try:
            FM.fit_eval_fm("rand_st_base", dd, 0, CH, log=lambda m: None, dro_eta=0.5, **bad, **kw)
            raise SystemExit(f"FAIL: dro + {why} was accepted")
        except AssertionError as e:
            assert "T2-D645" in str(e) or "T2-D642" in str(e) or "T2-D253" in str(e), (why, e)
    print("3. dro refused with wtr / sw / micro > 1")
    src = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert '+ (f"dro{args.fm_dro:g}" if getattr(args, "fm_dro", 0.0) else ""))' in src and '"--fm-dro", type=float, default=0.0' in src
    print("4. t2p_bench: tag 'dro<eta>' only when --fm-dro is given (default 0 = the old tag)")
    print("OK")


if __name__ == "__main__":
    main()

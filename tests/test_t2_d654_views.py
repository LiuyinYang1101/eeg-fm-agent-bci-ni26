#!/usr/bin/env python
"""T2-D654 (Track3): the eval-views dump (NI26_VIEW_DUMP=K), CPU, a random-init ST-base (1 layer, 2 epochs, user-balanced + mixup + aug, i.e. W25's C form):
  1. views absent == the pre-D654 t2p_fm.py (REF_COMMIT) BITWISE (prob_last), and with NI26_FEAT_DUMP=1 + NI26_VIEW_DUMP=4 the fit, prob_last and
     the identity feature dump are bitwise those of NI26_FEAT_DUMP=1 alone (the dump never touches the fit, the identity or the global RNG);
  2. the views are the PINNED fixed ones: deterministic, gains in [exp(-0.2), exp(0.2)] from Generator(65400 + v), shifts (15, -15, 30); K = 2's
     view == K = 4's first view; fm_apply_view == torch.roll(x * gain) (TV.augment's gather convention);
  3. an IDENTITY view (gain 1, shift 0) through fm_view_dump reproduces fm_feat_te and prob_last bit for bit (the dump uses the evaluated path);
     real views move the features; probabilities are rows of a simplex; shapes (K - 1, N, D) / (K - 1, N, 3);
  4. the same on the dcmean SIDE path (the side vectors of the identity window);
  5. NI26_VIEW_DUMP without NI26_FEAT_DUMP is refused; t2p_bench writes fm_view_feat / fm_view_prob / fm_view_gain / fm_view_shift for the main
     member and the continuation twin.
  python tests/test_t2_d654_views.py  ->  "OK"
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

REF_COMMIT = "1bcbe297"                   # the last t2p_fm.py before T2-D654
CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def env(**kv):
    for k in ("NI26_FEAT_DUMP", "NI26_VIEW_DUMP"):
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in kv.items()})


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(654)
    nn_, nt = 48, 12
    yy, yt = (np.arange(nn_) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    utr = (np.arange(nn_) % 4).astype(np.int64)
    w = np.where(utr == 0, 1 / 3, 1.0).astype(np.float32)
    d = dict(Xtr=rng.standard_normal((nn_, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=utr, wtr=w)
    kw = dict(epochs=2, bs=24, lr_enc=1e-4, lr_head=3e-2, wd=0.05, llrd=0.75, device="cpu", layer=1, calib="gain", ls=0.1, mixup=0.4, aug=(0.2, 30, 0.1))
    q = lambda **k: FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw, **k)   # noqa: E731
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_fm_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_fm.py"], capture_output=True, text=True, check=True).stdout)
        assert "T2-D654" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_fm_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_fm_ref"] = OLD
        spec.loader.exec_module(OLD)
        env()
        r_new = q()
        r_old = OLD.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    finally:
        ref.unlink(missing_ok=True)
    assert np.array_equal(r_new["prob_last"], r_old["prob_last"]) and "view_dump" not in r_new
    env(NI26_FEAT_DUMP=1)
    r_f = q()
    env(NI26_FEAT_DUMP=1, NI26_VIEW_DUMP=4)
    r_v = q()
    assert np.array_equal(r_v["prob_last"], r_f["prob_last"]) and np.array_equal(r_v["feat_dump"]["feat"], r_f["feat_dump"]["feat"])
    assert np.array_equal(r_v["prob_last"], r_new["prob_last"])
    vd = r_v["view_dump"]
    D = r_v["feat_dump"]["feat"].shape[1]
    assert vd["feat"].shape == (3, nt, D) and vd["prob"].shape == (3, nt, 3) and np.allclose(vd["prob"].sum(-1), 1, atol=1e-5)
    print(f"1. views absent == t2p_fm.py at {REF_COMMIT} bitwise; NI26_VIEW_DUMP=4 leaves the fit / prob_last / fm_feat_te bitwise; dump {vd['feat'].shape}")
    s4, s2 = FM.fm_view_spec(len(CH), 4), FM.fm_view_spec(len(CH), 2)
    assert [s for _, s in s4] == [15, -15, 30] and len(s2) == 1 and torch.equal(s2[0][0], s4[0][0]) and s2[0][1] == s4[0][1]
    assert all(torch.equal(a[0], b[0]) for a, b in zip(s4, FM.fm_view_spec(len(CH), 4)))
    assert all(float(g.min()) >= np.exp(-0.2) - 1e-7 and float(g.max()) <= np.exp(0.2) + 1e-7 and g.shape == (len(CH),) for g, _ in s4)
    assert not torch.equal(s4[0][0], s4[1][0]) and np.array_equal(vd["shift"], [15, -15, 30]) and np.allclose(vd["gain"], np.stack([g.numpy() for g, _ in s4]))
    X = torch.as_tensor(d["Xte"])
    for g, s in s4:
        assert torch.equal(FM.fm_apply_view(X, g, s), torch.roll(X * g.view(1, -1, 1), s, -1))
    print("2. the pinned fixed views: deterministic, gains in [e^-0.2, e^0.2], shifts (15, -15, 30), K = 2 == K = 4's first; apply == roll(x * gain)")
    env(NI26_FEAT_DUMP=1)
    rs = q(return_model=True)
    net = rs["net_last"]                                  # the model object that produced prob_last (copied before prob_best's reload)
    assert net is not None, sorted(rs)
    Xt = torch.as_tensor(d["Xte"])
    sh0, g0 = FM.VIEW_SHIFTS, FM.VIEW_GAIN
    try:
        FM.VIEW_SHIFTS, FM.VIEW_GAIN = (0,), 0.0
        st0 = torch.get_rng_state()
        idv = FM.fm_view_dump(net, Xt, None, 2)
        assert torch.equal(st0, torch.get_rng_state()), "the global RNG moved"
    finally:
        FM.VIEW_SHIFTS, FM.VIEW_GAIN = sh0, g0
    assert np.array_equal(idv["feat"][0], rs["feat_dump"]["feat"]) and np.array_equal(idv["prob"][0], rs["prob_last"]), "identity view != the evaluated path"
    real = FM.fm_view_dump(net, Xt, None, 4)
    assert float(np.abs(real["feat"] - rs["feat_dump"]["feat"][None]).max()) > 0
    print(f"3. an identity view reproduces fm_feat_te + prob_last bitwise (global RNG untouched); real views move the features "
          f"(max |d| {float(np.abs(real['feat'] - rs['feat_dump']['feat'][None]).max()):.3g})")
    d2 = dict(d, side_tr=rng.standard_normal((nn_, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
              side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    env(NI26_FEAT_DUMP=1, NI26_VIEW_DUMP=2)
    r_s = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=lambda m: None, side="dcmean", return_model=True, **kw)
    env(NI26_FEAT_DUMP=1)
    r_s0 = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=lambda m: None, side="dcmean", **kw)
    assert np.array_equal(r_s["prob_last"], r_s0["prob_last"]) and r_s["view_dump"]["prob"].shape == (1, nt, 3)
    net_s = r_s["net_last"]
    try:
        FM.VIEW_SHIFTS, FM.VIEW_GAIN = (0,), 0.0
        ids = FM.fm_view_dump(net_s, Xt, torch.as_tensor(d2["side_te"]), 2)
    finally:
        FM.VIEW_SHIFTS, FM.VIEW_GAIN = sh0, g0
    assert np.array_equal(ids["prob"][0], r_s["prob_last"]), "side path: identity view != prob_last"
    print("4. the dcmean side path: the dump leaves prob_last bitwise; an identity view (identity side vectors) reproduces prob_last bitwise")
    env(NI26_VIEW_DUMP=2)
    try:
        q()
        raise SystemExit("FAIL: NI26_VIEW_DUMP without NI26_FEAT_DUMP accepted")
    except AssertionError as e:
        assert "T2-D654" in str(e), e
    env()
    src = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    for who in ('r["view_dump"]', 'rc_["view_dump"]'):
        for key in ("fm_view_feat=", "fm_view_prob=", "fm_view_gain=", "fm_view_shift="):
            assert f"{key}{who}" in src, (who, key)
    print("5. NI26_VIEW_DUMP without NI26_FEAT_DUMP refused; t2p_bench writes the four view keys for the member and its continuation twin")
    print("OK")


if __name__ == "__main__":
    main()

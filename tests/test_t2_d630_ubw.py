#!/usr/bin/env python
"""T2-D630 (the pooled-training composition), CPU:
  1. t2p_bench.folds: 'ss8' = the 8 single-session users held out one at a time (test = all their windows; train = every other window, so P04 /
     P10 train on all 3 sessions); 'rt' = rt1_P04 / rt1_P10 (X trains on session 1 only, test sessions 2-3) and rt2_* (X trains on sessions 1-2,
     test session 3); no overlap, no val; the single-fold names resolve to one fold.
  2. fit_eval without 'wtr' == the pre-D630 train.py (pinned commit c48aa838) BITWISE (state_last and prob_last), FBCNet B3 + ShallowNet, r1;
  3. 'wtr' all ones == the plain CE to float tolerance (the self-normalised weighted mean = the batch mean); a global rescale of wtr (x 7.3) gives
     the same run to float tolerance (k cancels); 1/3-weights on some rows change the run; the guarded branches refuse wtr (mixup).
  python tests/test_t2_d630_ubw.py  ->  "OK"
"""
import importlib.util
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26 import train as T  # noqa: E402
import t2p_bench as B  # noqa: E402

REF_COMMIT = "c48aa838"


def data(seed=0, n=96, C=8, L=480):
    r = np.random.default_rng(seed)
    d = dict(Xtr=r.standard_normal((n, C, L)).astype(np.float32), ytr=r.integers(0, 3, n), Xva=r.standard_normal((0, C, L)).astype(np.float32),
             yva=r.integers(0, 3, 0), cva=np.array([], dtype=object), Xte=r.standard_normal((24, C, L)).astype(np.float32),
             yte=r.integers(0, 3, 24), cte=np.array(["c"] * 24))
    d["ytr"][:3] = [0, 1, 2]
    return d


def run(mod, model, recipe, **extra):
    r = mod.fit_eval(model, dict(data(), **extra), recipe, 0, device="cpu", return_state=True)
    return r["state_last"], r["prob_last"]


def maxd(a, b):
    return max(float((a[k].double() - b[k].double()).abs().max()) for k in a)


def main():
    # 1. folds
    _, idx, _ = B.load_export(str(Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"))
    sub = idx["subject"].astype(str).str.split("/").str[-1].to_numpy()
    ses = idx["session"].astype(str).to_numpy()
    F = B.folds(idx, "ss8")
    assert [f[0] for f in F] == [f"ss8_{u}" for u in ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")], [f[0] for f in F]
    for name, trm, vam, tem in F:
        u = name.split("_")[1]
        assert not vam.any() and not (trm & tem).any() and set(sub[tem]) == {u} and tem.sum() == (sub == u).sum()
        assert (trm | tem).all() and set(ses[trm & (sub == "P04")]) == {"1", "2", "3"} and set(ses[trm & (sub == "P10")]) == {"1", "2", "3"}
        assert len(B.folds(idx, name)) == 1 and np.array_equal(B.folds(idx, name)[0][1], trm)
    R = B.folds(idx, "rt")
    assert [f[0] for f in R] == ["rt1_P04", "rt1_P10", "rt2_P04", "rt2_P10"]
    for name, trm, vam, tem in R:
        k, x = name.split("_")
        o = "P10" if x == "P04" else "P04"
        assert not vam.any() and not (trm & tem).any() and (trm | tem).all() and set(sub[tem]) == {x}
        assert set(ses[tem]) == ({"2", "3"} if k == "rt1" else {"3"}) and set(ses[trm & (sub == x)]) == ({"1"} if k == "rt1" else {"1", "2"})
        assert set(ses[trm & (sub == o)]) == {"1", "2", "3"}
    print("1. ss8: 8 folds (each single-session user held out; P04 / P10 train on 3 sessions); rt: rt1 / rt2 x P04 / P10; no overlap, no val")
    # 2. absent wtr == the pre-D630 train.py bitwise
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "train_ref.py"
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:src/ni26/train.py"], capture_output=True, text=True, check=True).stdout)
        spec = importlib.util.spec_from_file_location("ni26.train_ref", ref)
        old = importlib.util.module_from_spec(spec)
        sys.modules["ni26.train_ref"] = old
        spec.loader.exec_module(old)
        assert "T2-D630" not in ref.read_text(), "the reference already has T2-D630"
        r1 = replace(T.RECIPES["r1"], epochs=2)
        for model in ("fbcnet_bboth", "shallow"):
            s_new, p_new = run(T, model, r1)
            s_old, p_old = run(old, model, r1)
            assert maxd(s_new, s_old) == 0.0 and np.array_equal(p_new, p_old), (model, maxd(s_new, s_old))
    print("2. no wtr == the pre-D630 train.py (c48aa838) bitwise: state_last and prob_last, FBCNet B3 + ShallowNet")
    # 3. the weighted branch
    n = 96
    for model in ("fbcnet_bboth", "shallow"):
        s0, p0 = run(T, model, r1)
        s1, p1 = run(T, model, r1, wtr=np.ones(n, np.float32))
        sk, pk = run(T, model, r1, wtr=np.full(n, 7.3, np.float32))
        w = np.ones(n, np.float32); w[::3] = 1 / 3
        sw, pw = run(T, model, r1, wtr=w)
        swk, pwk = run(T, model, r1, wtr=w * 2.9)
        d1, dk, dw, dwk = maxd(s0, s1), maxd(s1, sk), maxd(s0, sw), maxd(sw, swk)
        assert d1 < 1e-4 and np.abs(p0 - p1).max() < 1e-4, (model, d1)
        assert dk < 1e-4, (model, dk)        # a rescale of uniform weights: the same run (dwk below is reported only: rounding of 1/3 x 2.9
                                             # differs at 1e-7 per loss and Adam amplifies it over the steps; the loss-level identity is asserted in 4.)
        assert dw > 1e-3 and np.abs(p0 - pw).max() > 1e-4, (model, dw)
        print(f"3. {model}: wtr = 1 vs plain CE max |d state| {d1:.1e}; x7.3 rescale {dk:.1e}; 1/3-weights move the run ({dw:.3f}); their x2.9 rescale {dwk:.1e} (reported)")
    import torch
    import torch.nn.functional as F_
    g = torch.Generator().manual_seed(630)
    lg, yy = torch.randn(64, 3, generator=g), torch.randint(0, 3, (64,), generator=g)
    ww = torch.where(torch.rand(64, generator=g) < 0.4, torch.tensor(1 / 3), torch.tensor(1.0))
    ce = F_.cross_entropy(lg, yy, label_smoothing=0.1, reduction="none")
    l1, l2 = (ww * ce).sum() / ww.sum(), (2.9 * ww * ce).sum() / (2.9 * ww).sum()
    l3 = (ww / ww.mean() * ce).mean()
    assert abs(float(l1 - l2)) < 1e-6 and abs(float(l1 - l3)) < 1e-6, (float(l1), float(l2), float(l3))
    print(f"4. the loss: sum w ce / sum w == its x2.9 rescale == the mean-1 normalised mean (w / mean w) * ce ({float(l1):.6f}, max |d| "
          f"{max(abs(float(l1 - l2)), abs(float(l1 - l3))):.1e})")
    try:
        T.fit_eval("shallow", dict(data(), wtr=np.ones(n, np.float32)), replace(T.RECIPES["r1"], epochs=1, mixup=0.2), 0, device="cpu")
        raise SystemExit("wtr with mixup was accepted")
    except AssertionError:
        pass
    print("OK")


if __name__ == "__main__":
    main()

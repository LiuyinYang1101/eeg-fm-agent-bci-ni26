#!/usr/bin/env python
"""T2-D588: consistency regularisation between two independently augmented views of each FM target window (t2p_fm.sym_kl;
fit_eval_fm(cons=w): loss 0.5 (CE1 + CE2) + w x 0.5 (KL(p1 || p2) + KL(p2 || p1))), CPU:
  1. sym_kl: 0 for equal logits, symmetric, equal to the closed form on a hand case, gradients reach both inputs;
  2. OFF = bitwise: fit_eval_fm (random-init ST-base, 1 layer, gain-jitter aug + mixup) with cons=0.0 equals the call without the argument;
  3. cons 1.0 runs, is finite, moves the output and logs the per-epoch KL term; the two views DIFFER (independent augmentation: the logged
     first-step mean |x1 - x2| > 0); with no augmentation on, cons is refused (the two views would be identical);
  4. with the dcmean side head and mixup (D557's form: rc + aug + tilt + mixup + side) it runs and logs the KL term.
  python tests/test_t2p_fm_cons.py  ->  "OK"
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(588)
    # 1. sym_kl
    a = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    k = FM.sym_kl(a, b)
    p, q = torch.softmax(a.detach().double(), -1), torch.softmax(b.detach().double(), -1)
    ref = 0.5 * ((p * (p / q).log()).sum(-1).mean() + (q * (q / p).log()).sum(-1).mean())
    assert abs(float(k.detach()) - float(ref)) < 1e-6 and abs(float(FM.sym_kl(b, a).detach()) - float(k.detach())) < 1e-6 and float(FM.sym_kl(a, a).detach()) == 0.0
    k.backward()
    assert float(a.grad.abs().sum()) > 0 and float(b.grad.abs().sum()) > 0
    print(f"1. sym_kl = the closed form ({float(k.detach()):.5f}), symmetric, 0 on equal logits, gradients reach both views")
    # 2. off = bitwise
    nn, nt = 96, 30
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64))
    kw = dict(epochs=2, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.0, mixup=0.8, aug=(0.2, 30, 0.1))
    r_a = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    r_b = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, cons=0.0, **kw)
    assert np.array_equal(r_a["prob_last"], r_b["prob_last"]), "cons=0.0 is not bitwise"
    print("2. cons 0 = bitwise the call without the argument")
    # 3. on: runs, moves, logs; the views differ; no augmentation -> refused
    logs = []
    r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, cons=1.0, **kw)
    assert not np.array_equal(r_a["prob_last"], r_c["prob_last"]) and np.isfinite(r_c["prob_last"]).all()
    kl = [float(re.search(r"cons KL ([0-9.e+-]+)", m).group(1)) for m in logs if "cons KL" in m]
    vd = [m for m in logs if "consistency views" in m]
    assert len(kl) == 2 and all(np.isfinite(kl)) and kl[0] > 0, kl
    dv = float(re.search(r"mean \|x1 - x2\| ([0-9.e+-]+)", vd[0]).group(1))
    assert dv > 1e-3, vd
    assert any("consistency: two independently augmented views" in m for m in logs)
    pv = [m for m in logs if "T2-D588 (A): per-view batch" in m]    # the H200 OOM fix: bs 32 -> 16 per view, the steps per epoch doubled
    assert len(pv) == 1 and "per-view batch 16 (32 forwards per step)" in pv[0] and pv[0].endswith(f"steps per epoch {-(-len(d['Xtr']) // 16)}"), pv
    try:
        FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, cons=1.0, **dict(kw, aug=None))
        raise SystemExit("cons without any augmentation was not refused")
    except AssertionError as e:
        assert "no augmentation" in str(e), e
    print(f"3. cons 1.0 runs and moves the output; KL per epoch {kl}; the views differ (mean |x1 - x2| {dv:.3g}); per-view batch 16 of bs 32 "
          f"({pv[0].split('; ')[-1]}); no aug -> refused")
    # 4. D557's form with the side head: rc + aug + tilt + mixup + dcmean side
    d2 = dict(d, side_tr=rng.standard_normal((nn, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
              side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    logs4 = []
    r_s = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=logs4.append, side="dcmean", rc=0.75, tilt=0.15, cons=1.0, **kw)
    assert np.isfinite(r_s["prob_last"]).all() and sum("cons KL" in m for m in logs4) == 2
    r_s0 = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=lambda m: None, side="dcmean", rc=0.75, tilt=0.15, **kw)
    assert not np.array_equal(r_s0["prob_last"], r_s["prob_last"])
    print("4. D557's form (rc + aug + tilt + mixup) with the dcmean side head: runs, logs the KL term, moves the output")
    assert math.isfinite(kl[-1])
    print("OK")


if __name__ == "__main__":
    main()

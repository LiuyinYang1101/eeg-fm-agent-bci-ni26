#!/usr/bin/env python
"""T2-D596: the WORD-margin loss in the FM fine-tune (t2p_fm.word_margin; fit_eval_fm(wmargin=)), CPU:
  1. word_margin = the closed form on hand-set logits: WORD rows add softplus(0.5 - (z_W - z_MI)) and softplus(0.5 - (z_W - z_SUB)), MI rows
     softplus(0.5 - (z_MI - z_W)), SUB rows softplus(0.5 - (z_SUB - z_W)); the mean runs over the PAIRS (a WORD row weighs twice an MI / SUB
     row, the registered form); a large correct margin -> ~0, a wrong one -> large; gradients reach the logits;
  2. wmargin 0 = bitwise the call without the argument (the bench / final default);
  3. wmargin 0.3 with mixup + the dcmean side head runs, is finite, moves the output, logs the margin term with the same-class share and the
     logits-identity line; refused with consistency.
  python tests/test_t2p_fm_wmargin.py  ->  "OK"
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]
sp = lambda x: math.log1p(math.exp(x))  # noqa: E731


def main():
    torch.set_num_threads(2)
    lo = torch.tensor([[1.0, 0.2, 2.0],      # WORD row: z_W 2.0, z_MI 1.0, z_SUB 0.2
                       [0.5, 0.1, 1.5],      # MI row: z_MI 0.5 vs z_W 1.5 (wrong side)
                       [0.0, 3.0, 0.1]],     # SUB row: z_SUB 3.0 vs z_W 0.1 (a large correct margin)
                      requires_grad=True)
    y = torch.tensor([2, 0, 1])
    want = [sp(0.5 - (2.0 - 1.0)), sp(0.5 - (2.0 - 0.2)), sp(0.5 - (0.5 - 1.5)), sp(0.5 - (3.0 - 0.1))]
    got = FM.word_margin(lo, y)
    assert abs(float(got) - sum(want) / 4) < 1e-6, (float(got), sum(want) / 4)          # the mean over the 4 PAIRS, not the 3 rows
    assert abs(float(got) - (sum(want[:2]) / 2 + want[2] + want[3]) / 3) > 1e-3          # (a per-row mean would differ)
    got.backward()
    assert lo.grad is not None and float(lo.grad.abs().sum()) > 0
    assert float(FM.word_margin(torch.tensor([[0.0, 0.0, 9.0]]), torch.tensor([2]))) < 1e-3
    assert abs(float(FM.word_margin(torch.tensor([[9.0, 0.0, 0.0]]), torch.tensor([2]))) - (sp(9.5) + sp(0.5)) / 2) < 1e-5   # a wrong WORD row
    print(f"1. word_margin = the closed form over the 4 pairs ({float(got):.5f}); a WORD row has two pairs; gradients flow")
    rng = np.random.default_rng(596)
    nn, nt = 96, 30
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64),
             side_tr=rng.standard_normal((nn, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
             side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    kw = dict(epochs=2, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.0, mixup=0.8, side="dcmean")
    r_a = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    r_b = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, wmargin=0.0, **kw)
    assert np.array_equal(r_a["prob_last"], r_b["prob_last"]), "wmargin 0 is not bitwise"
    print("2. wmargin 0 = bitwise the call without the argument")
    logs = []
    r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, wmargin=0.3, **kw)
    assert np.isfinite(r_c["prob_last"]).all() and not np.array_equal(r_a["prob_last"], r_c["prob_last"])
    ep = [m for m in logs if "wmargin" in m and "same-class rows" in m]
    assert len(ep) == 2 and any("SAME logits tensor as the CE" in m for m in logs), logs[-4:]
    try:
        FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, wmargin=0.3, cons=1.0, **dict(kw, aug=(0.2, 30, 0.1)))
        raise SystemExit("wmargin with consistency was not refused")
    except AssertionError as e:
        assert "wmargin" in str(e), e
    print(f"3. wmargin 0.3 with mixup + the side head runs and moves the output ({ep[-1].split('; ')[-1]}); with consistency refused")
    print("OK")


if __name__ == "__main__":
    main()

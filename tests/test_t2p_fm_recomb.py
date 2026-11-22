#!/usr/bin/env python
"""T2-D583: within-user, within-class segment recombination of the FM target windows (t2p_fm.recomb_*; fit_eval_fm(recomb=(S, p))), CPU:
  1. recomb_masks: S nonnegative masks that sum to 1 at every sample; the joins cross-fade over `fade` samples (no hard step);
  2. recomb_batch draws every segment from the window's OWN (user, class) group of the training windows (checked with hard masks on windows
     that encode their group), never from another user or class, and POSITION-PRESERVING (segment s = the donor's same time slot); p 0
     leaves the batch untouched;
  3. OFF = bitwise: fit_eval_fm on a random-init ST-base (1 layer) with recomb=None equals the call without the argument; recomb (4, 0.5)
     runs, moves the output and logs a recombined-window count;
  4. the side vectors (Track3 LOW on 9de94fd7): recomb_batch(..., side) gives a recombined window the slot-mass-weighted mean of its donors'
     side vectors (checked on constant windows, where that equals the recombined window's own mean, soft joins included), keeps an untouched
     window's own vector, and leaves the x path bitwise the call without side; fit_eval_fm with side dcmean + recomb (4, 0.5) runs.
  python tests/test_t2p_fm_recomb.py  ->  "OK"
"""
import json
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
    # 1. masks
    M = FM.recomb_masks(480, 4, 6.0, dtype=torch.float64)
    assert M.shape == (4, 480) and float(M.min()) >= 0 and float((M.sum(0) - 1).abs().max()) < 1e-12
    for j in (120, 240, 360):
        assert 0 < float(M[j // 120 - 1, j]) < 1 and 0 < float(M[j // 120, j - 1]) < 1, j   # a soft join on both sides
    assert float(M[0, :117].min()) == 1.0 and float(M[3, 363:].min()) == 1.0
    print("1. masks: a partition of unity, 6-sample raised-cosine joins")
    # 2. the pool: windows encode their (user, class) group; with hard masks every recombined segment keeps its group's code
    rng = np.random.default_rng(0)
    n, C, L = 120, 4, 480
    u = rng.integers(0, 3, n)
    y = rng.integers(0, 3, n)
    code = (10 * u + y).astype(np.float32)
    X = torch.as_tensor(np.repeat(code[:, None, None], C, 1).repeat(L, 2) + 0.01 * rng.standard_normal((n, C, L)).astype(np.float32))
    rb = FM.recomb_setup(X, torch.as_tensor(y), u, 4, 1.0, 0.0, 0, log=lambda m: None)
    b = torch.arange(n)
    xr = FM.recomb_batch(X[b], b, rb)
    assert rb["count"] == n
    seg = xr.reshape(n, C, 4, 120).round().mean((1, 3))      # each segment's code
    assert torch.equal(seg, torch.as_tensor(code)[:, None].expand(n, 4).round()), "a segment came from another user / class"
    moved = float((xr - X).abs().amax())
    assert moved > 0.005, moved                               # segments were swapped (the small noise differs)
    # position-preserving (Track3): segment s of the new window = the SAME 1-s slot of its donor (cue-locked windows, Lotte 2015)
    Xt = torch.as_tensor(np.repeat((1000.0 * code)[:, None, None], C, 1).repeat(L, 2) + np.arange(L, dtype=np.float32)[None, None, :])
    rbt = FM.recomb_setup(Xt, torch.as_tensor(y), u, 4, 1.0, 0.0, 1, log=lambda m: None)
    xt = FM.recomb_batch(Xt[b], b, rbt)
    assert torch.equal(torch.remainder(xt, 1000.0), torch.arange(L, dtype=xt.dtype).expand_as(xt)), "a segment moved to another time slot"
    rb0 = FM.recomb_setup(X, torch.as_tensor(y), u, 4, 1e-12, 0.0, 0, log=lambda m: None)
    assert torch.equal(FM.recomb_batch(X[b], b, rb0), X[b])
    print("2. every segment from the window's own (user, class) training group; p ~0 leaves the batch untouched")
    # 3. off = bitwise; on = runs, moves, counts
    nn, nt = 96, 30
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64))
    kw = dict(epochs=2, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.0)
    logs = []
    r_a = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    r_b = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, recomb=None, **kw)
    assert np.array_equal(r_a["prob_last"], r_b["prob_last"]), "recomb=None is not bitwise"
    r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, recomb=(4, 0.5), **kw)
    assert not np.array_equal(r_a["prob_last"], r_c["prob_last"]) and np.isfinite(r_c["prob_last"]).all()
    cnt = [m for m in logs if "recombined" in m]
    assert cnt and any("segment recombination" in m for m in logs), logs[-3:]
    print(f"3. recomb=None bitwise; recomb (4, 0.5) runs and moves the output ({cnt[-1].split(';')[-1].strip()})")
    # 4. side vectors follow the donors
    cst = rng.standard_normal(n).astype(np.float32)
    Xc = torch.as_tensor(np.repeat(cst[:, None, None], C, 1).repeat(L, 2))
    Sc = torch.as_tensor(np.stack([cst, -2 * cst], 1))                   # 2-d side vectors, linear in the window's constant
    ra = FM.recomb_setup(Xc, torch.as_tensor(y), u, 4, 0.5, 6.0, 7, log=lambda m: None)
    rs = FM.recomb_setup(Xc, torch.as_tensor(y), u, 4, 0.5, 6.0, 7, log=lambda m: None)
    xa = FM.recomb_batch(Xc[b], b, ra)
    xs, ss = FM.recomb_batch(Xc[b], b, rs, Sc)
    assert torch.equal(xa, xs) and ra["count"] == rs["count"] and 0 < rs["count"] < n, "the side argument changed the x path"
    tm = xs.mean((1, 2))                                                 # the recombined window's true mean
    assert float((ss[:, 0] - tm).abs().max()) < 1e-5 and float((ss[:, 1] + 2 * tm).abs().max()) < 1e-5, "side != donors' slot-weighted mean"
    same = (xs - Xc[b]).abs().amax((1, 2)) == 0
    assert torch.equal(ss[same], Sc[b][same]) and int((~same).sum()) == rs["count"]
    assert abs(float(rs["mw"].sum()) - 1) < 1e-6 and float((rs["mw"] - 0.25).abs().max()) < 2e-3   # 120.5 / 120 / 120 / 119.5 samples (discrete fades)
    d2 = dict(d, side_tr=rng.standard_normal((nn, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
              side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    logs2 = []
    r_s = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=logs2.append, side="dcmean", recomb=(4, 0.5), **kw)
    assert np.isfinite(r_s["prob_last"]).all() and any("recombined" in m for m in logs2)
    print(f"4. side vectors: the donors' slot-mass-weighted mean ({rs['count']} / {n} recombined; = the recombined window's mean), untouched "
          "rows keep theirs, x path unchanged; side dcmean + recomb runs")
    print("OK")


if __name__ == "__main__":
    main()

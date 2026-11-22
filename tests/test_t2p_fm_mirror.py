#!/usr/bin/env python
"""T2-D618: the left-right mirror of the FM target windows (t2p_fm.mirror_setup / mirror_batch; fit_eval_fm(mirror=p)), CPU:
  1. the contract's 41 names give 18 homologous pairs + the 5 midline names fixed (C3 <-> C4, PO7 <-> PO8, TP7 <-> TP8, AF3 <-> AF4, Cz fixed);
     the map is an involution; a montage with an unpaired lateral channel is refused;
  2. p 1: every window x -> x[perm] and its dcmean side z (B, 82) -> both 41-blocks permuted the same way; p ~0 leaves the batch untouched;
     p 0.5 mirrors ~half (counted) and leaves the others bitwise; Sb None passes through;
  3. the side permutation is the right one: t2p_bench.side_raw(X[:, perm]) == side_raw(X)[:, p2] exactly, and side_z (per-feature causal z)
     commutes with p2 when its training prior is permuted with it;
  4. OFF = bitwise: fit_eval_fm on a random-init ST-base (1 layer) with mirror=0 equals the call without the argument; mirror 0.5 runs, moves
     the output and logs the mirrored-window count; it composes with the dcmean side head.
  python tests/test_t2p_fm_mirror.py  ->  "OK"
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(618)
    # 1. the map
    logs = []
    mr = FM.mirror_setup(1.0, CH, 0, log=logs.append)
    perm = mr["perm"].numpy()
    ix = {c: i for i, c in enumerate(CH)}
    assert len(CH) == 41 and mr["pairs"] == 18 and np.array_equal(perm[perm], np.arange(41))
    for a_, b_ in (("C3", "C4"), ("PO7", "PO8"), ("TP7", "TP8"), ("AF3", "AF4"), ("C1", "C2"), ("FC5", "FC6"), ("F1", "F2")):
        assert perm[ix[a_]] == ix[b_] and perm[ix[b_]] == ix[a_], (a_, b_)
    fixed = sorted(CH[i] for i in range(41) if perm[i] == i)
    assert fixed == sorted(["Fz", "FCz", "Cz", "CPz", "Pz"]), fixed
    try:
        FM.mirror_setup(1.0, ["C3", "C4", "P3", "Cz"], 0, log=lambda m: None)
        raise SystemExit("an unpaired lateral channel (P3 without P4) was accepted")
    except AssertionError:
        pass
    print(f"1. 18 homologous pairs + 5 midline fixed, an involution; unpaired montages refused ({logs[0][:60]}...)")
    # 2. the batch op
    n, T = 64, 120
    X = torch.as_tensor(rng.standard_normal((n, 41, T)).astype(np.float32))
    S = torch.as_tensor(rng.standard_normal((n, 82)).astype(np.float32))
    p2 = np.concatenate([perm, perm + 41])
    Y, Sy = FM.mirror_batch(X, S, mr)
    assert mr["count"] == n and torch.equal(Y, X[:, perm]) and torch.equal(Sy, S[:, p2])
    mr0 = FM.mirror_setup(1e-12, CH, 0, log=lambda m: None)
    Y0, S0 = FM.mirror_batch(X, S, mr0)
    assert Y0 is X and S0 is S and mr0["count"] == 0
    mr5 = FM.mirror_setup(0.5, CH, 1, log=lambda m: None)
    for _ in range(20):
        Z, Sz = FM.mirror_batch(X, S, mr5)
    frac = mr5["count"] / mr5["seen"]
    flipped = (Z - X).abs().amax((1, 2)) > 0
    assert 0.42 < frac < 0.58 and mr5["seen"] == 20 * n and 0 < int(flipped.sum()) < n, frac
    assert torch.equal(Z[flipped], X[flipped][:, perm]) and torch.equal(Z[~flipped], X[~flipped])
    assert torch.equal(Sz[flipped], S[flipped][:, p2]) and torch.equal(Sz[~flipped], S[~flipped])
    Zn, Sn = FM.mirror_batch(X, None, mr5)
    assert Sn is None and Zn.shape == X.shape
    print(f"2. p 1 -> x[perm] + side z [perm, perm + 41]; p ~0 untouched; p 0.5 mirrored {mr5['count']} / {mr5['seen']} ({frac:.3f}), "
          "the others bitwise; no side passes through")
    # 3. the side permutation matches the side features
    import t2p_bench as B
    Xr = rng.standard_normal((90, 41, T)) + rng.standard_normal((90, 41, 1)) * 3
    F0, F1 = B.side_raw(Xr, "dcmean"), B.side_raw(Xr[:, perm], "dcmean")
    assert F0.shape == (90, 82) and np.array_equal(F1[:, :41], F0[:, perm]) and np.abs(F1 - F0[:, p2]).max() < 1e-12, "side_raw(mirrored) != the permuted side_raw"
    idx = pd.DataFrame(dict(stream=np.repeat([0, 1, 2], 30), order=np.tile(np.arange(30), 3)))
    tr = np.arange(30)
    Z0, Z1 = B.side_z(F0, idx, tr, "dcmean"), B.side_z(F0[:, p2], idx, tr, "dcmean")
    assert np.allclose(Z1, Z0[:, p2], atol=1e-6), "side_z does not commute with the block permutation"
    print("3. side_raw(x[perm]) == side_raw(x)[:, p2] (block 1 exactly, block 2 to 1e-12: the channel-mean summation order); side_z commutes with p2 (prior permuted with it)")
    # 4. off = bitwise; on = runs, moves, counts; with the dcmean side head
    nn, nt = 96, 30
    yy, yt = (np.arange(nn) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn) % 2).astype(np.int64))
    kw = dict(epochs=2, bs=32, lr_enc=1e-4, lr_head=3e-2, wd=0.0, device="cpu", layer=1, calib="gain", ls=0.0)
    r_a = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    r_b = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, mirror=0.0, **kw)
    assert np.array_equal(r_a["prob_last"], r_b["prob_last"]), "mirror=0 is not bitwise"
    logs = []
    r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, mirror=0.5, **kw)
    assert not np.array_equal(r_a["prob_last"], r_c["prob_last"]) and np.isfinite(r_c["prob_last"]).all()
    cnt = [m for m in logs if "mirrored" in m]
    assert cnt and any("left-right mirror" in m and "18 homologous pairs" in m for m in logs), logs[-3:]
    d2 = dict(d, side_tr=rng.standard_normal((nn, 82)).astype(np.float32), side_va=np.zeros((0, 82), np.float32),
              side_te=rng.standard_normal((nt, 82)).astype(np.float32))
    r_s = FM.fit_eval_fm("rand_st_base", d2, 0, CH, log=lambda m: None, side="dcmean", mirror=0.5, **kw)
    assert np.isfinite(r_s["prob_last"]).all()
    print(f"4. mirror=0 bitwise; mirror 0.5 runs and moves the output ({cnt[-1].split(';')[-1].strip()}); with the side head too")
    print("OK")


if __name__ == "__main__":
    main()

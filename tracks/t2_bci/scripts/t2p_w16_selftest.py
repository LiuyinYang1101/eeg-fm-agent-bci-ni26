#!/usr/bin/env python
"""T2-D516: template W16 (W15 + the window-mean family) selftest, CPU:
  1. DCFamily (the template's per-stream features + causal z + tree arrays) on REAL public streams, window by window == the final models'
     probabilities computed in batch by scripts/local/t2_d516_final.py's functions (<= 1e-9; outputs/t2_d516/dcfam.npz);
  2. a synthetic three-family W13-shape model: W16 without cfg["dcfam"] == W15, every window, bitwise;
  3. with cfg["dcfam"] (weight 0.075): every window's output == (1 - w) * W15's per-family pool + w * the family's own Sinkhorn of its
     probabilities (recomputed independently); stream B after stream A on a fresh deepcopy == stream B alone (no state crosses streams).
  python tracks/t2_bci/scripts/t2p_w16_selftest.py  ->  "OK"
"""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "local"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
O = Path.home() / "workspace/outputs/neural_interface_challenge"


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W15, W16 = load("t2p_submission_w15"), load("t2p_submission_w16")
torch.set_num_threads(2)


def main():
    import t2_d516_final as FIN
    z = np.load(O / "t2_d516/dcfam.npz")
    arrs = {k: z[k] for k in z.files}
    # 1. the template's family == the batch reference on real streams
    F = FIN.dc_features(FIN.X)
    Z = FIN.causal_z(F, arrs["mu"], arrs["sd"])
    ref = np.mean([FIN.softmax(FIN.eval_arrays(Z, *(arrs[f"s{s}_{nm}"] for nm in ("feat", "thr", "left", "right", "dleft", "leaf")))) for s in (1, 2, 3)], 0)
    worst = 0.0
    for st in np.unique(FIN.STREAM)[:4]:
        r = np.flatnonzero(FIN.STREAM == st)
        r = r[np.argsort(FIN.ORD[r], kind="stable")]
        fam = W16.DCFamily(arrs, [1, 2, 3], 10.0, 3)
        got = np.stack([fam(np.asarray(FIN.X[i], np.float32)) for i in r])
        worst = max(worst, float(np.abs(got - ref[r]).max()))
    assert worst <= 1e-9, worst
    print(f"1. DCFamily window by window == the batch reference on 4 real streams (max |dp| {worst:.1e})")
    # 2 / 3. synthetic W13-shape models
    C, L, K = 41, 480, 3
    CH = json.loads((O / "t2_d516/dcfam_meta.json").read_text())["ch_names"]
    assert len(CH) == C
    torch.manual_seed(0)
    fam_ = lambda: [W16.build_net(dict(braindecode_class="ShallowFBCSPNet", ch_names=CH, n_classes=K, n_times=L, sfreq=120.0)).eval() for _ in range(2)]  # noqa: E731
    nets, nets2, nets3 = fam_(), fam_(), fam_()
    st_ = [0.36, 0.30, 0.34]
    cfg = dict(ch_names=CH, sfreq=120.0, n_times=L, n_classes=K, align=dict(mode="sz", scale=20.0),
               cnn2=dict(align=dict(mode="ea", scale=20.0), weight=0.3), cnn3=dict(align=dict(mode="ea", scale=20.0), weight=0.2),
               tta=dict(kind="sinkhorn", iters=10, n0=3, lam=0.95, per_family=dict(cnn=st_, cnn2=st_, cnn3=st_)))
    meta = dict(ch_names=CH, sfreq=120.0, n_chans=C)
    m15 = W15.StreamModel(nets, cfg, meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    m16o = W16.StreamModel(nets, cfg, meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    w = 0.075
    m16 = W16.StreamModel(nets, dict(cfg, dcfam=dict(file="dcfam.npz", weight=w, seeds=[1, 2, 3], n0=10)), meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    m16.set_dcfam(arrs)

    def run(model, Xs):
        mm = copy.deepcopy(model)
        post = []
        for x in Xs:
            mm.predict(torch.as_tensor(x[None]))
            post.append(np.asarray(mm.last_p))
        return np.stack(post)

    rows = np.flatnonzero(FIN.STREAM == np.unique(FIN.STREAM)[0])
    rows = rows[np.argsort(FIN.ORD[rows], kind="stable")][:50]
    S1 = np.asarray(FIN.X[rows], np.float32)
    p15, p16o, p16 = run(m15, S1), run(m16o, S1), run(m16, S1)
    assert np.array_equal(p15, p16o), "W16 without dcfam != W15"
    print("2. W16 without cfg['dcfam'] == W15 (bitwise, 50 real windows)")
    fam = W16.DCFamily(arrs, [1, 2, 3], 10.0, 3)
    sk = W16.SinkhornCorrector(10, 3, 3, 0.95, np.asarray(arrs["start"], np.float64))
    exp = np.stack([(1 - w) * a + w * sk(fam(x)) for a, x in zip(p15, S1)])
    d = float(np.abs(exp - p16).max())
    assert d <= 1e-12, d
    rows2 = np.flatnonzero(FIN.STREAM == np.unique(FIN.STREAM)[1])
    rows2 = rows2[np.argsort(FIN.ORD[rows2], kind="stable")][:40]
    S2 = np.asarray(FIN.X[rows2], np.float32)
    alone = run(m16, S2)
    mA = copy.deepcopy(m16)
    for x in S1:
        mA.predict(torch.as_tensor(x[None]))
    mB = copy.deepcopy(m16)
    after = []
    for x in S2:
        mB.predict(torch.as_tensor(x[None]))
        after.append(np.asarray(mB.last_p))
    assert np.array_equal(np.stack(after), alone), "stream B after stream A must equal stream B alone"
    print(f"3. W16 with dcfam (w {w}): output == (1 - w) W15 + w SK(family) (max |d| {d:.1e}); stream B after A == B alone (bitwise)")
    print("OK")


if __name__ == "__main__":
    main()

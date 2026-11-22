#!/usr/bin/env python
"""T2-D510 / D511: template W15 (in-stream adaptation) selftest, CPU, no package files:
  1. _w15_finetune == ni26.train.finetune(net, X, y, peruser_for(r1_pulx_vp2_e180_vr), seed): the same weights, bitwise (the research copy recipe);
  2. _prefix_decode's LAST row == t2p_report.causal_sinkhorn at every prefix length (start vector, lambda 0.95); its rows = the D511 teacher's;
  3. _conf_half == t2p_bench's 'conf' selection;
  4. a synthetic three-family W13-shape StreamModel (ShallowNet members, per-family decode, copies off, no FM):
     (a) W15 without cfg["instream"] == W14, every window (labels and decoded posterior, bitwise);
     (b) W15 WITH instream on a stream shorter than the first refit == W14 (bitwise);
     (c) on a longer stream: identical before the first refit; the refits fire at the configured points; the shipped members are unchanged;
     (d) deepcopy isolation: stream B on a fresh deepcopy after stream A == stream B alone (no state crosses streams);
     (e) both label forms ('conf', 'confR') run; the run budget (max_run_s 0) stops every refit after the first.
  python tracks/t2_bci/scripts/t2p_w15_selftest.py  ->  "OK"
"""
import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ni26 import kit  # noqa: E402
from ni26 import train as T  # noqa: E402

kit.use_track("bci_decoding")


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W14, W15 = load("t2p_submission_w14"), load("t2p_submission_w15")
torch.set_num_threads(2)
C, L, K = 41, 480, 3
CH = [f"E{i}" for i in range(C)]


def main():
    rng = np.random.default_rng(0)
    # 1. the copy recipe
    net = T.build_model("shallow", C, K, L, 120.0)
    X = torch.as_tensor(rng.standard_normal((40, C, L)).astype(np.float32))
    y = torch.as_tensor(rng.integers(0, K, 40))
    rec = T.peruser_for(T.RECIPES["r1_pulx_vp2_e180_vr"])
    rec = replace(rec, epochs=3)
    a = T.finetune(net, X, y, rec, 1234)
    b = W15._w15_finetune(net, X, y, rec.epochs, rec.lr, rec.wd, rec.bs, rec.label_smoothing, 1234)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()), "W15 finetune != ni26.train.finetune"
    print(f"1. _w15_finetune == ni26.train.finetune (bitwise; lr {rec.lr:g}, wd {rec.wd:g}, bs {rec.bs}, ls {rec.label_smoothing:g})")
    # 2. the prefix decode
    from t2p_report import causal_sinkhorn
    P = rng.dirichlet(np.ones(K), 70)
    start = np.array([0.4, 0.25, 0.35])
    ref = causal_sinkhorn(P, (np.zeros(len(P)), np.arange(len(P))), 10, 3, 0.95, start=start)
    for c in (1, 2, 30, 70):
        rows = [start.copy() for _ in range(3)] + list(P[:c])
        D = W15._prefix_decode(rows, 10, 0.95, K, 3)
        assert D.shape == (c, K) and np.abs(D[-1] - ref[c - 1]).max() < 1e-12, (c, np.abs(D[-1] - ref[c - 1]).max())
    print("2. _prefix_decode: last row == causal_sinkhorn at every prefix (start, lambda 0.95)")
    # 3. the confident half
    post = rng.dirichlet(np.ones(K), 61)
    ka, ya = W15._conf_half(post)
    yb, cf = post.argmax(1), post.max(1)
    keep = np.zeros(len(post), bool)
    for c_ in np.unique(yb):
        jj = np.flatnonzero(yb == c_)
        keep[jj[np.argsort(-cf[jj], kind="stable")[:int(np.ceil(len(jj) / 2))]]] = True
    assert np.array_equal(ka, np.flatnonzero(keep)) and np.array_equal(ya, yb[keep])
    print(f"3. _conf_half == t2p_bench 'conf' ({len(ka)} of {len(post)} kept)")
    # 4. a synthetic W13-shape model
    torch.manual_seed(0)
    fam = lambda: [W15.build_net(dict(braindecode_class="ShallowFBCSPNet", ch_names=CH, n_classes=K, n_times=L, sfreq=120.0)).eval() for _ in range(2)]  # noqa: E731
    nets, nets2, nets3 = fam(), fam(), fam()
    st = [0.36, 0.30, 0.34]
    cfg = dict(ch_names=CH, sfreq=120.0, n_times=L, n_classes=K, align=dict(mode="sz", scale=20.0),
               cnn2=dict(align=dict(mode="ea", scale=20.0), weight=0.3), cnn3=dict(align=dict(mode="ea", scale=20.0), weight=0.2),
               tta=dict(kind="sinkhorn", iters=10, n0=3, lam=0.95, per_family=dict(cnn=st, cnn2=st, cnn3=st)))
    meta = dict(ch_names=CH, sfreq=120.0, n_chans=C)
    ins = dict(families=dict(cnn2=dict(member=0, epochs=2, lr=1e-3), cnn3=dict(member=1, epochs=2, lr=1e-3)), refits=[20, 40], r=0.5,
               labels="confR", bs=32, wd=0.01, label_smoothing=0.1, seed=7)
    m14 = W14.StreamModel(nets, cfg, meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    m15o = W15.StreamModel(nets, cfg, meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    m15 = W15.StreamModel(nets, dict(cfg, instream=ins), meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    sd0 = {k: v.clone() for k, v in nets2[0].state_dict().items()}

    def run(model, Xs):
        mm = copy.deepcopy(model)
        lab, post = [], []
        for x in Xs:
            lab.append(int(mm.predict(torch.as_tensor(x[None]))[0]))
            post.append(np.asarray(mm.last_p))
        return np.array(lab), np.stack(post), mm

    S1 = (rng.standard_normal((60, C, L)) * 20).astype(np.float32)
    S2 = (rng.standard_normal((45, C, L)) * 25).astype(np.float32)
    l14, p14, _ = run(m14, S1)
    l15, p15, _ = run(m15o, S1)
    assert np.array_equal(l14, l15) and np.array_equal(p14, p15), "W15 without instream != W14"
    print("4a. W15 without cfg['instream'] == W14 (bitwise, 60 windows)")
    ls_, ps_, mm = run(m15, S1[:18])
    assert np.array_equal(ls_, l14[:18]) and np.array_equal(ps_, p14[:18]) and all(v is None for v in mm.is_net.values())
    print("4b. W15 with instream, an 18-window stream (< the first refit 20) == W14 (bitwise); no copy fitted")
    la, pa, mm = run(m15, S1)
    assert np.array_equal(pa[:20], p14[:20]), "identical before the first refit"
    assert all(v is not None for v in mm.is_net.values()) and not np.array_equal(pa[20:], p14[20:]), "the refits fired and moved the output"
    assert all(torch.equal(v, sd0[k]) for k, v in nets2[0].state_dict().items()), "the shipped member must never change"
    print(f"4c. W15 with instream, 60 windows: == W14 before window 20; copies fitted; {int((la != l14).sum())} labels differ after; shipped member unchanged")
    _, pB_alone, _ = run(m15, S2)
    _, _, mA = run(m15, S1)
    mB = copy.deepcopy(m15)
    pB = []
    for x in S2:
        mB.predict(torch.as_tensor(x[None]))
        pB.append(np.asarray(mB.last_p))
    assert np.array_equal(np.stack(pB), pB_alone), "stream B after stream A must equal stream B alone"
    print("4d. deepcopy isolation: stream B after stream A == stream B alone (bitwise)")
    m15c = W15.StreamModel(nets, dict(cfg, instream=dict(ins, labels="conf")), meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    _, pc, mc = run(m15c, S1)
    assert np.array_equal(pc[:20], p14[:20]) and all(v is not None for v in mc.is_net.values())
    m15b = W15.StreamModel(nets, dict(cfg, instream=dict(ins, max_run_s=0.0)), meta, (), nets2, (), (), nets3, ()).eval().stack_members()
    mb = copy.deepcopy(m15b)
    nets_at = []
    for t, x in enumerate(S1):
        mb.predict(torch.as_tensor(x[None]))
        if t in (25, 50):
            nets_at.append(id(mb.is_net["cnn2"]))
    assert nets_at[0] == nets_at[1] and mb.gshared.get("is_off"), "max_run_s 0: the refit at 40 must not fire"
    print("4e. labels 'conf' runs (== W14 before 20, copies fitted); max_run_s 0 stops the refit at 40")
    print("OK")


if __name__ == "__main__":
    main()

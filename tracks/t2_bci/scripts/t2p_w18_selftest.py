#!/usr/bin/env python
"""T2-D525 / D524: template W18 (W17 + two separable switches: cfg["imcoh"], cfg["fm_side"]) selftest, CPU, real windows:
  1. imcoh_features (one window) == scripts/local/t2_d487_wpli.features (the batch the reads used), float32-exact on 24 real windows;
  2. ICFamily over a real stream == the reader's pipeline (t2_d487_wpli.recentre + z + sklearn LogisticRegression.predict_proba, fitted here on
     lc9_0's training rows exactly as scripts/local/t2_d525_imcoh.py) (<= 1e-9);
  3. a synthetic W13-shape model with two STUB FM members (+ the real dcfam and side nets): W18 with neither switch == W17, bitwise;
  4. cfg["imcoh"] (w 0.03): W18 == W17's decode mixed independently, (1 - w) * W17 + w * a fresh SinkhornCorrector(start) of ICFamily (<= 1e-12);
     it moves the output; stream B after stream A (deepcopy + reset) == stream B alone (bitwise);
  5. cfg["fm_side"] (a head on FM member 0 only): W18 == W17 whose member-0 net adds W s + b at its LOGITS (s = t2p_bench.side_z's batch z)
     (<= 1e-6: float32 softmax vs float64); a zero head == no head (<= 1e-7); set_fm_side refuses a head list with no head;
  6. ms per window of the imcoh family (CPU).
  python tracks/t2_bci/scripts/t2p_w18_selftest.py  ->  "OK"
"""
import copy
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts/local"))
from ni26 import kit  # noqa: E402
import t2p_bench as B  # noqa: E402
import t2_d487_wpli as WP  # noqa: E402

kit.use_track("bci_decoding")
O = Path.home() / "workspace/outputs/neural_interface_challenge"
SD = O / "t2_d517_final"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W17, W18 = load("t2p_submission_w17"), load("t2p_submission_w18")
torch.set_num_threads(2)


class Stub(torch.nn.Module):
    """a stand-in FM member: (1, C, T) -> (1, K) logits (the template only needs a module)."""

    def __init__(self, C, K, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = torch.nn.Linear(C, K)
        with torch.no_grad():
            self.lin.weight.copy_(torch.randn(K, C, generator=g) * 2.0)
            self.lin.bias.copy_(torch.randn(K, generator=g))

    def forward(self, z):
        return self.lin(z.float().mean(-1))


class Plus(torch.nn.Module):
    """member 0 with the head added at its logits (the independent reference for test 5): s set per window from the batch z."""

    def __init__(self, base, W_, b_):
        super().__init__()
        self.base, self.W_, self.b_, self.s = base, torch.as_tensor(W_, dtype=torch.float32), torch.as_tensor(b_, dtype=torch.float32), None

    def forward(self, z):
        return self.base(z) + (self.W_ @ torch.as_tensor(self.s, dtype=torch.float32) + self.b_)[None]


def main():
    X, idx, meta = B.load_export(str(EXP))
    STREAM, ORD = idx["stream"].to_numpy(), idx["order"].to_numpy()
    CH = list(meta["ch_names"])
    C, L, K = len(CH), int(X.shape[-1]), 3
    streams = np.unique(STREAM)
    rows = np.flatnonzero(STREAM == streams[0])
    rows = rows[np.argsort(ORD[rows], kind="stable")]
    # 1. features
    ci = [CH.index(n) for n in W18.IC_NODES]
    assert tuple(W18.IC_NODES) == tuple(WP.NODES) and tuple(W18.IC_BANDS) == tuple(WP.BANDS.values())
    ref_f = WP.features(np.asarray(X[rows[:24]]), CH, float(meta["sfreq"]))[0]
    got_f = np.stack([W18.imcoh_features(np.asarray(X[i], np.float32), ci, float(meta["sfreq"])) for i in rows[:24]])
    d1 = float(np.abs(got_f - ref_f.astype(np.float64)).max())
    assert d1 <= 1e-6, d1
    print(f"1. imcoh_features == t2_d487_wpli.features on 24 real windows (max |d| {d1:.1e})")
    # 2. ICFamily == the reader's pipeline (lc9_0 LR, as t2_d525_imcoh.py)
    from sklearn.linear_model import LogisticRegression
    Fe = np.load(O / "t2_d487/features.npz")["imcoh"].astype(np.float64)
    Fr = WP.recentre(Fe, idx)
    ((_, trm, _, _),) = B.folds(idx, "lc9_0")
    tr = np.flatnonzero(trm)
    mu, sd = Fr[tr].mean(0), Fr[tr].std(0) + 1e-9
    clf = LogisticRegression(C=WP.C_LR, max_iter=3000).fit(((Fr - mu) / sd)[tr], idx["y"].to_numpy().astype(int)[tr])
    arrs = dict(mu=mu, sd=sd, coef=clf.coef_, intercept=clf.intercept_, start=np.array([0.36, 0.30, 0.34]))
    icf = W18.ICFamily(arrs, ci, float(meta["sfreq"]))
    ref_p = clf.predict_proba(((Fr - mu) / sd)[rows])
    t0 = time.time()
    got_p = np.stack([icf(np.asarray(X[i], np.float32)) for i in rows])
    ms_ic = 1000 * (time.time() - t0) / len(rows)
    d2 = float(np.abs(got_p - ref_p).max())
    assert d2 <= 1e-9, d2
    print(f"2. ICFamily over a real stream ({len(rows)} windows) == recentre + z + LogisticRegression.predict_proba (max |dp| {d2:.1e})")
    # the synthetic model (W17's selftest shape + two stub FM members)
    blob = torch.load(SD / "side.pt", map_location="cpu", weights_only=True)
    g = np.load(SD / "side_golden.npz")
    scale = float(blob["scale"])
    z = np.load(O / "t2_d516/dcfam.npz")
    dca = {k: z[k] for k in z.files}
    torch.manual_seed(0)
    fam_ = lambda cls: [W17.build_net(dict(braindecode_class=cls, ch_names=CH, n_classes=K, n_times=L, sfreq=120.0)).eval() for _ in range(2)]  # noqa: E731
    nets, nets2, nets3 = fam_("ShallowFBCSPNet"), fam_("ShallowFBCSPNet"), fam_("IFNet")
    fm = [Stub(C, K, 1).eval(), Stub(C, K, 2).eval()]
    st_ = [0.36, 0.30, 0.34]
    al = dict(mode="ea", scale=scale, n0=0.0, shrink=0.05, prior=None)
    fmm = [dict(family="st", align=dict(mode="sz", scale=scale, shrink=0.05)), dict(family="st", align=dict(mode="sz", scale=scale, shrink=0.05), car=True)]
    cfg = dict(ch_names=CH, sfreq=120.0, n_times=L, n_classes=K, align=dict(mode="none", scale=scale), fm_members=fmm, fm_weight=0.35,
               cnn2=dict(align=al, weight=0.2, braindecode_class="ShallowFBCSPNet", model_kwargs={}),
               cnn3=dict(align=al, weight=0.1, braindecode_class="IFNet", model_kwargs={}),
               tta=dict(kind="sinkhorn", iters=10, n0=3, lam=0.95, per_family=dict(cnn=st_, cnn2=st_, cnn3=st_, fm=st_)),
               dcfam=dict(file="dcfam.npz", weight=0.075, seeds=[1, 2, 3], n0=10),
               side=dict(file="side.pt", kind="dcmean", r=1.0 / 3.0, families=["cnn2", "cnn3"]))
    mt = dict(ch_names=CH, sfreq=120.0, n_chans=C)

    def build(mod, cfg_, fm_=fm):
        return mod.StreamModel(nets, cfg_, mt, fm_, nets2, (), (), nets3, ()).eval().stack_members().set_dcfam(dca).set_side(blob, "cpu")

    def run(model, rr, hook=None):
        mm = copy.deepcopy(model)
        post = []
        for i in rr:
            if hook is not None:
                hook(mm, i)
            mm.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
            post.append(np.asarray(mm.last_p))
        return np.stack(post)

    r50 = rows[:50]
    # 3. neither switch == W17
    m17, m18o = build(W17, cfg), build(W18, cfg)
    p17, p18o = run(m17, r50), run(m18o, r50)
    assert np.array_equal(p17, p18o), "W18 without its switches != W17"
    print("3. W18 without cfg['imcoh'] / cfg['fm_side'] == W17 (bitwise, 50 real windows; dcfam, side nets, two stub FM members)")
    # 4. imcoh
    w = 0.03
    m18i = build(W18, dict(cfg, imcoh=dict(file="imcoh.npz", weight=w))).set_imcoh(arrs)
    p18i = run(m18i, r50)
    icr = W18.ICFamily(arrs, ci, float(meta["sfreq"]))
    cor = W17.SinkhornCorrector(10, 3, K, 0.95, np.asarray(arrs["start"], np.float64))
    mref = copy.deepcopy(m17)
    orig = mref._fam_decode
    cur = {}

    def patched(parts):
        return (1 - w) * orig(parts) + w * cor(cur["p"])

    mref._fam_decode = patched
    exp = []
    for i in r50:
        cur["p"] = icr(np.asarray(X[i], np.float32))
        mref.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
        exp.append(np.asarray(mref.last_p))
    d4 = float(np.abs(np.stack(exp) - p18i).max())
    moved = float(np.abs(p18i - p17).max())
    assert d4 <= 1e-12 and moved > 1e-4, (d4, moved)
    rows2 = np.flatnonzero(STREAM == streams[1])
    rows2 = rows2[np.argsort(ORD[rows2], kind="stable")][:40]
    alone = run(m18i, rows2)
    mA = copy.deepcopy(m18i)
    for i in r50:
        mA.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
    mB = copy.deepcopy(mA)
    mB.reset_state()
    after = []
    for i in rows2:
        mB.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
        after.append(np.asarray(mB.last_p))
    assert np.array_equal(np.stack(after), alone), "imcoh: stream B after stream A (deepcopy + reset) must equal stream B alone"
    print(f"4. imcoh (w {w}): W18 == (1 - w) W17 + w Sinkhorn(ICFamily) (max |d| {d4:.1e}; moved vs W17 {moved:.4f}); B after A == B alone (bitwise)")
    # 5. fm_side on member 0
    rng = np.random.default_rng(0)
    W_, b_ = rng.standard_normal((K, 2 * C)) * 0.3, rng.standard_normal(K) * 0.3
    head = dict(kind="dcmean", n0=10.0, mu=blob["mu"], sd=blob["sd"], weight=torch.as_tensor(W_, dtype=torch.float32), bias=torch.as_tensor(b_, dtype=torch.float32))
    W32, b32 = head["weight"].double().numpy(), head["bias"].double().numpy()
    m18s = build(W18, dict(cfg, fm_side=True)).set_fm_side([head, None])
    p18s = run(m18s, r50)
    plus = Plus(fm[0], W32, b32)
    m17p = build(W17, cfg, fm_=[plus, fm[1]])

    def setz(mm, i):
        mm.fm_nets[0].s = g["Z"][i]

    p17p = run(m17p, r50, hook=setz)
    d5 = float(np.abs(p18s - p17p).max())
    moved5 = float(np.abs(p18s - p17).max())
    zero = dict(head, weight=torch.zeros(K, 2 * C), bias=torch.zeros(K))
    p18z = run(build(W18, dict(cfg, fm_side=True)).set_fm_side([zero, None]), r50)
    d5z = float(np.abs(p18z - p17).max())
    assert d5 <= 1e-6 and moved5 > 1e-4 and d5z <= 1e-7, (d5, moved5, d5z)
    try:
        build(W18, dict(cfg, fm_side=True)).set_fm_side([None, None])
        raise SystemExit("set_fm_side accepted a head list without a head")
    except AssertionError:
        pass
    print(f"5. fm_side (member 0): W18 == W17 with W s + b added at member 0's logits (max |d| {d5:.1e}; moved {moved5:.4f}); zero head == no head "
          f"(max |d| {d5z:.1e}); a head list without a head is refused")
    print(f"6. imcoh family: {ms_ic:.2f} ms per window (CPU, features + re-centring + LR)")
    print("OK")


if __name__ == "__main__":
    main()

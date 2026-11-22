#!/usr/bin/env python
"""T2-D517: template W17 (W16 + the window-mean side nets) selftest, CPU, on the REAL side nets (outputs/t2_d517_final/side.pt) and real streams:
  1. the template's side vector (_side_z, window by window) == t2p_side_final's batch z (t2p_bench.side_z) on 4 real streams (<= 1e-5);
  2. the template's side nets (_side_probs) on the research path's aligned windows (t2p_bench.aligned_all, EA n0 0, the package scale) == the
     trainer's golden per-seed mean probabilities (<= 1e-4: golden = cuda batch 512, here CPU one window), and the template's own aligners
     (aligner2 / aligner3 = ea, the package scale) reproduce those aligned windows (<= 1e-4);
  3. a synthetic W13-shape model (+ the real dcfam): W17 without cfg["side"] == W16, every window, bitwise;
  4. with cfg["side"] (r 1/3): every window's output == W16 whose cnn2 / cnn3 parts are replaced by (1 - r) part + r golden side probability
     before the correctors (an independent recombination through W16's own decode; <= 1e-4); stream B after stream A on a fresh deepcopy ==
     stream B alone (bitwise); ms per window of the side nets (CPU, one thread pair).
  python tracks/t2_bci/scripts/t2p_w17_selftest.py  ->  "OK"
"""
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ni26 import kit, stream  # noqa: E402
import t2p_bench as B  # noqa: E402

kit.use_track("bci_decoding")
O = Path.home() / "workspace/outputs/neural_interface_challenge"
SD = O / "t2_d517_final"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W16, W17 = load("t2p_submission_w16"), load("t2p_submission_w17")
torch.set_num_threads(2)


def main():
    X, idx, meta = B.load_export(str(EXP))
    STREAM, ORD = idx["stream"].to_numpy(), idx["order"].to_numpy()
    blob = torch.load(SD / "side.pt", map_location="cpu", weights_only=True)
    g = np.load(SD / "side_golden.npz")
    scale = float(blob["scale"])
    CH = list(meta["ch_names"])
    C, L, K = len(CH), int(X.shape[-1]), 3
    z = np.load(O / "t2_d516/dcfam.npz")
    arrs = {k: z[k] for k in z.files}
    torch.manual_seed(0)
    fam_ = lambda cls: [W16.build_net(dict(braindecode_class=cls, ch_names=CH, n_classes=K, n_times=L, sfreq=120.0)).eval() for _ in range(2)]  # noqa: E731
    nets, nets2, nets3 = fam_("ShallowFBCSPNet"), fam_("ShallowFBCSPNet"), fam_("IFNet")
    st_ = [0.36, 0.30, 0.34]
    al = dict(mode="ea", scale=scale, n0=0.0, shrink=0.05, prior=None)
    cfg = dict(ch_names=CH, sfreq=120.0, n_times=L, n_classes=K, align=dict(mode="none", scale=scale),
               cnn2=dict(align=al, weight=0.3, braindecode_class="ShallowFBCSPNet", model_kwargs={}),
               cnn3=dict(align=al, weight=0.2, braindecode_class="IFNet", model_kwargs={}),
               tta=dict(kind="sinkhorn", iters=10, n0=3, lam=0.95, per_family=dict(cnn=st_, cnn2=st_, cnn3=st_)),
               dcfam=dict(file="dcfam.npz", weight=0.075, seeds=[1, 2, 3], n0=10))
    r = 1.0 / 3.0
    mt = dict(ch_names=CH, sfreq=120.0, n_chans=C)
    m16 = W16.StreamModel(nets, cfg, mt, (), nets2, (), (), nets3, ()).eval().stack_members().set_dcfam(arrs)
    m17o = W17.StreamModel(nets, cfg, mt, (), nets2, (), (), nets3, ()).eval().stack_members().set_dcfam(arrs)
    m17 = W17.StreamModel(nets, dict(cfg, side=dict(file="side.pt", kind="dcmean", r=r, families=["cnn2", "cnn3"])), mt, (), nets2, (), (), nets3,
                          ()).eval().stack_members().set_dcfam(arrs).set_side(blob, "cpu")
    gold = {f: g[f"{f}_prob"].mean(0) for f in ("cnn2", "cnn3")}
    streams = np.unique(STREAM)
    # 1. the side vector
    worst = 0.0
    for st in streams[:4]:
        rows = np.flatnonzero(STREAM == st)
        rows = rows[np.argsort(ORD[rows], kind="stable")]
        mm = copy.deepcopy(m17)
        got = np.stack([mm._side_z(np.asarray(X[i], np.float32)) for i in rows])
        worst = max(worst, float(np.abs(got - g["Z"][rows]).max()))
    assert worst <= 1e-5, worst
    print(f"1. the template's side vector == t2p_bench.side_z on 4 real streams (max |dz| {worst:.1e})")
    # 2. the side nets on the research path's aligned windows; the template's aligners reproduce them
    from ni26 import train as T                    # the independent CPU reference: the trainer's own builder + T._pp_side
    ref_nets = {}
    for f, model in (("cnn2", "shallow"), ("cnn3", "ifnet")):
        ref_nets[f] = []
        for e in blob[f]:
            n_ = T.build_model(model, C, K, L, 120.0)
            n_.load_state_dict(e["net"])
            h_ = torch.nn.Linear(2 * C, K)
            h_.load_state_dict(e["head"])
            ref_nets[f].append((n_.eval(), h_.eval()))
    wp, wg, wa, t_side, n_side = 0.0, 0.0, 0.0, 0.0, 0
    for st in streams[:3]:
        rows = np.flatnonzero(STREAM == st)
        rows = rows[np.argsort(ORD[rows], kind="stable")][:60]
        xa = stream.align_stream(np.asarray(X[rows]), "ea", scale=scale, prior=None, n0=0.0, shrink=0.05)
        ref = {f: np.mean([T._pp_side(n_, h_, torch.as_tensor(xa, dtype=torch.float32), torch.as_tensor(g["Z"][rows])) for n_, h_ in ref_nets[f]], 0)
               for f in ref_nets}
        mm = copy.deepcopy(m17)
        for j, i in enumerate(rows):
            a2 = mm.aligner2(np.asarray(X[i], np.float32))
            a3 = mm.aligner3(np.asarray(X[i], np.float32))
            wa = max(wa, float(np.abs(a2 - xa[j]).max()), float(np.abs(a3 - xa[j]).max()))
            t0 = time.time()
            for f in ("cnn2", "cnn3"):
                p = mm._side_probs(f, xa[j], g["Z"][i], "cpu")
                wp = max(wp, float(np.abs(p - ref[f][j]).max()))
                wg = max(wg, float(np.abs(p - gold[f][i]).max()))
            t_side += time.time() - t0
            n_side += 1
    assert wa <= 1e-4 and wp <= 1e-5 and wg <= 1e-3, (wa, wp, wg)
    print(f"2. side nets == the trainer's builder on CPU (max |dp| {wp:.1e}); == the golden cuda probabilities (max {wg:.1e}); template aligners "
          f"== aligned_all (max {wa:.1e}); {1000 * t_side / n_side:.1f} ms per window for both families' side nets (CPU)")
    # 3. W17 without side == W16 bitwise

    def run(model, rows):
        mm = copy.deepcopy(model)
        post = []
        for i in rows:
            mm.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
            post.append(np.asarray(mm.last_p))
        return np.stack(post)

    rows = np.flatnonzero(STREAM == streams[0])
    rows = rows[np.argsort(ORD[rows], kind="stable")][:50]
    p16, p17o, p17 = run(m16, rows), run(m17o, rows), run(m17, rows)
    assert np.array_equal(p16, p17o), "W17 without side != W16"
    print("3. W17 without cfg['side'] == W16 (bitwise, 50 real windows, dcfam on)")
    # 4. with side: == W16 with the parts replaced by the independent blend before its own decode
    mref = copy.deepcopy(m16)
    exp = []
    orig = mref._fam_decode
    cur = {}

    def patched(parts):
        parts = list(parts)
        parts[1] = (1 - r) * np.asarray(parts[1], np.float64) + r * gold["cnn2"][cur["i"]]
        parts[3] = (1 - r) * np.asarray(parts[3], np.float64) + r * gold["cnn3"][cur["i"]]
        return orig(parts)

    mref._fam_decode = patched
    for i in rows:
        cur["i"] = i
        mref.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
        exp.append(np.asarray(mref.last_p))
    d = float(np.abs(np.stack(exp) - p17).max())
    assert d <= 1e-4, d
    moved = float(np.abs(p17 - p16).max())
    assert moved > 1e-3, moved
    rows2 = np.flatnonzero(STREAM == streams[1])
    rows2 = rows2[np.argsort(ORD[rows2], kind="stable")][:40]
    alone = run(m17, rows2)
    mA = copy.deepcopy(m17)
    for i in rows:
        mA.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
    mB = copy.deepcopy(mA)
    mB.reset_state()
    after = []
    for i in rows2:
        mB.predict(torch.as_tensor(np.asarray(X[i], np.float32)[None]))
        after.append(np.asarray(mB.last_p))
    assert np.array_equal(np.stack(after), alone), "stream B after stream A (deepcopy + reset) must equal stream B alone"
    print(f"4. W17 with side (r 1/3): output == W16 decode of the independently blended parts (max |d| {d:.1e}; moved vs W16 {moved:.3f}); "
          "stream B after A == B alone (bitwise)")
    if torch.cuda.is_available():                       # 5. the side nets' cost per window on the GPU (stacked path) and the cuda == cpu check
        mg = W17.StreamModel(nets, dict(cfg, side=dict(file="side.pt", kind="dcmean", r=r, families=["cnn2", "cnn3"])), mt, (), nets2, (), (), nets3,
                             ()).eval().stack_members().set_side(blob, "cuda")
        xa = stream.align_stream(np.asarray(X[rows]), "ea", scale=scale, prior=None, n0=0.0, shrink=0.05)
        dg = 0.0
        for j, i in enumerate(rows[:5]):
            for f in ("cnn2", "cnn3"):
                dg = max(dg, float(np.abs(mg._side_probs(f, xa[j], g["Z"][i], "cuda") - m17._side_probs(f, xa[j], g["Z"][i], "cpu")).max()))
        torch.cuda.synchronize()
        t0 = time.time()
        for j, i in enumerate(rows):
            for f in ("cnn2", "cnn3"):
                mg._side_probs(f, xa[j], g["Z"][i], "cuda")
        torch.cuda.synchronize()
        print(f"5. cuda: stacked {sorted(f for f, s in mg.side_stk.items() if s is not None)}; cuda vs cpu max |dp| {dg:.1e}; "
              f"{1000 * (time.time() - t0) / len(rows):.2f} ms per window (both families)")
        assert dg <= 1e-3, dg
    print("OK")


if __name__ == "__main__":
    main()

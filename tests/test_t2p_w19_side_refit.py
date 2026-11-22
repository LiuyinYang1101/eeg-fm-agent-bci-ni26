#!/usr/bin/env python
"""T2-D539b (template W19) GOLDEN TEST of the in-stream side-head re-fit (the manager's condition 2), CPU:
  1. W19's _side_fit_head = scripts/local/t2_d539_read.py fit_head (the reader's source, exec'd) on the real 819 dumps: max |dW|, |db| = 0;
  2. the template's own path (_side_blend -> _side_record; _side_refit at 60 / 100 / 150 with the reader's confR rows / labels from the 819
     teacher) on every test stream of lc9_0 / draw 819, both side families (5 ShallowNet / 5 IFNet nets, their dumped logits replayed by
     lookup nets), loop AND stacked paths: each window's family side probability = the reader's (side_prob_h at lam 1, mean over the nets)
     to <= 1e-6, and the re-fitted heads = the reader's to <= 1e-6;
  3. W18 bitwise: without cfg side.refit, and on every window before the first refit (incl. a stream cut to 50 windows), W19 = W18 exactly;
  4. a fit that raises / does not converge / returns non-finite keeps the trained head (= W18's output, no exception); the _instream_refit
     hook is wrapped (source check).
Needs outputs/t2_d540 (the 819 chain's side dumps + teacher).
  python tests/test_t2p_w19_side_refit.py  ->  "OK"
"""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
torch.set_num_threads(2)
O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
CKS = (60, 100, 150)
LAM = 1.0
DEV = torch.device("cpu")


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tracks/t2_bci/submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W18, W19 = load("t2p_submission_w18"), load("t2p_submission_w19")
# the reader's own functions, from its source (the reader itself is not run)
src = (ROOT / "scripts/local/t2_d539_read.py").read_text()
RD = {"np": np}
exec("from scipy.optimize import minimize", RD)
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("fit_head", "conf_half", "softmax"):
        exec(compile(ast.Module([node], []), "t2_d539_read(fn)", "exec"), RD)
idx = pd.read_csv(EXP / "index.csv")
STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()


class LookupNet(torch.nn.Module):
    """replays one side net's dumped logits: z (1, 1, 1) holds the test-row position."""

    def __init__(self, L):
        super().__init__()
        self.L = torch.as_tensor(L, dtype=torch.float32)

    def forward(self, z):
        return self.L[int(round(float(z.reshape(-1)[0])))][None]


def family(fam_dir, model, f=0, d=819):
    files = [O / f"t2_d540/{fam_dir}/runs/lc9_{f}__{model}__ea__n00__r1_pulx_vp2_e180_vr__sidedcmean__s{d + k * 1000}.npz" for k in range(5)]
    zs = [np.load(p) for p in files]
    te = zs[0]["test_idx"].astype(np.int64)
    assert all((z["test_idx"] == te).all() for z in zs)
    return te, [dict(L=z["side_logit_last"].astype(np.float64), W=z["side_w"].astype(np.float64), b=z["side_b"].astype(np.float64),
                     S=z["side_te"].astype(np.float64)) for z in zs]


def make(W, nets_by_fam, refit, stacked):
    m = W.StreamModel.__new__(W.StreamModel)
    torch.nn.Module.__init__(m)
    m.side, m.side_stk, m.side_hw = {}, {}, {}
    D = nets_by_fam["cnn2"][0]["W"].shape[1]
    m.side_mu = np.zeros(D)
    m.side_r = 1.0 / 3.0
    for f, nets in nets_by_fam.items():
        mem = []
        for e in nets:
            head = torch.nn.Linear(D, 3)
            with torch.no_grad():
                head.weight.copy_(torch.as_tensor(e["W"], dtype=torch.float32))
                head.bias.copy_(torch.as_tensor(e["b"], dtype=torch.float32))
            mem.append((LookupNet(e["L"]).eval(), head.eval()))
        m.side[f] = mem
        m.side_stk[f] = (lambda zt, mm=mem: torch.stack([n(zt) for n, _ in mm])) if stacked else None
        m.side_hw[f] = (torch.stack([h.weight.detach() for _, h in mem]), torch.stack([h.bias.detach() for _, h in mem]))
    if W is W19:
        m.side_refit = dict(lam=LAM) if refit else None
        if refit:
            m.side_W0 = {f: np.stack([h.weight.detach().double().numpy() for _, h in mm]) for f, mm in m.side.items()}
            m.side_b0 = {f: np.stack([h.bias.detach().double().numpy() for _, h in mm]) for f, mm in m.side.items()}
            m.side_refit_stats = dict(fits=0, kept_trained=0, errors=0)
    m.reset_side()
    return m


def window(m, pos, S_row):
    """one window through _side_blend; -> {family: its side probability} (backed out of the blend; parts start uniform)."""
    m._side_now = np.asarray(S_row, np.float32)
    zin = np.full((1, 1), float(pos), np.float32)
    u = np.full(3, 1.0 / 3.0)
    parts = [None, u.copy(), None, u.copy()]
    m._side_blend(parts, {"cnn2": zin, "cnn3": zin}, DEV)
    return {"cnn2": (parts[1] - (1 - m.side_r) * u) / m.side_r, "cnn3": (parts[3] - (1 - m.side_r) * u) / m.side_r}


def main():
    te, sh = family("side_sh", "shallow")
    te2, ifn = family("side_ifn", "ifnet")
    assert (te == te2).all()
    S0 = sh[0]["S"]
    assert all(np.abs(e["S"] - S0).max() == 0 for e in sh + ifn)
    tz = np.load(O / "t2_d540/teacher_p1/lc9_0__s819.npz")
    tpos = {int(r): k for k, r in enumerate(tz["rows"].astype(np.int64))}
    post = {c: tz[f"post_r{c}"].astype(np.float64) for c in CKS}
    fams = {"cnn2": sh, "cnn3": ifn}
    # 1. the fit itself: identical code on identical inputs
    a = np.arange(60)
    ya, keep = RD["conf_half"](np.stack([post[60][tpos[int(r)]] for r in te[a]]))
    W_r, b_r, ok_r = RD["fit_head"](sh[0]["L"][a][keep], S0[a][keep], ya[keep].astype(np.int64), sh[0]["W"], sh[0]["b"], LAM)
    W_t, b_t, ok_t = W19._side_fit_head(sh[0]["L"][a][keep], S0[a][keep], ya[keep].astype(np.int64), sh[0]["W"], sh[0]["b"], LAM)
    assert ok_r and ok_t and np.abs(W_r - W_t).max() == 0 and np.abs(b_r - b_t).max() == 0
    print("1. _side_fit_head = the reader's fit_head exactly (max |dW| 0)")
    # 2. + 3. every stream of the fold, loop and stacked paths
    worst_p, worst_h, n_win, n_fit = 0.0, 0.0, 0, 0
    for stacked in (False, True):
        for s in np.unique(STREAM[te]):
            j = np.flatnonzero(STREAM[te] == s)
            j = j[np.argsort(ORD[te][j], kind="stable")]
            for cut in ((len(j), 50) if s == np.unique(STREAM[te])[0] else (len(j),)):
                jj = j[:cut]
                m19, m18, m19off = make(W19, fams, True, stacked), make(W18, fams, False, stacked), make(W19, fams, False, stacked)
                heads = {f: [(e["W"], e["b"]) for e in nets] for f, nets in fams.items()}     # the reader's current heads
                for t, pos in enumerate(jj):
                    if t in CKS:                         # = the template's refit before window t (t windows delivered)
                        a = jj[:t]
                        pp = np.stack([post[t][tpos[int(r)]] for r in te[a]])
                        fin = np.isfinite(pp).all(1)
                        yk, kp = RD["conf_half"](pp[fin])
                        rows, ys = np.arange(t)[fin][kp], yk[kp].astype(np.int64)
                        m19._side_refit(rows, ys, DEV)
                        n_fit += 1
                        for f, nets in fams.items():
                            heads[f] = [RD["fit_head"](e["L"][a[fin][kp]], S0[a[fin][kp]], ys, e["W"], e["b"], LAM)[:2] for e in nets]
                            Wt, bt = m19.side_hw_now[f]
                            worst_h = max(worst_h, max(float(np.abs(Wt[i].double().numpy() - heads[f][i][0]).max()) for i in range(5)),
                                          max(float(np.abs(bt[i].double().numpy() - heads[f][i][1]).max()) for i in range(5)))
                    p19, p18, poff = window(m19, pos, S0[pos]), window(m18, pos, S0[pos]), window(m19off, pos, S0[pos])
                    for f, nets in fams.items():
                        ref = np.mean([RD["softmax"]((e["L"][pos] + S0[pos] @ heads[f][i][0].T + heads[f][i][1])[None])[0]
                                       for i, e in enumerate(nets)], 0)
                        worst_p = max(worst_p, float(np.abs(p19[f] - ref).max()))
                        assert np.array_equal(poff[f], p18[f]), "3. W19 without refit != W18"
                        if t < CKS[0]:
                            assert np.array_equal(p19[f], p18[f]), f"3. W19 before the first refit != W18 (t {t})"
                    n_win += 1
    assert worst_p <= 1e-6 and worst_h <= 1e-6, (worst_p, worst_h)
    print(f"2. {n_win} windows (loop + stacked), {n_fit} refit points: max |p - reader| {worst_p:.2e}, max |head - reader| {worst_h:.2e}")
    print("3. W19 = W18 bitwise without refit, before the first refit and on a 50-window stream")
    # 4. failures keep the trained head
    s = np.unique(STREAM[te])[0]
    j = np.flatnonzero(STREAM[te] == s)
    j = j[np.argsort(ORD[te][j], kind="stable")]
    real = W19._side_fit_head
    for mode in ("raise", "noconv", "nan"):
        def bad(*a_, mode=mode):
            if mode == "raise":
                raise np.linalg.LinAlgError("forced")
            W_, b_, _ = real(*a_)
            return (W_, b_, False) if mode == "noconv" else (W_ * np.nan, b_, True)
        W19._side_fit_head = bad
        m19, m18 = make(W19, fams, True, True), make(W18, fams, False, True)
        for t, pos in enumerate(j[:80]):
            if t == 60:
                m19._side_refit(np.arange(60), np.arange(60) % 3, DEV)
            p19, p18 = window(m19, pos, S0[pos]), window(m18, pos, S0[pos])
            assert all(np.array_equal(p19[f], p18[f]) for f in fams), mode
        st = m19.side_refit_stats
        assert (st["errors"] if mode == "raise" else st["kept_trained"]) == 10, (mode, st)
    W19._side_fit_head = real
    s19 = (ROOT / "tracks/t2_bci/submission/t2p_submission_w19.py").read_text()
    assert "self._side_refit(rows, ya, dev)\n            except Exception as e_:" in s19, "the _instream_refit hook must be wrapped"
    print("4. a raising / non-converged / non-finite fit keeps the trained heads (= W18), never raises; the hook is wrapped")
    print("OK")


if __name__ == "__main__":
    main()

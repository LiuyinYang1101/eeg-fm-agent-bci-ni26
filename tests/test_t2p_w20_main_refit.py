#!/usr/bin/env python
"""T2-D555 (template W20 = W19 + cfg fm_main_refit) test of the in-stream JOINT main + side head refit, CPU, synthetic (no read numbers):
  1. GOLDEN: the template's path (StreamModel._fm_head_apply -> _fm_refit_step -> _fm_refit -> _fm_main_refit at 60 / 100 / 150, labels from
     is_hist_post, 'conf' path) on a 190-window stream with 5 FM members (4 headed, 1 head-less; per-window pooled features as the hooks
     deliver them) = an independent replay with the D555 READER's functions (scripts/local/t2_d540a_stack_read.py fit_head / conf_half /
     softmax): logits = [feat, s] W^T + b from the trained [W_main | W_side], b_main + b_side (the main head alone for the head-less member):
     every window after the first refit point to <= 1e-6;
  2. W19 bitwise: W20 without cfg fm_main_refit == W19 on every window; with it, every window before the first refit point == W19;
  3. a fit that raises / does not converge / returns non-finite keeps that member's trained heads (== W19's output for it), no exception;
  4. source: set_fm_side asserts fm_main_refit replaces fm_side_refit and needs the eager path; the hooks are registered only when on.
  python tests/test_t2p_w20_main_refit.py  ->  "OK"
"""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
CKS = (60, 100, 150)
K, DM, DS, NW, NM = 3, 64, 82, 190, 5


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tracks/t2_bci/submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W19, W20 = load("t2p_submission_w19"), load("t2p_submission_w20")
RS = {}
src = open(ROOT / "scripts/local/t2_d540a_stack_read.py").read()
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("fit_head", "conf_half", "softmax"):
        exec(compile(ast.Module([node], []), "t2_d540a_stack_read(fn)", "exec"),
             dict(np=np, minimize=__import__("scipy.optimize", fromlist=["minimize"]).minimize), RS)
rng = np.random.default_rng(555)
FE = rng.normal(0, 1, (NM, NW, DM))                       # pooled features per member per window
MAIN = [(rng.normal(0, 0.1, (K, DM)), rng.normal(0, 0.1, K)) for _ in range(NM)]
S = rng.normal(0, 1, (NW, DS))
POST = rng.dirichlet(np.ones(K) * 0.7, NW)
SIDE = [(rng.normal(0, 0.05, (K, DS)), rng.normal(0, 0.1, K)) for _ in range(NM - 1)] + [None]
S32 = S.astype(np.float32).astype(np.float64)
L = np.stack([FE[i] @ MAIN[i][0].T + MAIN[i][1] for i in range(NM)])
P = np.exp(L - L.max(2, keepdims=True))
P = P / P.sum(2, keepdims=True)


def stub(mod, on, fit=None):
    m = mod.StreamModel.__new__(mod.StreamModel)
    m.cfg = {"n_classes": K}
    m.is_cfg = {"labels": "conf", "refits": list(CKS)}
    m.is_refits = list(CKS)
    m.gshared = {"g": {}}
    m.fm_heads = list(SIDE)
    m.fm_refit = None
    if on:                                               # set_fm_side's D555 block, as the template runs it (hooks replaced by _fm_feat)
        m.fm_main = dict(lam=1.0)
        m.fm_refit = dict(lam=1.0)
        m.fm_refit_stats = dict(fits=0, kept_trained=0, errors=0)
        m.fm_main0 = list(MAIN)
        m._fm_feat = [None] * NM
        m.reset_fm_refit()
        m.is_hist_post, m.is_hist_ok = [], []
    if fit is not None:
        mod._side_fit_head = fit
    return m


def run(m, n=NW):
    outs = []
    for c in range(n):
        m.is_n = c
        m._side_now = S[c].astype(np.float32)
        if getattr(m, "fm_main", None) is not None:
            m._fm_feat = [FE[i, c].astype(np.float32).astype(np.float64) for i in range(NM)]
        outs.append(np.stack(m._fm_head_apply([P[i, c].astype(np.float32) for i in range(NM)])))
        if hasattr(m, "is_hist_post"):
            m.is_hist_post.append(POST[c])
            m.is_hist_ok.append(True)
    return np.stack(outs)


def replay():
    out = np.full((NW, NM, K), np.nan)
    for i in range(NM):
        Wm, bm = MAIN[i]
        h = SIDE[i]
        fe = FE[i].astype(np.float32).astype(np.float64)
        X = fe if h is None else np.concatenate([fe, S32], 1)
        W0 = Wm if h is None else np.concatenate([Wm, h[0]], 1)
        b0 = bm if h is None else bm + h[1]
        for k, c in enumerate(CKS):
            e = CKS[k + 1] if k + 1 < len(CKS) else NW
            ya, keep = RS["conf_half"](POST[:c])
            sel = np.arange(c)[keep]
            W_, b_, _ = RS["fit_head"](np.zeros((len(sel), K)), X[sel], ya[keep].astype(np.int64), W0, b0, 1.0)
            out[c:e, i] = RS["softmax"](X[c:e] @ W_.T + b_)
    return out


def main():
    ref19 = run(stub(W19, False))
    off20 = run(stub(W20, False))
    assert np.array_equal(ref19, off20), "W20 without fm_main_refit != W19"
    m = stub(W20, True)
    on20 = run(m)
    assert np.array_equal(on20[:CKS[0]], ref19[:CKS[0]]), "before the first refit point the joint path must equal W19 bitwise"
    rep = replay()
    dp = float(np.abs(on20[CKS[0]:] - rep[CKS[0]:]).max())
    print(f"1. golden: template vs the reader's joint replay, windows {CKS[0]}..{NW - 1}: max |dp| {dp:.2e}; stats {m.fm_refit_stats}", flush=True)
    assert dp <= 1e-6, dp
    assert m.fm_refit_stats == dict(fits=NM * len(CKS), kept_trained=0, errors=0), m.fm_refit_stats
    print("2. W19 bitwise: W20 with the refit off, and with it on before window 60", flush=True)
    real = W20._side_fit_head
    for bad, kind in ((lambda *a: (_ for _ in ()).throw(RuntimeError("boom")), "errors"),
                      (lambda Ls, Ss, ys, W0, b0, lam: (W0 + 1.0, b0, False), "kept_trained"),
                      (lambda Ls, Ss, ys, W0, b0, lam: (W0 * np.nan, b0, True), "kept_trained")):
        mb = stub(W20, True, fit=bad)
        ob = run(mb)
        W20._side_fit_head = real
        assert np.array_equal(ob, ref19), kind
        assert mb.fm_refit_stats[kind] == NM * len(CKS), (kind, mb.fm_refit_stats)
    print("3. failed / non-converged / non-finite joint fits keep the trained heads (== W19), no exception", flush=True)
    s20 = (ROOT / "tracks/t2_bci/submission/t2p_submission_w20.py").read_text()
    seg = s20[s20.index("        mr = self.cfg.get(\"fm_main_refit\")"):s20.index("    def reset_fm_refit")]
    assert "assert self.fm_refit is None" in seg and "assert not self.gpu_graph" in seg and "register_forward_pre_hook" in seg
    assert seg.index("if mr is not None:") < seg.index("register_forward_pre_hook")
    print("4. source: replaces fm_side_refit, eager path only, hooks only when on", flush=True)
    print("OK")


if __name__ == "__main__":
    main()

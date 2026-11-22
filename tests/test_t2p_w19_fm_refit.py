#!/usr/bin/env python
"""T2-D553 (template W19 + cfg fm_side_refit) test of the in-stream FM side-head refit, CPU, synthetic (no read numbers):
  1. GOLDEN: the template's own path (StreamModel._fm_head_apply -> _fm_refit_step -> _fm_refit at 60 / 100 / 150, labels from is_hist_post,
     'conf' path) on a 190-window stream with 5 FM members (4 headed, 1 head-less) = an independent replay with the D553 READER's functions
     (scripts/local/t2_d540a_stack_read.py: fit_head / conf_half / softmax, exec'd from its source; the reader fits on the pre-head logits L,
     the template on log p = L - lse(L), the same objective): every window's member probabilities to <= 1e-6, the re-fitted heads to <= 1e-6;
  2. W18 / W19 bitwise: without cfg fm_side_refit (fm_refit None) and, with it, on every window before the first refit point, the outputs ==
     W18's _fm_head_apply exactly;
  3. a fit that raises / does not converge / returns non-finite keeps the trained head (that member's outputs == W18's, no exception), and a
     failure of the whole refit keeps the trained heads for the stream;
  4. source checks: reset_state calls reset_fm_refit; the refit path never raises out of predict (try / except around _fm_refit).
  python tests/test_t2p_w19_fm_refit.py  ->  "OK"
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
K, D, NW = 3, 82, 190


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tracks/t2_bci/submission" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W18, W19 = load("t2p_submission_w18"), load("t2p_submission_w19")
RS = {}
src = open(ROOT / "scripts/local/t2_d540a_stack_read.py").read()
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("fit_head", "conf_half", "softmax"):
        exec(compile(ast.Module([node], []), "t2_d540a_stack_read(fn)", "exec"), dict(np=np, minimize=__import__("scipy.optimize", fromlist=["minimize"]).minimize), RS)
for k in ("fit_head", "conf_half", "softmax"):
    assert k in RS, k

rng = np.random.default_rng(553)
L = rng.normal(0, 1.5, (5, NW, K))                       # 5 members' pre-head logits
S = rng.normal(0, 1, (NW, D))
POST = rng.dirichlet(np.ones(K) * 0.7, NW)               # the stream's teacher posteriors ('conf' path: is_hist_post)
HEADS = [(rng.normal(0, 0.05, (K, D)), rng.normal(0, 0.1, K)) for _ in range(4)] + [None]
P = np.exp(L - L.max(2, keepdims=True))
P = P / P.sum(2, keepdims=True)


def stub(mod, refit, fit=None):
    m = mod.StreamModel.__new__(mod.StreamModel)
    m.cfg = {"n_classes": K} if not refit else {"n_classes": K, "fm_side_refit": {"lam": 1.0}}
    m.is_cfg = {"labels": "confR", "refits": list(CKS)}
    m.is_refits = list(CKS)
    m.gshared = {"g": {}}
    m.fm_heads = list(HEADS)
    if refit:                                            # set_fm_side's refit block, as the template runs it
        m.fm_refit = dict(lam=1.0)
        m.fm_refit_stats = dict(fits=0, kept_trained=0, errors=0)
        m.reset_fm_refit()
        m.is_cfg = dict(m.is_cfg, labels="conf")       # the test feeds the labels through is_hist_post
        m.is_hist_post, m.is_hist_ok = [], []
    if fit is not None:
        mod._side_fit_head = fit
    return m


def run(m, n=NW):
    outs = []
    for c in range(n):
        m.is_n = c
        m._side_now = S[c].astype(np.float32)
        o = m._fm_head_apply([P[i, c].astype(np.float32) for i in range(5)])
        outs.append(np.stack(o))
        if hasattr(m, "is_hist_post"):
            m.is_hist_post.append(POST[c])
            m.is_hist_ok.append(True)
    return np.stack(outs)                                # (n, 5, K)


def reader_replay():
    """the reader's loop (fm_refit_probs) on the same stream: L, S, the teacher rows = POST[:c] at each refit point."""
    out = np.zeros((NW, 5, K))
    heads_at = {}
    for i in range(5):
        h = HEADS[i]
        if h is None:
            out[:, i] = P[i]
            continue
        s32 = S.astype(np.float32).astype(np.float64)
        p = RS["softmax"](L[i] + s32 @ h[0].T + h[1])
        for k, c in enumerate(CKS):
            e = CKS[k + 1] if k + 1 < len(CKS) else NW
            ya, keep = RS["conf_half"](POST[:c])
            sel = np.arange(c)[keep]
            W_, b_, _ = RS["fit_head"](L[i][sel], s32[sel], ya[keep].astype(np.int64), h[0], h[1], 1.0)
            heads_at[(i, c)] = (W_, b_)
            p[c:e] = RS["softmax"](L[i][c:e] + s32[c:e] @ W_.T + b_)
        out[:, i] = p
    return out, heads_at


def main():
    ref18 = run(stub(W18, False))
    off19 = run(stub(W19, False))
    assert np.array_equal(ref18, off19), "W19 without fm_side_refit != W18"
    m = stub(W19, True)
    on19 = run(m)
    assert np.array_equal(on19[:CKS[0]], ref18[:CKS[0]]), "before the first refit point the refit path must equal W18 bitwise"
    rep, heads_at = reader_replay()
    dp = float(np.abs(on19 - rep).max())
    W_, b_ = m.fm_heads_now[0]
    dW = max(float(np.abs(W_ - heads_at[(0, CKS[-1])][0]).max()), float(np.abs(b_ - heads_at[(0, CKS[-1])][1]).max()))
    print(f"1. golden: template vs the reader's replay max |dp| {dp:.2e}; last heads (member 0) |dW|, |db| {dW:.2e}; "
          f"stats {m.fm_refit_stats}", flush=True)
    assert dp <= 1e-6 and dW <= 1e-6, (dp, dW)
    assert m.fm_refit_stats == dict(fits=12, kept_trained=0, errors=0), m.fm_refit_stats
    assert np.array_equal(on19[:, 4], ref18[:, 4]), "the head-less member must be untouched"
    print("2. W18 bitwise: refit off, and refit on before window 60", flush=True)
    real = W19._side_fit_head
    for bad, kind in ((lambda *a: (_ for _ in ()).throw(RuntimeError("boom")), "errors"),
                      (lambda Ls, Ss, ys, W0, b0, lam: (W0 + 1.0, b0, False), "kept_trained"),
                      (lambda Ls, Ss, ys, W0, b0, lam: (W0 * np.nan, b0, True), "kept_trained")):
        mb = stub(W19, True, fit=bad)
        ob = run(mb)
        W19._side_fit_head = real
        assert np.allclose(ob, ref18, rtol=0, atol=0), kind
        assert mb.fm_refit_stats[kind] == 12, (kind, mb.fm_refit_stats)
    mw = stub(W19, True)
    mw._fm_refit = lambda c: (_ for _ in ()).throw(ValueError("whole refit"))
    ow = run(mw)
    assert np.array_equal(ow, ref18) and mw.fm_refit_stats["errors"] == 3, mw.fm_refit_stats
    print("3. failed / non-converged / non-finite fits and a failed refit keep the trained heads (== W18), no exception", flush=True)
    s19 = (ROOT / "tracks/t2_bci/submission/t2p_submission_w19.py").read_text()
    assert "self.reset_fm_refit()" in s19[s19.index("    def reset_state"):s19.index("    def _fam_decode")]
    seg = s19[s19.index("    def _fm_refit_step"):s19.index("    def _fm_refit(self, c)")]
    assert "try:\n                self._fm_refit(c)\n            except Exception" in seg
    print("4. source: reset_state resets the FM refit; the refit is wrapped", flush=True)
    print("OK")


if __name__ == "__main__":
    main()

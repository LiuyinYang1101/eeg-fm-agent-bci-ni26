#!/usr/bin/env python
"""T2-D560 (template W20 = W19 + cfg fm_proto) test of the per-stream class prototypes + the head-less member's main-head refit, CPU,
synthetic (no read numbers):
  1. GOLDEN: the template's path (StreamModel._fm_head_apply -> _fm_refit_step -> _fm_refit (D553) -> _fm_proto_refit at 60 / 100 / 150,
     labels from is_hist_post, -> _fm_proto_apply) on a 190-window stream with 5 FM members (4 headed, 1 head-less; per-window pooled
     features as the hooks deliver them) = THE LAB's OWN member function (scripts/local/t2_lab_ideas.py _member, proto 0.5 = what D560
     screened and confirmed) on a fake context holding the same stream: every member, every window, <= 1e-6;
  2. W19 bitwise: W20 without fm_proto == W19 (fm_side_refit on) on every window; with it, every window before the first refit == W19;
  3. a head-less fit that raises keeps the trained main head (== the lab's hl 'skip'), stats count it; a window with a non-finite feature
     gets W19's output; no exception;
  4. source: set_fm_side asserts fm_proto needs fm_side_refit and no fm_main_refit (the CUDA-graph path is allowed); hooks only when on;
  5. the buffer path (the hook body copies into a static buffer, _fm_head_apply refreshes the features from it) == the direct path bitwise.
  python tests/test_t2p_w20_proto.py  ->  "OK"
"""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")
CKS = (60, 100, 150)
K, DM, DS, NW, NM = 3, 64, 82, 190, 5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SUB = ROOT / "tracks/t2_bci/submission"
W19, W20 = load("t2p_submission_w19", SUB / "t2p_submission_w19.py"), load("t2p_submission_w20", SUB / "t2p_submission_w20.py")
LAB = load("t2_lab_ideas", ROOT / "scripts/local/t2_lab_ideas.py")
RS = {}
src = open(ROOT / "scripts/local/t2_d540a_stack_read.py").read()
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("fit_head", "softmax"):
        exec(compile(ast.Module([node], []), "t2_d540a_stack_read(fn)", "exec"),
             dict(np=np, minimize=__import__("scipy.optimize", fromlist=["minimize"]).minimize), RS)
rng = np.random.default_rng(560)
FE = rng.normal(0, 1, (NM, NW, DM)).astype(np.float32).astype(np.float64)   # pooled features per member per window (as the hooks give)
MAIN = [(rng.normal(0, 0.15, (K, DM)), rng.normal(0, 0.1, K)) for _ in range(NM)]
S = rng.normal(0, 1, (NW, DS))
S32 = S.astype(np.float32).astype(np.float64)
POST = rng.dirichlet(np.ones(K) * 0.7, NW)
SIDE = [(rng.normal(0, 0.05, (K, DS)), rng.normal(0, 0.1, K)) for _ in range(NM - 1)] + [None]
L = np.stack([FE[i] @ MAIN[i][0].T + MAIN[i][1] for i in range(NM)])
P = np.exp(L - L.max(2, keepdims=True))
P = P / P.sum(2, keepdims=True)


def stub(mod, proto, nan_at=None):
    m = mod.StreamModel.__new__(mod.StreamModel)
    m.cfg = {"n_classes": K}
    m.is_cfg = {"labels": "conf", "refits": list(CKS)}
    m.is_refits = list(CKS)
    m.gshared = {"g": {}}
    m.fm_heads = list(SIDE)
    m.fm_refit = dict(lam=1.0)                           # D553 (W19's fm_side_refit lam 1), as set_fm_side sets it
    m.fm_refit_stats = dict(fits=0, kept_trained=0, errors=0)
    m.fm_main = None
    if proto:                                            # set_fm_side's D560 block, as the template runs it (hooks replaced by _fm_feat)
        m.fm_proto = dict(w=0.5, lam_hl=1.0)
        m.fm_proto_stats = dict(refits=0, hl_fits=0, kept_trained=0, errors=0)
        m.fm_main0 = list(MAIN)
        m._fm_feat = [None] * NM
    m.nan_at = nan_at
    m.reset_fm_refit()
    m.is_hist_post, m.is_hist_ok = [], []
    return m


def run(m, n=NW):
    outs = []
    for c in range(n):
        m.is_n = c
        m._side_now = S[c].astype(np.float32)
        if getattr(m, "fm_proto", None) is not None:
            m._fm_feat = [FE[i, c].copy() for i in range(NM)]
            if m.nan_at == c:
                m._fm_feat[1][3] = np.nan
        outs.append(np.stack(m._fm_head_apply([P[i, c].astype(np.float32) for i in range(NM)])))
        m.is_hist_post.append(POST[c])
        m.is_hist_ok.append(True)
    return np.stack(outs)


class Ctx:
    CKS = CKS
    stream, order = np.zeros(NW, np.int64), np.arange(NW)
    fit_head = staticmethod(lambda *a: RS["fit_head"](*a))
    softmax = staticmethod(lambda x: RS["softmax"](x))

    @staticmethod
    def dumps(path):
        i = int(path[1:])
        d = dict(rows=np.arange(NW), feat=FE[i], Wm=MAIN[i][0], bm=MAIN[i][1], z=S32, p=None)
        if SIDE[i] is not None:
            d.update(Ws=SIDE[i][0], bs=SIDE[i][1])
        return d

    @staticmethod
    def teacher(f, d):
        return {r: r for r in range(NW)}, {c: POST for c in CKS}


def lab(hl="refit"):
    LAB._FM.clear()
    return np.stack([LAB._member(Ctx, 0, 0, f"m{i}", proto=0.5, hl=hl) for i in range(NM)], 1)    # (NW, NM, K)


def main():
    ref19 = run(stub(W19, False))
    off20 = run(stub(W20, False))
    assert np.array_equal(ref19, off20), "W20 without fm_proto != W19"
    m = stub(W20, True)
    on20 = run(m)
    assert np.array_equal(on20[:CKS[0]], ref19[:CKS[0]]), "before the first refit point fm_proto must equal W19 bitwise"
    lb = lab()
    dp = float(np.abs(on20 - lb).max())
    print(f"1. golden: template vs the LAB's _member (proto 0.5), every member / window: max |dp| {dp:.2e}; stats {m.fm_proto_stats}", flush=True)
    assert dp <= 1e-6, dp
    assert m.fm_proto_stats == dict(refits=3, hl_fits=3, kept_trained=0, errors=0), m.fm_proto_stats
    assert float(np.abs(on20[CKS[0]:] - ref19[CKS[0]:]).max()) > 1e-3, "fm_proto must move the output after the first refit"
    print("2. W19 bitwise: fm_proto off, and on before window 60", flush=True)
    real = W20._side_fit_head

    def bad(Ls, Ss, ys, W0, b0, lam):                    # raise only for the head-less MAIN-head fit (features as Ss, no fixed logits)
        if Ss.shape[1] == DM:
            raise RuntimeError("boom")
        return real(Ls, Ss, ys, W0, b0, lam)
    W20._side_fit_head = bad
    try:
        mb = stub(W20, True)
        ob = run(mb)
    finally:
        W20._side_fit_head = real
    ds = float(np.abs(ob - lab("skip")).max())
    assert ds <= 1e-6 and mb.fm_proto_stats == dict(refits=3, hl_fits=3, kept_trained=0, errors=3), (ds, mb.fm_proto_stats)
    mn = stub(W20, True, nan_at=120)
    on_ = run(mn)
    W19r = run(stub(W19, False))
    assert np.array_equal(on_[120], W19r[120]) and np.allclose(on_[121:], on20[121:], atol=1e-6, rtol=0), "a non-finite feature window"
    print(f"3. a raising head-less fit -> trained main head + prototypes (== the lab's hl 'skip', |dp| {ds:.1e}); a non-finite feature -> "
          "W19's output for that window", flush=True)
    s20 = (SUB / "t2p_submission_w20.py").read_text()
    seg = s20[s20.index("        pr = self.cfg.get(\"fm_proto\")"):s20.index("    def reset_fm_refit")]
    for a in ("assert self.fm_main is None", "assert self.fm_refit is not None", "register_forward_pre_hook", "self._fm_feat_copy(j, inp[0])"):
        assert a in seg, a
    seg_p = seg[seg.index("if pr is not None:"):]
    assert "gpu_graph" not in seg_p, "D560 must run on W19 b2's CUDA-graph path (the hooks copy into static buffers)"
    assert seg.index("if pr is not None:") < seg_p.index("register_forward_pre_hook") + seg.index("if pr is not None:")
    print("4. source: needs fm_side_refit, no fm_main_refit; graph path allowed (buffer hooks); hooks only when on", flush=True)
    mb2 = stub(W20, True)                                 # 5. the buffer path: the hook body -> static buffers -> _fm_head_apply's refresh
    mb2._fm_feat_buf = [None] * NM
    outs = []
    for c in range(NW):
        mb2.is_n = c
        mb2._side_now = S[c].astype(np.float32)
        for i in range(NM):
            mb2._fm_feat_copy(i, torch.from_numpy(FE[i, c].astype(np.float32))[None])
        mb2._fm_feat = [np.full(DM, np.nan)] * NM        # stale on purpose: the refresh must replace it from the buffers
        outs.append(np.stack(mb2._fm_head_apply([P[i, c].astype(np.float32) for i in range(NM)])))
        mb2.is_hist_post.append(POST[c])
        mb2.is_hist_ok.append(True)
    db = float(np.abs(np.stack(outs) - on20).max())
    assert db == 0.0, db
    print(f"5. buffer path (hook copy -> static buffer -> refresh) == the direct path bitwise (|dp| {db:.1e})", flush=True)
    print("OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""T2-D563 (template W21 = W20 + cfg fm_knn) test of the per-stream kNN datastore vote, CPU, synthetic (no read numbers):
  1. GOLDEN: StreamModel._knn_apply window by window (space both, beta 0.5 = 0.25 per space, k 10, t0 30) on a 150-window stream (5 members'
     pooled features, the 82-d side vector, W20's final outputs) = THE LAB's own vote (scripts/local/t2_lab_ideas.py _knn_stream, the
     knn idea's combination: log p + beta / 2 * log p_kNN per space, softmax) on the same stream: every window <= 1e-9; windows < t0 untouched;
  2. W20 bitwise: W21 without fm_knn == W20 on the D560 golden stream (tests/test_t2p_w20_proto.py's stub, every window);
  3. a window with a non-finite feature gets no term and is never a neighbour (its output == its input; the next windows equal a replay
     that drops it from the datastore); stats count voted / skipped;
  4. source: set_fm_side asserts fm_knn needs fm_proto (D560's buffers); the vote is applied on both predict paths (graph and eager);
  5. review F4 (W21 only): a failed head-less fit -> that member's W19 behaviour (== the lab's hl 'skip' + prototypes on the headed members);
  6. review F5 (W21 only): an error in the prototype apply -> W19's FM output; F4 / F5 are output-identical in normal cases (part 2: == W20).
  python tests/test_t2p_w21_knn.py  ->  "OK"
"""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from ni26 import kit  # noqa: E402

kit.use_track("bci_decoding")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SUB = ROOT / "tracks/t2_bci/submission"
W21 = load("t2p_submission_w21", SUB / "t2p_submission_w21.py")
LAB = load("t2_lab_ideas", ROOT / "scripts/local/t2_lab_ideas.py")
P20 = load("test_t2p_w20_proto", ROOT / "tests/test_t2p_w20_proto.py")
RS = {}
src = open(ROOT / "scripts/local/t2_d540a_stack_read.py").read()
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("conf_half", "softmax"):
        exec(compile(ast.Module([node], []), "t2_d540a_stack_read(fn)", "exec"), dict(np=np), RS)
rng = np.random.default_rng(563)
N, NM, DM, DS = 150, 5, 64, 82
FE = rng.normal(0, 1, (NM, N, DM)).astype(np.float32).astype(np.float64)
SV = rng.normal(0, 1, (N, DS))
P = rng.dirichlet(np.ones(3) * 1.5, N)
KN = dict(space="both", beta=0.5, k=10, t0=30)


class Ctx:
    conf_half = staticmethod(lambda pp: RS["conf_half"](pp))


def stub(fe_nan_at=None):
    m = W21.StreamModel.__new__(W21.StreamModel)
    m.cfg = {"n_classes": 3}
    m.fm_refit, m.fm_proto = None, None
    m.fm_knn = dict(spaces=["fm", "side"], beta=0.5, k=10, t0=30, floor=None)
    m.fm_knn_stats = dict(windows=0, voted=0, skipped=0, errors=0)
    m.reset_fm_refit()
    m.nan_at = fe_nan_at
    return m


def run(m, rows=range(N)):
    out = []
    for t in rows:
        m._fm_feat = [FE[i, t].copy() for i in range(NM)]
        if m.nan_at == t:
            m._fm_feat[2][5] = np.nan
        m._side_now = SV[t].astype(np.float32)
        out.append(m._knn_apply(P[t].copy()))
    return np.stack(out)


def lab(rows):
    rows = np.asarray(list(rows))
    X = np.hstack([FE[i][rows] / (np.linalg.norm(FE[i][rows], axis=1, keepdims=True) + 1e-9) for i in range(NM)])
    S32 = SV[rows].astype(np.float32).astype(np.float64)
    lg = np.log(np.maximum(P[rows], 1e-30))
    for sp, Z in (("fm", X), ("side", S32)):
        pk = LAB._knn_stream(Ctx, Z, P[rows], sp, KN["k"], KN["t0"], None)
        mk = np.isfinite(pk).all(1)
        lg[mk] = lg[mk] + (KN["beta"] / 2) * np.log(pk[mk])
    return RS["softmax"](lg)


def main():
    m = stub()
    got = run(m)
    exp = lab(range(N))
    d = float(np.abs(got - exp).max())
    print(f"1. golden: template vs the LAB's vote, every window: max |dp| {d:.2e}; stats {m.fm_knn_stats}", flush=True)
    assert d <= 1e-9, d
    assert np.array_equal(got[:KN["t0"]], P[:KN["t0"]]), "windows before t0 must be untouched"
    assert m.fm_knn_stats == dict(windows=N, voted=N - KN["t0"], skipped=KN["t0"], errors=0), m.fm_knn_stats
    on20 = P20.run(P20.stub(P20.W20, True))
    on21 = P20.run(P20.stub(W21, True))
    assert np.array_equal(on20, on21), "W21 without fm_knn != W20"
    print("2. W20 bitwise: W21 without fm_knn == W20 on the D560 golden stream", flush=True)
    bad = 70
    mb = stub(fe_nan_at=bad)
    ob = run(mb)
    assert np.array_equal(ob[bad], P[bad]), "the non-finite window must get no term"
    keep_rows = [t for t in range(N) if t != bad]
    exp2 = lab(keep_rows)                                 # the datastore without the bad window = the lab on the stream without it
    d2 = float(np.abs(np.delete(ob, bad, 0)[bad:] - exp2[bad:]).max())
    assert d2 <= 1e-9, d2
    assert mb.fm_knn_stats["skipped"] == KN["t0"] + 1 and mb.fm_knn_stats["errors"] == 0, mb.fm_knn_stats
    print(f"3. a non-finite feature window: no term, never a neighbour (the later windows == the lab without it, |dp| {d2:.1e})", flush=True)
    # 5. review F4: a raising head-less fit -> that member's W19 behaviour (no refit, no prototypes); the headed members keep theirs
    real = W21._side_fit_head

    def bad(Ls, Ss, ys, W0, b0, lam):
        if Ss.shape[1] == P20.DM:
            raise RuntimeError("boom")
        return real(Ls, Ss, ys, W0, b0, lam)
    W21._side_fit_head = bad
    try:
        mf = P20.stub(W21, True)
        of = P20.run(mf)
    finally:
        W21._side_fit_head = real
    P20.LAB._FM.clear()
    exp4 = np.stack([P20.LAB._member(P20.Ctx, 0, 0, f"m{i}", proto=0.5, hl="skip", proto_scope="headed") for i in range(P20.NM)], 1)
    d4 = float(np.abs(of - exp4).max())
    assert d4 <= 1e-6 and mf.fm_proto_stats["errors"] == 3, (d4, mf.fm_proto_stats)
    # 6. review F5: an error inside the prototype apply -> W19's FM output for that window, counted, no exception
    mg = P20.stub(W21, True)
    on = P20.run(mg)
    outs = [np.asarray(x, np.float64) for x in P20.P[:, P20.NW - 1]]
    mg.fm_proto_now = ([np.zeros((2, 7))] * P20.NM, mg.fm_proto_now[1])       # a wrong-shape prototype -> the apply raises
    e0 = mg.fm_proto_stats["errors"]
    got = mg._fm_proto_apply(outs)
    assert all(np.array_equal(g_, o_) for g_, o_ in zip(got, outs)) and mg.fm_proto_stats["errors"] == e0 + 1
    print(f"5. F4: a failed head-less fit -> that member's W19 behaviour (== the lab's hl 'skip' + headed-only prototypes, |dp| {d4:.1e}); "
          "6. F5: an apply error -> W19's FM output, counted; both output-identical in normal cases (part 2)", flush=True)
    s21 = (SUB / "t2p_submission_w21.py").read_text()
    seg = s21[s21.index('        kn = self.cfg.get("fm_knn")'):s21.index("    def reset_fm_refit")]
    assert "assert self.fm_proto is not None" in seg
    assert s21.count("p = self._knn_apply(p)") == 2, "the vote on both predict paths"
    print("4. source: fm_knn needs fm_proto; applied on both predict paths", flush=True)
    print("OK")


if __name__ == "__main__":
    main()

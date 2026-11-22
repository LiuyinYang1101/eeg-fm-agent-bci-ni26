#!/usr/bin/env python
"""Self-test: the streaming aligner (what predict() runs, one window at a time) equals the vectorised training
form for every mode, and EA whitening behaves (aligned stream covariance -> identity for the full-stream mode)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from ni26 import stream  # noqa: E402

rng = np.random.default_rng(0)
N, C, T = 40, 41, 480
A = rng.normal(size=(C, C))
X = np.einsum("cd,ndt->nct", A, rng.normal(size=(N, C, T))) * 15 + rng.normal(size=(N, C, 1)) * 30
prior = stream.trace_normalised_mean(stream.window_covs(X - X.mean(-1, keepdims=True)))
for mode, kw in [("none", dict(scale=12.0)), ("wz", {}), ("sz", {}), ("ea", {}), ("ea", dict(prior=prior, n0=5.0))]:
    V = stream.align_stream(X, mode, **kw)
    al = stream.StreamAligner(mode, **kw)
    S = np.stack([al(x) for x in X])
    err = float(np.abs(V - S).max() / (np.abs(V).max() + 1e-12))
    print(f"{mode:6s} {str({k: v for k, v in kw.items() if k != 'prior'}):14s} max rel err {err:.2e}")
    assert err < 1e-5, (mode, err)
al.reset(); assert al.n == 0 and al.sum_cov is None
F = stream.align_stream(X, "eafull", shrink=0.0)
Cm = stream.window_covs(F).mean(0)
print("eafull mean cov - I max abs", float(np.abs(Cm - np.eye(C)).max()))
assert np.abs(Cm - np.eye(C)).max() < 1e-3

# the package's aligner is a verbatim copy of ni26.stream's (t2p_submission.py "verbatim ... keep in sync" block)
import inspect  # noqa: E402
tpl = (Path(__file__).resolve().parents[1] / "submission" / "t2p_submission.py").read_text()
blk = tpl.split("# ------------------------------------------------------------------- verbatim from src/ni26/stream.py (keep in sync)\n")[1]
blk = blk.split("# ------------------------------------------------------------------- end verbatim")[0]
assert blk.strip() == (inspect.getsource(stream._inv_sqrt) + "\n\n" + inspect.getsource(stream.StreamAligner)).strip(), \
    "t2p_submission.py's StreamAligner / _inv_sqrt drifted from src/ni26/stream.py"
print("package aligner == ni26.stream (verbatim)")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t2p_route  # noqa: E402
fblk = tpl.split("# ------------------------------------------------------------------- verbatim from tracks/t2_bci/scripts/t2p_route.py (keep in sync)\n")[1]
fblk = fblk.split("# ------------------------------------------------------------------- end verbatim fingerprint")[0]
assert fblk.strip() == inspect.getsource(t2p_route.fingerprint).strip(), "t2p_submission.py's fingerprint drifted from t2p_route.py"
print("package fingerprint == t2p_route.fingerprint (verbatim)")

# B24 guards: a corrupted window never poisons the session's statistics; a x1e4 spike does not squash later windows
for mode in ("sz", "ea"):
    ref = stream.StreamAligner(mode)
    clean = [ref(x) for x in X]
    for name, fn in [("nan10", lambda x: np.where(rng.random(x.shape) < 0.1, np.nan, x)),
                     ("inf", lambda x: np.where(rng.random(x.shape) < 0.01, np.where(rng.random(x.shape) < 0.5, np.inf, -np.inf), x)),
                     ("zero", lambda x: 0 * x)]:
        al = stream.StreamAligner(mode)
        Y = [al(fn(x) if t == 5 else x) for t, x in enumerate(X)]
        assert all(np.isfinite(y).all() for y in Y), (mode, name)
        # the bad window never entered the statistics: later windows = a clean stream without window 5
        al2 = stream.StreamAligner(mode)
        Z = [al2(x) for t, x in enumerate(X) if t != 5]
        err = max(float(np.abs(Y[t] - Z[t - 1]).max()) for t in range(6, N))
        assert err < 1e-5, (mode, name, err)
    print(f"{mode}: NaN / inf / zero window -> finite, statistics untouched")
for mode in ("sz", "ea"):                                             # T2-D116: EA has the spike guard too
    ref = stream.StreamAligner(mode)
    clean = [ref(x) for x in X]
    for t_sp in (0, 5):                                                # a x1e4 spike at the session start or later
        al = stream.StreamAligner(mode)
        Y = [al(x * (1e4 if t == t_sp else 1)) for t, x in enumerate(X)]
        late = np.stack(Y[t_sp + 3:]).std() / np.stack(clean[t_sp + 3:]).std()
        assert np.isfinite(np.stack(Y)).all() and 0.8 < late < 1.25, (mode, t_sp, late)
        print(f"{mode}: x1e4 spike at window {t_sp} -> later windows' scale ratio {late:.3f} (recovered)")
    al = stream.StreamAligner(mode)                                    # a flat channel all session: finite output
    Y = [al(np.where(np.arange(C)[:, None] == 3, 0.0, x)) for x in X]
    assert np.isfinite(np.stack(Y)).all() and np.abs(np.stack(Y)[:, 3]).max() < 1e-6, mode
    print(f"{mode}: flat channel -> finite, the flat channel stays 0")
# review 2026-10-16 (T2 P1d): with a prior (n0 > 0) the EA spike guard corrects the prior's trace scale too
ref = stream.StreamAligner("ea", prior=prior, n0=5.0)
clean = [ref(x) for x in X]
for t_sp in (0, 5):
    al = stream.StreamAligner("ea", prior=prior, n0=5.0)
    Y = [al(x * (1e4 if t == t_sp else 1)) for t, x in enumerate(X)]
    late = np.stack(Y[t_sp + 3:]).std() / np.stack(clean[t_sp + 3:]).std()
    assert np.isfinite(np.stack(Y)).all() and 0.8 < late < 1.25, ("ea+prior", t_sp, late)
    print(f"ea + prior n0 5: x1e4 spike at window {t_sp} -> later windows' scale ratio {late:.3f} (recovered)")

# the package's PriorCorrector: = t2p_report.causal_prior_correction on clean outputs; n0 = 0 with a broken first output
# stays finite (review T2 P1d)
from ni26 import kit  # noqa: E402
import importlib.util  # noqa: E402
kit.use_track("bci_decoding")
spec = importlib.util.spec_from_file_location("t2p_tpl", Path(__file__).resolve().parents[1] / "submission" / "t2p_submission.py")
tplm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tplm)
from t2p_report import causal_prior_correction  # noqa: E402
P = rng.dirichlet(np.ones(3), 50)
for n0 in (3.0, 0.0):
    pc = tplm.PriorCorrector(1.0, n0, 3)
    Q = np.stack([pc(p) for p in P])
    R = causal_prior_correction(P, (np.zeros(len(P)), np.arange(len(P))), 1.0, n0)
    assert np.abs(Q - R).max() < 1e-12, (n0, np.abs(Q - R).max())
    pc.reset()
    Q = np.stack([pc(np.full(3, np.nan) if t == 0 else p) for t, p in enumerate(P)])
    assert np.isfinite(Q).all(), n0
    print(f"PriorCorrector n0 {n0:g}: == causal_prior_correction; NaN first output -> finite")
# T2-D286: the package's SinkhornCorrector (SK10, the user's approval 2026-10-27) = t2p_report.causal_sinkhorn = T2-D285's read; iters 1 =
# PriorCorrector(1, 3); reset / deepcopy give independent streams; a broken output stays finite; tta kind selects it
import copy  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "local"))
import t2_d285_quota as D285  # noqa: E402
for P_ in (rng.dirichlet(np.ones(3), 60), rng.dirichlet(np.ones(3) * 0.3, 200)):
    for it in (1, 3, 10):
        sk = tplm.SinkhornCorrector(it, 3, 3)
        Q = np.stack([sk(p) for p in P_])
        R = causal_sinkhorn(P_, (np.zeros(len(P_)), np.arange(len(P_))), it, 3)
        assert np.abs(Q - R).max() < 1e-12, (it, np.abs(Q - R).max())
        assert np.abs(R - D285.sinkhorn(P_, it)).max() < 1e-12, it
    R1 = causal_prior_correction(P_, (np.zeros(len(P_)), np.arange(len(P_))), 1.0, 3.0)
    assert np.abs(causal_sinkhorn(P_, (np.zeros(len(P_)), np.arange(len(P_))), 1, 3) - R1).max() < 1e-12, "iters 1 != the prior"
sk = tplm.SinkhornCorrector(10, 3, 3)
A = [sk(p) for p in P[:20]]
sk2 = copy.deepcopy(sk)
sk2.reset()
B = [sk2(p) for p in P[:20]]
assert np.array_equal(np.stack(A), np.stack(B)) and len(sk.rows) == 23, "reset / deepcopy: streams are not independent"
sk.reset()
Q = np.stack([sk(np.full(3, np.nan) if t in (0, 7) else p) for t, p in enumerate(P)])
assert np.isfinite(Q).all()
src_ = inspect.getsource(tplm.StreamModel.__init__)            # tta kind "sinkhorn" -> SK, "prior" -> the prior, else none
assert 'if t.get("kind") == "sinkhorn":' in src_ and 'SinkhornCorrector(t["iters"], t["n0"]' in src_ and 'PriorCorrector(t["alpha"], t["n0"]' in src_
print("SinkhornCorrector (SK1 / 3 / 10): == causal_sinkhorn == T2-D285's read; SK1 == the prior; reset / deepcopy independent; NaN -> finite")
# load_net (review T2 P1a): a worker braindecode / mne that builds other FIR lengths adopts the trained taps
import torch  # noqa: E402
from ni26 import train as TR  # noqa: E402
cls_, kw_ = TR.MODEL_SPECS["fbcnet_bboth"]
cfgn = dict(ch_names=[f"c{i}" for i in range(C)], n_classes=3, n_times=T, sfreq=120.0, braindecode_class=cls_,
            model_kwargs={k: [list(b) for b in v] for k, v in kw_.items()})
src_net = tplm.build_net(cfgn)
sd = {k: v.clone() for k, v in src_net.state_dict().items()}
sd["spectral_filtering.fir_list.0"] = torch.randn(401) * 1e-2
sd["spectral_filtering.fir_a_list.0"] = torch.nn.functional.pad(torch.ones(1), (0, 400))
net = tplm.load_net(tplm.build_net(cfgn), sd).eval()
assert tuple(net.state_dict()["spectral_filtering.fir_list.0"].shape) == (401,)
with torch.no_grad():
    o = net(torch.randn(2, C, T))
assert o.shape == (2, 3) and torch.isfinite(o).all()
bad = dict(sd, **{"final_layer.bias": torch.zeros(4)})
try:
    tplm.load_net(tplm.build_net(cfgn), bad)
    raise AssertionError("a non-FIR shape mismatch must still fail")
except RuntimeError:
    pass
print("load_net: FIR length mismatch -> trained taps adopted, forward finite; other mismatches still fail")
print("OK")

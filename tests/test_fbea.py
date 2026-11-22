"""T2-D424: the filter-bank EA mode of ni26.stream.align_stream (bench only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ni26 import stream  # noqa: E402


def _stream(n=12, c=6, t=480, seed=0):
    r = np.random.default_rng(seed)
    A = r.normal(size=(c, c))
    return np.einsum("cd,ndt->nct", A, r.normal(size=(n, c, t)) + np.sin(np.arange(t) * 2 * np.pi * 10 / 120)[None, None, :])


def test_bands_partition_the_window():
    X = _stream()
    X = X - X.mean(-1, keepdims=True)
    Xb = stream.band_split(X)
    assert Xb.shape == (len(stream.FB_EDGES),) + X.shape
    np.testing.assert_allclose(Xb.sum(0), X, atol=1e-9)
    for a in range(len(Xb)):                       # disjoint rfft supports -> zero sample cross-covariance
        for b in range(a + 1, len(Xb)):
            assert abs(np.einsum("nct,ndt->", Xb[a], Xb[b])) < 1e-6 * np.einsum("nct,nct->", X, X)


def test_one_band_is_ea():
    X = _stream()
    for kw in (dict(), dict(prior=np.eye(6) / 6, n0=3.0)):
        ea = stream.align_stream(X, "ea", **kw)
        fb = stream.align_stream(X, "fbea", edges=(0.0,), prior=None if "prior" not in kw else kw["prior"][None], n0=kw.get("n0", 0.0))
        np.testing.assert_allclose(fb, ea, rtol=1e-4, atol=1e-5)


def test_causal_and_unit_power():
    X = _stream(n=20, seed=1)
    full = stream.align_stream(X, "fbea")
    part = stream.align_stream(X[:9], "fbea")
    np.testing.assert_allclose(full[:9], part, atol=1e-6)          # window t uses windows 1..t only
    p = (full[5:].astype(np.float64) ** 2).mean(-1).sum(-1)          # total power per window ~ C, as EA
    assert np.all((p > 0.3 * 6) & (p < 3 * 6)), p


def test_prior_shape():
    X = _stream()
    P = stream.fb_prior(X - X.mean(-1, keepdims=True))
    assert P.shape == (len(stream.FB_EDGES), 6, 6)
    np.testing.assert_allclose(np.trace(P, axis1=1, axis2=2), 1.0, rtol=1e-9)
    out = stream.align_stream(X, "fbea", prior=P, n0=3.0)
    assert np.isfinite(out).all()


def test_eab_full_band_is_ea_and_causal():
    X = _stream(n=15, seed=2)
    np.testing.assert_allclose(stream.align_stream(X, "eab", band=(0.0, 1e9)), stream.align_stream(X, "ea"), rtol=1e-4, atol=1e-5)
    full, part = stream.align_stream(X, "eab"), stream.align_stream(X[:7], "eab")
    np.testing.assert_allclose(full[:7], part, atol=1e-6)
    assert np.isfinite(full).all()


def test_eal_causal_finite():
    X = _stream(n=15, seed=3)
    full, part = stream.align_stream(X, "eal"), stream.align_stream(X[:6], "eal")
    np.testing.assert_allclose(full[:6], part, atol=1e-6)
    assert np.isfinite(full).all() and not np.allclose(full, stream.align_stream(X, "eab"))


def test_eaf_gamma1_is_ea_and_causal():
    X = _stream(n=15, seed=4)
    np.testing.assert_allclose(stream.align_stream(X, "eaf", gamma=1.0), stream.align_stream(X, "ea"), rtol=1e-4, atol=1e-5)
    full, part = stream.align_stream(X, "eaf"), stream.align_stream(X[:8], "eaf")
    np.testing.assert_allclose(full[:8], part, atol=1e-6)
    assert np.isfinite(full).all() and not np.allclose(full, stream.align_stream(X, "ea"))

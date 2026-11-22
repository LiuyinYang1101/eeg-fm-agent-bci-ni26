#!/usr/bin/env python
"""T2-D244: a member temperature in t2p_pkgshape (spec member field `temp`), CPU:
  1. temper(P, 1) returns P itself (off = bitwise unchanged); temper(P, T) = softmax(log P / T) per row, rows sum to 1, NaN rows stay NaN
     (the windows outside a protocol's test / val);
  2. on files on disk (T2-D242's spec, kit + seen1, 3 indices): a member copy with temp 1 gives the anchor's rows exactly; with temp 2 the rows differ.
  python tests/test_pkgshape_temp.py  ->  "OK"
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
from t2p_pkgshape import temper  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    rng = np.random.default_rng(0)
    P = rng.dirichlet(np.ones(3), 50)
    P[:5] = np.nan
    assert temper(P, 1.0) is P
    for T in (1.5, 2.0, 3.0):
        Q = temper(P, T)
        L = np.log(P[5:]) / T
        S = np.exp(L - L.max(1, keepdims=True))
        assert np.allclose(Q[5:], S / S.sum(1, keepdims=True), atol=1e-12) and np.allclose(Q[5:].sum(1), 1) and np.isnan(Q[:5]).all(), T
        assert (Q[5:].max(1) <= P[5:].max(1) + 1e-12).all(), "T > 1 softens"
    print("temper: T 1 = the same array; T > 1 = softmax(log p / T), softer, NaN rows kept")
    m = json.load(open(O / "t2_d242/spec.json"))["members"]
    mem = {k: m[k] for k in ("fbc_rc75", "sh_vr", "stl", "mor")}
    mem["stl_t1"], mem["stl_t2"] = dict(m["stl"], temp=1.0), dict(m["stl"], temp=2.0)
    line = lambda s: {"fbc_rc75": 0.4, "sh_vr": 0.3, s: 0.2, "mor": 0.1}
    spec = {"n": 3, "anchor": "c_rc75", "bar": "d141", "members": mem, "arms": {"c_rc75": line("stl"), "c_t1": line("stl_t1"), "c_t2": line("stl_t2")}}
    with tempfile.TemporaryDirectory() as td:
        sp, out = Path(td) / "spec.json", Path(td) / "res.json"
        sp.write_text(json.dumps(spec))
        subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(sp), "--out", str(out)],
                       check=True, capture_output=True)
        rows = json.load(open(out))["rows"]
    get = lambda a: [{k: v for k, v in r.items() if k != "arm"} for r in rows if r["arm"] == a]
    assert get("c_t1") == get("c_rc75") and len(get("c_rc75")) == 6, "temp 1 = the anchor exactly"
    assert get("c_t2") != get("c_rc75"), "temp 2 changes the package"
    print("pkgshape: a temp-1 member copy reproduces the anchor's rows exactly; temp 2 differs")
    print("OK")


if __name__ == "__main__":
    main()

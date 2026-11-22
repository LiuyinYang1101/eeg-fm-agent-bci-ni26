#!/usr/bin/env python
"""T2-D550: the hard guard against TRUE-LABEL ORACLE outputs (scripts/local/t2_oracle_guard.py), CPU, no data:
  1. names: code-less in-stream suffixes (isS, isSr<k>, is<N>, is<N>r<k>; also after a recipe tag such as _kd) are oracles; the teacher-label
     codes (pr / pc / pa / ps), is0 (the base), isT (the base with times) and ordinary member names are not;
  2. the manifest: res.twin_suffix 'isS' / 'is60' under an innocent file name is refused; 'isSpr' / 'is0' pass;
  3. the readers' own need() / load() (scripts/local/t2_d532_read.py, compiled from its source) refuse a planted oracle by name and by manifest;
  4. every reader that loads member files calls the guard (source check): D532 need / load, D536 side_prob, the stack reader's side_prob_h,
     the D550 pool screen.
  python tests/test_t2_oracle_guard.py  ->  "OK"
"""
import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
import t2_oracle_guard as G  # noqa: E402

ORACLE = ["lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vrisS__s813.npz", "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vris60__s792.npz",
          "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vrisSr1__s1.npz", "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdisS__s811.npz",
          "lc9_0__ifnet__ea__n00__r1_pulx_vp2_e180_vris100r2__s1.npz"]
FINE = ["lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vrisSpr__s816.npz", "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vris60pc__s1.npz",
        "lc9_0__ifnet__ea__n00__r1_pulx_vp2_e180_vris0__s821.npz", "lc9_0__ifnet__ea__n00__r1_pulx_vp2_e180_vrisT__s821.npz",
        "lc9_0__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__s821.npz", "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vrisSpa__s1.npz",
        "lc9_0__shallow__ea__n00__r1_pulx_vp2_e180_vrisSps__s1.npz",
        "lc9_0__fm-st_large-mean__sz__n00__fmE100b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75tl0.15sddcmean__sidedcmean__s828.npz"]


def npz(d, name, suffix):
    p = Path(d) / name
    np.savez(p, test_idx=np.arange(3), prob_last=np.full((3, 3), 1 / 3), res=json.dumps(dict(twin_suffix=suffix)))
    return p


def main():
    for n in ORACLE:
        assert G.oracle_reason(n), n
    for n in FINE:
        assert G.oracle_reason(n) is None, n
    print(f"1. names: {len(ORACLE)} oracle forms refused, {len(FINE)} legitimate names pass")
    with tempfile.TemporaryDirectory() as td:
        for suf, bad in (("isS", True), ("is60", True), ("isSr1", True), ("isSpr", False), ("is0", False), ("isT", False), (None, False)):
            p = npz(td, f"lc9_0__shallow__x__s{abs(hash(str(suf))) % 1000}.npz", suf)
            z = np.load(p)
            assert bool(G.oracle_reason(p, z)) == bad, (suf, G.oracle_reason(p, z))
        print("2. manifest: twin_suffix isS / is60 / isSr1 refused under innocent names; isSpr / is0 / isT / none pass")
        src = (ROOT / "scripts/local/t2_d532_read.py").read_text()
        ns = {"np": np, "Path": Path, "refuse_oracle": G.refuse_oracle}
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name in ("need", "load"):
                exec(compile(ast.Module([node], []), "t2_d532_read(fn)", "exec"), ns)
        good = npz(td, FINE[0], "isSpr")
        assert ns["load"](good)[1].shape == (3, 3)
        for p in (npz(td, ORACLE[0], "isS"), npz(td, "lc9_0__shallow__innocent__s5.npz", "isS")):
            for fn in ("need", "load"):
                if fn == "need" and "innocent" in p.name:
                    continue                             # need() checks the name only; load() the manifest too
                try:
                    ns[fn](p)
                except SystemExit as e:
                    assert "oracle guard" in str(e), e
                else:
                    raise AssertionError(f"{fn} accepted the oracle {p.name}")
        print("3. the D532 reader's need() / load() refuse a planted oracle (name and manifest); a legitimate file loads")
    checks = {"scripts/local/t2_d532_read.py": ["refuse_oracle(path)", "refuse_oracle(path, z)"],
              "scripts/local/t2_d536_read.py": ["refuse_oracle(path, z)"],
              "scripts/local/t2_d540a_stack_read.py": ["refuse_oracle(path, z)"],
              "scripts/local/t2_d550_pool_screen.py": ["oracle_reason(p)", "oracle_reason(p, z)"]}
    for f, needles in checks.items():
        s = (ROOT / f).read_text()
        for nd in needles:
            assert nd in s, (f, nd)
    print("4. every member-loading reader calls the guard")
    print("OK")


if __name__ == "__main__":
    main()

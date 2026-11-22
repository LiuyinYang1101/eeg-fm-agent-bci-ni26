#!/usr/bin/env python
"""Backlog 69: FM -> CNN distillation (Recipe.kd / kd_t / kd_ext -> ni26.train.fit_eval), CPU:
  1. r1_pulxx_vp2_e180_vr_rc75_kd / _kdx differ from the FBCNet rc75 line, and r1_pulx_vp2_e180_vr_kd / _kdx from the ShallowNet vr line, in
     kd (0.5) / kd_ext (0 / 1) only; their per-user copies are the lines'; only these four recipes set kd;
  2. fit_eval with kd 0 = r1 bitwise (FBCNet B3 and ShallowNet), even when data carries ptr / Xkd / pkd;
  3. kd 0.5 (PROTEUS only, and + external windows) runs, moves the weights, keeps the plain net's state keys / shapes;
  4. the external batch is forwarded with BatchNorm in eval mode: external windows at 1000x the scale leave the running statistics on the
     PROTEUS scale (running_var within 10x of the kd-in run's, not ~1e6x);
  5. guards: kd with mixup or user_adv refused; a ptr of the wrong length refused.
  python tests/test_kd.py  ->  "OK"
"""
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402


def bn_var(state):
    return {k: v for k, v in state.items() if k.endswith("running_var")}


def main():
    want = {"r1_pulxx_vp2_e180_vr_rc75": ("r1_pulxx_vp2_e180_vr_rc75_kd", "r1_pulxx_vp2_e180_vr_rc75_kdx"),
            "r1_pulx_vp2_e180_vr": ("r1_pulx_vp2_e180_vr_kd", "r1_pulx_vp2_e180_vr_kdx")}
    for base, names in want.items():
        b = T.RECIPES[base]
        for n, ext in zip(names, (0, 1)):
            r = T.RECIPES[n]
            diff = {k for k, v in asdict(r).items() if v != asdict(b)[k]}
            assert diff == ({"name", "kd"} | ({"kd_ext"} if ext else set())) and r.kd == 0.5 and r.kd_t == 2.0 and r.kd_ext == ext, (n, diff)
            assert T.peruser_for(r) == T.peruser_for(b) and T.peruser_for(r).kd == 0, "the copies must be unchanged"
    assert sorted(n for n, r in T.RECIPES.items() if r.kd) == sorted(sum(want.values(), ())), "only the four kd recipes"
    print("recipes: the four kd recipes differ in kd / kd_ext only; copies unchanged")

    d = data()
    rng = np.random.default_rng(1)
    d["ptr"] = rng.dirichlet(np.ones(3), len(d["ytr"])).astype(np.float32)
    d["Xkd"] = d["Xtr"][::-1].copy()
    d["pkd"] = rng.dirichlet(np.ones(3), len(d["Xkd"])).astype(np.float32)
    plain = replace(T.RECIPES["r1"], epochs=2)
    for m in ("fbcnet_bboth", "shallow"):
        a = T.fit_eval(m, {k: v for k, v in d.items() if k not in ("ptr", "Xkd", "pkd")}, plain, 0, device="cpu", return_state=True)
        b = T.fit_eval(m, d, replace(plain, kd=0.0, kd_ext=1), 0, device="cpu", return_state=True)
        assert all(torch.equal(a["state_last"][k], b["state_last"][k]) for k in a["state_last"]) and \
            np.array_equal(a["prob_last"], b["prob_last"]), f"{m}: kd 0 must be r1 bitwise"
        ki = T.fit_eval(m, d, replace(plain, kd=0.5), 0, device="cpu", return_state=True)
        kx = T.fit_eval(m, d, replace(plain, kd=0.5, kd_ext=1), 0, device="cpu", return_state=True)
        for nm, c in (("kd", ki), ("kdx", kx)):
            sa, sc = a["state_last"], c["state_last"]
            assert sa.keys() == sc.keys() and all(sa[k].shape == sc[k].shape for k in sa), (m, nm)
            moved = max(float((sa[k].float() - sc[k].float()).abs().max()) for k in sa if sa[k].is_floating_point())
            assert moved > 1e-5, (m, nm, moved)
            print(f"{m}: {nm} moves the weights by up to {moved:.1e}; state keys / shapes = the plain net's")
        big = dict(d, Xkd=d["Xkd"] * 1000.0)
        kb = T.fit_eval(m, big, replace(plain, kd=0.5, kd_ext=1), 0, device="cpu", return_state=True)
        vi, vb = bn_var(ki["state_last"]), bn_var(kb["state_last"])
        assert vi, f"{m}: no BatchNorm found"
        r = max(float((vb[k] / vi[k].clamp_min(1e-12)).max()) for k in vi)
        assert r < 10, f"{m}: the external forward must not update BatchNorm (running_var ratio {r:.1e})"
        print(f"{m}: external windows x1000 leave BatchNorm on the PROTEUS scale (max running_var ratio {r:.2f})")
    for bad in (dict(mixup=0.4), dict(user_adv=0.3)):
        try:
            T.fit_eval("shallow", d, replace(plain, kd=0.5, **bad), 0, device="cpu")
            raise SystemExit(f"kd + {bad} must be refused")
        except AssertionError as e:
            assert "backlog 69" in str(e) or "T2-D293" in str(e), e
    try:
        T.fit_eval("shallow", dict(d, ptr=d["ptr"][:-1]), replace(plain, kd=0.5), 0, device="cpu")
        raise SystemExit("a ptr of the wrong length must be refused")
    except AssertionError:
        pass
    print("guards: kd + mixup / user_adv refused; a ptr of the wrong length refused")
    print("OK")


if __name__ == "__main__":
    main()

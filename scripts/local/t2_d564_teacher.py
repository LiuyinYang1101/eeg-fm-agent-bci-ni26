#!/usr/bin/env python
"""T2-D564 (registered 2026-11-17 before any number; zero GPU): the P1 teacher files with the window-mean (DC) family pooled in =
scripts/local/t2_d512_teacher.py EXACTLY, with R / OUT / DRAWS redirected and one term injected into the prefix posteriors (each edit asserted
to occur once): post_r<c> = [sum_k W_k * prefix_decode(family k's rows <= c) + w_t * prefix_decode(the DC bag's rows <= c, its cross-fitted
start)], renormalised. The DC bag = the chain's pred_s*_dc.npz (te_f / bag_f; the start = the OTHER folds' stranger mean, as the readers'
DCVEC). The full-stream `post` is unchanged (the lab reads post_r only).
    python scripts/local/t2_d564_teacher.py --chain t2_d555 --draws 825 --wt 0.075
"""
import argparse
import glob
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--chain", required=True, choices=("t2_d555", "t2_d557", "t2_d558"))
ap.add_argument("--draws", type=int, required=True, choices=(825, 828, 831))
ap.add_argument("--wt", type=float, required=True, choices=(0.0, 0.075, 0.15))   # 0.0 = the identity check (== the shipped files)
a = ap.parse_args()
assert {"t2_d555": 825, "t2_d557": 828, "t2_d558": 831}[a.chain] == a.draws
O = Path.home() / "workspace/outputs/neural_interface_challenge"
dcf = glob.glob(str(O / a.chain / "pred_s*_dc.npz"))
assert len(dcf) == 1, dcf
out = f"t2_d564_w{int(round(a.wt * 1000)):03d}_{a.draws}"
SRC = Path(__file__).resolve().parent / "t2_d512_teacher.py"
src = SRC.read_text()
tru = [l for l in src.split("\n") if l.startswith("TRU = {f: [u for u in set(U[MIX[(f, DRAWS[0])][0]])")]
assert len(tru) == 1, tru
inj = (tru[0] + "\n"
       f"_zdc = np.load({dcf[0]!r})\n"
       "DCP, DCV = {}, {}\n"
       "for _f in range(10):\n"
       "    _te, _p = _zdc[f'te_{_f}'], _zdc[f'bag_{_f}'].astype(np.float64)\n"
       "    _p = _p / _p.sum(1, keepdims=True)\n"
       "    _pos = {int(r): k for k, r in enumerate(_te)}\n"
       "    DCP[_f] = _p[np.array([_pos[int(r)] for r in MIX[(_f, DRAWS[0])][0]])]\n"
       "for _f in range(10):\n"
       "    _v = np.mean([(_zdc[f'bag_{_g}'] / _zdc[f'bag_{_g}'].sum(1, keepdims=True))[U[_zdc[f'te_{_g}']] == TRU[_g]].mean(0) for _g in range(10) if _g != _f], 0)\n"
       "    DCV[_f] = _v / _v.sum()\n"
       f"WT = {a.wt}\n"
       "print(f'[d564] DC term in the teacher at w_t {WT}', flush=True)")
for x, y in (('R = O / "t2_d512"\n', f'R = O / "{a.chain}"\n'),
             ('OUT = O / "t2_d512/teacher_p1"\n', f'OUT = O / "{out}/teacher_p1"\n'),
             ("DRAWS = (810, 811, 812)\n", f"DRAWS = {(a.draws, a.draws + 1, a.draws + 2)}\n"),
             (tru[0], inj),
             ("            q = sum(W[k] * prefix_decode(mx[k][h][j[:c]], V[k]) for k in W)\n",
              "            q = sum(W[k] * prefix_decode(mx[k][h][j[:c]], V[k]) for k in W) + WT * prefix_decode(DCP[f][h][j[:c]], DCV[f])\n")):
    assert src.count(x) == 1, (x[:60], src.count(x))
    src = src.replace(x, y)
exec(compile(src, str(SRC), "exec"), {"__file__": str(SRC), "__name__": "__main__"})

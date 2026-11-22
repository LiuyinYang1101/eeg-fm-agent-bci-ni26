#!/usr/bin/env python
"""T2-D600 (registered 4bec8ce1 before any number; descriptive, no decision): does in-stream adaptation cause the returning users' decline with
stream position? W21 in the lab (the t2_d571_lpw20d557_<d0> root, mix --idea w20_fb, NI26_LAB_PRE_R=1, C with NI26_LAB_D557=1; ASSERTED == the
D598 W21 mix file) against four UNADAPTED variants, each one switch in the lab's own code:
  (a) noP1  R1 = 0: the sh / ifn parts = R alone on every window (the template's pre-refit part);
  (b) noFM  the FM part = the members' trained heads (the unadapted part fm_feedback starts from: no side / proto refits);
  (c) noH   D539b's CNN side-head refits off (H_ON False);
  (d) none  (a) + (b) + (c);
  (e) raw   (Track3's addition, before numbers) W21's parts with the per-family causal Sinkhorn (and the DC part's) replaced by the raw pool:
            whether decoder state, not the models, drives a late returning dip.
Track3's second addition (before numbers): position is confounded with RUN TYPE (each session = Acq, Acq, Online x 4; D285), so the same tables
are keyed by run type (Acq = runs 1-2, Onl = runs 3-6; index.csv task) and by run number (1-6) as well as by position.
Per position interval (window order 0-59 / 60-99 / 100-149 / >= 150) x group (strangers; returning; P04 / P10 x session 2 / 3): the cell-mean
balanced accuracy on the interval's windows (the reader's cells()) and window accuracy, averaged over folds x draws.
    mem-guard env OMP_NUM_THREADS=4 python scripts/local/t2_d600_pos.py --set C|D   ->  outputs/t2_d600_<set>.json + printed tables
"""
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

T0 = time.time()
S = sys.argv[sys.argv.index("--set") + 1]
assert S in ("C", "D"), S
D0 = {"C": 831, "D": 834}[S]
for k in [k for k in os.environ if k.startswith("NI26_")]:
    del os.environ[k]
os.environ.update(NI26_D559_ROOT=f"t2_d571_lpw20d557_{D0}", NI26_D559_CKS="60,100,150", NI26_LAB_PRE_R="1")
if S == "C":
    os.environ["NI26_LAB_D557"] = "1"
O = os.path.expanduser("~/workspace/outputs/neural_interface_challenge")
sys.argv = ["t2_lab.py", "mix", "--set", S, "--base", "w20", "--idea", "w20_fb", "--out", "/dev/null"]
src = open("scripts/local/t2_lab.py").read()
CUT = '\nif MODE == "mix":'
assert src.count(CUT) == 1, "the lab's mix branch moved"
g = {"__name__": "d600", "__file__": os.path.abspath("scripts/local/t2_lab.py")}
exec(compile(src.split(CUT)[0], "t2_lab_head(d600)", "exec"), g)
assert g["PRE_R"] and g["BASE_W20"], (g["PRE_R"], g["BASE_W20"])
IDEAS, Ctx, lab_mix, parts_of, configure = g["IDEAS"], g["Ctx"], g["lab_mix"], g["parts_of"], g["configure"]
U, SES, TRU, ORD, Y, cells, IX, DRAWS = g["U"], g["SES"], g["TRU"], g["ORD"], g["Y"], g["cells"], g["IX"], g["DRAWS"]
ON, W18 = list(g["ON"]), g["W18"]
assert "D539b" in ON, ON

# ---- W21 (adapted): the lab's mix --idea w20_fb, asserted == D598's W21 file ----
reg = [(n, fn, p) for n, fn, p in IDEAS.registry() if n == "w20_fb"]
assert reg                                                # = the lab's mix mode: the FIRST registry entry of that name (asserted == D598 below)
r_ = reg[0][1](Ctx, **reg[0][2]) or {}
P21, W21 = r_.get("P", g["P_ON"]), r_.get("W", g["W_ON"])
assert not r_.get("dec") and not r_.get("post")
M = {"W21": lab_mix(P21, W21)}
z = np.load(f"{O}/t2_lab/pair_mix_{S}_d598_w21_{S}.npz")
dref = max(float(np.abs(M["W21"][k][1] - z[f"mx_{k[0]}_{k[1]}"]).max()) for k in M["W21"])
assert all(np.array_equal(M["W21"][k][0], z[f"ix_{k[0]}_{k[1]}"]) for k in M["W21"]) and dref <= 1e-12, dref
print(f"[d600] {S}: W21 lab mix == D598's W21 file (max |d| {dref:.1e}); level {g['level'](M['W21'])} ({time.time() - T0:.0f} s)", flush=True)


def cnn_parts(r1, h_on):
    """the CNN family parts under R1 = r1 and D539b on / off (the FM part is never taken from here)."""
    r0 = g["R1"]
    g["R1"] = r1
    configure(ON if h_on else [x for x in ON if x != "D539b"])
    try:
        return {(f, d): {k: v for k, v in parts_of(f, d, W18).items() if k != "fm"} for f in range(10) for d in DRAWS}
    finally:
        g["R1"] = r0
        configure(ON)


chk = cnn_parts(g["R1"], True)                           # the identity: the recomputed CNN parts == W21's (only fm differs in fm_feedback's P)
dchk = max(float(np.abs(chk[k][fam] - P21[k][fam]).max()) for k in chk for fam in chk[k])
assert set(chk[(0, DRAWS[0])]) == {"fbc", "sh", "ifn"} and dchk == 0.0, dchk
noP1, noH, noboth = cnn_parts(0.0, True), cnn_parts(g["R1"], False), cnn_parts(0.0, False)
base_fm = {}
for f in range(10):                                      # = fm_feedback's unadapted FM part (the members' trained heads, side heads included)
    mem = []
    for p in Ctx.fm_paths[f]:
        dm = Ctx.dumps(p)
        lg0 = dm["feat"] @ dm["Wm"].T + dm["bm"] + (dm["z"] @ dm["Ws"].T + dm["bs"] if "Ws" in dm else 0.0)
        mem.append(Ctx.softmax(lg0))
    base_fm[f] = np.mean(mem, 0)


def swap(cnn=None, fm=False):
    return {k: dict(v, **(cnn[k] if cnn else {}), **({"fm": base_fm[k[0]]} if fm else {})) for k, v in P21.items()}


for nm, P_ in (("noP1", swap(noP1)), ("noFM", swap(fm=True)), ("noH", swap(noH)), ("none", swap(noboth, fm=True))):
    M[nm] = lab_mix(P_, W21)
    print(f"[d600] {S}: {nm} level {g['level'](M[nm])} ({time.time() - T0:.0f} s)", flush=True)
def raw_mix(P, Wt):
    """= lab_mix without any decode: (1 - W_DC) sum_k W_k P_k + W_DC dc, the DC bag row-normalised as lab_mix feeds it to its decoder."""
    out = {}
    for f in range(10):
        ix = IX[f]
        te, p = g["DC"][f]
        pos = {int(r): k for k, r in enumerate(te)}
        q = p[np.array([pos[int(r)] for r in ix])]
        q = q / q.sum(1, keepdims=True)
        for d in DRAWS:
            out[(f, d)] = (ix, (1 - g["W_DC"]) * sum(Wt[fam] * P[(f, d)][fam] for fam in Wt) + g["W_DC"] * q)
    return out


M["raw"] = raw_mix(P21, W21)
print(f"[d600] {S}: raw level {g['level'](M['raw'])} ({time.time() - T0:.0f} s)", flush=True)
VARS = ("noP1", "noFM", "noH", "none", "raw")
for nm in VARS:                                          # the switch must change something, and only after the first refit for noP1 / noFM's refits
    dd = max(float(np.abs(M[nm][k][1] - M["W21"][k][1]).max()) for k in M[nm])
    assert dd > 1e-6, (nm, dd)
pre = {nm: max(float(np.abs(M[nm][k][1] - M["W21"][k][1])[ORD[M[nm][k][0]] < 60].max()) for k in M[nm]) for nm in M}
print(f"[d600] {S}: max |variant - W21| on windows < 60 (pre-refit; the decode starts / FM proto may differ): {pre}", flush=True)

IV = ("0-59", "60-99", "100-149", "150+")
RUN = g["idx"]["run"].to_numpy().astype(int)
RT = np.where(g["idx"]["task"].astype(str).str.startswith("Acquisition").to_numpy(), 0, 1)
assert set(RUN[RT == 0]) <= {1, 2} and set(RUN[RT == 1]) <= {3, 4, 5, 6}, "run type != runs 1-2 / 3-6"
KEYS = {"pos": (IV, lambda ix: np.digitize(ORD[ix], [60, 100, 150])), "runtype": (("Acq", "Onl"), lambda ix: RT[ix]),
        "run": (("r1", "r2", "r3", "r4", "r5", "r6"), lambda ix: RUN[ix] - 1)}


def prof(Mx, key):
    labs, kf = KEYS[key]
    out = defaultdict(list)
    for (f, d), (ix, mx) in Mx.items():
        pred, iv = mx.argmax(1), kf(ix)
        st = U[ix] == TRU[f]
        ret = ~st & np.isin(SES[ix], ("2", "3"))
        grp = {"stranger": st, "returning": ret}
        for u in np.unique(U[ix][ret]):
            for s_ in ("2", "3"):
                grp[f"{u}_s{s_}"] = ret & (U[ix] == u) & (SES[ix] == s_)
        for gn, gm in grp.items():
            for k in range(len(labs)):
                m = gm & (iv == k)
                if not m.any():
                    continue
                out[(gn, labs[k], "acc")].append(float((pred[m] == Y[ix][m]).mean()))
                c = cells(pred[m], ix[m])
                if c:
                    out[(gn, labs[k], "cba")].append(float(np.mean(list(c.values()))))
    return {k: (round(float(np.mean(v)), 4), len(v)) for k, v in out.items()}


res = {"set": S, "draws": list(DRAWS), "check_vs_d598": dref, "pre_refit_max_diff": pre, "levels": {nm: g["level"](Mx) for nm, Mx in M.items()}}
for key in KEYS:
    R = {nm: prof(Mx, key) for nm, Mx in M.items()}
    res[f"profile_{key}"] = {nm: {"|".join(k): v[0] for k, v in R[nm].items()} for nm in R}
    res[f"n_units_{key}"] = {"|".join(k): v[1] for k, v in R["W21"].items()}
    labs = KEYS[key][0]
    groups = ["stranger", "returning"] + sorted({k[0] for k in R["W21"]} - {"stranger", "returning"})
    for met in ("cba", "acc"):
        print(f"\n[d600] {S} {met} by {key} (columns {labs}); W21 level, then W21 - variant", flush=True)
        for gn in groups:
            row = [R["W21"].get((gn, iv, met), (np.nan, 0))[0] for iv in labs]
            print(f"  {gn:10s} W21   " + "  ".join(f"{x:.4f}" for x in row), flush=True)
            for nm in VARS:
                dl = [R["W21"].get((gn, iv, met), (np.nan,))[0] - R[nm].get((gn, iv, met), (np.nan,))[0] for iv in labs]
                print(f"  {gn:10s} -{nm:5s}" + "  ".join(f"{x:+.4f}" for x in dl), flush=True)
json.dump(res, open(f"{O}/t2_d600_{S}.json", "w"), indent=1, default=str)
print(f"\n[d600] {S} done ({time.time() - T0:.0f} s): outputs/t2_d600_{S}.json", flush=True)

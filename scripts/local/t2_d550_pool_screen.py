#!/usr/bin/env python
"""T2-D550 (registered before any number): the A45 POOL SCREEN. Every lc9 member output on disk vs the W19 stack's WRONG stranger windows on
D536's draws 816-818. The stack reader's head (scripts/local/t2_d540a_stack_read.py, --draws 816) is exec'd unchanged; the W19 stack = its
read(["D539b", "D543", "D546b"]); every file the stack reads is recorded (load / side_prob wrapped) and excluded from the pool.
Per candidate (ten strangers' windows, pooled over the 3 draws): accuracy on the stack-wrong windows (the ranking key); P(stack wrong | member
wrong); fixes = member right where its family's stack part is wrong (fm -> the FM slot; fbcnet -> fbc; shallow -> sh; ifnet -> ifn; others ->
the W-weighted CNN pool; raw parts, argmax), breaks = the reverse, and the share of the fixes on stack-wrong windows per fold half; per stranger
and per class (word first). FLAG iff the fix share on stack-wrong windows > 0.5 on both halves AND the cross-fitted mix estimate (the member as
its own family at w in {0.05, 0.10, 0.20}, the stack's families x (1 - w), the stack's own per-family causal Sinkhorn; w picked on one half's
strangers, read on the other) has no stranger < -0.020; the mix estimate runs for at most the top 20 passing the fix-share condition.
    mem-guard env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/local/t2_d550_pool_screen.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
sys.argv = [sys.argv[0], "--draws", "816", "--items", "D539b,D543,D546b"]
SRC = open("scripts/local/t2_d540a_stack_read.py").read()
CUT = "\nBASE = read([])\n"
assert SRC.count(CUT) == 1, "the stack reader's tail moved"
exec(compile(SRC.split(CUT)[0], "t2_d540a_stack_read_head(d550)", "exec"))
assert DSET == 816 and DRAWS == (816, 817, 818)
ON = ["D539b", "D543", "D546b"]
from t2_oracle_guard import oracle_reason  # noqa: E402
CLS = {2: "word", 0: "MI", 1: "sub"}
USED = {str(Path(DCF).resolve())}
_load0, _sp0 = load, side_prob


def load(path):
    USED.add(str(Path(path).resolve()))
    return _load0(path)


def side_prob(path, f, mode):
    USED.add(str(Path(path).resolve()))
    return _sp0(path, f, mode)


STACK = read(ON)
lv = level(STACK)
configure(ON)
W19W = dict(W)
PARTS = {(f, d): parts_of(f, d, W18) for f in range(10) for d in DRAWS}
configure([])
print(f"D550: W19 stack at 816-818 level {lv}; {len(USED)} files read by the stack ({time.time() - T0:.0f} s)", flush=True)
SM = {f: U[IX[f]] == TRU[f] for f in range(10)}
YS = {f: Y[IX[f]][SM[f]] for f in range(10)}
SP = {k: v[1][SM[k[0]]].argmax(1) for k, v in STACK.items()}
cw = {k: W19W[k] for k in ("fbc", "sh", "ifn")}
REFP = {}
for fam in ("fm", "fbc", "sh", "ifn", "other"):
    for (f, d), P in PARTS.items():
        q = P[fam] if fam != "other" else sum(cw[k] * P[k] for k in cw) / sum(cw.values())
        REFP[(fam, f, d)] = np.asarray(q)[SM[f]].argmax(1)


def family(name):
    n = name.lower()
    return "fm" if "fm-" in n else "fbc" if "fbcnet" in n else "sh" if "shallow" in n else "ifn" if "ifnet" in n else "other"


# ---- the pool
O_ = Path.home() / "workspace/outputs/neural_interface_challenge"
f0s = sorted(set(str(p) for p in O_.glob("**/lc9_0__*.npz") if "/t3" not in str(p)))
cands, skipped = [], {"fold_missing": 0, "in_stack": 0, "oracle": 0, "no_prob": 0, "rows": 0}
for p0 in f0s:
    paths = [p0.replace("/lc9_0__", f"/lc9_{f}__") for f in range(10)]
    if not all(Path(p).exists() for p in paths):
        skipped["fold_missing"] += 1
        continue
    if any(str(Path(p).resolve()) in USED for p in paths):
        skipped["in_stack"] += 1
        continue
    if any(oracle_reason(p) for p in paths):           # T2-D550: a TRUE-label in-stream copy (t2p_bench sel 'true', code ''): an oracle, never a member
        skipped["oracle"] += 1
        continue
    cands.append(paths)
print(f"D550: {len(f0s)} fold-0 files; {len(cands)} candidates; skipped {skipped}", flush=True)


def probs_of(paths):
    out = {}
    for f, p in enumerate(paths):
        z = np.load(p)
        if oracle_reason(p, z):                           # T2-D550: the manifest too
            return "oracle"
        if "test_idx" not in z.files or "prob_last" not in z.files:
            return "no_prob"
        try:
            q = realign(z["test_idx"].astype(np.int64), z["prob_last"].astype(np.float64), IX[f])
        except Exception:
            return "rows"
        if q.shape != (len(IX[f]), 3) or not np.isfinite(q).all():
            return "rows"
        out[f] = q
    return out


rows = []
for paths in cands:
    pr = probs_of(paths)
    if isinstance(pr, str):
        skipped[pr] += 1
        continue
    fam = family(Path(paths[0]).name)
    agg = {h: dict(fix=0, fix_sw=0) for h in (0, 1)}
    tot = dict(n=0, sw=0, r_sw=0, mw=0, mw_sw=0, fix=0, fix_sw=0, brk=0, right=0)
    pu, pc = {}, {k: [0, 0] for k in CLS}
    for f in range(10):
        pm = pr[f][SM[f]].argmax(1)
        y = YS[f]
        mw = pm != y
        for d in DRAWS:
            sw = SP[(f, d)] != y
            rw = REFP[(fam, f, d)] != y
            fix, brk = ~mw & rw, mw & ~rw
            tot["n"] += len(y); tot["sw"] += int(sw.sum()); tot["r_sw"] += int((~mw & sw).sum()); tot["mw"] += int(mw.sum())
            tot["mw_sw"] += int((mw & sw).sum()); tot["fix"] += int(fix.sum()); tot["fix_sw"] += int((fix & sw).sum()); tot["brk"] += int(brk.sum())
            tot["right"] += int((~mw).sum())
            h = 0 if f < 5 else 1
            agg[h]["fix"] += int(fix.sum()); agg[h]["fix_sw"] += int((fix & sw).sum())
            u = TRU[f]
            a_ = pu.setdefault(u, [0, 0]); a_[0] += int((~mw & sw).sum()); a_[1] += int(sw.sum())
            for k in CLS:
                pc[k][0] += int((~mw & sw & (y == k)).sum()); pc[k][1] += int((sw & (y == k)).sum())
    fs = [agg[h]["fix_sw"] / max(agg[h]["fix"], 1) for h in (0, 1)]
    rows.append(dict(name=str(Path(paths[0]).relative_to(O_)).replace("lc9_0__", "lc9_{f}__"), family=fam,
                     acc=round(tot["right"] / tot["n"], 4), acc_on_stack_wrong=round(tot["r_sw"] / tot["sw"], 4),
                     p_stack_wrong_given_wrong=round(tot["mw_sw"] / max(tot["mw"], 1), 4),
                     fix_share_stack_wrong=round(tot["fix_sw"] / max(tot["fix"], 1), 4), fix_share_h0=round(fs[0], 4), fix_share_h1=round(fs[1], 4),
                     fixes=tot["fix"], breaks=tot["brk"], fixshare_both=bool(fs[0] > 0.5 and fs[1] > 0.5),
                     on_stack_wrong_by_class={CLS[k]: round(v[0] / max(v[1], 1), 4) for k, v in pc.items()},
                     on_stack_wrong_by_stranger={u: round(v[0] / max(v[1], 1), 4) for u, v in sorted(pu.items())}, _paths=paths))
R = pd.DataFrame(rows).sort_values("acc_on_stack_wrong", ascending=False)
print(f"D550: screened {len(R)} candidates ({time.time() - T0:.0f} s); skipped {skipped}; pass the fix-share condition: {int(R.fixshare_both.sum())}", flush=True)
print(R[["name", "family", "acc", "acc_on_stack_wrong", "p_stack_wrong_given_wrong", "fix_share_h0", "fix_share_h1", "fixes", "breaks"]].head(25).to_string(), flush=True)


# ---- the cross-fitted mix estimate (top 20 passing the fix-share condition)
def mix_with(pr, w):
    global W, parts_of
    configure(ON)
    W0 = dict(W)
    W = {k: v * (1.0 - w) for k, v in W0.items()}
    W["x"] = w
    po0 = parts_of

    def po(f, d, parts, po0=po0):
        P = dict(po0(f, d, parts))
        P["x"] = pr[f]
        return P
    parts_of = po
    try:
        return mix(W18)
    finally:
        parts_of = po0
        configure([])


flags, mixres = [], {}
top = R[R.fixshare_both].head(20)
for _, r in top.iterrows():
    pr = probs_of(r["_paths"])
    per_w = {}
    for w in (0.05, 0.10, 0.20):
        mx = mix_with(pr, w)
        per_w[w] = per_stranger(mx, STACK)
    h0 = [TRU[f] for f in range(5)]
    h1 = [TRU[f] for f in range(5, 10)]
    w0 = max(per_w, key=lambda w: np.mean([per_w[w].get(u, 0.0) for u in h0]))     # picked on half 0, read on half 1
    w1 = max(per_w, key=lambda w: np.mean([per_w[w].get(u, 0.0) for u in h1]))
    read = {**{u: per_w[w1][u] for u in h0}, **{u: per_w[w0][u] for u in h1}}
    cf = dict(w_pick_h0=w0, w_pick_h1=w1, read_per_stranger=read, read_ten=round(float(np.mean(list(read.values()))), 4),
              read_min=round(float(min(read.values())), 4), per_w={str(w): v for w, v in per_w.items()})
    flag = bool(cf["read_min"] >= -0.020)
    mixres[r["name"]] = dict(cf, flag=flag)
    print(f"D550 mix {r['name']}: picks {w0} / {w1}; cross-fitted ten {cf['read_ten']:+.4f}, min stranger {cf['read_min']:+.4f} -> "
          f"{'FLAG' if flag else 'no flag'}", flush=True)
    if flag:
        flags.append(r["name"])
out = dict(stack_level=lv, n_files_in_stack=len(USED), candidates=len(R), skipped=skipped, n_fixshare_both=int(R.fixshare_both.sum()),
           ranked=[{k: v for k, v in r.items() if k != "_paths"} for r in R.to_dict("records")], mix=mixres, flags=flags)
json.dump(out, open(O_ / "t2_d550_pool_screen_run2.json", "w"), indent=1, default=str)
print(f"D550: FLAGS {flags if flags else 'none'}; wrote {O_ / 't2_d550_pool_screen_run2.json'} ({time.time() - T0:.0f} s)", flush=True)

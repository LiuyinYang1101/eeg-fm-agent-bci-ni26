#!/usr/bin/env python
"""T2-D614 STEP 1, the cached screen (registered b76c164f before any number; decided on SET A): is the deployable ROUTED IFNet copy a better
start than the pooled member / the current student on the windows after the first refit? Set A's caches:
  U   = the P1 bench's pooled twin (p1_ifn ...vris0): the student's current start;
  A   = the deployed P1 student (p1_ifn ...vrisSpr);
  C_k = the routed per-user copy: cnn_ifn ...vrtv8cpu's prob_users_test[k_t], k_t = t2p_fpcalib's own causal fingerprint route
        (fit_space on the fold's training rows, stream_route_every(every 1), the shipped fp_proteus.npy);
  R   = the routed part (route_ifn ...vrtv8cfb2) — ASSERTED == blend_route(shared, copies, k, r) <= 1e-6 (r recovered from {1/3, 1/2}).
Per group (strangers = users outside the fold's training users; returning = the training users' later sessions; P04 / P10), on windows of
order >= 60: the accuracy of U, A, C_k, R and the C-vs-A fixes / breaks. Gate: returning acc(C_k) - acc(A) >= +0.02 AND strangers
acc(C_k) - acc(U) >= -0.02 -> CONTINUE (a report to the manager; no code before their call); else CLOSED.
    mem-guard env OMP_NUM_THREADS=2 python scripts/local/t2_d614_diag.py --set A   ->  outputs/t2_d614_diag_<set>.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
sys.path.insert(0, str(ROOT / "src"))
import t2p_bench as B  # noqa: E402
import t2p_fpcalib as FP  # noqa: E402

S = sys.argv[sys.argv.index("--set") + 1]
D0 = {"A": 825, "B": 828, "C": 831, "D": 834}[S]
assert S == "A" or "--descriptive" in sys.argv, "T2-D614: the gate is set A; B-D only with --descriptive after the decision"
O = Path.home() / "workspace/outputs/neural_interface_challenge"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
TAG = "w20d557" if D0 in (831, 834) else "w20"
LP = O / f"t2_d571_lp{TAG}_{D0}"
CNN, RTE, P1 = (LP / "cnn_ifn/runs").resolve(), (LP / "route_ifn").resolve(), (LP / "p1_ifn/runs").resolve()
X, idx, meta = B.load_export(str(E))
del X
subj, ses, order, y = idx["subject"].to_numpy(), idx["session"].astype(str).to_numpy(), idx["order"].to_numpy(), idx["y"].to_numpy()
Fp = np.load(O / "t2_d124b/fp_proteus.npy")
STEM = "ifnet__ea__n00__r1_pulx_vp2_e180_vr"
CT = defaultdict(lambda: defaultdict(int))
CHK = dict(route_max=0.0, r=set(), units=0)


def get(path, te):
    z = np.load(path)
    pos = {int(r): j for j, r in enumerate(z["test_idx"])}
    return z, np.asarray(z["prob_last"], np.float64)[np.array([pos[int(r)] for r in te])]


for f in range(10):
    proto = f"lc9_{f}"
    (_, trm, _, tem), = B.folds(idx, proto)
    tr, te = np.flatnonzero(trm), np.flatnonzero(tem)
    proj, mu, users = FP.fit_space(Fp[tr], subj[tr])
    k = np.zeros(len(te), int)
    pos = {int(r): j for j, r in enumerate(te)}
    for g_ in B.stream_groups(idx, te):
        k[[pos[int(r)] for r in g_]] = FP.stream_route_every(proj(Fp[g_]), mu, 1)
    for d in (D0, D0 + 1, D0 + 2):
        pu = np.load(CNN / f"{proto}__{STEM}tv8cpu__s{d}.npz")
        assert list(pu["users"]) == list(users) and np.array_equal(pu["test_idx"], te), (f, d)
        sh, Pu = pu["prob_last"].astype(np.float64), pu["prob_users_test"].astype(np.float64)
        _, Rf = get(RTE / f"{proto}__{STEM}tv8cfb2__s{d}.npz", te)
        dev = {r: float(np.abs(FP.blend_route(sh, Pu, k, r) - Rf).max()) for r in (1 / 3, 0.5)}
        r_ = min(dev, key=dev.get)
        CHK["route_max"] = max(CHK["route_max"], dev[r_])
        CHK["r"].add(round(r_, 4))
        Ck = Pu[np.where(k >= 0, k, 0), np.arange(len(te))]
        _, A = get(P1 / f"{proto}__{STEM}isSpr__s{d}.npz", te)
        _, U0 = get(P1 / f"{proto}__{STEM}is0__s{d}.npz", te)
        yt, late = y[te], order[te] >= 60
        stranger = ~np.isin(subj[te], users)
        returning = ~stranger & np.isin(ses[te], ("2", "3"))
        groups = {"stranger": stranger, "returning": returning}
        for u in ("P04", "P10"):
            groups[f"returning_{u}"] = returning & np.char.endswith(subj[te].astype(str), u)
        for gn, gm in groups.items():
            m = gm & late & (k >= 0)
            if not m.any():
                continue
            C_ = CT[gn]
            C_["n"] += int(m.sum())
            for nm, Pm in (("U", U0), ("A", A), ("C", Ck), ("R", Rf)):
                C_[f"ok_{nm}"] += int((Pm.argmax(1) == yt)[m].sum())
            ca, cc = A.argmax(1) == yt, Ck.argmax(1) == yt
            C_["fix"] += int((cc & ~ca & m).sum())
            C_["break"] += int((~cc & ca & m).sum())
            C_["route_own"] += int((np.array(users)[np.where(k >= 0, k, 0)] == subj[te])[m].sum())
        CHK["units"] += 1
CHK["r"] = sorted(CHK["r"])
print(f"[d614] {S}: checks {CHK}", flush=True)
assert CHK["route_max"] <= 1e-6, ("the recomputed route != the route file", CHK)
res = {"set": S, "checks": CHK, "rates": {}}
for gn, v in CT.items():
    acc = {nm: round(v[f"ok_{nm}"] / v["n"], 4) for nm in ("U", "A", "C", "R")}
    res["rates"][gn] = dict(n=v["n"], **{f"acc_{nm}": a for nm, a in acc.items()}, C_minus_A=round(acc["C"] - acc["A"], 4),
                            C_minus_U=round(acc["C"] - acc["U"], 4), fix=v["fix"], brk=v["break"], route_own_share=round(v["route_own"] / v["n"], 4))
    print(f"[d614] {S} {gn}: {res['rates'][gn]}", flush=True)
if S == "A":
    rr, ss = res["rates"]["returning"], res["rates"]["stranger"]
    res["gate"] = dict(returning_C_minus_A=rr["C_minus_A"], stranger_C_minus_U=ss["C_minus_U"],
                       CONTINUE=bool(rr["C_minus_A"] >= 0.02 and ss["C_minus_U"] >= -0.02))
    print(f"[d614] GATE {res['gate']}", flush=True)
json.dump(res, open(O / f"t2_d614_diag_{S}.json", "w"), indent=1)

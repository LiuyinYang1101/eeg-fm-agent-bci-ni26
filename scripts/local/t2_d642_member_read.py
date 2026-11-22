#!/usr/bin/env python
"""T2-D642 (W25) MEMBER SCREEN reader (registered + amended in docs/DECISIONS_T2.md before any number): per FM slot (W21's fm_0..5 = D578's
lab lines, seeds 900 + slot), arms B (unweighted), C (--user-balance, the same seed: PAIRED), T (B at seed + 50: the seed-noise floor), on
ss8 (the 8 single-session strangers) + rt1 / rt2 (P04 / P10), from the fetched sofia job dirs outputs/sofia_s1/ni26-t2p-d642-<slot>_<arm>-<id>.
  member alone = the member's prob_last (the cont slots fm_2 / fm_4: their continuation twin 'cb') -> causal Sinkhorn per stream (10 / 3 /
  0.95, uniform start) -> cell BA: strangers on u's windows (ss8_u), rt1 on X's sessions 2-3, rt2 on X's session 3.
  Per slot: dS = mean_u (C - B), n_ge0, worst; d(rt1) / d(rt2) = mean over X of C - B. Floors (AMENDED, Track3): ONE shared floor per
  measure (S / rt1 / rt2) = the RMS over the six slots of mean(T - B); a delta smaller than its floor reads as ZERO, but a cost below -0.010 is
  never zeroed (only toward the bar). PASS (per slot) = dS >= +0.008 with >= 6 / 8 >= 0 and the worst >= -0.02, AND d(rt1) >= -0.010
  AND d(rt2) >= -0.010. The MI slot fm_0 is reported separately with the same bars.
  NOMINATION (the set): >= 4 of 6 slots with raw dS > 0 AND the pooled six-slot dS (the mean over slots) >= +0.008; a slot ENTERS only if it
  passes its own bars. REFUSES a partial read (every slot x arm x fold x file present).
  AMENDED x4 (the pre-written CONDITIONAL amendment (B)): if outputs/t2_d642/AMENDMENT_B_TRIGGERED exists (the trigger, its time and the
  cancelled T arrays, written when triggered), the T arm is read for s0-s2 only and the floors = the RMS over those THREE slots; else six.
  AMENDED x7 (W26, the selection-free replicate): --replicate (after the C' arrays, ni26-t2p-d642a-<slot>_Cp) reports per slot the member-alone
  C' - T (paired at T's seeds 950 + slot; DESCRIPTIVE: slots are nominated on C - B only) -> members_replicate.json, whose raw rt2 deltas over
  the entering slots enter the combined set read's d(ii).
  python scripts/local/t2_d642_member_read.py [--replicate]  (threads capped by the caller) -> outputs/t2_d642/RESULT_D642_members.md + members.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402
from t2p_report import causal_sinkhorn  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
OUT = O / "t2_d642"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
SLOTS = ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]
CONT = {"s2_plaincont", "s4_carcont"}
AMB = (Path.home() / "workspace/outputs/neural_interface_challenge/t2_d642/AMENDMENT_B_TRIGGERED").exists()   # T2-D642 amendment (B)
T_SLOTS = SLOTS[:3] if AMB else SLOTS                  # the slots whose T arm is read (the floors' basis)
STRANGERS = ("P01", "P02", "P03", "P05", "P06", "P07", "P08", "P09")
FOLDS = [f"ss8_{u}" for u in STRANGERS] + ["rt1_P04", "rt1_P10", "rt2_P04", "rt2_P10"]


def job_dir(slot, arm):
    """the read ran as one Slurm ARRAY per (slot, arm) = 6 task dirs ni26-t2p-d642a-<slot>_<arm>-<task job id> (the 8-GPU jobs were cancelled)."""
    pre = 'd644' if arm in ('E', 'U') else ('d645' if arm == 'R' else ('d646' if arm in ('X', 'Y') else ('d647' if arm == 'H' else ('d648' if arm in ('V', 'W') else 'd642a'))))   # H = D647
    ds = sorted(S.glob(f"ni26-t2p-{pre}-{slot}_{arm}-*"))
    assert len(ds) == 6, f"REFUSED: {slot} {arm}: {len(ds)} fetched array task dirs (need 6)"
    return ds


def member_file(ds, slot, arm, fold, seed):
    fs = [f for d in ds for f in (d / "runs").glob(f"{fold}__fm-*__s{seed}.npz") if ("cb__" in f.name) == (slot in CONT) and ("__ubw" in f.name) == (arm in ("C", "Cp", "U", "Y", "H", "W"))   # Cp = W26's C'; U = C + chemb; Y = D646's C; H = D647's half dose
          and ("__ubwp05" in f.name) == (arm == "H")
          and ("ce0.0001" in f.name) == (arm in ("E", "U")) and ("dro0.003" in f.name) == (arm == "R")]
    assert len(fs) == 1, f"REFUSED: {slot} {arm} {fold} s{seed}: {[f.name for f in fs]}"
    return fs[0]


def main():
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        seed = 900 + SLOTS.index(slot) + (50 if arm in ("T", "Cp") else 0)
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, seed))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]

    res = {}
    for slot in SLOTS:
        L = {arm: dict(S={u: level(slot, arm, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, arm, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, arm, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")})
             for arm in (("B", "C", "T") if slot in T_SLOTS else ("B", "C"))}

        def d(a, b, k):
            return {q: L[a][k][q] - L[b][k][q] for q in L[a][k]}
        dS, r1, r2 = d("C", "B", "S"), d("C", "B", "r1"), d("C", "B", "r2")
        tb = {k: float(np.mean(list(d("T", "B", k).values()))) for k in ("S", "r1", "r2")} if "T" in L else None   # this slot's T - B (one floor draw)
        raw = dict(dS=float(np.mean(list(dS.values()))), n_ge0=int(sum(v >= 0 for v in dS.values())), worst=float(min(dS.values())),
                   d_rt1=float(np.mean(list(r1.values()))), d_rt2=float(np.mean(list(r2.values()))))
        res[slot] = dict(levels={a: {k: float(np.mean(list(v.values()))) for k, v in L[a].items()} for a in L}, raw=raw, t_minus_b=tb,
                         per_user=dS, per_X_rt1=r1, per_X_rt2=r2)
    # the floors (Track3's LOW, amended before any number): ONE shared floor per measure = the RMS over the six slots of mean(T - B)
    FL = {k: float(np.sqrt(np.mean([res[s_]["t_minus_b"][k] ** 2 for s_ in T_SLOTS]))) for k in ("S", "r1", "r2")}   # amendment (B): s0-s2 only

    def z_(x, f, cost=False):
        """sub-floor = zero, but only TOWARD the bar: a cost below -0.010 is never zeroed (never across the bar)."""
        if abs(x) < f and not (cost and x < -0.010):
            return 0.0
        return x
    for slot in SLOTS:
        raw = res[slot]["raw"]
        zS, z1, z2 = z_(raw["dS"], FL["S"]), z_(raw["d_rt1"], FL["r1"], cost=True), z_(raw["d_rt2"], FL["r2"], cost=True)
        checks = dict(dS=zS >= 0.008, n6=raw["n_ge0"] >= 6, worst=raw["worst"] >= -0.02, rt1=z1 >= -0.010, rt2=z2 >= -0.010)
        res[slot].update(floors=FL, checks=checks, passes=bool(all(checks.values())))
    n_pos = sum(res[s]["raw"]["dS"] > 0 for s in SLOTS)
    pooled = float(np.mean([res[s]["raw"]["dS"] for s in SLOTS]))
    nominated = bool(n_pos >= 4 and pooled >= 0.008)
    entering = [s for s in SLOTS if nominated and res[s]["passes"]]
    out = dict(slots=res, sign=dict(n_pos=n_pos, pooled_dS=pooled), nominated=nominated, entering=entering)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.joinpath("members.json").write_text(json.dumps(out, indent=1, default=float))
    L_ = ["# T2-D642 RESULT — the member screen (user-balanced FM members, ship composition ss8 + rt; member alone)", "",
          f"Registered + amended before any number. C − B paired (same seed); floors = the RMS over {'the THREE slots s0–s2 (amendment (B) TRIGGERED)' if AMB else 'the six slots'} of mean(T − B) per measure (T = B "
          f"at seed + 50): S {FL['S']:.4f}, rt1 {FL['r1']:.4f}, rt2 {FL['r2']:.4f}; sub-floor deltas read as zero, never across the −0.010 cost bar.", "",
          f"**Set: {n_pos} / 6 slots ΔS > 0; pooled six-slot ΔS {pooled:+.4f} (bar +0.008) → {'NOMINATED' if nominated else 'NOT nominated'}; entering: {entering or 'none'}**", ""]
    for s in SLOTS:
        r = res[s]
        L_.append(f"- **{s}**{' (MI, separate)' if s == 's0_mi' else ''}: ΔS {r['raw']['dS']:+.4f} ({r['raw']['n_ge0']} / 8, worst {r['raw']['worst']:+.4f}); "
                  f"rt1 {r['raw']['d_rt1']:+.4f}, rt2 {r['raw']['d_rt2']:+.4f}; floors S {r['floors']['S']:.4f} / rt1 {r['floors']['r1']:.4f} / rt2 {r['floors']['r2']:.4f}; "
                  f"levels B {r['levels']['B']['S']:.4f} / C {r['levels']['C']['S']:.4f}{' / T ' + format(r['levels']['T']['S'], '.4f') if 'T' in r['levels'] else ''} → {'PASS' if r['passes'] else 'fail'} {r['checks']}")
    OUT.joinpath("RESULT_D642_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def replicate():
    """W26: the member-alone C' - T per slot (descriptive; refuses a partial replicate)."""
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, 950 + SLOTS.index(slot)))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]
    m = json.loads(OUT.joinpath("members.json").read_text())       # the C - B read (nomination) must exist first
    res = {}
    for slot in SLOTS:
        L = {arm: dict(S={u: level(slot, arm, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, arm, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, arm, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")}) for arm in ("T", "Cp")}
        d = {k: {q: L["Cp"][k][q] - L["T"][k][q] for q in L["T"][k]} for k in ("S", "r1", "r2")}
        res[slot] = dict(raw=dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())),
                                  worst=float(min(d["S"].values())), d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values())))),
                         per_user=d["S"], entering=slot in m["entering"])
    OUT.joinpath("members_replicate.json").write_text(json.dumps(dict(slots=res, entering=m["entering"]), indent=1, default=float))
    L_ = ["# T2-D642 x7 (W26) — the member-alone REPLICATE C' − T (descriptive; nomination stays on C − B)", ""]
    L_ += [f"- {s}{' (ENTERING)' if res[s]['entering'] else ''}: ΔS {res[s]['raw']['dS']:+.4f} ({res[s]['raw']['n_ge0']} / 8, worst "
           f"{res[s]['raw']['worst']:+.4f}); rt1 {res[s]['raw']['d_rt1']:+.4f}, rt2 {res[s]['raw']['d_rt2']:+.4f}" for s in SLOTS]
    OUT.joinpath("RESULT_D642_members_replicate.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


W27_SLOTS = ["s1_stlsd", "s3_stlcarsd"]                # T2-D644: fm_1 / fm_3 (astra)


def w27(arm="E"):
    """T2-D644 (W27): the member screen E (--fm-chemb-lr 1e-4) vs D642's B (the same seeds 900 + slot: paired) on fm_1 / fm_3, member alone,
    D642's member bars with D642's registered floors (members.json; sub-floor -> zero, never across -0.010). A slot that passes ENTERS W27's
    full-chain read (tags g / e on the decision cell). -> members_w27.json + RESULT_D644_members.md
    arm "U" (--u; the combination rule's (b): C + chemb vs B, the same bars) -> members_u.json + RESULT_D644_members_U.md"""
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, 900 + SLOTS.index(slot)))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]
    FL = json.loads(OUT.joinpath("members.json").read_text())["slots"][SLOTS[0]]["floors"]   # D642's registered floors (one per measure)

    def z_(x, f, cost=False):
        return 0.0 if abs(x) < f and not (cost and x < -0.010) else x
    res = {}
    for slot in (["s1_stlsd"] if arm == "R" else W27_SLOTS):   # T2-D645's DRO arm: s1 only
        L = {a_: dict(S={u: level(slot, a_, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, a_, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, a_, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")}) for a_ in ("B", arm)}
        d = {k: {q: L[arm][k][q] - L["B"][k][q] for q in L["B"][k]} for k in ("S", "r1", "r2")}
        raw = dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())), worst=float(min(d["S"].values())),
                   d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values()))))
        zS, z1, z2 = z_(raw["dS"], FL["S"]), z_(raw["d_rt1"], FL["r1"], True), z_(raw["d_rt2"], FL["r2"], True)
        checks = dict(dS=zS >= 0.008, n6=raw["n_ge0"] >= 6, worst=raw["worst"] >= -0.02, rt1=z1 >= -0.010, rt2=z2 >= -0.010)
        res[slot] = dict(raw=raw, per_user=d["S"], floors=FL, checks=checks, passes=bool(all(checks.values())),
                         levels={a: {k: float(np.mean(list(v.values()))) for k, v in L[a].items()} for a in L})
        if arm == "R":                                    # T2-D645's registered descriptives (Track3 LOW): R - C (paired, seed 901) + the final q
            LC = dict(S={u: level(slot, "C", f"ss8_{u}", U == u) for u in STRANGERS},
                      r1={x: level(slot, "C", f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                      r2={x: level(slot, "C", f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")})
            res[slot]["R_minus_C"] = {k: float(np.mean([L["R"][k][q] - LC[k][q] for q in LC[k]])) for k in ("S", "r1", "r2")}
            qs = []
            for d_ in job_dir(slot, "R"):
                for lf in sorted(d_.glob("arm*_shard*_*.log")):
                    qs += [ln.split("T2-D645 FM-DRO final q:", 1)[1].strip() for ln in lf.read_text().splitlines() if "T2-D645 FM-DRO final q:" in ln]
            res[slot]["final_q"] = qs
    entering = [s for s in res if res[s]["passes"]]
    OUT.joinpath({"E": "members_w27.json", "U": "members_u.json", "R": "members_dro.json"}[arm]).write_text(json.dumps(dict(slots=res, entering=entering, nominated=bool(entering)), indent=1, default=float))
    L_ = [f"# {'T2-D645' if arm == 'R' else 'T2-D644'} RESULT — {dict(E='W27 member screen (E = --fm-chemb-lr 1e-4', U='the combination arm U (C + chemb', R='the FM-DRO member screen (R = --fm-dro 0.003')[arm]} vs D642's B, paired seeds; member alone; D642's bars + floors)", "",
          f"Floors (D642's): S {FL['S']:.4f}, rt1 {FL['r1']:.4f}, rt2 {FL['r2']:.4f}. **Entering: {entering or 'none'}**", ""]
    L_ += [f"- {s}: ΔS {res[s]['raw']['dS']:+.4f} ({res[s]['raw']['n_ge0']} / 8, worst {res[s]['raw']['worst']:+.4f}); rt1 {res[s]['raw']['d_rt1']:+.4f}, "
           f"rt2 {res[s]['raw']['d_rt2']:+.4f} → {'PASS' if res[s]['passes'] else 'fail'} {res[s]['checks']}" for s in res]
    for s in res:
        if "R_minus_C" in res[s]:
            rc = res[s]["R_minus_C"]
            L_ += [f"- {s} descriptive: R − C (the static-balance comparator, paired) ΔS {rc['S']:+.4f}, rt1 {rc['r1']:+.4f}, rt2 {rc['r2']:+.4f}; "
                   f"final q ({len(res[s]['final_q'])} fold fits): " + " | ".join(res[s]["final_q"][:12])]
    OUT.joinpath({"E": "RESULT_D644_members.md", "U": "RESULT_D644_members_U.md", "R": "RESULT_D645_members.md"}[arm]).write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d647():
    """T2-D647 (W28 = ALL6-HALF) stage 1: the member-alone H - T (H = T's lines + --user-balance-pow 0.5, the same seeds 950 + slot: paired), all six
    slots, + the descriptive C' - T (members_replicate.json, the same seeds). The PINNED KILL: stage 1 PASSES only if the pooled six-slot dS(H - T)
    >= +0.006 AND the pooled six-slot rt2(H - T) > -0.007 (0.6 x / 0.5 x of C' - T's pooled +0.0100 / -0.0145). -> members_d647.json + RESULT_D647_members.md"""
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, 950 + SLOTS.index(slot)))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]
    rep = json.loads(OUT.joinpath("members_replicate.json").read_text())["slots"]
    res = {}
    for slot in SLOTS:
        L = {arm: dict(S={u: level(slot, arm, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, arm, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, arm, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")}) for arm in ("T", "H")}
        d = {k: {q: L["H"][k][q] - L["T"][k][q] for q in L["T"][k]} for k in ("S", "r1", "r2")}
        res[slot] = dict(raw=dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())),
                                  worst=float(min(d["S"].values())), d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values())))),
                         per_user=d["S"], cprime_minus_t=rep[slot]["raw"])
    pS = float(np.mean([res[s_]["raw"]["dS"] for s_ in SLOTS]))
    p2 = float(np.mean([res[s_]["raw"]["d_rt2"] for s_ in SLOTS]))
    cS = float(np.mean([rep[s_]["raw"]["dS"] for s_ in SLOTS]))
    c2 = float(np.mean([rep[s_]["raw"]["d_rt2"] for s_ in SLOTS]))
    passes = bool(pS >= 0.006 and p2 > -0.007)
    OUT.joinpath("members_d647.json").write_text(json.dumps(dict(slots=res, pooled_dS=pS, pooled_rt2=p2, cprime_pooled_dS=cS, cprime_pooled_rt2=c2,
                                                                 stage1_passes=passes, entering=SLOTS if passes else [], nominated=passes), indent=1, default=float))
    L_ = [f"# T2-D647 (W28 = ALL6-HALF) STAGE 1 — the member-alone H − T at seeds 950 + slot (paired); C′ − T the descriptive comparator", "",
          f"**Pinned kill: pooled ΔS(H − T) {pS:+.4f} (bar ≥ +0.006) AND pooled rt2(H − T) {p2:+.4f} (bar > −0.007) → "
          f"{'STAGE 1 PASSES (the set read + replicate next)' if passes else 'KILLED (no set read, no donors)'}**; C′ − T pooled ΔS {cS:+.4f}, rt2 {c2:+.4f}", ""]
    L_ += [f"- {s_}: H − T ΔS {res[s_]['raw']['dS']:+.4f} ({res[s_]['raw']['n_ge0']} / 8, worst {res[s_]['raw']['worst']:+.4f}); rt1 {res[s_]['raw']['d_rt1']:+.4f}, "
           f"rt2 {res[s_]['raw']['d_rt2']:+.4f} | C′ − T ΔS {rep[s_]['raw']['dS']:+.4f}, rt2 {rep[s_]['raw']['d_rt2']:+.4f}" for s_ in SLOTS]
    OUT.joinpath("RESULT_D647_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d646():
    """T2-D646: the member-alone Y - X (D642's C vs B at the FRESH seeds 1000 + slot), all six slots; descriptive (its raw rt2 over the six slots
    feeds the D646 set read's d(ii)) -> members_d646.json + RESULT_D646_members.md"""
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, 1000 + SLOTS.index(slot)))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]
    res = {}
    for slot in SLOTS:
        L = {arm: dict(S={u: level(slot, arm, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, arm, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, arm, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")}) for arm in ("X", "Y")}
        d = {k: {q: L["Y"][k][q] - L["X"][k][q] for q in L["X"][k]} for k in ("S", "r1", "r2")}
        res[slot] = dict(raw=dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())),
                                  worst=float(min(d["S"].values())), d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values())))),
                         per_user=d["S"])
    pooled = float(np.mean([res[s]["raw"]["dS"] for s in SLOTS]))
    OUT.joinpath("members_d646.json").write_text(json.dumps(dict(slots=res, pooled_dS=pooled, entering=SLOTS, nominated=True), indent=1, default=float))
    L_ = [f"# T2-D646 — the member-alone Y − X (C vs B at the fresh seeds 1000 + slot; descriptive): pooled six-slot ΔS {pooled:+.4f}", ""]
    L_ += [f"- {s}: ΔS {res[s]['raw']['dS']:+.4f} ({res[s]['raw']['n_ge0']} / 8, worst {res[s]['raw']['worst']:+.4f}); rt1 {res[s]['raw']['d_rt1']:+.4f}, "
           f"rt2 {res[s]['raw']['d_rt2']:+.4f}" for s in SLOTS]
    OUT.joinpath("RESULT_D646_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d648():
    """T2-D648: the member-alone W - V (D642's C vs B at the FRESH seeds 1100 + slot), all six slots; descriptive (its raw rt2 over the six slots
    feeds the D648 set read's d(ii)) -> members_d648.json + RESULT_D648_members.md"""
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    STREAM, ORD = idx.stream.to_numpy(), idx.order.to_numpy()

    def level(slot, arm, fold, keep):
        z = np.load(member_file(job_dir(slot, arm), slot, arm, fold, 1100 + SLOTS.index(slot)))
        te, p = z["test_idx"], z["prob_last"].astype(np.float64)
        D = causal_sinkhorn(p / p.sum(1, keepdims=True), (STREAM[te], ORD[te]), iters=10, n0=3, lam=0.95)
        k = keep[te]
        assert k.any(), (slot, arm, fold)
        return cell_balanced_accuracy(Y[te][k], D[k].argmax(1), CELL[te][k])[0]
    res = {}
    for slot in SLOTS:
        L = {arm: dict(S={u: level(slot, arm, f"ss8_{u}", U == u) for u in STRANGERS},
                       r1={x: level(slot, arm, f"rt1_{x}", (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                       r2={x: level(slot, arm, f"rt2_{x}", (U == x) & (SES == "3")) for x in ("P04", "P10")}) for arm in ("V", "W")}
        d = {k: {q: L["W"][k][q] - L["V"][k][q] for q in L["V"][k]} for k in ("S", "r1", "r2")}
        res[slot] = dict(raw=dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())),
                                  worst=float(min(d["S"].values())), d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values())))),
                         per_user=d["S"])
    pooled = float(np.mean([res[s]["raw"]["dS"] for s in SLOTS]))
    OUT.joinpath("members_d648.json").write_text(json.dumps(dict(slots=res, pooled_dS=pooled, entering=SLOTS, nominated=True), indent=1, default=float))
    L_ = [f"# T2-D648 — the member-alone W − V (C vs B at the fresh seeds 1100 + slot; descriptive): pooled six-slot ΔS {pooled:+.4f}", ""]
    L_ += [f"- {s}: ΔS {res[s]['raw']['dS']:+.4f} ({res[s]['raw']['n_ge0']} / 8, worst {res[s]['raw']['worst']:+.4f}); rt1 {res[s]['raw']['d_rt1']:+.4f}, "
           f"rt2 {res[s]['raw']['d_rt2']:+.4f}" for s in SLOTS]
    OUT.joinpath("RESULT_D648_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))



def _dirs(pre, slot, arm, n):
    ds = sorted(S.glob(f"ni26-t2p-{pre}-{slot}_{arm}-*"))
    assert len(ds) == n, f"REFUSED: {pre} {slot} {arm}: {len(ds)} fetched array task dirs (need {n})"
    return ds


def _file(ds, slot, fold, seed, must, mustnot):
    """one member file per (fold, seed): the cont slots' 'cb' twin; every tag in must present, every tag in mustnot absent (the arm's identity)."""
    fs = [f for d in ds for f in (d / "runs").glob(f"{fold}__fm-*__s{seed}.npz") if ("cb__" in f.name) == (slot in CONT)
          and all(m in f.name for m in must) and not any(m in f.name for m in mustnot)]
    assert len(fs) == 1, f"REFUSED: {slot} {fold} s{seed} must {must} not {mustnot}: {[f.name for f in fs]}"
    return fs[0]


def _idx():
    idx = pd.read_csv(EXP / "index.csv", low_memory=False)
    return dict(U=idx.subject.astype(str).str[-3:].to_numpy(), SES=idx.session.astype(str).to_numpy(), Y=idx.y.to_numpy(), CELL=idx.cell.astype(str).to_numpy(),
                ST=idx.stream.to_numpy(), OD=idx.order.to_numpy(), RUN=idx.run.to_numpy())


def _level(I, f, keep):
    z = np.load(f)
    te, p = z["test_idx"], z["prob_last"].astype(np.float64)
    D = causal_sinkhorn(p / p.sum(1, keepdims=True), (I["ST"][te], I["OD"][te]), iters=10, n0=3, lam=0.95)
    k = keep[te]
    assert k.any(), f.name
    return cell_balanced_accuracy(I["Y"][te][k], D[k].argmax(1), I["CELL"][te][k])[0]


def _ship_pair(I, slots, A, B):
    """member alone, ship composition (ss8 + rt): per slot Delta = A - B on the strangers (ss8_u), rt1 (X's sessions 2-3), rt2 (X's session 3).
    A / B = dict(pre, arm, seed0, must, mustnot); the seed = seed0 + the slot index."""
    U, SES = I["U"], I["SES"]
    res = {}
    for slot in slots:
        k = SLOTS.index(slot)
        L = {}
        for nm, a in (("A", A), ("B", B)):
            ds = _dirs(a["pre"], slot, a["arm"], 6)
            fl = lambda fold: _file(ds, slot, fold, a["seed0"] + k, a["must"], a["mustnot"])   # noqa: E731
            L[nm] = dict(S={u: _level(I, fl(f"ss8_{u}"), U == u) for u in STRANGERS},
                         r1={x: _level(I, fl(f"rt1_{x}"), (U == x) & np.isin(SES, ("2", "3"))) for x in ("P04", "P10")},
                         r2={x: _level(I, fl(f"rt2_{x}"), (U == x) & (SES == "3")) for x in ("P04", "P10")})
        d = {q: {u: L["A"][q][u] - L["B"][q][u] for u in L["B"][q]} for q in ("S", "r1", "r2")}
        res[slot] = dict(dS=float(np.mean(list(d["S"].values()))), n_ge0=int(sum(v >= 0 for v in d["S"].values())), worst=float(min(d["S"].values())),
                         d_rt1=float(np.mean(list(d["r1"].values()))), d_rt2=float(np.mean(list(d["r2"].values()))), per_user=d["S"])
    return res


def d649(rep=False):
    """T2-D649 (W29) MEMBER SCREEN, member alone, G (C + --fm-grl lambda max) vs C at the FRESH seeds 1200 + slot, all six slots, ss8 + rt. G = ONLY the
    arm the smoke picked by ENGAGEMENT (outputs/t2_d649/lambda_pick.json, rule 3); the other G arms were cancelled + deleted unread (it refuses if any
    unpicked G dir is still fetched). THE PINNED KILL: KILL unless the pooled (six-slot mean) dS(G - C) >= +0.004 AND the pooled rt2(G - C) >= -0.005.
    -> outputs/t2_d649/members_d649.json + RESULT_D649_members.md
    --d649 --rep: the SAME read on the replicate members (d649r, 1300 + slot) -> members_d649rep.json, DESCRIPTIVE (its raw rt2 = read 2's d(ii));
    it needs the screen read (members_d649.json) ALIVE first."""
    OUTd = O / "t2_d649"
    pre, seed0, sfx = ("d649r", 1300, "rep") if rep else ("d649", 1200, "")
    if rep:
        assert json.loads((OUTd / "members_d649.json").read_text())["alive"], "REFUSED: the screen KILLED D649; the replicate stays unread"
    pk = json.loads((OUTd / "lambda_pick.json").read_text())
    assert pk["engaged"] and pk["lambda_max"] is not None, "REFUSED: the smoke did not engage (D371: not tested); nothing to read"
    lam = float(pk["lambda_max"])
    arm = {0.1: "G01", 0.3: "G03", 1.0: "G1"}[lam]
    left = [d.name for a_ in ("G01", "G03", "G1") if a_ != arm for d in S.glob(f"ni26-t2p-d649-s*_{a_}-*")]
    assert not left, f"REFUSED: unpicked G dirs are present locally (must be deleted unread first): {left[:3]}"
    I = _idx()
    res = _ship_pair(I, SLOTS, dict(pre=pre, arm=arm, seed0=seed0, must=["__ubw", f"grl{lam:g}"],
                              mustnot=[f"grl{lam:g}{c}" for c in "0123456789."]),   # the cb twin: ...grl0.3cb__
                     dict(pre=pre, arm="C", seed0=seed0, must=["__ubw"], mustnot=["grl"]))
    pS = float(np.mean([res[s]["dS"] for s in SLOTS])); pr2 = float(np.mean([res[s]["d_rt2"] for s in SLOTS])); pr1 = float(np.mean([res[s]["d_rt1"] for s in SLOTS]))
    # AMENDED 2026-11-20 16:31:32 (the owner, via the manager, before any number: "any small gain is plausible and should be included"): the member
    # gate = the SIGN gate pooled dS > 0 (was dS >= +0.004 AND rt2 >= -0.005; a non-positive dS kills under every version, so the kill branch stays safe)
    alive = (pS > 0) if not rep else None             # the gate is the SCREEN's (read 1) only
    OUTd.joinpath(f"members_d649{sfx}.json").write_text(json.dumps(dict(lambda_max=lam, arm=arm, slots=res, pooled_dS=pS, pooled_rt1=pr1, pooled_rt2=pr2,
                                                                   kill_rule="the sign gate pooled dS > 0 (2026-11-20 16:31:32; was dS >= +0.004 AND rt2 >= -0.005)", alive=alive), indent=1, default=float))
    L_ = [f"# T2-D649 — the member {'replicate (descriptive)' if rep else 'screen'} G (lambda max {lam:g}, arm {arm}) − C at the fresh seeds {seed0} + slot, member alone", "",
          f"**Pooled six-slot ΔS {pS:+.4f}, rt2 {pr2:+.4f} (rt1 {pr1:+.4f})"
          + (" (descriptive; read 2's d(ii))**" if rep else f" → the member SIGN gate (pooled ΔS > 0; the owner 16:31:32): {'ALIVE → the set read + replicate' if alive else 'KILLED'}**"), ""]
    L_ += [f"- {s}: ΔS {res[s]['dS']:+.4f} ({res[s]['n_ge0']} / 8, worst {res[s]['worst']:+.4f}); rt1 {res[s]['d_rt1']:+.4f}, rt2 {res[s]['d_rt2']:+.4f}" for s in SLOTS]
    OUTd.joinpath(f"RESULT_D649_members{sfx}.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d652():
    """T2-D652 (W30 = W25's C lines + vp 2) MEMBER GATE (pinned before any number; Track3's pin of the MI slot: both member gates pool the 5 changed
    slots s1-s5 only). GATE 1 first: Track3's D651 result (pooled C - B on rtl) must be >= 0, else REFUSED (W30 stops unread). Then
      (a) strangers dS(V - C') = the ship-composition member read, V = d652 at 950 + slot vs C' = d642a Cp at 950 + slot (same seeds, not run-paired);
      (b) dRL(V - C) on rtl = per slot x single-session user u: prob_last on u's last two runs -> causal Sinkhorn over those rows -> cell BA (D651's
          read form), V = d652rl at 1400 + slot vs C = D651's C (d651) at 1400 + slot, paired; pooled over 5 x 8.
    MEMBER GATE: dS(V - C') >= +0.003 AND dRL(V - C) >= -0.002 (both means over s1-s5). -> outputs/t2_d652/members_d652.json + RESULT_D652_members.md"""
    OUTd = O / "t2_d652"
    g1 = None                                            # AMENDED 16:29 (R2): gate 1 DROPPED (D651 descriptive only)
    I = _idx()
    S5 = SLOTS[1:]
    res = _ship_pair(I, S5, dict(pre="d652", arm="V", seed0=950, must=["__ubw", "vp2"], mustnot=[]),
                     dict(pre="d642a", arm="Cp", seed0=950, must=["__ubw"], mustnot=["vp2", "__ubwp05"]))
    U, RUN = I["U"], I["RUN"]
    rl = {}
    has_rl = all(len(list(S.glob(f"ni26-t2p-d652rl-{s}_V-*"))) == 4 for s in S5)   # the rtl leg is DESCRIPTIVE (16:29): optional (capacity, 17:2x)
    for slot in (S5 if has_rl else []):
        k = SLOTS.index(slot)
        dv, dc = _dirs("d652rl", slot, "V", 4), _dirs("d651", slot, "C", 4)
        d = {}
        for u in STRANGERS:
            runs = np.unique(RUN[U == u])
            keep = (U == u) & np.isin(RUN, runs[-2:])
            fv = _file(dv, slot, f"rtl_{u}", 1400 + k, ["__ubw", "vp2"], [])
            fc = _file(dc, slot, f"rtl_{u}", 1400 + k, ["__ubw"], ["vp2"])
            for f in (fv, fc):
                te = np.load(f)["test_idx"]
                assert set(U[te]) == {u} and set(RUN[te].tolist()) == set(runs[-2:].tolist()), (slot, u, f.name, "not u's last two runs")
            d[u] = _level(I, fv, keep) - _level(I, fc, keep)
        rl[slot] = dict(dRL=float(np.mean(list(d.values()))), n_ge0=int(sum(v >= 0 for v in d.values())), per_user=d)
    pS = float(np.mean([res[s]["dS"] for s in S5])); pRL = float(np.mean([v for s in S5 for v in rl[s]["per_user"].values()])) if has_rl else None
    pr2 = float(np.mean([res[s]["d_rt2"] for s in S5])); pr1 = float(np.mean([res[s]["d_rt1"] for s in S5]))
    ok = pS > 0                                          # AMENDED 16:31:32 (the owner): the member SIGN gate (was dS >= +0.003 AND dRL >= -0.002; rtl / rt1 descriptive)
    OUTd.mkdir(parents=True, exist_ok=True)
    n_rl = sum(len(rl[s]["per_user"]) for s in rl)
    assert n_rl in (0, 40), n_rl                         # Track3: W30's dRL pools the 5 changed slots x 8 users (0 = the descriptive leg skipped)
    OUTd.joinpath("members_d652.json").write_text(json.dumps(dict(gate1_d651=g1, slots=res, rtl=rl, pooled_dS=pS, pooled_dRL=pRL, n_rl_pairs=n_rl, pooled_rt1=pr1, pooled_rt2=pr2,
                                                                   gate="the sign gate pooled dS(V - C') > 0 over s1-s5 (2026-11-20 16:31:32; gate 1 dropped 16:29)", passes=ok), indent=1, default=float))
    L_ = [f"# T2-D652 (W30) — the member gate (the SIGN gate pooled ΔS > 0, 16:31:32; gate 1 dropped 16:29), V (+ vp 2) on the five headed slots, member alone", "",
          f"**ΔS(V − C′) {pS:+.4f} (> 0?) → {'PASS → the set read + replicate vs W25-ALL6' if ok else 'FAIL: W30 closed'}** (ΔRL(V − C) {'n/a (rtl cut)' if pRL is None else f'{pRL:+.4f}'}, descriptive) "
          f"(descriptive: rt1 {pr1:+.4f}, rt2 {pr2:+.4f})", ""]
    L_ += [f"- {s}: ΔS {res[s]['dS']:+.4f} ({res[s]['n_ge0']} / 8, worst {res[s]['worst']:+.4f}); "
           + (f"ΔRL {rl[s]['dRL']:+.4f} ({rl[s]['n_ge0']} / 8); " if s in rl else "")
           + f"rt1 {res[s]['d_rt1']:+.4f}, rt2 {res[s]['d_rt2']:+.4f}" for s in S5]
    OUTd.joinpath("RESULT_D652_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d649rl():
    """T2-D649 (W29): W29's OWN dRL, used ONLY by the W29 x W30 rule (never by D649's own adoption rule): G_picked on rtl at 1400 + slot (d649rl) vs
    D651's C (d651) at 1400 + slot, ALL SIX slots (s0 has a G member), D651's read form (u's last two runs -> causal Sinkhorn -> cell BA), pooled 6 x 8.
    REFUSES unless the screen is ALIVE and all 48 pairs exist. -> outputs/t2_d649/members_d649rl.json + RESULT_D649rl_members.md"""
    OUTd = O / "t2_d649"
    assert json.loads((OUTd / "members_d649.json").read_text())["alive"], "REFUSED: the screen KILLED D649; the rtl arm stays unread"
    pk = json.loads((OUTd / "lambda_pick.json").read_text())
    assert pk["engaged"] and pk["lambda_max"] is not None, "REFUSED: not engaged"
    lam = float(pk["lambda_max"]); arm = {0.1: "G01", 0.3: "G03", 1.0: "G1"}[lam]
    I = _idx(); U, RUN = I["U"], I["RUN"]
    rl = {}
    for slot in SLOTS:
        k = SLOTS.index(slot)
        dg, dc = _dirs("d649rl", slot, arm, 4), _dirs("d651", slot, "C", 4)
        d = {}
        for u in STRANGERS:
            runs = np.unique(RUN[U == u])
            keep = (U == u) & np.isin(RUN, runs[-2:])
            fg = _file(dg, slot, f"rtl_{u}", 1400 + k, ["__ubw", f"grl{lam:g}"], [f"grl{lam:g}{c}" for c in "0123456789."])
            fc = _file(dc, slot, f"rtl_{u}", 1400 + k, ["__ubw"], ["vp2", "grl"])
            for f in (fg, fc):
                te = np.load(f)["test_idx"]
                assert set(U[te]) == {u} and set(RUN[te].tolist()) == set(runs[-2:].tolist()), (slot, u, f.name, "not u's last two runs")
            d[u] = _level(I, fg, keep) - _level(I, fc, keep)
        rl[slot] = dict(dRL=float(np.mean(list(d.values()))), n_ge0=int(sum(v >= 0 for v in d.values())), per_user=d)
    allp = [v for s in SLOTS for v in rl[s]["per_user"].values()]
    assert len(allp) == 48, len(allp)                    # Track3: W29's dRL pools its 6 changed slots x 8 users
    pRL = float(np.mean(allp))
    OUTd.joinpath("members_d649rl.json").write_text(json.dumps(dict(lambda_max=lam, arm=arm, slots=rl, pooled_dRL=pRL, n_rl_pairs=len(allp)), indent=1, default=float))
    L_ = [f"# T2-D649 (W29) — its own ΔRL for the W29 × W30 rule: G (λ {lam:g}) − D651's C on rtl at 1400 + slot, six slots × 8 users: {pRL:+.4f}", ""]
    L_ += [f"- {s}: ΔRL {rl[s]['dRL']:+.4f} ({rl[s]['n_ge0']} / 8 ≥ 0)" for s in SLOTS]
    OUTd.joinpath("RESULT_D649rl_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def d652b():
    """T2-D652 (W30): the SECOND pair's member read (descriptive; the kappa denominator's read 2 + read 2's d(ii)): V = d652b at 1100 + slot vs D648's
    W (d648, = D642's C at 1100 + slot), the five headed slots, ship composition. Needs W30's member gate PASSED (members_d652.json).
    -> outputs/t2_d652/members_d652b.json + RESULT_D652b_members.md"""
    OUTd = O / "t2_d652"
    assert json.loads((OUTd / "members_d652.json").read_text())["passes"], "REFUSED: W30's member gate FAILED; the second pair stays unread"
    S5 = SLOTS[1:]
    res = _ship_pair(_idx(), S5, dict(pre="d652b", arm="V", seed0=1100, must=["__ubw", "vp2"], mustnot=[]),
                     dict(pre="d648", arm="W", seed0=1100, must=["__ubw"], mustnot=["vp2", "__ubwp05"]))
    pS = float(np.mean([res[s]["dS"] for s in S5])); pr2 = float(np.mean([res[s]["d_rt2"] for s in S5])); pr1 = float(np.mean([res[s]["d_rt1"] for s in S5]))
    OUTd.joinpath("members_d652b.json").write_text(json.dumps(dict(slots=res, pooled_dS=pS, pooled_rt1=pr1, pooled_rt2=pr2), indent=1, default=float))
    L_ = [f"# T2-D652 (W30) — the second pair V1100 − W1100, member alone over s1–s5 (descriptive): ΔS {pS:+.4f}, rt1 {pr1:+.4f}, rt2 {pr2:+.4f}", ""]
    L_ += [f"- {s}: ΔS {res[s]['dS']:+.4f} ({res[s]['n_ge0']} / 8, worst {res[s]['worst']:+.4f}); rt1 {res[s]['d_rt1']:+.4f}, rt2 {res[s]['d_rt2']:+.4f}" for s in S5]
    OUTd.joinpath("RESULT_D652b_members.md").write_text("\n".join(L_) + "\n")
    print("\n".join(L_))


def stack():
    """THE STACK READ's member read (the d(ii) of its OLD sealed): the stack arm (setread/bw21/stack_spec.json: G / V / J; s0 = C's for V, G's for J)
    vs C at 1800 + slot (the tstk fits), identity features (the views are the set read's), ship composition; d_rt2 over the CHANGED slots (V: s1-s5;
    G / J: six). -> outputs/t2_stack/members_stack.json"""
    spec = json.loads((O / "t2_d642/setread/bw21/stack_spec.json").read_text())
    arm = spec["arm"]
    RULE = {"C": (["__ubw"], ["grl", "vp2"]), "G": (["__ubw", "grl0.1"], ["vp2"] + [f"grl0.1{c}" for c in "0123456789"]),
            "V": (["__ubw", "vp2"], ["grl"]), "J": (["__ubw", "vp2", "grl0.1"], [f"grl0.1{c}" for c in "0123456789"])}
    I = _idx()
    res = {}
    for slot in SLOTS:
        a_ = ({"G": "G", "V": "C", "J": "G"}[arm] if slot == SLOTS[0] and arm != "C" else arm)
        if a_ == "C":
            continue                                     # an unchanged slot (a V-stack's s0): delta 0 by construction
        res.update(_ship_pair(I, [slot], dict(pre="stk", arm=a_, seed0=1800, must=RULE[a_][0], mustnot=RULE[a_][1]),
                              dict(pre="stk", arm="C", seed0=1800, must=RULE["C"][0], mustnot=RULE["C"][1])))
    ch = list(res)
    out = dict(arm=arm, view=spec["view"], changed=ch, slots=res, pooled_dS=float(np.mean([res[s]["dS"] for s in ch])),
               d_rt2_changed=float(np.mean([res[s]["d_rt2"] for s in ch])), d_rt1_changed=float(np.mean([res[s]["d_rt1"] for s in ch])))
    (O / "t2_stack").mkdir(exist_ok=True)
    (O / "t2_stack/members_stack.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[stack members] arm {arm}: changed slots {ch}; pooled dS {out['pooled_dS']:+.4f}, rt2 {out['d_rt2_changed']:+.4f}, rt1 {out['d_rt1_changed']:+.4f}")


if __name__ == "__main__":
    a_ = sys.argv[1:]
    if "--d646" in a_:
        d646()
        sys.exit(0)
    if "--d647" in a_:
        d647()
        sys.exit(0)
    if "--d648" in a_:
        d648()
        sys.exit(0)
    if "--d649" in a_:
        d649(rep="--rep" in a_)
        sys.exit(0)
    if "--d652b" in a_:
        d652b()
        sys.exit(0)
    if "--d649rl" in a_:
        d649rl()
        sys.exit(0)
    if "--stack" in a_:
        stack()
        sys.exit(0)
    if "--d652" in a_:
        d652()
        sys.exit(0)
    replicate() if "--replicate" in a_ else (w27() if "--w27" in a_ else (w27("U") if "--u" in a_ else (w27("R") if "--dro" in a_ else main())))

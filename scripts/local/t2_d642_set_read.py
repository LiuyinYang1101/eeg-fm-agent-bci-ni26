#!/usr/bin/env python
"""T2-D642 FULL-CHAIN SET READ, the reader (design + bar pinned in docs/DECISIONS_T2.md before any number). REFUSES unless the identity PASSED
(outputs/t2_d642/setread/identity.log) AND every tag's students are spliced ('### done d642 tag <t>' for bs cs bp04 cp04 bp10 cp10 in
setread/chain.log; no partial read).
  Mixes (W21's lab form: `t2_lab.py mix --set E --base w20 --idea w20_fb`, NI26_LAB_PRE_R=1, CKS 60,100,150) on each tag's lp root
  t2_d642_lp<tag>_<cell>_860 with NI26_LAB_FMSPLICE = its fmspl dir; tags p04 / p10 with NI26_LAB_FMSTART=freeze (the id root's records, Track3 Q2).
  Root s chronological + the decode-only reorders (NI26_LAB_ORDER runs_perm / in_run, seeds 6310-6312; PARTIAL); roots p04 / p10 chronological.
  Scored on the 8 spliced lc9 folds, each root ONLY on its own rows: s = TRU[f]'s rows (cell BA per user, mean over draws 860-862); p04 / p10 =
  that user's sessions 2-3 rows (cell BA per fold x draw, mean). C - B paired (the same student seeds 880-882).
  THE BAR (pinned): strangers dS (root s, C - B) >= +0.004 with >= 6 / 8 users >= 0 and dS >= 0 under BOTH reorders (the mean over the 3 seeds of
  each kind); returning d = the mean of the p04 (P04) / p10 (P10) deltas >= -0.003; 0.57 dS + 0.38 d(returning) + 0.06 d(ii) >= +0.002, d(ii) =
  the member screen's rt2 delta of the ENTERING slots (members.json raw d_rt2, their mean; member alone), entering the sum as min(d(ii), 0)
  (Track3 LOW 1: a member-alone value counts only as a COST in a full-stack sum; the raw value reported). Raw deltas (no floor zeroing); D638's
  noise floors (nz vs the reference root) are reported beside them.
  BOTH CELLS (the manager, before any D642 number): bw21 and bw23 each read from its own outputs/t2_d642/setread/<cell>/; the W25 DECISION uses
  the cell of the owner's final base (bw23 if USER_LINES item 166 is typed, else bw21); the other cell is descriptive.
  AMENDED x7 (W26): --read first (B vs C, tags b* / c*) | replicate (T vs C', tags t* / d*; the decision cell only; the cell's identity and FM-start
  records reused) -> setread/<cell>/set_<read>.json + RESULT_D642_set_<read>.md (each read's own bar line is DESCRIPTIVE under x7);
  --combine -> the DECISION: the registered bars on the MEAN of the two reads (per-user means first; >= 6 / 8 on them; each reorder kind on the
  mean of the two reads' means; returning the mean; d(ii) = the mean of the two member screens' raw rt2 over the entering slots, as min(., 0))
  AND the replicate's strangers dS > 0 -> setread/<cell>/set_combined.json + RESULT_D642_set_combined.md (each read printed alone beside).
  mem-guard env OMP_NUM_THREADS=4 ... python scripts/local/t2_d642_set_read.py --cell bw21|bw23 --read first|replicate [--jobs 2] | --combine
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ni26.metrics import cell_balanced_accuracy  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
A_ = {k: sys.argv[i + 1] for i, k in enumerate(sys.argv[:-1]) if k in ("--cell", "--read", "--jobs", "--set")}   # value options (--combine is a flag)
CELL_ = A_.get("--cell", "")
assert CELL_ in ("bw21", "bw23", "cw21", "bw21fm40", "bw21fm55"), "--cell bw21|bw23|cw21|bw21fm40|bw21fm55"   # bw21fm40 = T2-D655 (the FM share 0.40)
IDEA = "w20_fb_fm40" if CELL_.endswith("fm40") else "w20_fb_fm55" if CELL_.endswith("fm55") else "w20_fb"   # T2-D658: + fm55   # T2-D655: the teacher and the mix move together (t2_lab's guard)
SR = O / "t2_d642" / "setread" / CELL_
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
PY = sys.executable
READ = A_.get("--read", "first")
assert READ in ("first", "replicate", "w27", "all6", "all6rep", "d646", "d648", "d649", "d649rep", "w30", "w30rep", "d654", "d654rep", "stack"), READ
BA, TE = {"first": ("b", "c"), "replicate": ("t", "d"), "w27": ("g", "e"), "all6": ("h", "k"), "all6rep": ("m", "n"), "d646": ("x", "y"), "d648": ("v", "w"),
          "d649": ("f", "a"), "d649rep": ("j", "i"), "w30": ("n", "q"), "w30rep": ("w", "r"),
          "d654": ("2", "3"), "d654rep": ("5", "6"), "stack": ("o", "z")}[READ]   # stack = C 1800 (v1) vs the stack arm (+ D654's arm)   # T2-D649 (C vs G) / T2-D652 (base vs V) / T2-D654 (v1 vs the picked arm)
# T2-D654: the stage-1 member json per read (d(ii) = the PICKED arm's member rt2 at that seed set; the pick = the 1600 json's)
D654M = {"d654": "t2_d654/members_d654_d654a_1600.json", "d654rep": "t2_d654/members_d654_stk_1800.json"}   # read 2 = C@1800 (17:2x re-plan)
SLOTS6 = ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]
ALL6 = READ in ("all6", "all6rep", "d646", "d648", "d649", "d649rep", "w30", "w30rep", "d654", "d654rep", "stack")     # D642 x9 / T2-D646 / D648 / D649 / D652: every slot spliced (bw21 only)
# T2-D649 / D652: the member json behind each read (its d(ii) = the raw member rt2 over the CHANGED slots: six for W29, s1-s5 for W30, Track3 pin 1)
NEWM = {"d649": ("t2_d649/members_d649.json", ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]),
        "d649rep": ("t2_d649/members_d649rep.json", ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]),
        "w30": ("t2_d652/members_d652.json", ["s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]),
        "w30rep": ("t2_d652/members_d652b.json", ["s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"])}
SET_ = A_.get("--set", "s2")                           # --combine: s2 (first + replicate) | all6 (all6 + all6rep)
assert SET_ in ("s2", "all6"), SET_
TAGS = (f"{BA}s", f"{TE}s", f"{BA}p04", f"{TE}p04", f"{BA}p10", f"{TE}p10")
DRAWS = (860, 861, 862)
USERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
SPLICED = [f for f in range(10) if USERS[f] not in ("P04", "P10")]
STRANGERS = [USERS[f] for f in SPLICED]
REORDERS = [f"{k}:{s}" for k in ("runs_perm", "in_run") for s in (6310, 6311, 6312)]


def refuse_unless_complete():
    idl = (SR / "identity.log").read_text() if (SR / "identity.log").exists() else ""
    assert "[d642 identity] PASS: the no-op root's chronological mix" in idl and "[d642 identity] PASS teachers" in idl, "REFUSED: the identity has not PASSED"
    ch = (SR / "chain.log").read_text() if (SR / "chain.log").exists() else ""
    missing = [t for t in TAGS if f"### done d642 tag {t} " not in ch]
    assert not missing, f"REFUSED: partial tags (no partial read): {missing}"
    m = json.loads((O / "t2_d642" / ("members_w27.json" if READ == "w27" else "members.json")).read_text())
    assert m["nominated"] and m["entering"], "REFUSED: no set nominated"
    assert READ != "d646" or (O / "t2_d642" / "members_d646.json").exists(), "REFUSED: the D646 read needs members_d646.json (--d646)"
    assert READ != "d648" or (O / "t2_d642" / "members_d648.json").exists(), "REFUSED: the D648 read needs members_d648.json (--d648)"
    if READ == "stack":                                 # THE STACK READ: its spec (>= 2 included parts) + its member read (d(ii))
        assert (SR / "stack_spec.json").exists() and (O / "t2_stack/members_stack.json").exists(), "REFUSED: stack_spec / members_stack missing"
    if READ in D654M:                                   # T2-D654: stage 1's pinned pick must exist (None = closed, no stage 2)
        p1 = json.loads((O / D654M["d654"]).read_text())
        assert p1["pick"], "REFUSED: D654 stage 1 has no pick (closed)"
        assert p1["pick"] in json.loads((O / D654M[READ]).read_text())["arms"], (READ, "the stage-1 json lacks the pick")
    if READ in NEWM:                                    # T2-D649 / D652: the member gate must have PASSED before any set read
        mj = json.loads((O / NEWM[READ][0]).read_text())
        assert READ not in ("d649", "d649rep") or json.loads((O / "t2_d649/members_d649.json").read_text())["alive"], "REFUSED: D649's member screen KILLED"
        assert READ not in ("w30", "w30rep") or json.loads((O / "t2_d652/members_d652.json").read_text())["passes"], "REFUSED: W30's member gate FAILED"
        assert set(NEWM[READ][1]) <= set(mj["slots"]), (READ, sorted(mj["slots"]))
    assert READ not in ("replicate", "all6rep") or (O / "t2_d642" / "members_replicate.json").exists(), \
        "REFUSED: a replicate read needs members_replicate.json (--replicate)"   # Track3: the first-type reads (first / w27 / all6) never need it
    if ALL6:
        assert CELL_ == "bw21", "D642 x9: ALL6 on the decision cell bw21 only"
        m = dict(m, entering=SLOTS6)                     # every slot (the member screen's nomination is the precondition)
    for t in TAGS:
        man = json.loads((SR / f"fmspl_{t}" / "manifest.json").read_text())
        assert man["slots"] == m["entering"], (t, man["slots"], m["entering"])
        assert "the lab loads EXACTLY the manifest's" in (SR / f"teach_{t}.log").read_text(), (t, "Track3 LOW 2's path check missing")
    return m


def mix_path(tag, order):
    return SR / f"mix_{tag}_{order.replace(':', '_')}.npz"


def run_mix(tag, order):
    out = mix_path(tag, order)
    if not out.exists():
        env = {k: v for k, v in os.environ.items() if not k.startswith("NI26_")}
        env.update(NI26_LAB_PRE_R="1", NI26_D559_ROOT=f"t2_d642_lp{tag}_{CELL_}_860", NI26_D559_CKS="60,100,150", NI26_LAB_FMSPLICE=str(SR / f"fmspl_{tag}"))
        if tag[1] == "p":
            env["NI26_LAB_FMSTART"] = f"freeze:{SR / 'fmstart.npz'}"
        if order != "chrono":
            env["NI26_LAB_ORDER"] = order
        with open(SR / f"mix_{tag}.log", "a") as log:
            subprocess.run([PY, "-u", "scripts/local/t2_lab.py", "mix", "--set", "E", "--base", "w20", "--idea", IDEA, "--out", str(out)],
                           env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    z = np.load(out)
    assert str(z["idea"]) == IDEA and str(z["base"]) == "w20" and str(z["order"]) == ("" if order == "chrono" else order), (tag, order)
    return out


def main():
    m = refuse_unless_complete()
    jobs = int(A_.get("--jobs", "2"))
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    runs = [(t, "chrono") for t in TAGS] + [(t, o) for o in REORDERS for t in (f"{BA}s", f"{TE}s")]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda to: run_mix(*to), runs))

    def cba(rows, p):
        return cell_balanced_accuracy(Y[rows], p.argmax(1), CELL[rows])[0]

    def score_s(tag, order):
        z = np.load(mix_path(tag, order))
        st = {u: [] for u in STRANGERS}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[k], mx[k]))
        return st                                        # per user: 3 draws

    def score_x(tag, x):
        z = np.load(mix_path(tag, "chrono"))
        v = []
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                v.append(cba(ix[k], mx[k]))
        return v                                         # per (fold, draw): 24

    def ds(a, b):
        du = {u: float(np.mean(np.subtract(a[u], b[u]))) for u in STRANGERS}
        dd = [float(np.mean([a[u][j] - b[u][j] for u in STRANGERS])) for j in range(len(DRAWS))]
        return dict(dS=float(np.mean(list(du.values()))), n_ge0=int(sum(v >= 0 for v in du.values())), min=float(min(du.values())), per_user=du, per_draw=dd)

    S = {(t, o): score_s(t, o) for t in (f"{BA}s", f"{TE}s") for o in ["chrono"] + REORDERS}
    ch = ds(S[(f"{TE}s", "chrono")], S[(f"{BA}s", "chrono")])
    orders = {k: float(np.mean([ds(S[(f"{TE}s", f"{k}:{s}")], S[(f"{BA}s", f"{k}:{s}")])["dS"] for s in (6310, 6311, 6312)])) for k in ("runs_perm", "in_run")}
    X = {x: dict(B=score_x(f"{BA}p{x[1:]}", x), C=score_x(f"{TE}p{x[1:]}", x)) for x in ("P04", "P10")}
    dx = {x: float(np.mean(np.subtract(X[x]["C"], X[x]["B"]))) for x in X}
    d_ret = float(np.mean(list(dx.values())))
    if READ == "stack":                                  # THE STACK READ: the stack arm's member rt2 vs C at 1800 over its CHANGED slots
        d_ii = float(json.loads((O / "t2_stack/members_stack.json").read_text())["d_rt2_changed"])
    elif READ in D654M:                                  # T2-D654: the picked arm's member rt2 (all six slots change)
        pk = json.loads((O / D654M["d654"]).read_text())["pick"]
        d_ii = float(json.loads((O / D654M[READ]).read_text())["arms"][pk]["d_rt2"])
    elif READ in NEWM:                                   # T2-D649 / D652: the generic member read's json (no 'raw' level), the CHANGED slots
        mj = json.loads((O / NEWM[READ][0]).read_text())
        d_ii = float(np.mean([mj["slots"][s]["d_rt2"] for s in NEWM[READ][1]]))
    else:
        ms = m if READ in ("first", "w27", "all6") else json.loads((O / "t2_d642" / ({"d646": "members_d646.json", "d648": "members_d648.json"}.get(READ, "members_replicate.json"))).read_text())
        d_ii = float(np.mean([ms["slots"][s]["raw"]["d_rt2"] for s in m["entering"]]))
    sealed = 0.57 * ch["dS"] + 0.38 * d_ret + 0.06 * min(d_ii, 0.0)                      # Track3 LOW 1: d(ii) as a cost only
    checks = dict(dS=ch["dS"] >= 0.004, n6=ch["n_ge0"] >= 6, runs_perm=orders["runs_perm"] >= 0, in_run=orders["in_run"] >= 0,
                  returning=d_ret >= -0.003, sealed=sealed >= 0.002)
    passes = bool(all(checks.values()))
    lev = dict(bs=float(np.mean([np.mean(v) for v in S[(f"{BA}s", "chrono")].values()])), cs=float(np.mean([np.mean(v) for v in S[(f"{TE}s", "chrono")].values()])),
               **{f"{a}{x}": float(np.mean(X[x][a])) for x in X for a in ("B", "C")})
    d638 = json.loads((O / "t2_d638" / "result.json").read_text())["noise"]
    out = dict(cell=CELL_, read=READ, tags=list(TAGS), entering=m["entering"], levels=lev, strangers=ch, orders=orders, returning=dict(per_X=dx, d=d_ret),
               d_ii=d_ii, sealed=sealed, checks=checks, passes=passes, d638_floors=dict(S=d638["floor_S"], i=d638["floor_i"]))
    SR.joinpath(f"set_{READ}.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"# T2-D642 RESULT — the full-chain SET read, {READ} ({dict(first='B vs C', replicate='T vs C′ (W26 replicate)', w27='B vs E (T2-D644 W27)', all6='B vs C, ALL six slots (×9)', all6rep='T vs C′, ALL six slots (×9 replicate)', d646='X vs Y = B vs C at seeds 1000+, ALL six slots (T2-D646 veto read)', d648='V vs W = B vs C at seeds 1100+, ALL six slots (T2-D648, the fourth read)', d649='C vs G_picked at 1200+ (T2-D649 W29, read 1)', d649rep='C vs G_picked at 1300+ (T2-D649, read 2)', w30='C′ (n) vs V at 950+ (T2-D652 W30, read 1; s0 = n)', w30rep='W (w) vs V at 1100+ (T2-D652, read 2; s0 = w)', d654='v1 vs the picked view arm, 1600 base (T2-D654 read 1)', d654rep='v1 vs the picked view arm, 1700 base (T2-D654 read 2)', stack='C 1800 (v1) vs THE STACK (the stack arm + the D654 arm if included)')[READ]}; the bar line below is "
         "DESCRIPTIVE under ×7: the decision = --combine)", "",
         f"Registered before any number. Cell {out['cell']}; entering slots {m['entering']}; identity PASSED (teachers + mix bitwise). "
         "Roots p04 / p10: FM starts frozen to the id root's (Track3 Q2; a stated approximation). Order evidence decode-only (PARTIAL).", "",
         f"**Verdict: {'PASS' if passes else 'FAIL'}** {checks}", "",
         f"- strangers (root s): ΔS {ch['dS']:+.4f} ({ch['n_ge0']} / 8 ≥ 0, min {ch['min']:+.4f}); per draw {', '.join(f'{v:+.4f}' for v in ch['per_draw'])}; "
         f"per user {', '.join(f'{u} {v:+.4f}' for u, v in ch['per_user'].items())}",
         f"- reorders (mean of 3 seeds): runs_perm {orders['runs_perm']:+.4f}, in_run {orders['in_run']:+.4f}",
         f"- returning: P04 {dx['P04']:+.4f}, P10 {dx['P10']:+.4f} → Δ {d_ret:+.4f}; Δ(ii) (member screen rt2, entering slots, raw) {d_ii:+.4f}",
         f"- sealed arithmetic 0.57·ΔS + 0.38·Δret + 0.06·min(Δ(ii), 0) = {sealed:+.4f} (bar +0.002)",
         f"- levels: strangers B {lev['bs']:.4f} / C {lev['cs']:.4f}; P04 B {lev['BP04']:.4f} / C {lev['CP04']:.4f}; P10 B {lev['BP10']:.4f} / C {lev['CP10']:.4f}",
         f"- D638's noise floors (descriptive): |ΔS| {d638['floor_S']:.4f}, |Δ(i)| {d638['floor_i']:.4f}"]
    if READ == "d646":                                    # T2-D646's pinned VETO rule
        veto = bool(ch["dS"] < 0 and sealed < 0)
        out["d646_veto"] = veto
        SR.joinpath(f"set_{READ}.json").write_text(json.dumps(out, indent=1, default=float))
        L.insert(2, f"**T2-D646 VETO RULE (pinned): VETO iff strangers ΔS < 0 AND sealed < 0; here ΔS {ch['dS']:+.4f}, sealed {sealed:+.4f} → "
                    f"{'VETO: the manager withdraws item 167, W21 ships' if veto else 'NO VETO: item 167 stays recommended (the size estimate refined)'}**")
    SR.joinpath(f"RESULT_D642_set_{READ}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def combine():
    """D642 x7 (W26): the DECISION = the registered bars on the MEAN of the two reads AND the replicate's strangers dS > 0."""
    a, b = (json.loads((SR / f"set_{r}.json").read_text()) for r in (("first", "replicate") if SET_ == "s2" else ("all6", "all6rep")))
    ret_bar = -0.003 if SET_ == "s2" else 0.0             # D642 x9: ALL6's returning bar is >= 0
    assert a["cell"] == b["cell"] == CELL_ and a["entering"] == b["entering"], (a["cell"], b["cell"], a["entering"], b["entering"])
    pu = {u: (a["strangers"]["per_user"][u] + b["strangers"]["per_user"][u]) / 2 for u in STRANGERS}
    dS = float(np.mean(list(pu.values())))
    n6 = int(sum(v >= 0 for v in pu.values()))
    orders = {k: (a["orders"][k] + b["orders"][k]) / 2 for k in ("runs_perm", "in_run")}
    d_ret = (a["returning"]["d"] + b["returning"]["d"]) / 2
    d_ii = (a["d_ii"] + b["d_ii"]) / 2
    sealed = 0.57 * dS + 0.38 * d_ret + 0.06 * min(d_ii, 0.0)
    checks = dict(dS=dS >= 0.004, n6=n6 >= 6, runs_perm=orders["runs_perm"] >= 0, in_run=orders["in_run"] >= 0, returning=d_ret >= ret_bar,
                  sealed=sealed >= 0.002, replicate_dS_pos=b["strangers"]["dS"] > 0)
    passes = bool(all(checks.values()))
    out = dict(cell=CELL_, set=SET_, entering=a["entering"], dS=dS, n_ge0=n6, per_user=pu, orders=orders, d_ret=d_ret, d_ii=d_ii, sealed=sealed, checks=checks,
               passes=passes, first=dict(dS=a["strangers"]["dS"], n_ge0=a["strangers"]["n_ge0"], d_ret=a["returning"]["d"], sealed=a["sealed"]),
               replicate=dict(dS=b["strangers"]["dS"], n_ge0=b["strangers"]["n_ge0"], d_ret=b["returning"]["d"], sealed=b["sealed"]))
    SR.joinpath(f"set_combined_{SET_}.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"# T2-D642 ×7 / ×9 DECISION ({SET_}) — the full-chain set read, the MEAN of the two reads (B vs C, T vs C′) + the replicate alone", "",
         f"Cell {CELL_}; entering slots {a['entering']}. **Verdict: {'PASS' if passes else 'FAIL'}** {checks}", "",
         f"- mean of the two reads: strangers ΔS {dS:+.4f} ({n6} / 8 ≥ 0 on per-user means; per user {', '.join(f'{u} {v:+.4f}' for u, v in pu.items())}); "
         f"reorders runs_perm {orders['runs_perm']:+.4f} / in_run {orders['in_run']:+.4f}; returning {d_ret:+.4f}; Δ(ii) {d_ii:+.4f}; sealed {sealed:+.4f}",
         f"- first read alone (B vs C): ΔS {a['strangers']['dS']:+.4f} ({a['strangers']['n_ge0']} / 8); returning {a['returning']['d']:+.4f}; sealed {a['sealed']:+.4f}",
         f"- replicate alone (T vs C′): ΔS {b['strangers']['dS']:+.4f} ({b['strangers']['n_ge0']} / 8); returning {b['returning']['d']:+.4f}; sealed {b['sealed']:+.4f}"]
    SR.joinpath(f"RESULT_D642_set_combined_{SET_}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def decide648():
    """T2-D648's PINNED decision on the UNSELECTED reads 2-4 (the W26 replicate all6rep, D646, D648; read 1 = the discovery read, excluded):
    W25-ALL6 stays recommended iff the MEAN sealed arithmetic >= 0 AND the MEAN strangers dS > 0; else the manager recommends W21."""
    R3 = {r: json.loads((SR / f"set_{r}.json").read_text()) for r in ("all6rep", "d646", "d648")}
    assert all(v["cell"] == CELL_ == "bw21" for v in R3.values()), {r: v["cell"] for r, v in R3.items()}
    dS = {r: v["strangers"]["dS"] for r, v in R3.items()}
    se = {r: v["sealed"] for r, v in R3.items()}
    rt = {r: v["returning"]["d"] for r, v in R3.items()}
    mS, mse, mrt = (float(np.mean(list(x.values()))) for x in (dS, se, rt))
    stays = bool(mse >= 0 and mS > 0)
    out = dict(reads=list(R3), dS=dS, sealed=se, returning=rt, mean_dS=mS, mean_sealed=mse, mean_returning=mrt, w25_stays=stays)
    SR.joinpath("decide_d648.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# T2-D648 — the PINNED decision on the UNSELECTED ALL6 reads 2–4 (read 1 = the discovery read, excluded)", "",
         f"**mean sealed {mse:+.4f} (≥ 0?) AND mean strangers ΔS {mS:+.4f} (> 0?) → "
         f"{'W25-ALL6 STAYS RECOMMENDED' if stays else 'the manager recommends W21'}**; mean returning {mrt:+.4f}", ""]
    L += [f"- {r}: ΔS {dS[r]:+.4f}, returning {rt[r]:+.4f}, sealed {se[r]:+.4f}" for r in R3]
    SR.joinpath("RESULT_D648_decision.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def d650():
    """T2-D650: W25-ALL6 + D630's balanced pooled CNN nets = the ALL6 balanced FM (k tags) on cell cw21 vs the SAME k tags on bw21 (W25-ALL6's
    read), paired by draw / student seed; d(ii) = D638's CNN-stage cw21 - bw21 rt2 (as a cost); the reorders on root s. -> setread/cw21/set_d650.json"""
    assert CELL_ == "cw21", "T2-D650 runs on --cell cw21"
    idl = (SR / "identity.log").read_text() if (SR / "identity.log").exists() else ""
    assert "[d642 identity] PASS: the no-op root's chronological mix" in idl, "REFUSED: the cw21 identity has not PASSED"
    ch = (SR / "chain.log").read_text() if (SR / "chain.log").exists() else ""
    assert all(f"### done d642 tag {t} " in ch for t in ("ks", "kp04", "kp10")), "REFUSED: partial cw21 k tags"
    jobs = int(A_.get("--jobs", "2"))
    runs = [(t, "chrono") for t in ("ks", "kp04", "kp10")] + [("ks", o) for o in REORDERS]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda to: run_mix(*to), runs))
    SB = O / "t2_d642" / "setread" / "bw21"
    for t in ("ks", "kp04", "kp10"):
        assert (SB / f"mix_{t}_chrono.npz").exists(), ("the bw21 k mixes", t)
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()

    def cba(rows, p):
        return cell_balanced_accuracy(Y[rows], p.argmax(1), CELL[rows])[0]

    def sc_s(path):
        z = np.load(path); st = {u: [] for u in STRANGERS}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64); k = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[k], mx[k]))
        return st

    def sc_x(path, x):
        z = np.load(path); v = []
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64); k = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                v.append(cba(ix[k], mx[k]))
        return v
    a, b = sc_s(SR / "mix_ks_chrono.npz"), sc_s(SB / "mix_ks_chrono.npz")
    du = {u: float(np.mean(np.subtract(a[u], b[u]))) for u in STRANGERS}
    dS = float(np.mean(list(du.values()))); n6 = int(sum(v >= 0 for v in du.values()))
    orders = {}
    for kk in ("runs_perm", "in_run"):
        vals = []
        for sd in (6310, 6311, 6312):
            o = f"{kk}:{sd}"
            fb = SB / f"mix_ks_{o.replace(':', '_')}.npz"
            assert fb.exists(), ("the bw21 ks reorder mix", o)
            aa, bb = sc_s(SR / f"mix_ks_{o.replace(':', '_')}.npz"), sc_s(fb)
            vals.append(float(np.mean([np.mean(np.subtract(aa[u], bb[u])) for u in STRANGERS])))
        orders[kk] = float(np.mean(vals))
    dx = {x: float(np.mean(np.subtract(sc_x(SR / f"mix_kp{x[1:]}_chrono.npz", x), sc_x(SB / f"mix_kp{x[1:]}_chrono.npz", x)))) for x in ("P04", "P10")}
    d_ret = float(np.mean(list(dx.values())))
    d_ii = float(json.loads((O / "t2_d638" / "result.json").read_text())["pre_pairs"]["cw21-bw21"]["d_ii"])
    sealed = 0.57 * dS + 0.38 * d_ret + 0.06 * min(d_ii, 0.0)
    out = dict(cell="cw21 vs bw21 (k tags)", strangers=dict(dS=dS, n_ge0=n6, per_user=du), orders=orders, returning=dict(per_X=dx, d=d_ret), d_ii=d_ii,
               sealed=sealed, positive=bool(dS > 0 and sealed > 0))
    SR.joinpath("set_d650.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# T2-D650 — W25-ALL6 + D630's balanced pooled CNN nets (cw21 + k) vs W25-ALL6 (bw21 + k), the first read", "",
         f"- strangers ΔS {dS:+.4f} ({n6} / 8 ≥ 0; per user {', '.join(f'{u} {v:+.4f}' for u, v in du.items())}); reorders runs_perm {orders['runs_perm']:+.4f} / in_run {orders['in_run']:+.4f}",
         f"- returning: P04 {dx['P04']:+.4f}, P10 {dx['P10']:+.4f} → {d_ret:+.4f}; Δ(ii) (D638 CNN stage cw21 − bw21) {d_ii:+.4f}; sealed {sealed:+.4f}",
         f"- **a replicate {'IS' if out['positive'] else 'is NOT'} run** (registered: only if strangers ΔS > 0 and sealed > 0); adoption = the mean of two reads (D649's bar)"]
    SR.joinpath("RESULT_D650_set.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


# ===== THE OWNER'S INCLUSION RULE (2026-11-20 16:31:32, via the manager, before any D649 / D652 / D654 number) + R1 / R2 (16:29): Track3's
# lead/diag_word_2026-11-18/inclusion_decide_draft.py, verbatim below (include -> _include, final -> _final). sealed* / kappa / dRL / the D651
# fallback are WITHDRAWN (16:29); the metric = the OLD arithmetic ('sealed' in each set_<read>.json); 0.57 / 0.38 / 0.06 = 60 / 40 / 6 of 106.
READ2_DS_LEG = True                                    # Track3's recommended addition (the manager may drop it: then rule 2 = his three legs)


def _include(SR, r1, r2, name, extra=None):
    """RULE 2 for one part from setread/bw21/set_<r1>.json and set_<r2>.json (r2 = the fresh replicate) -> decide_<name>.json + .md."""
    R1, R2 = (json.loads((SR / f"set_{r}.json").read_text()) for r in (r1, r2))
    assert R1["cell"] == R2["cell"] == "bw21", (R1["cell"], R2["cell"])
    dS = float(np.mean([R1["strangers"]["dS"], R2["strangers"]["dS"]]))
    old = float(np.mean([R1["sealed"], R2["sealed"]]))
    ret = float(np.mean([R1["returning"]["d"], R2["returning"]["d"]]))
    pu1, pu2 = R1["strangers"]["per_user"], R2["strangers"]["per_user"]
    assert sorted(pu1) == sorted(pu2) and len(pu1) == 8, (sorted(pu1), sorted(pu2))
    pu = {u: 0.5 * (pu1[u] + pu2[u]) for u in pu1}
    loso = {u: float(np.mean([v for w, v in pu.items() if w != u])) for u in pu}
    checks = dict(mean_dS=dS > 0, mean_old_sealed=old > 0, read2_old_sealed=R2["sealed"] > 0)
    if READ2_DS_LEG:
        checks["read2_dS"] = R2["strangers"]["dS"] > 0
    inc = bool(all(checks.values()))
    out = dict(part=name, included=inc, checks=checks, mean_dS=dS, mean_old_sealed=old, mean_returning=ret,
               read2=dict(dS=R2["strangers"]["dS"], old_sealed=R2["sealed"]), read1=dict(dS=R1["strangers"]["dS"], old_sealed=R1["sealed"]),
               reported=dict(loso_min=min(loso.values()), loso_worst_drop=min(loso, key=loso.get), per_user_two_read=pu),
               rule="the owner's inclusion rule (2026-11-20 16:31:32) on the OLD arithmetic" + (" + Track3's read-2 dS leg" if READ2_DS_LEG else ""),
               **(extra or {}))
    (SR / f"decide_{name}.json").write_text(json.dumps(out, indent=1, default=float))
    L = [f"# {name} — the inclusion rule on its two set reads ({r1}, {r2} = fresh), OLD arithmetic", "",
         f"**{'INCLUDED' if inc else 'NOT included'}**: mean ΔS {dS:+.4f} (> 0?), mean OLD sealed {old:+.4f} (> 0?), read 2 OLD sealed {R2['sealed']:+.4f} (> 0?)"
         + (f", read 2 ΔS {R2['strangers']['dS']:+.4f} (> 0?)" if READ2_DS_LEG else ""), "",
         f"- reported: returning (full-chain rt1) {ret:+.4f}; leave-one-stranger-out min {min(loso.values()):+.4f} (drop {min(loso, key=loso.get)})"]
    (SR / f"RESULT_{name}_decision.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return out


def _final(SR, parts, stack_read=None, stack_cut=False):
    """RULE 3. parts = {"W29": decide dict | status string, "W30": ..., "D654": ...} (a string = not engaged / killed / closed; 'pending' refuses).
    stack_read = the stack's set_<read>.json dict at 1800 (needed iff >= 2 parts are included), or stack_cut = True (capacity) -> the best single."""
    assert not any(v == "pending" for v in parts.values()), {k: v for k, v in parts.items() if v == "pending"}
    inc = {k: v for k, v in parts.items() if isinstance(v, dict) and v["included"]}
    stack = None
    if not inc:
        pick, why = "W25-ALL6", "no part included: the fallback = D648's registered rule (W25-ALL6; a tie within noise vs W21)"
    elif len(inc) == 1:
        pick = next(iter(inc))
        why = f"one part included: {pick} alone"
    else:
        best = max(inc, key=lambda k: inc[k]["mean_old_sealed"])
        if stack_read is None:
            assert stack_cut, "REFUSED: >= 2 parts are included and the stack read is pending (or pass stack_cut=True if capacity cut it)"
            pick, why = best, f"the stack read was CUT (capacity): the best single included part {best}"
        else:
            assert stack_read["cell"] == "bw21", stack_read["cell"]
            stack = dict(parts=sorted(inc), dS=stack_read["strangers"]["dS"], old_sealed=stack_read["sealed"], returning=stack_read["returning"]["d"])
            ok = stack["dS"] > 0 and stack["old_sealed"] > 0
            pick = ("STACK(" + "+".join(sorted(inc)) + ")") if ok else best
            why = (f"the stack read at 1800: ΔS {stack['dS']:+.4f}, OLD sealed {stack['old_sealed']:+.4f} -> "
                   + ("the stack ships" if ok else f"the stack fails: the best single included part {best}"))
    out = dict(pick=pick, why=why, stack=stack, parts={k: (v if isinstance(v, str) else dict(included=v["included"], mean_dS=v["mean_dS"],
                                                                                            mean_old_sealed=v["mean_old_sealed"])) for k, v in parts.items()},
               never="a stack not read as a stack; a combination outside the one stack", d651="descriptive only")
    (SR / "final_pick.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# T2 — THE FINAL PICK (the owner's inclusion rule, one stack; OLD arithmetic)", "", f"**{pick}** — {why}", ""]
    for k, v in parts.items():
        L.append(f"- {k}: {v}" if isinstance(v, str) else
                 f"- {k}: {'included' if v['included'] else 'not included'} (mean ΔS {v['mean_dS']:+.4f}, OLD sealed {v['mean_old_sealed']:+.4f})")
    L += ["", "Promotion / upload: the owner's typed line only; every package / timing gate unchanged; incomplete at Nov 21 14:00 = the fallback."]
    (SR / "RESULT_final_pick.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return out




def decide649():
    """W29 = the part G: the inclusion rule on its two set reads (d649, d649rep = the fresh 1300 read)."""
    return _include(SR, "d649", "d649rep", "d649")


def decide652():
    """W30 = the part V: the inclusion rule on its two set reads (w30 = vs C' at 950, w30rep = the fresh V1100 vs W)."""
    return _include(SR, "w30", "w30rep", "d652")


def decide654():
    """D654 = the picked view arm: the inclusion rule on its two set reads (d654 = 1600, d654rep = the fresh 1700)."""
    pk = json.loads((O / D654M["d654"]).read_text())["pick"]
    assert pk, "REFUSED: D654 stage 1 has no pick (closed)"
    return _include(SR, "d654", "d654rep", "d654", extra=dict(pick=pk))


def _status(name):
    """a part's status for _final: its decide dict, or a string (not engaged / killed / closed), or 'pending' (refused)."""
    jl = lambda p_: json.loads(p_.read_text()) if p_.exists() else None   # noqa: E731
    cut = jl(SR / f"cut_{name}.json")                   # Track3 (20:0x): a part CUT at the cut-off is recorded explicitly (never 'pending' forever)
    if cut is not None:
        return f"cut for capacity (never read): {cut.get('why', '')}"
    if name == "W29":                                   # CUT for capacity 2026-11-20 17:2x (the manager): engaged at lambda 0.1, never read
        return "cut for capacity (not tested; engaged at lambda 0.1, never read)"
        pk = jl(O / "t2_d649/lambda_pick.json")
        if pk is None:
            return "pending"
        if not pk["engaged"]:
            return "not engaged (D371: not tested)"
        m = jl(O / "t2_d649/members_d649.json")
        if m is None:
            return "pending"
        if not m["alive"]:
            return "killed (the member sign gate)"
        return jl(SR / "decide_d649.json") or "pending"
    if name == "W30":
        m = jl(O / "t2_d652/members_d652.json")
        if m is None:
            return "pending"
        if not m["passes"]:
            return "closed (the member sign gate)"
        return jl(SR / "decide_d652.json") or "pending"
    m = jl(O / D654M["d654"])
    if m is None:
        return "pending"
    if not m["pick"]:
        return "closed (stage 1: no eligible arm with member dS > 0)"
    return jl(SR / "decide_d654.json") or "pending"



def d658():
    """T2-D658 (registered 2026-11-21 18:50:58 before any number): RH's FM share 0.55 (this cell, bw21fm55; teacher AND mix = tw_fm(0.55), D655's
    form) vs 0.35 (bw21's existing w30 read) on W30's OWN q tags (V at 950), paired by draw (the same student seeds); the OLD arithmetic with
    d(ii) == 0 (no member changes). INFORMATION for the owner's choice: RH is offered iff its composite vs plain W30 >= -0.003 (and gated by 21:45).
    Chronological only. -> setread/bw21fm55/d658.json + RESULT_D658.md"""
    assert CELL_ == "bw21fm55", "T2-D658 runs on --cell bw21fm55"
    import t2_lab_ideas as IDX                           # Track3's pin 2: the lab's tw_fm(0.55) == the RH package's family shares
    tw = IDX.tw_fm(0.55)
    pk = IDX.RH_PACKAGE_SHARES
    pk_main = 1.0 - pk["fm"] - pk["cnn2"] - pk["cnn3"]
    assert abs(tw["fm"] - pk["fm"]) <= 1e-4 and abs(tw["sh"] - pk["cnn2"]) <= 1e-4 and abs(tw["ifn"] - pk["cnn3"]) <= 1e-4 and abs(tw["fbc"] - pk_main) <= 1e-4, (tw, pk, pk_main)
    S35 = O / "t2_d642" / "setread" / "bw21"
    ch = (SR / "chain.log").read_text() if (SR / "chain.log").exists() else ""
    TG = ["qs", "qp04", "qp10"]
    assert all(f"### done d642 tag {t} " in ch for t in TG), "REFUSED: partial fm55 tags"
    for t_ in TG:
        assert (S35 / f"mix_{t_}_chrono.npz").exists(), t_
    with ThreadPoolExecutor(int(A_.get("--jobs", "2"))) as ex:
        list(ex.map(lambda t_: run_mix(t_, "chrono"), TG))
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()
    cba = lambda rows, p_: cell_balanced_accuracy(Y[rows], p_.argmax(1), CELL[rows])[0]   # noqa: E731

    def score_s(sr, tag):
        z = np.load(sr / f"mix_{tag}_chrono.npz")
        st = {u: [] for u in STRANGERS}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[k], mx[k]))
        return st

    def score_x(sr, tag, x):
        z = np.load(sr / f"mix_{tag}_chrono.npz")
        v = []
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                v.append(cba(ix[k], mx[k]))
        return v
    a, b = score_s(SR, "qs"), score_s(S35, "qs")
    du = {u: float(np.mean(np.subtract(a[u], b[u]))) for u in STRANGERS}
    dd = [float(np.mean([a[u][j] - b[u][j] for u in STRANGERS])) for j in range(len(DRAWS))]
    dS = float(np.mean(list(du.values())))
    dx = {x: float(np.mean(np.subtract(score_x(SR, f"qp{x[1:]}", x), score_x(S35, f"qp{x[1:]}", x)))) for x in ("P04", "P10")}
    dret = float(np.mean(list(dx.values())))
    comp = 0.57 * dS + 0.38 * dret
    lev = dict(strangers_055=float(np.mean([np.mean(v) for v in a.values()])), strangers_035=float(np.mean([np.mean(v) for v in b.values()])))
    out = dict(dS=dS, per_user=du, per_draw=dd, n_ge0=int(sum(v >= 0 for v in du.values())), returning=dret, per_X=dx, composite=comp,
               offer_leg=comp >= -0.003, levels=lev, tw_fm_055=tw, rule="RH offered iff composite vs plain W30 >= -0.003 AND gated by 21:45 AND no failed gate")
    (SR / "d658.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# T2-D658 — RH (FM share 0.55) vs plain W30 (0.35) on W30's q tags (bw21fm55 vs bw21), chronological, OLD arithmetic (d(ii) = 0)", "",
         f"**composite 0.57 dS + 0.38 dret = {comp:+.4f}** (offer leg >= -0.003: {'MET' if comp >= -0.003 else 'NOT met'}); strangers dS {dS:+.4f} "
         f"({out['n_ge0']} / 8 >= 0; draws {', '.join(f'{v:+.4f}' for v in dd)}); returning {dret:+.4f} (P04 {dx['P04']:+.4f}, P10 {dx['P10']:+.4f})", "",
         "Per user (0.55 - 0.35): " + ", ".join(f"{u} {v:+.4f}" for u, v in du.items()), "",
         f"Levels: strangers 0.35 {lev['strangers_035']:.4f} -> 0.55 {lev['strangers_055']:.4f}. Caveat: one unreplicated read; a share step re-tuned on the panel lost before (D591)."]
    (SR / "RESULT_D658.md").write_text("\n".join(L) + "\n")
    print(f"=== D658 RH vs W30: composite {comp:+.4f} (offer leg {'MET' if comp >= -0.003 else 'NOT met'}); dS {dS:+.4f} ({out['n_ge0']} / 8); returning {dret:+.4f}", flush=True)


def d655():
    """T2-D655 (registered before any number; Track3's tightened rule): the FM share 0.40 (this cell, bw21fm40) vs 0.35 (bw21's existing reads), on the
    three W25 ALL6 roots of the balanced members (n = all6rep, y = d646, w = d648), paired by draw; the OLD arithmetic with d(ii) == 0. INCLUDE iff
    (i) the three-root mean strangers dS > 0; (ii) >= 6 / 8 strangers >= 0 on the three-root mean per user; (iii) leave-one-stranger-out: the mean of
    the other 7 > 0 for EVERY left-out stranger; (iv) the three-root mean returning (rt1, P04 / P10 s2-3) >= 0; (v) roots 2 (y) and 3 (w) EACH alone
    dS > 0 AND OLD sealed > 0; (vi) for runs_perm and in_run (seeds 6310-6312, root s) the three-root mean dS >= 0. -> setread/bw21fm40/decide_d655.json"""
    assert CELL_ == "bw21fm40", "T2-D655 runs on --cell bw21fm40"
    S35 = O / "t2_d642" / "setread" / "bw21"
    ch = (SR / "chain.log").read_text() if (SR / "chain.log").exists() else ""
    TG = [f"{r}{x}" for r in "nyw" for x in ("s", "p04", "p10")]
    assert all(f"### done d642 tag {t} " in ch for t in TG), "REFUSED: partial fm40 tags"
    for t_ in TG:                                        # the 0.35 side: the existing reads' files (never recomputed here)
        assert (S35 / f"mix_{t_}_chrono.npz").exists(), t_
    for r in "nyw":
        for o in REORDERS:
            assert (S35 / f"mix_{r}s_{o.replace(':', '_')}.npz").exists(), (r, o)
    jobs = int(A_.get("--jobs", "2"))
    runs = [(t_, "chrono") for t_ in TG] + [(f"{r}s", o) for r in "nyw" for o in REORDERS]
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda to: run_mix(*to), runs))
    idx = pd.read_csv(EXP / "index.csv")
    U, SES, Y, CELL = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy(), idx.y.to_numpy(), idx.cell.to_numpy()

    def cba(rows, p):
        return cell_balanced_accuracy(Y[rows], p.argmax(1), CELL[rows])[0]

    def path(sr, tag, order):
        return sr / f"mix_{tag}_{order.replace(':', '_')}.npz"

    def score_s(sr, tag, order):
        z = np.load(path(sr, tag, order))
        st = {u: [] for u in STRANGERS}
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = U[ix] == USERS[f]
                st[USERS[f]].append(cba(ix[k], mx[k]))
        return st

    def score_x(sr, tag, x):
        z = np.load(path(sr, tag, "chrono"))
        v = []
        for f in SPLICED:
            for d in DRAWS:
                ix, mx = z[f"ix_{f}_{d}"], z[f"mx_{f}_{d}"].astype(np.float64)
                k = (U[ix] == x) & np.isin(SES[ix], ("2", "3"))
                v.append(cba(ix[k], mx[k]))
        return v
    R = {}
    for r, name in (("n", "all6rep"), ("y", "d646"), ("w", "d648")):
        a, b = score_s(SR, f"{r}s", "chrono"), score_s(S35, f"{r}s", "chrono")
        du = {u: float(np.mean(np.subtract(a[u], b[u]))) for u in STRANGERS}
        dS = float(np.mean(list(du.values())))
        dx = {x: float(np.mean(np.subtract(score_x(SR, f"{r}p{x[1:]}", x), score_x(S35, f"{r}p{x[1:]}", x)))) for x in ("P04", "P10")}
        dret = float(np.mean(list(dx.values())))
        od = {}
        for k in ("runs_perm", "in_run"):
            vals = []
            for s_ in (6310, 6311, 6312):
                a_, b_ = score_s(SR, f"{r}s", f"{k}:{s_}"), score_s(S35, f"{r}s", f"{k}:{s_}")
                vals.append(float(np.mean([np.mean(np.subtract(a_[u], b_[u])) for u in STRANGERS])))
            od[k] = float(np.mean(vals))
        R[name] = dict(root=r, dS=dS, per_user=du, returning=dret, per_X=dx, sealed=0.57 * dS + 0.38 * dret, orders=od)
    pu = {u: float(np.mean([R[k]["per_user"][u] for k in R])) for u in STRANGERS}
    mS, mret = float(np.mean(list(pu.values()))), float(np.mean([R[k]["returning"] for k in R]))
    loso = {u: float(np.mean([v for w_, v in pu.items() if w_ != u])) for u in STRANGERS}
    mo = {k: float(np.mean([R[n]["orders"][k] for n in R])) for k in ("runs_perm", "in_run")}
    legs = {"i_mean_dS": mS > 0, "ii_6of8": sum(v >= 0 for v in pu.values()) >= 6, "iii_loso": all(v > 0 for v in loso.values()),
            "iv_returning": mret >= 0, "v_root2": R["d646"]["dS"] > 0 and R["d646"]["sealed"] > 0, "v_root3": R["d648"]["dS"] > 0 and R["d648"]["sealed"] > 0,
            "vi_runs_perm": mo["runs_perm"] >= 0, "vi_in_run": mo["in_run"] >= 0}
    inc = bool(all(legs.values()))
    out = dict(part="D655", included=inc, legs=legs, roots=R, mean_dS=mS, per_user=pu, n_ge0=int(sum(v >= 0 for v in pu.values())), loso=loso, mean_returning=mret,
               mean_old_sealed=float(np.mean([R[k]["sealed"] for k in R])), orders=mo, caveat="ONE panel read with FM-seed replication (shared CNN cell, student seeds, users)")
    SR.joinpath("decide_d655.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# T2-D655 — the FM share 0.40 vs 0.35 on W25-ALL6, three W25 ALL6 roots (ONE panel read with FM-seed replication)", "",
         f"**{'INCLUDED' if inc else 'NOT included'}** — legs {legs}", "",
         f"- three-root mean ΔS {mS:+.4f} ({out['n_ge0']} / 8 ≥ 0; per user {', '.join(f'{u} {v:+.4f}' for u, v in pu.items())}); LOSO min {min(loso.values()):+.4f}; "
         f"returning {mret:+.4f}; orders runs_perm {mo['runs_perm']:+.4f} / in_run {mo['in_run']:+.4f}; mean OLD sealed {out['mean_old_sealed']:+.4f}"]
    L += [f"- {k} (root {v['root']}): ΔS {v['dS']:+.4f}, returning {v['returning']:+.4f} (P04 {v['per_X']['P04']:+.4f}, P10 {v['per_X']['P10']:+.4f}), OLD sealed {v['sealed']:+.4f}, "
          f"orders {v['orders']}" for k, v in R.items()]
    SR.joinpath("RESULT_D655.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

def final():
    """RULE 3 (one stack; never a stack not read as a stack): --final [--stack-cut]; the stack read = setread/bw21/set_stack.json when present."""
    # the capacity re-plan (the manager 17:2x): W30 and D654 cannot be read as a stack (V fits with dumps are unaffordable), so >= 2 included parts
    # = the best single by mean OLD sealed (rule 3's fallback): stack_cut is ALWAYS on now
    return _final(SR, {k: _status(k) for k in ("W29", "W30", "D654")}, stack_read=None, stack_cut=True)


if __name__ == "__main__":
    a_ = sys.argv[1:]
    if "--final" in a_:
        final()
        sys.exit(0)
    if "--d655" in a_:
        d655()
    if "--d658" in a_:
        d658()
        sys.exit(0)
    if "--decide654" in a_:
        decide654()
        sys.exit(0)
    if "--decide649" in a_:
        decide649()
        sys.exit(0)
    if "--decide652" in a_:
        decide652()
        sys.exit(0)
    combine() if "--combine" in a_ else (decide648() if "--decide648" in a_ else (d650() if "--d650" in a_ else main()))

#!/usr/bin/env python
"""T2-D475 read (written before any number): common average reference (CAR) on the input of (A) the ST-L members and (B) the FBCNet member.
Both arms are paired with existing controls on the SAME seeds (ST-L: T2-D455 controls 472-485; FBCNet: T2-D419 control 324-335), so CAR is
the only difference. FBCNet files are first routed with t2p_fpcalib blend (r 1/3, tag 3, xtag __car) into outputs/t2_d475/route_fbc_car.
Shape C = T2-D459's (M7's weights; fresh CNN draws D419 / D429; D442 control FM). Typical = non-P04; kit + seen1, test + val; causal SK.
 A alone : three-net vote, CAR vs plain (12 draws).   A in C : all five FM members replaced by five CAR nets vs five plain nets (10 draws).
 B alone : one routed FBCNet, CAR vs plain (12).      B in C : the FBCNet slot, CAR vs plain (12 draws).
BAR (per arm, in C): typical >= +0.006, >= 9 / 12 (>= 8 / 10 for A), no typical user below -0.005. Alone is reported, not a bar.
    mem-guard python scripts/local/t2_d475_read.py [A|B]
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t2_cluster_ci import EXP, cell_rows  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
J = json.load(open(O / "t2_d442/spec.json"))
M = dict(J["members"])
for d, names in (("t2_d419", {"fbc_e180": "fbc_tv8c"}), ("t2_d429", {"sh_ctl": "sh_tv8c", "ifn_ctl": "ifn"})):
    Sp = json.load(open(O / d / "spec.json"))["members"]
    for src, dst in names.items():
        M[dst] = Sp[src]
C = J["arms"]["M7f"]
FM5 = [k for k in C if k.startswith("n_")]
idx = pd.read_csv(EXP / "index.csv")
key = ["proto", "split", "i", "cell"]
arms_to_read = sys.argv[1:] or ["A", "B"]


def stl(pat):
    out = {}
    for pr in ("kit", "seen1"):
        ds = glob.glob(f"{S}/{pat.format(pr=pr)}-[0-9]*")
        assert len(ds) == 1, (pat, pr, ds)
        f = glob.glob(f"{ds[0]}/runs/{pr}__fm-*__s472*.npz")
        assert len(f) == 1, f
        out[pr] = f[0].replace("__s472.npz", "__s{s}.npz")
    return out


def compare(spec, a, b, n, protos=("kit", "seen1")):
    R = {x: cell_rows(spec, x, n, idx, protos) for x in (a, b)}
    m = R[a].merge(R[b][key + ["ba"]], on=key, suffixes=("", "_b"))
    t = m[m.subj != "P04"]
    d = (t.ba - t.ba_b).groupby(t.i).mean()
    u = (m.ba - m.ba_b).groupby(m.subj).mean()
    return f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{n}) level {t.ba.mean():.3f} vs {t.ba_b.mean():.3f}  " + " ".join(f"{s} {v:+.4f}" for s, v in u.items())


if "A" in arms_to_read:
    pc, pp = stl("ni26-t2p-d475-stlcar-{pr}"), stl("ni26-t2p-d455-ctl-{pr}")
    for k in range(5):
        M[f"car{k}"] = {"offset": 472 + k, **pc}
        M[f"pl{k}"] = {"offset": 472 + k, **pp}
    arms = {"A_car3": {f"car{k}": 1 / 3 for k in range(3)}, "A_pl3": {f"pl{k}": 1 / 3 for k in range(3)},
            "A_C_car": {**{k: v for k, v in C.items() if k not in FM5}, **{f"car{k}": C[FM5[k]] for k in range(5)}},
            "A_C_pl": {**{k: v for k, v in C.items() if k not in FM5}, **{f"pl{k}": C[FM5[k]] for k in range(5)}}}
    spec = {"members": M, "arms": arms}
    print("A ST-L alone (3-net vote), CAR vs plain:", compare(spec, "A_car3", "A_pl3", 12), flush=True)
    print("A ST-L in C (all five FM members), CAR vs plain:", compare(spec, "A_C_car", "A_C_pl", 10), flush=True)
if "early" in arms_to_read:
    # EARLY LOOK (the manager's option 2; written 2026-11-13 before any number): kit only, seeds 472-476 trained on the local 5090
    # (scripts/local/t2_d475_stl_early.sh) vs T2-D455's H200 controls on the same seeds. DESCRIPTIVE, not the verdict (the sofia pair is).
    # Rule: single-net alone <= 0 on all five seeds AND the in-C draw <= 0 -> say so, stop expecting arm A; in-C draw >= +0.006 with
    # >= 4 / 5 seeds alone > 0 -> prepare the M7-CAR build spec while the sofia jobs wait; anything else -> wait for the sofia pair.
    ds = glob.glob(f"{S}/ni26-t2p-d475-stlcar-kitearly-5090/runs/kit__fm-*__s472.npz")
    assert len(ds) == 1, ds
    pc, pp = {"kit": ds[0].replace("__s472.npz", "__s{s}.npz")}, stl("ni26-t2p-d455-ctl-{pr}")
    for k in range(5):
        M[f"car{k}"] = {"offset": 472 + k, **pc}
        M[f"pl{k}"] = {"offset": 472 + k, **pp}
    arms = {"E_car1": {"car0": 1.0}, "E_pl1": {"pl0": 1.0}, "E_car3": {f"car{k}": 1 / 3 for k in range(3)}, "E_pl3": {f"pl{k}": 1 / 3 for k in range(3)},
            "E_C_car": {**{k: v for k, v in C.items() if k not in FM5}, **{f"car{k}": C[FM5[k]] for k in range(5)}},
            "E_C_pl": {**{k: v for k, v in C.items() if k not in FM5}, **{f"pl{k}": C[FM5[k]] for k in range(5)}}}
    spec = {"members": M, "arms": arms}
    print("EARLY kit, one net alone, CAR(5090) vs plain(H200), seeds 472-476:", compare(spec, "E_car1", "E_pl1", 5, ("kit",)), flush=True)
    print("EARLY kit, three-net vote alone (3 draws):", compare(spec, "E_car3", "E_pl3", 3, ("kit",)), flush=True)
    print("EARLY kit, in C (all five FM members, 1 draw):", compare(spec, "E_C_car", "E_C_pl", 1, ("kit",)), flush=True)
if any(a_ in arms_to_read for a_ in ("A13", "A13r", "A13n", "F4", "A13L", "A13Lr", "A13c", "A13cr", "A13j", "A13w", "A13m", "A13mr", "A13k", "A13e", "A13x", "A13lite")):
    # T2-D475d (the manager's addition, registered 2026-11-13 before the sofia pair's numbers; zero training): is CAR a second INPUT VIEW?
    # Every arm inside C with W13's decode (per-family causal SKλ, each family with its own start vector cross-fitted by T2-D476b's rule:
    # mean of the family's normalised mix over the OTHER split, all draws + protocols, per arm), 10 draws, ST-L seed of member k in draw i
    # = 472 + (i + k) mod 14 (the pair's and D455's 14 seeds; distinct seeds within a draw, as a build would train them).
    #   pl     : five plain (the base)               car5 : five CAR (all swapped)
    #   m32    : 3 plain + 2 CAR (equal runtime)     m23  : 2 plain + 3 CAR (equal runtime)
    #   add    : five plain + five CAR (CAR k at seed index i + 5 + k), FM family weight unchanged (0.35 / 10 each; + ~560 s, fits 3,400 s)
    # CAR INSIDE the fm family is the bar form: it is what the 4-slot W13 template decodes, and it isolates the input view; a separate CAR
    # family would also split the FM decode in two, which a 3 + 2 split of plain nets would do too. Descriptive: m32o / m23o / addo = the
    # same nets with CAR as its own family (weight = its nets' sum, own start vector); pl10 = ten plain nets (the 'more nets' control for add).
    # BAR (each of car5, m32, m23, add vs pl): typical >= +0.006, >= 8 / 10, no typical user below -0.005 (D475's). If several pass, the
    # largest typical gain, except that add must beat the best passing equal-runtime form by >= +0.002 (it doubles FM runtime); the pick is
    # reported with the four-form multiplicity. Any pass needs CAR inside the template for the CAR members = a template change (user's line).
    import numpy as np
    from t2_cluster_ci import causal_sinkhorn
    import os
    FAKE = bool(os.environ.get("D475_FAKE_CAR"))     # code check before the pair lands: plain files stand in for CAR, so car5 / m32 / m23 = pl
    pp = stl("ni26-t2p-d455-ctl-{pr}")              # and add = pl10 exactly (only the own-family variants may differ)
    pc = pp if FAKE else stl("ni26-t2p-d475-stlcar-{pr}")
    w_fm = sum(C[m] for m in FM5)
    CNN = [("fbc", "fbc_tv8c"), ("sh", "sh_tv8c"), ("ifn", "ifn")]

    def st(k):
        return lambda i: 472 + (i + k) % 14

    def arm13(n_pl, car_ks, own=False, n_tot=None):
        n_tot = n_tot or (n_pl + len(car_ks))
        w = w_fm / n_tot
        mem = [(f, C[m], ("M", m)) for f, m in CNN]
        mem += [("fm", w, ("pl", st(k))) for k in range(n_pl)]
        mem += [("car" if own else "fm", w, ("car", st(k))) for k in car_ks]
        return mem
    A13 = {"pl": arm13(5, []), "car5": arm13(0, range(5)), "m32": arm13(3, [3, 4]), "m23": arm13(2, [2, 3, 4]),
           "add": arm13(5, range(5, 10)), "m32o": arm13(3, [3, 4], True), "m23o": arm13(2, [2, 3, 4], True),
           "addo": arm13(5, range(5, 10), True), "pl10": arm13(10, []),
           # T2-D475f (the manager's ask, registered 2026-11-14 before its numbers): smaller adds at less runtime, same files, zero training
           "add3": arm13(5, range(5, 8)), "add2": arm13(5, range(5, 7))}
    N13 = 10
    F4 = "F4" in arms_to_read
    LAPM = "A13L" in arms_to_read or "A13Lr" in arms_to_read      # T2-D480 (registered 2026-11-14 before its numbers)
    CCPT = "A13c" in arms_to_read or "A13cr" in arms_to_read      # T2-D484 (registered 2026-11-14 06:27 before its numbers)
    LJM = "A13j" in arms_to_read or "A13k" in arms_to_read or "A13e" in arms_to_read   # T2-D486 (registered 2026-11-14 11:18 before its numbers)
    N0M = os.environ.get("D494_N0", "") if "A13e" in arms_to_read else ""   # T2-D494 (registered 2026-11-14 before its numbers): EA prior n0
    KDM = "A13k" in arms_to_read                                    # T2-D490 (registered 2026-11-14 before its numbers): D486's arms, kd nets
    WPM = "A13w" in arms_to_read                                    # T2-D487 (registered 2026-11-14 11:24 before its numbers)
    MIM = "A13m" in arms_to_read or "A13mr" in arms_to_read      # T2-D488 (registered 2026-11-14 16:36 before its numbers)
    XFM = os.environ.get("D492_FM", "") if "A13x" in arms_to_read else ""   # T2-D492 (registered 2026-11-14 before its numbers)
    LITE = "A13lite" in arms_to_read                                # T2-D496 (registered 2026-11-14 before its numbers): W14-lite
    REP ="A13r" in arms_to_read or "A13n" in arms_to_read or F4 or "A13Lr" in arms_to_read or "A13cr" in arms_to_read or "A13mr" in arms_to_read
    NETS = "A13n" in arms_to_read
    C2P, W4 = {}, (0.03, 0.05, 0.08, 0.12)
    POOL = {}
    if NETS:
        # T2-D475g (registered 2026-11-14 00:26 before its numbers): k CNN nets per family from a 24-seed pool, seed j of draw i = pool[(2i + j) mod 24]
        for fam, srcs in (("fbc", [("t2_d419/spec.json", "fbc_e180"), ("t2_d430/spec_member_fbc.json", "fbc_ctl")]),
                          ("sh", [("t2_d429/spec.json", "sh_ctl"), ("t2_d424/spec_member.json", "sh_ctl")]),
                          ("ifn", [("t2_d429/spec.json", "ifn_ctl"), ("t2_d408/spec_m7.json", "ifn")])):
            POOL[fam] = []
            for f, k in srcs:
                mm = json.load(open(O / f))["members"][k]
                POOL[fam] += [(mm, mm["offset"] + q) for q in range(12)]
        famw = {f: C[m] for f, m in CNN}

        def cnn(kn):
            return [(f, famw[f] / kn, ("P", f, j)) for f, _ in CNN for j in range(kn)]

        def fmk(n_pl, n_car):
            w = w_fm / (n_pl + n_car)
            return [("fm", w, ("pl", st(k))) for k in range(n_pl)] + [("fm", w, ("car", st(k))) for k in range(5, 5 + n_car)]
        A13 = {"ref": cnn(15) + fmk(5, 0), "a": cnn(15) + fmk(5, 3), "b": cnn(10) + fmk(5, 5), "c": cnn(10) + fmk(5, 3), "d": cnn(15) + fmk(5, 5)}
    elif F4:
        # T2-D479 (panel P11 F4, registered 2026-11-14 02:02 before its numbers): the C2L300 nets (T2-D463 raw / ann) as a fifth family
        for kind in ("raw", "ann"):
            C2P[kind] = {}
            for pr in ("kit", "seen1"):
                f_ = glob.glob(f"{S}/ni26-t2p-d463-{kind}-{pr}-[0-9]*/runs/{pr}__fm-*__s500.npz")
                assert len(f_) == 1, f_
                C2P[kind][pr] = f_[0].replace("__s500.npz", "__s{s}.npz")

        def scaled(mem, sc):
            return [(f, w * sc, src) for f, w, src in mem]
        bases = {"pl": arm13(5, []), "add3": arm13(5, range(5, 8))}
        A13 = dict(bases)
        for kind in ("raw", "ann"):
            for b_, mem in bases.items():
                for w in W4:
                    A13[f"{b_}_{kind}_{w}"] = scaled(mem, 1 - w) + [("c2", w / 3, ("C2", kind, j)) for j in range(3)]
    elif REP:
        # T2-D475e (registered 2026-11-14 before its numbers; zero training): the replicate of the passing add form before shipping. Other CNN
        # draws (T2-D476c's set F: FBCNet D430 control, ShallowNet D424 control, IFNet M7's stored nets) and other plain FM nets (T2-D442's five
        # shape-C members, the n_* files) replace the A13 base's; the CAR nets are the pair's (reused, seed index i + k). W13 decode as A13.
        #   rpl : F's CNNs + D442's five plain (M7's weights)     radd : rpl's plain at 0.035 each + five CAR at 0.035 each
        #   rpl10 (descriptive) : rpl's plain + five D455 plain (seed index i + k), all at 0.035 = the 'more nets' control
        # BAR (radd vs rpl): typical >= +0.003 with >= 8 / 10, no typical user below -0.005.
        M["fbc_tv8c"] = json.load(open(O / "t2_d430/spec_member_fbc.json"))["members"]["fbc_ctl"]
        M["sh_tv8c"] = json.load(open(O / "t2_d424/spec_member.json"))["members"]["sh_ctl"]
        M["ifn"] = json.load(open(O / "t2_d408/spec_m7.json"))["members"]["ifn"]
        base = [(f, C[m], ("M", m)) for f, m in CNN]
        A13 = {"rpl": base + [("fm", C[m], ("M", m)) for m in FM5],
               "radd": base + [("fm", w_fm / 10, ("M", m)) for m in FM5] + [("fm", w_fm / 10, ("car", st(k))) for k in range(5)],
               "rpl10": base + [("fm", w_fm / 10, ("M", m)) for m in FM5] + [("fm", w_fm / 10, ("pl", st(k))) for k in range(5)],
               "radd3": base + [("fm", w_fm / 8, ("M", m)) for m in FM5] + [("fm", w_fm / 8, ("car", st(k))) for k in range(3)],
               "radd2": base + [("fm", w_fm / 7, ("M", m)) for m in FM5] + [("fm", w_fm / 7, ("car", st(k))) for k in range(2)]}
    if LAPM:
        # T2-D480: the Laplacian nets (four sofia jobs per protocol, seeds 472-485); Lap member at seed index i + 7, the CAR ones i + 5, i + 6
        LAPF = {}
        for pr in ("kit", "seen1"):
            if os.environ.get("D480_FAKE_LAP"):        # code check before the nets land: the CAR files stand in -> m21L = add3L = add3 exactly
                LAPF[pr] = {sd: pc[pr].format(s=sd) for sd in range(472, 486)}
            else:
                LAPF[pr] = {int(f_.rsplit("__s", 1)[1][:-4]): f_ for f_ in
                            glob.glob(f"{S}/ni26-t2p-d480-stllap-{pr}-[0-9]-[0-9]*/runs/{pr}__fm-*__lap__s*.npz")}
            assert sorted(LAPF[pr]) == list(range(472, 486)), (pr, sorted(LAPF[pr]))
        base = [(f, C[m], ("M", m)) for f, m in CNN]          # A13Lr: M already holds F's CNNs (the REP block above)
        plain = [("M", m) for m in FM5] if "A13Lr" in arms_to_read else [("pl", st(k)) for k in range(5)]
        nm = "r" if "A13Lr" in arms_to_read else ""
        w8 = w_fm / 8
        A13 = {f"{nm}pl": base + [("fm", C[FM5[0]], src) for src in plain],
               f"{nm}add3": base + [("fm", w8, src) for src in plain] + [("fm", w8, ("car", st(k))) for k in (5, 6, 7)],
               f"{nm}m21L": base + [("fm", w8, src) for src in plain] + [("fm", w8, ("car", st(k))) for k in (5, 6)] + [("fm", w8, ("lap", st(7)))],
               f"{nm}add3L": base + [("fm", w8, src) for src in plain] + [("fm", w8, ("lap", st(k))) for k in (5, 6, 7)]}
    if CCPT:
        # T2-D484: the CAR members fine-tuned from the CAR-view CPT (four sofia parts per protocol, seeds 472-485) against T2-D475's CAR members
        # from the plain CPT at the SAME seed indices: A13 set CAR at i + 5 .. (add3: 5-7, add5: 5-9); A13r set CAR at i + 0 .. (radd3 / radd5).
        # Alone (descriptive): three CAR nets as the only family under W13's decode (indices i .. i + 2), CAR-CPT vs plain-CPT.
        # D484_FAKE=1 (code check before the parts land): the plain-CPT CAR files stand in -> every c arm = its plain-CPT twin exactly.
        CARC = {}
        for pr in ("kit", "seen1"):
            if os.environ.get("D484_FAKE"):
                CARC[pr] = {sd: pc[pr].format(s=sd) for sd in range(472, 486)}
            else:
                CARC[pr] = {int(f_.rsplit("__s", 1)[1][:-4]): f_ for f_ in
                            glob.glob(f"{S}/ni26-t2p-d484-stlcarcpt-{pr}-[0-9]-[0-9]*/runs/{pr}__fm-*__car__s*.npz")}
            assert sorted(CARC[pr]) == list(range(472, 486)), (pr, sorted(CARC[pr]))
        base = [(f, C[m], ("M", m)) for f, m in CNN]          # A13cr: M already holds F's CNNs (the REP block above)
        plain = [("M", m) for m in FM5] if "A13cr" in arms_to_read else [("pl", st(k)) for k in range(5)]
        nm = "r" if "A13cr" in arms_to_read else ""
        c0 = 0 if nm else 5
        A13 = {}
        for n_c in (3, 5):
            w = w_fm / (5 + n_c)
            for kind, tag in (("car", ""), ("carc", "c")):
                A13[f"{nm}add{n_c}{tag}"] = base + [("fm", w, src) for src in plain] + [("fm", w, (kind, st(c0 + k))) for k in range(n_c)]
        for kind, tag in (("car", ""), ("carc", "c")):
            A13[f"{nm}alone3{tag}"] = [("fm", 1 / 3, (kind, st(k))) for k in range(3)]
    if MIM:
        # T2-D488: ONE MI-expert ST-L in the five, in W14's shape (5 plain + 3 CAR, each w_fm / 8). T2-D455's MI nets (seeds 472-485) pair with its
        # controls on the same seeds, which are the A13 plain nets (pp), so a swap at one seed index isolates the MI first stage.
        #   A13 set : add3 (plain at i .. i+4, CAR at i+5 .. i+7) vs mi1 (the plain net at index i -> the MI net at that seed); mi2 descriptive
        #   A13r set: rctl1 = F's CNNs + D442's plain 2..5 + a D455 control at index i+3 + CAR at i .. i+2, vs rmi1 = the MI net at that seed
        # D488_FAKE=1 (code check): the MI entries point at the controls -> mi1 = mi2 = add3 and rmi1 = rctl1 exactly.
        pm = pp if os.environ.get("D488_FAKE") else stl("ni26-t2p-d455-mi-{pr}")
        base = [(f, C[m], ("M", m)) for f, m in CNN]          # A13mr: M already holds F's CNNs (the REP block above)
        w8 = w_fm / 8

        def car3(c0):
            return [("fm", w8, ("car", st(c0 + k))) for k in range(3)]
        if REP:
            rest = [("fm", w8, ("M", m)) for m in FM5[1:]]
            A13 = {"rctl1": base + rest + [("fm", w8, ("pl", st(3)))] + car3(0), "rmi1": base + rest + [("fm", w8, ("mi", st(3)))] + car3(0)}
        else:
            A13 = {"add3": base + [("fm", w8, ("pl", st(k))) for k in range(5)] + car3(5),
                   "mi1": base + [("fm", w8, ("mi", st(0)))] + [("fm", w8, ("pl", st(k))) for k in range(1, 5)] + car3(5),
                   "mi2": base + [("fm", w8, ("mi", st(k))) for k in range(2)] + [("fm", w8, ("pl", st(k))) for k in range(2, 5)] + car3(5)}
    if LJM:
        # T2-D486: the CNN slots' nets trained on 4-s crops at a random start in [0.5, 2.5] s re cue (the long export), paired with the A13 base's
        # controls on the same seeds (FBCNet D419 offset 324, ShallowNet / IFNet D429 offset 348), routed as those were; FM = W14's 5 + 3 (jW_*)
        # or five plain (j5_*). D486_FAKE=1 (code check before the nets land): the lj entries point at the controls -> every arm = its ref.
        LJF = {"fbc_lj": ("t2_d486/route_fbc_lj/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__lj05-25__s{{s}}.npz", 324),
               "sh_lj": ("t2_d486/route_sh_lj/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__lj05-25__s{{s}}.npz", 348),
               "ifn_lj": ("t2_d486/route_ifn_lj/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2__lj05-25__s{{s}}.npz", 348)}
        if KDM:    # T2-D490: the same slots, nets trained with 0.5 CE + 0.5 KL to out-of-fold cross-user soft targets (routed as the controls)
            LJF = {"fbc_lj": ("t2_d490/route_fbc_kd/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75_kdtv8cfb3__s{{s}}.npz", 324),
                   "sh_lj": ("t2_d490/route_sh_kd/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdtv8cfb2__s{{s}}.npz", 348),
                   "ifn_lj": ("t2_d490/route_ifn_kd/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vr_kdtv8cfb2__s{{s}}.npz", 348)}
        if KDM and os.environ.get("D490R"):   # T2-D490r: ShallowNet only, on T2-D423 / D424's control seeds 336-347 (its slot's base = that control)
            M["sh_tv8c"] = dict(json.load(open(O / "t2_d424/spec_member.json"))["members"]["sh_ctl"])
            assert M["sh_tv8c"]["offset"] == 336, M["sh_tv8c"]
            LJF = {"fbc_lj": (None, 324),
                   "sh_lj": ("t2_d490r/route_sh_kd/{p}__shallow__ea__n00__r1_pulx_vp2_e180_vr_kdtv8cfb2__s{{s}}.npz", 336),
                   "ifn_lj": (None, 348)}
        if os.environ.get("D504") and not KDM:   # T2-D504: only the IFNet slot, its copies re-blended at r 1 (FBCNet / ShallowNet = their controls)
            LJF = {"fbc_lj": (None, 324), "sh_lj": (None, 348),
                   "ifn_lj": ("t2_d504/route_ifn_r1/{p}__ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb1__s{{s}}.npz", 348)}
        if N0M:    # T2-D494: ShallowNet / IFNet at EA prior n0 = D494_N0 windows (FBCNet unaligned: its slot keeps the control)
            assert N0M in ("10", "30"), N0M
            LJF = {"fbc_lj": (None, 324),
                   "sh_lj": (f"t2_d494/route_sh_n0{N0M}/{{p}}__shallow__ea__n0{N0M}__r1_pulx_vp2_e180_vrtv8cfb2__s{{{{s}}}}.npz", 348),
                   "ifn_lj": (f"t2_d494/route_ifn_n0{N0M}/{{p}}__ifnet__ea__n0{N0M}__r1_pulx_vp2_e180_vrtv8cfb2__s{{{{s}}}}.npz", 348)}
        for (k_, (pat, o_)), (_, m_) in zip(LJF.items(), CNN):
            M[k_] = dict(M[m_]) if os.environ.get("D486_FAKE") or os.environ.get("D490_FAKE") or os.environ.get("D494_FAKE") or pat is None else {"offset": o_, **{p: str(O / pat.format(p=p)) for p in ("kit", "seen1")}}
            assert M[k_]["offset"] == M[m_]["offset"], (k_, M[k_]["offset"], M[m_]["offset"])
        LJ_OF = {"fbc_tv8c": "fbc_lj", "sh_tv8c": "sh_lj", "ifn": "ifn_lj"}
        w8 = w_fm / 8
        FMW = [("fm", w8, ("pl", st(k))) for k in range(5)] + [("fm", w8, ("car", st(k))) for k in (5, 6, 7)]
        FMP = [("fm", w_fm / 5, ("pl", st(k))) for k in range(5)]

        def cnn_j(swap):
            return [(f, C[m], ("M", LJ_OF[m] if f in swap else m)) for f, m in CNN]
        A13 = {"jW_ref": cnn_j(()) + FMW, "jW_all": cnn_j(("fbc", "sh", "ifn")) + FMW, "jW_fbc": cnn_j(("fbc",)) + FMW,
               "jW_sh": cnn_j(("sh",)) + FMW, "jW_ifn": cnn_j(("ifn",)) + FMW, "j5_ref": cnn_j(()) + FMP, "j5_all": cnn_j(("fbc", "sh", "ifn")) + FMP}
        for f, m in CNN:
            A13[f"a_{f}"] = [(f, 1.0, ("M", m))]
            A13[f"a_{f}lj"] = [(f, 1.0, ("M", LJ_OF[m]))]
    if WPM:
        # T2-D487: the lagged-coupling member (imaginary coherency; wPLI twin, descriptive), seed-free (one fit per protocol, the same file at
        # every draw), as its own family at w taken from the CNN share, on A13's base with W14's FM (5 plain + 3 CAR). D487_FAKE=1 (code check):
        # the member's file = the A13 base's ShallowNet file of seed 348 -> the reader runs end to end on a known member.
        for k_, kind in (("wp", "imcoh"), ("wpl", "wpli")):
            M[k_] = ({"offset": 0, **{p: M["sh_tv8c"][p].format(s=348) for p in ("kit", "seen1")}} if os.environ.get("D487_FAKE")
                     else {"offset": 0, **{p: str(O / f"t2_d487/{p}__{kind}.npz") for p in ("kit", "seen1")}})
        w8 = w_fm / 8
        FMW = [("fm", w8, ("pl", st(k))) for k in range(5)] + [("fm", w8, ("car", st(k))) for k in (5, 6, 7)]
        cnn_share = sum(C[m] for _, m in CNN)
        WW = (0.03, 0.05, 0.08, 0.12)

        def with_wp(w, k_="wp"):
            return [(f, C[m] * (cnn_share - w) / cnn_share, ("M", m)) for f, m in CNN] + FMW + ([("wp", w, ("M", k_))] if w > 0 else [])
        A13 = {"jW_ref": with_wp(0.0), **{f"w_{w}": with_wp(w) for w in WW}, **{f"v_{w}": with_wp(w, "wpl") for w in WW},
               "a_wp": [("wp", 1.0, ("M", "wp"))], "a_wpl": [("wp", 1.0, ("M", "wpl"))], **{f"a_{f}": [(f, 1.0, ("M", m))] for f, m in CNN}}
    if XFM:
        # T2-D492: the CPT'd REVE / LaBraM as its OWN family of five members (member k of draw i = seed 424 + (i + k) mod 12) added to W14's
        # shape (A13's CNNs + 5 plain + 3 CAR ST-L) at weight w, every other family x (1 - w); w = 0.08 is the bar, 0.03 / 0.05 / 0.12
        # descriptive. Files = the three part dirs per protocol (ni26-t2p-d492-<fm>-<proto>-<j>-<jobid>); the paired controls = T2-D452's
        # no-CPT files on the same seeds. D492_FAKE=1 (code check before the CPT'd members land): the CPT'd files = D452's.
        assert XFM in ("reve", "labram"), XFM
        ctl_dir = {"reve": "ni26-t2p-d452-reve-none-{pr}-[0-9]*", "labram": "ni26-t2p-d452-labram-sz-{pr}-[0-9]*"}[XFM]

        def xfiles(pat):
            out = {}
            for pr in ("kit", "seen1"):
                fs = sorted(glob.glob(f"{S}/{pat.format(pr=pr)}/runs/{pr}__fm-{XFM}-mean__*__s4[23][0-9].npz"))
                out[pr] = {int(f.rsplit("__s", 1)[1][:-4]): f for f in fs}
                assert sorted(out[pr]) == list(range(424, 436)), (pat, pr, sorted(out[pr]))
            return out
        XC = xfiles(ctl_dir)
        XF = XC if os.environ.get("D492_FAKE") else xfiles(f"ni26-t2p-d492-{XFM}-{{pr}}-[0-2]-[0-9]*")
        w8 = w_fm / 8
        FMW = [("fm", w8, ("pl", st(k))) for k in range(5)] + [("fm", w8, ("car", st(k))) for k in (5, 6, 7)]
        base = [(f, C[m], ("M", m)) for f, m in CNN] + FMW
        WX = (0.03, 0.05, 0.08, 0.12)

        def xs(k):
            return lambda i: 424 + (i + k) % 12

        def with_x(w, src="X"):
            return [(f, v * (1 - w), s_) for f, v, s_ in base] + ([("xfm", w / 5, (src, xs(k))) for k in range(5)] if w > 0 else [])
        A13 = {"x_ref": with_x(0.0), **{f"x_{w}": with_x(w) for w in WX}, **{f"c_{w}": with_x(w, "XC") for w in WX},
               "a_x5": [("xfm", 0.2, ("X", xs(k))) for k in range(5)], "a_c5": [("xfm", 0.2, ("XC", xs(k))) for k in range(5)],
               "a_x1": [("xfm", 1.0, ("X", xs(0)))], "a_c1": [("xfm", 1.0, ("XC", xs(0)))]}
    if LITE:
        # T2-D496: member SUBSETS of W14's FM family (5 plain at seed index i .. i+4 + 3 CAR at i+5 .. i+7), each subset an equal-weight
        # mean (the template's rule) at the FM family's weight: k plain (the first k) + the 3 CAR, and 2 plain + 2 CAR; w13 = the 5 plain.
        def lite(pl_ks, car_ks):
            n = len(pl_ks) + len(car_ks)
            return ([(f, C[m], ("M", m)) for f, m in CNN] + [("fm", w_fm / n, ("pl", st(k))) for k in pl_ks]
                    + [("fm", w_fm / n, ("car", st(k))) for k in car_ks])
        A13 = {"w14": lite(range(5), (5, 6, 7)), "w13": lite(range(5), ()), "k1c3": lite((0,), (5, 6, 7)), "k2c3": lite((0, 1), (5, 6, 7)),
               "k3c3": lite((0, 1, 2), (5, 6, 7)), "k4c3": lite((0, 1, 2, 3), (5, 6, 7)), "k2c2": lite((0, 1), (5, 6))}
    keys_ = idx.stream.to_numpy(), idx.order.to_numpy()
    MIX = {}
    for proto in ("kit", "seen1"):
        for i in range(N13):
            cache = {}

            def get(src):
                kind, x = src[0], src[1]
                if kind == "lap":                                  # T2-D480: the Laplacian net of this seed
                    f = LAPF[proto][x(i)]
                elif kind == "carc":                               # T2-D484: the CAR-CPT CAR net of this seed
                    f = CARC[proto][x(i)]
                elif kind in ("X", "XC"):                          # T2-D492: the CPT'd FM net (X) / its D452 control (XC) of this seed
                    f = (XF if kind == "X" else XC)[proto][x(i)]
                elif kind == "mi":                                 # T2-D488: the D455 MI-expert net of this seed
                    f = pm[proto].format(s=x(i))
                elif kind == "C2":                                   # T2-D479: C2L300 seed 500 + (i + j) mod 14
                    f = C2P[x][proto].format(s=500 + (i + src[2]) % 14)
                elif kind == "P":                                  # T2-D475g: the pooled CNN seed j of draw i
                    mm, sd = POOL[x][(2 * i + src[2]) % 24]
                    f = mm[proto].format(s=sd)
                else:
                    f = M[x][proto].format(s=M[x].get("offset", 0) + i) if kind == "M" else (pp if kind == "pl" else pc)[proto].format(s=x(i))
                if f not in cache:
                    z = np.load(f)
                    cache[f] = {"test": (z["test_idx"], z["prob_last"].astype(np.float64)), "val": (z["val_idx"], z["prob_val_last"].astype(np.float64))}
                return cache[f]
            for a, mem in A13.items():
                for split in ("test", "val"):
                    ix, fam, fw = None, {}, {}
                    for f, w, src in mem:
                        ix_, p = get(src)[split]
                        assert ix is None or (len(ix) == len(ix_) and (ix == ix_).all())
                        ix, fam[f], fw[f] = ix_, fam.get(f, 0.0) + w * p, fw.get(f, 0.0) + w
                    MIX[(a, proto, i, split)] = (ix, {f: v / v.sum(1, keepdims=True) for f, v in fam.items()}, fw)
        print(f"A13 {proto} loaded", flush=True)
    assert all(abs(sum(v[2].values()) - 1.0) < 1e-9 for v in MIX.values())
    VEC = {}
    for a in A13:
        for sp in ("test", "val"):
            other = [v for k, v in MIX.items() if k[0] == a and k[3] != sp]
            VEC[(a, sp)] = {f: np.concatenate([o[1][f] for o in other]).mean(0) for f in other[0][1]}
    rows = []
    for (a, proto, i, split), (ix, mx, fw) in MIX.items():
        kk = (keys_[0][ix], keys_[1][ix])
        P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
        y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
        for c in np.unique(cells):
            mk = cells == c
            rows.append(dict(arm=a, proto=proto, split=split, i=i, cell=c, subj=subj[mk][0],
                             ba=np.mean([np.mean(P[mk][y[mk] == k] == k) for k in np.unique(y[mk])])))
    R13 = pd.DataFrame(rows)
    (O / "t2_d475").mkdir(exist_ok=True)
    R13.to_csv(O / ("t2_d475/cells_" + ("a13m" + ("r" if REP else "") if MIM else "a13lite" if LITE else f"a13x{XFM}" if XFM else "a13w" if WPM else ("a13k" + ("r" if os.environ.get("D490R") else "") if KDM else "a13j504" if os.environ.get("D504") else f"a13e{N0M}" if N0M else "a13j") if LJM else "a13c" + ("r" if REP else "") if CCPT else "a13L" + ("r" if REP else "") if LAPM else "f4" if F4
                                       else "a13" + ("n" if NETS else "r" if REP else "")) + ("_fake" if FAKE or os.environ.get("D484_FAKE") or os.environ.get("D486_FAKE") or os.environ.get("D487_FAKE") or os.environ.get("D488_FAKE") or os.environ.get("D490_FAKE") or os.environ.get("D494_FAKE") or os.environ.get("D492_FAKE") else "") + ".csv"), index=False)

    def cmp13(a, b, bar=0.006):
        m = R13[R13.arm == a].merge(R13[R13.arm == b][key + ["ba"]], on=key, suffixes=("", "_b"))
        t = m[m.subj != "P04"]
        d = (t.ba - t.ba_b).groupby(t.i).mean()
        u = (m.ba - m.ba_b).groupby(m.subj).mean()
        ut = u.drop("P04", errors="ignore")
        ok = d.mean() >= bar and (d > 0).sum() >= 8 and ut.min() >= -0.005
        return (f"typical {d.mean():+.4f} ({int((d > 0).sum())}/{N13}) level {t.ba.mean():.4f} vs {t.ba_b.mean():.4f}  "
                + " ".join(f"{s} {v:+.4f}" for s, v in u.items()) + f"  -> {'PASS' if ok else 'no'}")
    if WPM:                              # T2-D487: alone, error correlation with the stack, and the cross-fitted in-stack bar
        ORD_W = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)

        def decoded(a, shuf=False):
            out = {}
            for (a_, proto, i, split), (ix, mx, fw) in MIX.items():
                if a_ == a:
                    kk = (keys_[0][ix], (ORD_W if shuf else keys_[1])[ix])
                    out[(proto, i, split)] = (ix, sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1))
            return out

        def cell_level(dec, splits=("test", "val"), raw=False):
            v = []
            for (proto, i, split), (ix, P) in dec.items():
                if split not in splits:
                    continue
                y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                v += [np.mean([np.mean(P[cells == c][y[cells == c] == k] == k) for k in np.unique(y[cells == c])]) for c in np.unique(cells) if subj[cells == c][0] != "P04"]
            return v if raw else float(np.mean(v))
        for a in ("a_wp", "a_wpl", "a_fbc", "a_sh", "a_ifn"):
            print(f"D487 alone {a}: typical level {R13[(R13.arm == a) & (R13.subj != 'P04')].ba.mean():.4f}  per user "
                  + " ".join(f"{u} {v:.3f}" for u, v in R13[R13.arm == a].groupby('subj').ba.mean().items()), flush=True)
        dref = decoded("jW_ref")
        for a in ("a_wp", "a_wpl", "a_fbc", "a_sh", "a_ifn"):
            da, cs = decoded(a), []
            for kd_, (ix, P) in dref.items():            # (not `key`: cmp13 merges on the global `key`)
                y, subj = idx.y.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                m_ = subj != "P04"
                e1, e2 = (P != y)[m_].astype(float), (da[kd_][1] != y)[m_].astype(float)
                cs.append(np.corrcoef(e1, e2)[0, 1])
            print(f"D487 error correlation with the stack (jW_ref), typical windows: {a} {np.mean(cs):+.3f}", flush=True)
        for fam_, pre in (("imcoh (the bar)", "w"), ("wpli (descriptive)", "v")):
            parts, wsel = [], {}
            for sp, other in (("test", "val"), ("val", "test")):
                lv = {w: R13[(R13.arm == f"{pre}_{w}") & (R13.split == other) & (R13.subj != "P04")].ba.mean() for w in WW}
                wsel[sp] = max(lv, key=lv.get)
                parts.append(R13[(R13.arm == f"{pre}_{wsel[sp]}") & (R13.split == sp)])
            R13 = pd.concat([R13, pd.concat(parts).assign(arm=f"{pre}_cf")])
            line = cmp13(f"{pre}_cf", "jW_ref", 0.003)
            print(f"D487 {fam_}: w* test {wsel['test']} / val {wsel['val']}: {pre}_cf vs jW_ref:", line if pre == "w" else line.replace("-> PASS", "").replace("-> no", ""), flush=True)
            if pre == "w":
                sh_cf = float(np.mean(sum((cell_level(decoded(f"w_{wsel[sp]}", True), (sp,), raw=True) for sp in ("test", "val")), [])))
                lose_cf = R13[(R13.arm == "w_cf") & (R13.subj != "P04")].ba.mean() - sh_cf
                lose_ref = R13[(R13.arm == "jW_ref") & (R13.subj != "P04")].ba.mean() - cell_level(decoded("jW_ref", True))
                print(f"D487 shuffle gate: w_cf loses {lose_cf:+.4f}, jW_ref {lose_ref:+.4f} -> " + ("ok" if lose_cf - lose_ref <= 0.003 else "FAILS"), flush=True)
            for w in WW:
                print(f"   (descriptive) {pre}_{w} vs jW_ref:", cmp13(f"{pre}_{w}", "jW_ref").split("  ")[0], flush=True)
    if LITE:                             # T2-D496: each lite form vs w14 (bar: >= -0.001 typical, no typical user below -0.005, shuffle gate)
        MS = {"w14": 8, "w13": 5, "k1c3": 4, "k2c3": 5, "k3c3": 6, "k4c3": 7, "k2c2": 4}
        ORD_S = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)
        lose = {}
        for a in MS:
            vs = []
            for (a_, proto, i, split), (ix, mx, fw) in MIX.items():
                if a_ != a:
                    continue
                kk = (keys_[0][ix], ORD_S[ix])
                P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
                y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                vs += [np.mean([np.mean(P[cells == c][y[cells == c] == k] == k) for k in np.unique(y[cells == c])])
                       for c in np.unique(cells) if subj[cells == c][0] != "P04"]
            lose[a] = R13[(R13.arm == a) & (R13.subj != "P04")].ba.mean() - float(np.mean(vs))
        for a, n in MS.items():
            if a == "w14":
                continue
            ms = 15.85 + 4.43 * n                    # measured: W13 (5 FM members) 38.0 ms, W14 (8) 51.3 ms on the H200 -> 4.43 ms / member
            m_ = R13[R13.arm == a].merge(R13[R13.arm == "w14"][key + ["ba"]], on=key, suffixes=("", "_b"))
            t_ = m_[m_.subj != "P04"]
            d_, u_ = (t_.ba - t_.ba_b).groupby(t_.i).mean(), (t_.ba - t_.ba_b).groupby(t_.subj).mean()
            okl = d_.mean() >= -0.001 and u_.min() >= -0.005 and lose[a] - lose["w14"] <= 0.003
            print(f"D496 {a} ({n} FM members, ≈ {ms:.1f} ms) vs w14: " + cmp13(a, "w14", -0.001).rsplit("  ->", 1)[0]
                  + f" | shuffle loses {lose[a]:+.4f} vs w14 {lose['w14']:+.4f} -> " + ("HOLDS" if okl else "does not hold"), flush=True)
        for a in ("k1c3", "k2c3", "k3c3", "k2c2"):
            print(f"D496 descriptive {a} vs w13:", cmp13(a, "w13", 0.0).replace("-> PASS", "").replace("-> no", ""), flush=True)
    if XFM:                              # T2-D492's bar: x_0.08 vs x_ref (+0.006, >= 8 / 10, no typical user below -0.005) + the shuffle gate
        print(f"D492 {XFM} BAR x_0.08 (CPT'd family at w 0.08) vs x_ref (W14's shape):", cmp13("x_0.08", "x_ref", 0.006), flush=True)
        ORD_S = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order
        lose = {}
        for a in ("x_ref", "x_0.08"):
            vs = []
            for (a_, proto, i, split), (ix, mx, fw) in MIX.items():
                if a_ != a:
                    continue
                kk = (keys_[0][ix], ORD_S[ix])
                P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
                y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                vs += [np.mean([np.mean(P[cells == c][y[cells == c] == k] == k) for k in np.unique(y[cells == c])])
                       for c in np.unique(cells) if subj[cells == c][0] != "P04"]
            lose[a] = R13[(R13.arm == a) & (R13.subj != "P04")].ba.mean() - float(np.mean(vs))
        print(f"D492 {XFM} shuffle gate: x_0.08 loses {lose['x_0.08']:+.4f}, x_ref {lose['x_ref']:+.4f} -> "
              + ("ok" if lose["x_0.08"] - lose["x_ref"] <= 0.003 else "FAILS"), flush=True)
        for w in (0.03, 0.05, 0.12):
            print(f"D492 {XFM} descriptive x_{w} vs x_ref:", cmp13(f"x_{w}", "x_ref", 0.006).replace("-> PASS", "").replace("-> no", ""), flush=True)
        for w in (0.03, 0.05, 0.08, 0.12):
            print(f"D492 {XFM} descriptive (no-CPT D452 family) c_{w} vs x_ref:", cmp13(f"c_{w}", "x_ref", 0.006).replace("-> PASS", "").replace("-> no", ""), flush=True)
        for a_, b_ in (("a_x5", "a_c5"), ("a_x1", "a_c1")):
            print(f"D492 {XFM} alone (CPT vs D452, paired seeds) {a_} vs {b_}:", cmp13(a_, b_, 0.006).replace("-> PASS", "").replace("-> no", ""), flush=True)
    TAGJ = ("D490r" if os.environ.get("D490R") else "D490") if KDM else "D504" if os.environ.get("D504") else f"D494 n0 {N0M}" if N0M else "D486"
    if LJM:                              # T2-D486's bar (T2-D490: the same bar): jW_all vs jW_ref (+0.003, >= 8 / 10, no typical user below -0.005) + the shuffle gate
        print(f"{TAGJ} BAR jW_all (all three CNN slots lj) vs jW_ref:", cmp13("jW_all", "jW_ref", 0.003), flush=True)
        ORD_S = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order (= D481 / D483 / D485)
        lose = {}
        for a in ("jW_ref", "jW_all"):
            vs = []
            for (a_, proto, i, split), (ix, mx, fw) in MIX.items():
                if a_ != a:
                    continue
                kk = (keys_[0][ix], ORD_S[ix])
                P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
                y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                vs += [np.mean([np.mean(P[cells == c][y[cells == c] == k] == k) for k in np.unique(y[cells == c])])
                       for c in np.unique(cells) if subj[cells == c][0] != "P04"]
            lose[a] = R13[(R13.arm == a) & (R13.subj != "P04")].ba.mean() - float(np.mean(vs))
        print(f"{TAGJ} shuffle gate: jW_all loses {lose['jW_all']:+.4f}, jW_ref {lose['jW_ref']:+.4f} -> "
              + ("ok" if lose["jW_all"] - lose["jW_ref"] <= 0.003 else "FAILS"), flush=True)
        for a_, b_ in (("jW_fbc", "jW_ref"), ("jW_sh", "jW_ref"), ("jW_ifn", "jW_ref"), ("j5_all", "j5_ref"),
                       ("a_fbclj", "a_fbc"), ("a_shlj", "a_sh"), ("a_ifnlj", "a_ifn")):
            print(f"{TAGJ} descriptive {a_} vs {b_}:", cmp13(a_, b_, 0.003).replace("-> PASS", "").replace("-> no", ""), flush=True)
    if CCPT:                             # T2-D484's bar: add3c vs add3 (+0.003, >= 8 / 10, no typical user below -0.005) on BOTH sets
        nm = "r" if REP else ""
        print(f"D484{' replicate set' if nm else ' A13 set'} BAR {nm}add3c vs {nm}add3:", cmp13(nm + "add3c", nm + "add3", 0.003), flush=True)
        for a_, b_ in (("add5c", "add5"), ("add3c", "add5"), ("alone3c", "alone3")):
            print(f"D484{' replicate set' if nm else ' A13 set'} descriptive {nm}{a_} vs {nm}{b_}:",
                  cmp13(nm + a_, nm + b_, 0.003).replace("-> PASS", "").replace("-> no", ""), flush=True)
    if MIM:                              # T2-D488's bar: mi1 vs add3 AND rmi1 vs rctl1 (+0.003, >= 8 / 10, no typical user below -0.005) + shuffle
        a_, b_ = ("rmi1", "rctl1") if REP else ("mi1", "add3")
        print(f"D488{' replicate set' if REP else ' A13 set'} BAR {a_} vs {b_}:", cmp13(a_, b_, 0.003), flush=True)
        ORD_M = np.random.default_rng(476).permutation(len(idx)).astype(np.int64)   # D476b's stress order (= D486)
        lose = {}
        for a in (a_, b_):
            vs = []
            for (a2, proto, i, split), (ix, mx, fw) in MIX.items():
                if a2 != a:
                    continue
                kk = (keys_[0][ix], ORD_M[ix])
                P = sum(fw[f] * causal_sinkhorn(mx[f], kk, 10, 3, 0.95, start=VEC[(a, split)][f]) for f in mx).argmax(1)
                y, cells, subj = idx.y.to_numpy()[ix], idx.cell.to_numpy()[ix], idx.subject.str[-3:].to_numpy()[ix]
                vs += [np.mean([np.mean(P[cells == c][y[cells == c] == k] == k) for k in np.unique(y[cells == c])])
                       for c in np.unique(cells) if subj[cells == c][0] != "P04"]
            lose[a] = R13[(R13.arm == a) & (R13.subj != "P04")].ba.mean() - float(np.mean(vs))
        print(f"D488 shuffle gate: {a_} loses {lose[a_]:+.4f}, {b_} {lose[b_]:+.4f} -> " + ("ok" if lose[a_] - lose[b_] <= 0.003 else "FAILS"), flush=True)
        if not REP:
            print("D488 descriptive mi2 vs add3:", cmp13("mi2", "add3", 0.003).replace("-> PASS", "").replace("-> no", ""), flush=True)
    if LAPM:                             # T2-D480's rule: m21L vs add3 on both sets (equal runtime); add3L = the Laplacian view alone
        nm = "r" if REP else ""
        for a_, b_ in (("m21L", "add3"), ("m21L", "pl"), ("add3", "pl"), ("add3L", "pl"), ("add3L", "add3")):
            print(f"D480{' replicate' if nm else ''} {a_} vs {b_}:", cmp13(nm + a_, nm + b_, 0.001).replace("-> PASS", "").replace("-> no", ""), flush=True)
    if F4:                               # the cross-fitted weight: w* chosen on the OTHER split's typical level, then this split at w*
        for kind in ("raw", "ann"):
            gains = {}
            for b_ in ("pl", "add3"):
                parts, wsel = [], {}
                for sp, other in (("test", "val"), ("val", "test")):
                    lv = {w: R13[(R13.arm == f"{b_}_{kind}_{w}") & (R13.split == other) & (R13.subj != "P04")].ba.mean() for w in W4}
                    wsel[sp] = max(lv, key=lv.get)
                    parts.append(R13[(R13.arm == f"{b_}_{kind}_{wsel[sp]}") & (R13.split == sp)])
                cf = pd.concat(parts).assign(arm=f"{b_}_{kind}_cf")
                R13 = pd.concat([R13, cf])
                line = cmp13(f"{b_}_{kind}_cf", b_)
                gains[b_] = float(line.split()[1])
                print(f"D479 {kind} on {b_}: w* test {wsel['test']} / val {wsel['val']}:", line, flush=True)
                for w in W4:
                    print(f"   (descriptive) {kind} on {b_} at fixed w {w}:", cmp13(f"{b_}_{kind}_{w}", b_).split("  ")[0], flush=True)
            red = gains["add3"] < 0.5 * gains["pl"] if gains["pl"] > 0 else None
            print(f"D479 {kind}: gain on pl {gains['pl']:+.4f}, on add3 {gains['add3']:+.4f} -> "
                  + ("redundant with the CAR view" if red else "not redundant" if red is not None else "no gain on pl to compare"), flush=True)
    if NETS:
        for a_, b_ in (("a", "ref"), ("b", "ref"), ("c", "ref"), ("d", "ref"), ("b", "a"), ("c", "a"), ("d", "a")):
            print(f"D475g {a_} vs {b_}:", cmp13(a_, b_, 0.003).replace("-> PASS", "").replace("-> no", ""), flush=True)
    elif REP and not F4 and not LAPM and not CCPT and not LJM and not WPM and not MIM and not XFM and not LITE:
        print("A13r REPLICATE radd vs rpl (bar +0.003):", cmp13("radd", "rpl", 0.003), flush=True)
        print("A13r descriptive radd vs rpl10:", cmp13("radd", "rpl10"), flush=True)
        print("A13r descriptive rpl10 vs rpl:", cmp13("rpl10", "rpl"), flush=True)
        for a in ("radd3", "radd2"):
            print(f"D475f REPLICATE {a} vs rpl (bar +0.003):", cmp13(a, "rpl", 0.003), flush=True)
        print("D475f descriptive radd3 vs radd:", cmp13("radd3", "radd"), flush=True)
    for a in (() if REP or LAPM or CCPT or LJM or WPM or MIM or XFM or LITE else ("add3", "add2")):
        print(f"D475f BAR {a} vs pl:", cmp13(a, "pl"), flush=True)
    if not REP and not LAPM and not CCPT and not LJM and not WPM and not MIM and not XFM and not LITE:
        print("D475f descriptive add3 vs add:", cmp13("add3", "add"), flush=True)
    for a in (() if REP or LAPM or CCPT or LJM or WPM or MIM or XFM or LITE else ("car5", "m32", "m23", "add")):
        print(f"A13 BAR {a} vs pl:", cmp13(a, "pl"), flush=True)
    for a, b in (() if REP or LAPM or CCPT or LJM or WPM or MIM or XFM or LITE else (("add", "pl10"), ("pl10", "pl"), ("m32o", "m32"), ("m23o", "m23"), ("addo", "add"))):
        print(f"A13 descriptive {a} vs {b}:", cmp13(a, b), flush=True)
if "B" in arms_to_read:
    M["fbc_car"] = {"offset": 324, **{p: str(O / f"t2_d475/route_fbc_car/{p}__fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3__car__s{{s}}.npz") for p in ("kit", "seen1")}}
    arms = {"B_car": {"fbc_car": 1.0}, "B_pl": {"fbc_tv8c": 1.0},
            "B_C_car": {("fbc_car" if k == "fbc_tv8c" else k): v for k, v in C.items()}, "B_C_pl": C}
    spec = {"members": M, "arms": arms}
    print("B FBCNet alone, CAR vs plain:", compare(spec, "B_car", "B_pl", 12), flush=True)
    print("B FBCNet in C, CAR vs plain:", compare(spec, "B_C_car", "B_C_pl", 12), flush=True)

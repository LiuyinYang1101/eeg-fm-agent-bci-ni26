#!/usr/bin/env python
"""T2-D642 FULL-CHAIN SET READ, the splicer (design pinned in docs/DECISIONS_T2.md before any number; Track3 reviews this file before any root
is read). On D638's chosen CNN cell <cell> (its base root t2_d638_<cell>_860 = the spliced routes, the base teacher and the base students, which use
D498's FIXED FM and stay the cell's own), the FM members of the ENTERING slots (outputs/t2_d642/members.json) are replaced by the D642 read's
ship-composition members, and the FM-dependent chain is regenerated: fbteach -> lpteach -> the final P1 students -> the mix (W21's lab form).
The lab's FM member files carry ONE model per lc9 fold (per-row prob / features / side rows + ONE main head fm_out_w / b + ONE side head
side_w / b, read by the in-stream refits), so one spliced file cannot hold both the stranger's ss8 model and the rt1 models: THREE roots per arm,
each scored ONLY on its own rows (t2_d642_set_read.py):
  tag <arm><root>, arm b (the B members: unweighted, seed 900 + slot) | c (C: --user-balance, the same seed) | t (T: unweighted, seed 950 + slot)
  | d (W26's C': --user-balance, seed 950 + slot), root
    s    in the 8 spliced folds, TRU[f]'s rows <- ss8_<TRU[f]>'s rows, the heads <- ss8_<TRU[f]>'s heads (P04 / P10 rows: set E's, UNSCORED);
    p04  P04 sessions 2-3 rows <- rt1_P04's rows, the heads <- rt1_P04's (TRU / P10 rows: set E's, UNSCORED);
    p10  the same for P10;
  tag id = the NO-OP identity: every slot, set E's own lc9 rows and heads written back through the same code path (s's groups).
  UNSCORED rows keep EVERY per-row array of set E's file; only the head arrays are the source's (the lab reads per-row FM values per stream,
  so an unscored stream never feeds a scored one, except through the FM START vectors: root s keeps the cross-fitted convention; roots p04 /
  p10 FREEZE them to the id root's values, NI26_LAB_FMSTART, Track3 Q2). Folds 3 / 9 (P04 / P10 held out) and non-entering slots: unspliced.
  BOTH CELLS bw21 and bw23 (the manager, before any D642 number): every file of a cell under outputs/t2_d642/setread/<cell>/, every root
  t2_d642_<fb|lp><tag>_<cell>_860; the shared input setread/fmpaths_setE.json.
  python scripts/local/t2_d642_setsplice.py fmspl <tag> <cell>      # outputs/t2_d642/setread/<cell>/fmspl_<tag>/ (+ manifest.json)
  python scripts/local/t2_d642_setsplice.py lproot <tag> <cell>     # t2_d642_lp<tag>_<cell>_860's links (route_* -> the cell's base root; id: its students)
  python scripts/local/t2_d642_setsplice.py p1t <tag> <cell>        # the final P1 teachers of the SCORED group only (t2p_bench --instream-pl)
  python scripts/local/t2_d642_setsplice.py students <tag> <cell>   # the cell's final students with the scored group's rows <- the new students
  python scripts/local/t2_d642_setsplice.py check-id <cell>         # id's fb / lp teachers == the cell's (t2_d638_fb / lp<cell>_860) BITWISE
  python scripts/local/t2_d642_setsplice.py check-paths <tag> <cell> # Track3 LOW 2: the lab's spliced paths == the manifest EXACTLY
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t2_d642_member_read import job_dir, member_file  # noqa: E402  (the screen's exact file rule: 6 array task dirs, cb twin, __ubw)
import t2_d642_member_read as MR  # noqa: E402  (T2-D649 / D652: the generic _dirs / _file of their member reads)

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
SRC = O / "t2_d615"                                    # set E's members (the CNN / side / DC links)
D638 = O / "t2_d638"
D642 = O / "t2_d642"
SR0 = D642 / "setread"                                 # the shared input fmpaths_setE.json
SR, CELL_ = None, None                                 # per cell: SR0 / <cell> (pin_cell)
DRAWS, SEEDS = (860, 861, 862), (880, 881, 882)        # the students' seeds = D638's (880 + j <-> draw 860 + j): B / C paired
# T2-D646: the x / y tags use the SAME student seeds 880-882 as reads 1-2 (Track3's correction, the manager: t2p_bench --seeds also retrains
# the students' base pooled nets, so fresh seeds would put the students on a different base than the routed part = a non-ship structure)


def sseeds(tag):
    return SEEDS
USERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
SPLICED = [f for f in range(10) if USERS[f] not in ("P04", "P10")]
SLOTS = ["s0_mi", "s1_stlsd", "s2_plaincont", "s3_stlcarsd", "s4_carcont", "s5_stlcarsd"]
FORMS_E = [[826, "pmi@848"], [827, "psd@843"], [828, "pcb@846"], [829, "psd@844"], [830, "pcb@847"], [831, "psd@845"]]   # W21's fm_0..5 = SLOTS
TAGS = ("id", "bs", "cs", "bp04", "cp04", "bp10", "cp10", "ts", "ds", "tp04", "dp04", "tp10", "dp10",   # t / d = W26's replicate (T vs C')
        "gs", "es", "gp04", "ep04", "gp10", "ep10", "us", "up04", "up10",                 # g / e = T2-D644's W27 (B vs E); u = U (C + chemb)
        "hs", "ks", "hp04", "kp04", "hp10", "kp10", "ms", "ns", "mp04", "np04", "mp10", "np10",   # D642 x9 ALL6: h / k = B / C, m / n = T / C', all six slots
        "xs", "ys", "xp04", "yp04", "xp10", "yp10",                                      # T2-D646: x / y = B / C at the fresh seeds 1000 + slot, all six
        "vs", "ws", "vp04", "wp04", "vp10", "wp10",                                      # T2-D648: v / w = B / C at the fresh seeds 1100 + slot, all six
        "fs", "as", "fp04", "ap04", "fp10", "ap10", "js", "is", "jp04", "ip04", "jp10", "ip10",   # T2-D649 (W29): f / a = C / G_picked at 1200, j / i = at 1300
        "qs", "qp04", "qp10", "rs", "rp04", "rp10",                                      # T2-D652 (W30): q = V at 950 (base n), r = V at 1100 (base w)
        *[f"{d}{r}" for d in "12345678oz" for r in ("s", "p04", "p10")])                  # T2-D654: 1 / 4 = the base splices of C at 1600 / 1700 (d654a / d654b);
                                                                                          # 2 / 5 = their v1 transforms, 3 / 6 = the picked arm's (t2_d654_viewmix fmspl)
                                                                                          # THE STACK READ (the owner's rule 3): 7 = the base splice of C at 1800 (stk),
                                                                                          # 8 = the stack arm's (stack_spec.json: G / V / J), o / z = their transforms
ARM = {"b": ("B", 0), "c": ("C", 0), "t": ("T", 50), "d": ("Cp", 50), "g": ("B", 0), "e": ("E", 0), "u": ("U", 0),
       "h": ("B", 0), "k": ("C", 0), "m": ("T", 50), "n": ("Cp", 50), "x": ("X", 100), "y": ("Y", 100),
       "v": ("V", 200), "w": ("W", 200)}   # tag letter -> (arm dir, seed offset)
ALL6L = "hkmnxyvwafijqr12345678oz"                           # the letters whose tags splice ALL six slots (D642 x9 / D646 / D648 / D649 / D652)
NEW_SRC = {"f": ("d649", "C", 1200), "a": ("d649", "G", 1200), "j": ("d649r", "C", 1300), "i": ("d649r", "G", 1300),
           "q": ("d652", "V", 950), "r": ("d652b", "V", 1100),   # T2-D649 / D652: letter -> (job-dir prefix, arm, seed0); the seed = seed0 + slot
           "1": ("d654a", "C", 1600), "4": ("stk", "C", 1800),     # T2-D654: the base splices (D642's C with the view dumps; fm_view_* not copied);
                                                                    # read 2 = the C@1800 arrays (the capacity re-plan 17:2x: the 1700 set never ran)
           "7": ("stk", "C", 1800), "8": ("stk", "STACK", 1800)}   # THE STACK READ: C at 1800 / the stack arm at 1800 (setread/<cell>/stack_spec.json)
D654_VIEWMIX = "2356oz"                            # T2-D654 / the stack read: transform tags, written by t2_d654_viewmix.py fmspl from a base splice
STK_RULE = {"C": (["__ubw"], ["grl", "vp2"]), "G": (["__ubw", "grl0.1"], ["vp2"] + [f"grl0.1{c}" for c in "0123456789"]),
            "V": (["__ubw", "vp2"], ["grl"]), "J": (["__ubw", "vp2", "grl0.1"], [f"grl0.1{c}" for c in "0123456789"])}   # the tstk arms' name tags
STK_S0 = {"G": "G", "V": "C", "J": "G"}            # Track3: no V / J member on s0 (no vp on MI): a V-stack's s0 = C's, a J-stack's s0 = G's
W30_BASE = {"q": "n", "r": "w"}                    # Track3's pin (3): W30's s0 = its base tag's s0 member (W25's C), byte-identical


def picked_g():
    """T2-D649: the G arm the smoke picked by ENGAGEMENT (lambda_pick.json, rule 3); never another."""
    pk = json.loads((O / "t2_d649" / "lambda_pick.json").read_text())
    assert pk["engaged"] and pk["lambda_max"] is not None, "REFUSED: the D649 smoke did not engage"
    lam = float(pk["lambda_max"])
    return {0.1: "G01", 0.3: "G03", 1.0: "G1"}[lam], lam


def new_src(tag, k, src):
    """the source member file of a T2-D649 / D652 tag for slot k, scored fold src."""
    L, slot = tag[0], SLOTS[k]
    if L in "78":                                     # THE STACK READ's base splices (the tstk fits at 1800 + slot)
        arm = "C" if L == "7" else json.loads((SR / "stack_spec.json").read_text())["arm"]
        a_ = STK_S0[arm] if (k == 0 and arm != "C") else arm
        must, mustnot = STK_RULE[a_]
        return MR._file(MR._dirs("stk", slot, a_, 6), slot, src, 1800 + k, must, mustnot)
    if L in W30_BASE and k == 0:                       # W30 has no V for the MI slot: the base tag's own s0 member (Track3 pin 3)
        barm, off = ARM[W30_BASE[L]]
        return member_file(job_dir(slot, barm), slot, barm, src, 900 + k + off)
    pre, arm, s0 = NEW_SRC[L]
    if arm == "G":
        arm, lam = picked_g()
        must, mustnot = ["__ubw", f"grl{lam:g}"], [f"grl{lam:g}{c}" for c in "0123456789."]
    elif arm == "C":
        must, mustnot = ["__ubw"], ["grl", "vp2"]
    else:
        must, mustnot = ["__ubw", "vp2"], []
    return MR._file(MR._dirs(pre, slot, arm, 6), slot, src, s0 + k, must, mustnot)
CELLS = ("bw21", "bw23", "cw21", "cw23")
ROUTES = ("route_fbc", "route_sh", "route_ifn")
P1 = {"sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr", "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vr"}
LINKS = ("cnn_fbc", "cnn_sh", "cnn_ifn", "side_sh", "side_ifn", "s10", "pred_s28-29-30_dc.npz")
ROW_KEYS = ("prob_last", "prob_best", "side_logit_last", "side_te", "fm_feat_te")
HEAD_KEYS = ("fm_out_w", "fm_out_b", "side_w", "side_b")
idx = pd.read_csv(EXP / "index.csv")
U, SES = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy()


def root(tag, kind):
    return O / f"t2_d642_{kind}{tag}_{CELL_}_860"


def base_cell(cell):
    """T2-D655: the D638 CNN cell under a set-read cell (bw21fm40 -> bw21: the same routes / students / base teacher root, the FM share 0.40)."""
    return cell[:-4] if cell.endswith(("fm40", "fm55")) else cell   # T2-D658: + fm55


def pin_cell(cell):
    """every file of the cell's set read under setread/<cell>/ and every root named with the cell (bw21 / bw23: the manager)."""
    global SR, CELL_
    assert cell in ("bw21", "bw23", "cw21", "bw21fm40", "bw21fm55"), cell   # T2-D658: + bw21fm55   # cw21 = T2-D650; bw21fm40 = T2-D655 (bw21's CNN cell, the FM share 0.40)
    SR, CELL_ = SR0 / cell, cell
    SR.mkdir(parents=True, exist_ok=True)


def scored(tag, f, te):
    """(the scored rows' mask over te, the source fold name) of tag's root in fold f (id = s's groups, no-op)."""
    r = "s" if tag == "id" else tag[1:]
    if r == "s":
        return U[te] == USERS[f], f"ss8_{USERS[f]}"
    x = r.upper()
    return (U[te] == x) & np.isin(SES[te], ("2", "3")), f"rt1_{x}"


def entering(tag):
    if tag == "id" or tag[0] in ALL6L:                     # the no-op identity; D642 x9 ALL6 / T2-D646 / D648 / D649 / D652 = every slot
        return list(range(6))
    m = json.loads((D642 / ("members_w27.json" if tag[0] in "geu" else "members.json")).read_text())   # T2-D644: W27's own entering slots (U: W27's slots)
    e = m["entering"]
    assert m["nominated"] and e, "REFUSED: no slot enters the set (members.json): no set read"
    return [SLOTS.index(s) for s in e]


def fm_paths():
    z = json.loads((SR0 / "fmpaths_setE.json").read_text())   # `t2_lab.py fmpaths --set E --base w20` with NO splice (the files the read loads)
    assert z["fmsplice"] == "" and z["forms"] == FORMS_E, (z["fmsplice"], z["forms"])
    names = [Path(p).name for v in z["paths"].values() for p in v]
    assert len(names) == 60 and len(set(names)) == 60, "T2-D642 (Track3 LOW 2): the 60 (slot, fold) FM basenames are not unique"
    return {int(f): v for f, v in z["paths"].items()}


def cmd_fmspl(tag, cell):
    pin_cell(cell)
    assert tag in TAGS, tag
    out = SR / f"fmspl_{tag}"
    assert not out.exists(), f"never overwrite a read's inputs: {out}"
    out.mkdir(parents=True)
    paths, slots, man = fm_paths(), entering(tag), []
    assert tag[0] not in ALL6L or slots == list(range(6)), (tag, slots, "D642 x9 / D646 / D648 / D649 / D652: ALL6's tags splice ALL six slots")   # the manager / Track3
    assert tag == "id" or tag[0] in ARM or tag[0] in NEW_SRC, (tag, "no source rule for this letter")
    assert tag[0] not in D654_VIEWMIX, (tag, "a D654 transform tag is written by t2_d654_viewmix.py fmspl, never spliced here")
    arm, off = ARM[tag[0]] if tag != "id" and tag[0] in ARM else (None, 0)   # the NEW_SRC letters (D649 / D652 / D654) resolve in new_src
    for f in SPLICED:
        for k in slots:
            p = Path(paths[f][k])
            z = np.load(p)
            te = z["test_idx"].astype(np.int64)
            m, src = scored(tag, f, te)
            assert m.sum() > 0, (tag, f, k)
            new = {key: z[key] for key in z.files}
            if tag == "id":                                  # the NO-OP: set E's own rows / heads through the same code path
                zs, srcp, sel = z, str(p), np.flatnonzero(m)
            else:
                srcp = (new_src(tag, k, src) if tag[0] in NEW_SRC else member_file(job_dir(SLOTS[k], arm), SLOTS[k], arm, src, 900 + k + off))
                zs = np.load(srcp)
                pos = {int(r): i for i, r in enumerate(zs["test_idx"].astype(np.int64))}
                assert set(pos) == set(te[m].tolist()), (tag, f, k, "the source rows != the scored group's rows")
                sel = np.array([pos[int(r)] for r in te[m]])
            ks = sorted(set(z.files) - {"res", "cont_curves"})
            kz = sorted(x for x in set(zs.files) - {"res", "cont_curves"} if not x.startswith("fm_view_"))   # T2-D654: the view dumps stay in the source (viewmix reads them)
            assert ks == kz, (tag, f, k, ks, sorted(zs.files))
            assert set(ks) <= set(ROW_KEYS + HEAD_KEYS + ("test_idx", "val_idx", "prob_val_last")), ks
            assert len(z["val_idx"]) == 0 and len(zs["val_idx"]) == 0, (tag, f, k, "val rows")
            for key in ROW_KEYS:
                if key in ks:
                    a = np.array(z[key], copy=True)
                    assert a.shape[0] == len(te) and zs[key].shape[1:] == a.shape[1:], (key, a.shape, zs[key].shape)
                    a[m] = zs[key][sel]
                    new[key] = a
            for key in HEAD_KEYS:
                if key in ks:
                    assert zs[key].shape == z[key].shape, (key, zs[key].shape, z[key].shape)
                    new[key] = np.array(zs[key], copy=True)
            res = json.loads(str(z["res"])) if "res" in z.files else {}
            res["d642_splice"] = dict(tag=tag, cell=cell, slot=SLOTS[k], fold=f, scored=src, n_scored=int(m.sum()), source=str(srcp))
            new["res"] = np.array(json.dumps(res))
            np.savez(out / p.name, **new)
            man.append(dict(fold=f, slot=SLOTS[k], file=p.name, source=str(srcp), scored=src, n_scored=int(m.sum())))
    if tag[0] == "8" and json.loads((SR / "stack_spec.json").read_text())["arm"] == "V":   # Track3: a V-stack's s0 == the C base's (tag 7), path + bytes
        import hashlib
        bm = json.loads((SR / f"fmspl_7{tag[1:]}" / "manifest.json").read_text())
        b0 = {e["fold"]: e["source"] for e in bm["files"] if e["slot"] == SLOTS[0]}
        for e in man:
            if e["slot"] == SLOTS[0]:
                assert e["source"] == b0[e["fold"]] and hashlib.sha256(Path(e["source"]).read_bytes()).digest() == \
                    hashlib.sha256(Path(b0[e["fold"]]).read_bytes()).digest(), (tag, e["fold"], "the V-stack's s0 != C's")
        print(f"[d642 splice] {tag}: the V-stack's s0 == the C base's (7{tag[1:]}) in all {len(b0)} folds (path + sha256)", flush=True)
    if tag[0] in W30_BASE:                             # Track3's pin (3): s0 == the base tag's s0 source per fold, path AND bytes
        import hashlib
        bm = json.loads((SR / f"fmspl_{W30_BASE[tag[0]]}{tag[1:]}" / "manifest.json").read_text())
        b0 = {e["fold"]: e["source"] for e in bm["files"] if e["slot"] == SLOTS[0]}
        h = lambda q: hashlib.sha256(Path(q).read_bytes()).hexdigest()   # noqa: E731
        for e in man:
            if e["slot"] == SLOTS[0]:
                assert e["source"] == b0[e["fold"]] and h(e["source"]) == h(b0[e["fold"]]), (tag, e["fold"], "W30's s0 != the base tag's s0")
        print(f"[d642 splice] {tag}: s0 == the base tag {W30_BASE[tag[0]]}{tag[1:]}'s s0 in all {len(b0)} folds (path + sha256; Track3 pin 3)", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(tag=tag, cell=cell, slots=[SLOTS[k] for k in slots], files=man), indent=1))
    print(f"[d642 splice] {out.name}: {len(man)} files ({len(SPLICED)} folds x slots {[SLOTS[k] for k in slots]}; scored group {'TRU <- ss8' if tag in ('id', 'bs', 'cs') else tag[1:].upper() + ' s2-3 <- rt1'}"
          f"{'; NO-OP' if tag == 'id' else ''})", flush=True)


def cmd_lproot(tag, cell):
    pin_cell(cell)
    r = root(tag, "lp")
    assert tag in TAGS and r.exists() and (r / "teacher_p1").exists() and not (r / "route_fbc").exists(), r   # lpteach created r / teacher_p1
    for x in LINKS:
        (r / x).symlink_to(SRC / x)
    for x in ROUTES:
        (r / x).symlink_to(O / f"t2_d638_{base_cell(cell)}_860" / x)
    if tag == "id":                                         # the id root's final students = the cell's (no GPU)
        for fam in P1:
            (r / f"p1_{fam}").symlink_to(O / f"t2_d638_lp{cell}_860" / f"p1_{fam}")
    print(f"[d642 splice] lp root {r.name}: links (route_* -> t2_d638_{cell}_860{'; students -> the cell' if tag == 'id' else ''})", flush=True)


def cmd_p1t(tag, cell):
    pin_cell(cell)
    assert tag in TAGS[1:], tag
    T = root(tag, "lp") / "teacher_p1"
    out = SR / "p1t" / tag
    assert not out.exists(), out
    for f in SPLICED:
        for j, d in enumerate(DRAWS):
            z = np.load(T / f"lc9_{f}__s{d}.npz")
            rows = z["rows"].astype(np.int64)
            keys = ["rows", "post"] + [k for k in z.files if k.startswith("post_r")]
            assert {"post_r60", "post_r100", "post_r150"} <= set(keys), z.files
            m, src = scored(tag, f, rows)
            dd = out / ("ss8" if src.startswith("ss8_") else f"f{f}")
            dd.mkdir(parents=True, exist_ok=True)
            np.savez(dd / f"{src}__s{sseeds(tag)[j]}.npz", **{k: z[k][m] for k in keys})
    print(f"[d642 splice] P1 teachers {out} (the scored group only: {'ss8, 8 strangers' if tag.endswith('s') else tag[1:].upper() + ' rt1 per fold'}) x 3 seeds", flush=True)


def cmd_students(tag, cell):
    pin_cell(cell)
    assert tag in TAGS[1:], tag
    r = root(tag, "lp")
    ref = O / f"t2_d638_lp{base_cell(cell)}_860"            # the cell's final students (unscored rows, folds 3 / 9; T2-D655: bw21's)
    xt = "__ubw" if cell.startswith("c") else ""            # the cell's nets: C = --user-balance (t2p_bench names it __ubw)
    raw = SR / "p1" / tag
    for fam, stem in P1.items():
        dd = r / f"p1_{fam}" / "runs"
        assert not dd.exists(), dd
        dd.mkdir(parents=True)
        for f in range(10):
            for j, d in enumerate(DRAWS):
                z0 = np.load(ref / f"p1_{fam}" / "runs" / f"lc9_{f}__{stem}isSpr__s{d}.npz")
                te, p = z0["test_idx"], z0["prob_last"].astype(np.float64)
                prov = f"the cell's final students ({cell})"
                if f in SPLICED:
                    m, src = scored(tag, f, te)
                    sub = "ss8" if src.startswith("ss8_") else f"f{f}"
                    z = np.load(raw / sub / fam / "runs" / f"{src}__{stem}isSpr{xt}__s{sseeds(tag)[j]}.npz")
                    pos = {int(x): i for i, x in enumerate(z["test_idx"])}
                    assert set(pos) == set(te[m].tolist()), (tag, fam, f, d)
                    p = p.copy()
                    p[m] = z["prob_last"].astype(np.float64)[[pos[int(x)] for x in te[m]]]
                    prov = f"the cell's final students; {src} rows <- the {tag} students (seed {sseeds(tag)[j]})"
                res = json.loads(str(z0["res"])) if "res" in z0.files else {}
                res["d642"] = prov
                np.savez_compressed(dd / f"lc9_{f}__{stem}isSpr__s{d}.npz", test_idx=te, prob_last=p.astype(np.float32), prob_best=p.astype(np.float32),
                                    val_idx=z0["val_idx"] if "val_idx" in z0.files else np.zeros(0, int),
                                    prob_val_last=z0["prob_val_last"] if "prob_val_last" in z0.files else np.zeros((0, 3), np.float32), res=json.dumps(res))
    print(f"[d642 splice] final students -> {r.name}/p1_*/runs (the scored group regenerated, the rest the cell's)", flush=True)


def cmd_check_paths(tag, cell):
    """Track3 LOW 2: `t2_lab.py fmpaths` run WITH NI26_LAB_FMSPLICE=fmspl_<tag> (setread/fmpaths_<tag>.json): the paths resolving into fmspl_<tag>
    are EXACTLY the manifest's files at their (fold, slot) positions, and every other path == set E's (no silent fallback, no double splice)."""
    pin_cell(cell)
    base, spl = fm_paths(), SR / f"fmspl_{tag}"
    z = json.loads((SR / f"fmpaths_{tag}.json").read_text())
    assert Path(z["fmsplice"]) == spl and z["forms"] == FORMS_E, (z["fmsplice"], z["forms"])
    man = json.loads((spl / "manifest.json").read_text())
    want = {(int(r["fold"]), SLOTS.index(r["slot"])): r["file"] for r in man["files"]}
    got = {}
    for f, v in z["paths"].items():
        for k, p in enumerate(v):
            if Path(p).parent == spl:
                got[(int(f), k)] = Path(p).name
            else:
                assert p == base[int(f)][k], (tag, f, k, p, "!= set E's path")
    assert got == want, (tag, sorted(set(got) ^ set(want))[:6], "the spliced paths != the manifest")
    assert all(want[key] == Path(base[key[0]][key[1]]).name for key in want), (tag, "a spliced file sits at another slot's basename")
    print(f"[d642 splice] {tag}: the lab loads EXACTLY the manifest's {len(want)} spliced files (folds x slots {man['slots']}); the other "
          f"{60 - len(want)} paths are set E's", flush=True)


def same_npz(a, b):
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files):
        return False, f"keys {sorted(za.files)} vs {sorted(zb.files)}"
    for k in za.files:
        if not np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f"):
            return False, k
    return True, ""


def cmd_check_id(cell):
    pin_cell(cell)
    bad = []
    for kind in ("fb", "lp"):
        mine, ref = root("id", kind) / "teacher_p1", O / f"t2_d638_{kind}{cell}_860" / "teacher_p1"
        files = sorted(p.name for p in ref.glob("lc9_*__s*.npz"))
        assert len(files) == 30 and sorted(p.name for p in mine.glob("lc9_*__s*.npz")) == files, (mine, ref)
        b0 = len(bad)
        for fn in files:
            ok, why = same_npz(mine / fn, ref / fn)
            if not ok:
                bad.append(f"{kind}/{fn}: {why}")
        print(f"[d642 identity] {mine.parent.name}/teacher_p1 vs {ref.parent.name}/teacher_p1: {'BITWISE' if len(bad) == b0 else 'DIFFERS'} (30 files)", flush=True)
    if bad:
        raise SystemExit("STOP: the D642 identity teachers differ: " + "; ".join(bad[:6]))
    print("[d642 identity] PASS teachers: the no-op FM splice reproduces the cell's fb / lp teachers BITWISE", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    {"fmspl": lambda: cmd_fmspl(sys.argv[2], sys.argv[3]), "lproot": lambda: cmd_lproot(sys.argv[2], sys.argv[3]),
     "p1t": lambda: cmd_p1t(sys.argv[2], sys.argv[3]), "students": lambda: cmd_students(sys.argv[2], sys.argv[3]),
     "check-id": lambda: cmd_check_id(sys.argv[2]), "check-paths": lambda: cmd_check_paths(sys.argv[2], sys.argv[3])}[c]()

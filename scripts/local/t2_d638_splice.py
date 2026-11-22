#!/usr/bin/env python
"""T2-D638 (registered + amended ×2 in docs/DECISIONS_T2.md before any number): the spliced lab roots of the full-stack 2 × 2 on set E
(draws 860-862, t2_d615's members; W21's lab form = an lp root read as `t2_lab.py mix --base w20 --idea w20_fb`, PRE_R, CKS 60,100,150).
Cells: id (the NO-OP identity: set E's own rows written back), bw21, bw23, cw21, cw23 (D630's nets B / C x the W21 / session-1 router;
route files from scripts/local/t2_d638_routes.py, seed 880 + j <-> draw 860 + j).
Roots: base O/t2_d638_<cell>_860 (spliced route_*, the base teacher, the base P1 students; fbteach reads it), fb O/t2_d638_fb<cell>_860
(teacher only), lp O/t2_d638_lp<cell>_860 (the read root: route_* -> the base root's, the LP teacher, the final P1 students).
Spliced folds = the 8 lc9 folds whose held-out user is single-session (f != 3, 9): TRU[f]'s rows <- the cell's ss8_TRU rows; P04 / P10
sessions 2-3 <- the cell's rt1_P04 / rt1_P10 rows; folds 3 / 9 unspliced (the reference files' rows).
  python scripts/local/t2_d638_splice.py base <cell>            # route_* + symlinks + the base teacher (t2_d512_teacher.py's logic, redirected)
  python scripts/local/t2_d638_splice.py p1t <cell> base|final   # the P1 teachers for t2p_bench: ss8_<u> (its fold's) + per fold f rt1_<X>
  python scripts/local/t2_d638_splice.py students <cell> base|final   # t2p_bench's isSpr outputs -> the root's lc9 isSpr files
  python scripts/local/t2_d638_splice.py lproot <cell>          # the lp root's links (id: the reference final students)
  python scripts/local/t2_d638_splice.py check-teachers         # identity: id's base / fb / lp teachers == the references BITWISE
  python scripts/local/t2_d638_splice.py check-ii               # identity (ii): t2p_bench lc9_0 / 860 with the reference teacher == its isSpr
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
SRC = O / "t2_d615"                                    # set E's members
REF_FB, REF_LP = O / "t2_d571_fbw20d557_860", O / "t2_d571_lpw20d557_860"   # W21's lab roots on set E
D638 = O / "t2_d638"
DRAWS, SEEDS = (860, 861, 862), (880, 881, 882)
USERS = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]   # lc9 fold f holds out USERS[f] (t2p_bench LC_BASE)
SPLICED = [f for f in range(10) if USERS[f] not in ("P04", "P10")]
CELLS = ("id", "nz", "bw21", "bw23", "cw21", "cw23")   # nz = the NOISE cell: id's no-op routes, BOTH student generations regenerated (lc9)
ROUTE = {"fbc": "fbcnet_bboth__none__n00__r1_pulxx_vp2_e180_vr_rc75tv8cfb3", "sh": "shallow__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2",
         "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vrtv8cfb2"}
P1 = {"sh": "shallow__ea__n00__r1_pulx_vp2_e180_vr", "ifn": "ifnet__ea__n00__r1_pulx_vp2_e180_vr"}
LINKS = ("cnn_fbc", "cnn_sh", "cnn_ifn", "side_sh", "side_ifn", "s10", "pred_s28-29-30_dc.npz")
idx = pd.read_csv(EXP / "index.csv")
U, SES = idx.subject.str[-3:].to_numpy(), idx.session.astype(str).to_numpy()


def root(cell, kind=""):
    return O / f"t2_d638_{kind}{cell}_860"


def groups(f, te):
    """the spliced groups of fold f's rows: (mask over te, source fold name)."""
    u = USERS[f]
    g = [(U[te] == u, f"ss8_{u}")]
    for x in ("P04", "P10"):
        g.append(((U[te] == x) & np.isin(SES[te], ("2", "3")), f"rt1_{x}"))
    cover = np.zeros(len(te), bool)
    for m, _ in g:
        assert not (cover & m).any()
        cover |= m
    assert cover.all(), (f, "fold rows outside TRU + P04 / P10 sessions 2-3")
    return g


def put(te, p, rows_src, p_src, m):
    """p[m] <- p_src at te[m]'s rows (rows_src must hold exactly te[m]'s rows)."""
    pos = {int(r): i for i, r in enumerate(rows_src)}
    assert set(pos) == {int(r) for r in te[m]}, "the source rows != the fold's group rows"
    p[m] = p_src[[pos[int(r)] for r in te[m]]]


def save(path, te, p, ref, prov):
    np.savez_compressed(path, test_idx=te, prob_last=p.astype(np.float32), prob_best=p.astype(np.float32),
                        val_idx=ref["val_idx"] if "val_idx" in ref.files else np.zeros(0, int),
                        prob_val_last=ref["prob_val_last"] if "prob_val_last" in ref.files else np.zeros((0, 3), np.float32),
                        res=json.dumps(dict(json.loads(str(ref["res"])) if "res" in ref.files else {}, d638=prov)))


def link_members(r):
    for x in LINKS:
        (r / x).symlink_to(SRC / x)


def cmd_base(cell):
    r = root(cell)
    assert cell in CELLS and not r.exists(), r
    r.mkdir()
    link_members(r)
    for fam, stem in ROUTE.items():
        (r / f"route_{fam}").mkdir()
        for f in range(10):
            for j, d in enumerate(DRAWS):
                ref = np.load(SRC / f"route_{fam}" / f"lc9_{f}__{stem}__s{d}.npz")
                te, p = ref["test_idx"], ref["prob_last"].astype(np.float64)
                prov = "unspliced (set E's rows)"
                if f in SPLICED:
                    q = p.copy()
                    for m, src in groups(f, te):
                        if cell in ("id", "nz"):             # the NO-OP splice: set E's own rows through the same code path
                            put(te, q, te[m], p[m], m)
                        else:
                            z = np.load(D638 / f"route_{cell}" / f"route_{fam}" / f"{src}__{stem}__s{SEEDS[j]}.npz")
                            put(te, q, z["test_idx"], z["prob_last"].astype(np.float64), m)
                    p, prov = q, f"spliced: TRU <- ss8, P04 / P10 s2-3 <- rt1 ({cell}, seed {SEEDS[j]})" if cell not in ("id", "nz") else f"no-op splice ({cell})"
                save(r / f"route_{fam}" / f"lc9_{f}__{stem}__s{d}.npz", te, p, ref, prov)
    # the base teacher: t2_d512_teacher.py EXACTLY with R / OUT / DRAWS redirected (= t2_d615_teacher.py's pattern)
    src = (Path(__file__).resolve().parent / "t2_d512_teacher.py").read_text()
    for a, b in (('R = O / "t2_d512"\n', f'R = O / "{r.name}"\n'), ('OUT = O / "t2_d512/teacher_p1"\n', f'OUT = O / "{r.name}/teacher_p1"\n'),
                 ("DRAWS = (810, 811, 812)\n", "DRAWS = (860, 861, 862)\n")):
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    exec(compile(src, str(Path(__file__).resolve().parent / "t2_d512_teacher.py"), "exec"),
         {"__file__": str(Path(__file__).resolve().parent / "t2_d512_teacher.py"), "__name__": "__main__"})
    n = len(list((r / "teacher_p1").glob("lc9_*__s*.npz")))
    assert n == 30, n
    if cell == "id":                                       # the id root's base students = set E's (no GPU)
        for fam in P1:
            (r / f"p1_{fam}").symlink_to(SRC / f"p1_{fam}")
    print(f"[d638 splice] base root {r.name}: route_* ({len(SPLICED)} spliced folds), links, base teacher (30 files)", flush=True)


def teacher_dir(cell, gen):
    return (root(cell) if gen == "base" else root(cell, "lp")) / "teacher_p1"


def cmd_p1t(cell, gen):
    assert cell in CELLS[2:] and gen in ("base", "final")
    T = teacher_dir(cell, gen)
    out = D638 / "p1t" / f"{gen}_{cell}"
    assert not out.exists(), out
    for f in SPLICED:
        for j, d in enumerate(DRAWS):
            z = np.load(T / f"lc9_{f}__s{d}.npz")
            rows = z["rows"].astype(np.int64)
            keys = ["rows", "post"] + [k for k in z.files if k.startswith("post_r")]
            assert {"post_r60", "post_r100", "post_r150"} <= set(keys), z.files
            for m, src in groups(f, rows):
                dd = out / ("ss8" if src.startswith("ss8_") else f"f{f}")
                dd.mkdir(parents=True, exist_ok=True)
                np.savez(dd / f"{src}__s{SEEDS[j]}.npz", **{k: z[k][m] for k in keys})
    print(f"[d638 splice] P1 teachers {out}: ss8 (8 strangers) + f<f>/rt1_P04, rt1_P10 for {len(SPLICED)} folds x 3 seeds", flush=True)


def cmd_students(cell, gen):
    assert cell in CELLS[2:] and gen in ("base", "final")
    r = root(cell) if gen == "base" else root(cell, "lp")
    refr = SRC if gen == "base" else REF_LP
    xt = "__ubw" if cell.startswith("c") else ""
    raw = D638 / "p1" / f"{gen}_{cell}"
    for fam, stem in P1.items():
        dd = r / f"p1_{fam}" / "runs"
        assert not dd.exists(), dd
        dd.mkdir(parents=True)
        for f in range(10):
            for j, d in enumerate(DRAWS):
                ref = np.load(refr / f"p1_{fam}" / "runs" / f"lc9_{f}__{stem}isSpr__s{d}.npz")
                te, p = ref["test_idx"], ref["prob_last"].astype(np.float64)
                prov = f"unspliced (the reference {gen} students)"
                if f in SPLICED:
                    q = np.full_like(p, np.nan)
                    for m, src in groups(f, te):
                        sub = "ss8" if src.startswith("ss8_") else f"f{f}"
                        z = np.load(raw / sub / fam / "runs" / f"{src}__{stem}isSpr{xt}__s{SEEDS[j]}.npz")
                        put(te, q, z["test_idx"], z["prob_last"].astype(np.float64), m)
                    assert np.isfinite(q).all()
                    p, prov = q, f"spliced {gen} students ({cell}, seed {SEEDS[j]})"
                save(dd / f"lc9_{f}__{stem}isSpr__s{d}.npz", te, p, ref, prov)
    print(f"[d638 splice] {gen} students -> {r.name}/p1_*/runs (isSpr, 30 files per family)", flush=True)


def cmd_linkp1(cell, gen):
    """the noise cell: its lc9 students (t2p_bench --protocols lc9 into outputs/t2_d638/p1/<gen>_nz/<fam>/runs) linked as the root's p1_<fam>."""
    assert cell == "nz" and gen in ("base", "final")
    r = root(cell) if gen == "base" else root(cell, "lp")
    for fam, stem in P1.items():
        d = D638 / "p1" / f"{gen}_nz" / fam
        n = len(list((d / "runs").glob(f"lc9_*__{stem}isSpr__s86[0-2].npz")))
        assert n == 30, (fam, n)
        (r / f"p1_{fam}").symlink_to(d)
    print(f"[d638 splice] noise cell: {gen} lc9 students linked into {r.name}", flush=True)


def cmd_lproot(cell):
    b, r = root(cell), root(cell, "lp")
    assert cell in CELLS and b.exists() and r.exists() and (r / "teacher_p1").exists(), r   # lpteach created r / teacher_p1
    link_members(r)
    for fam in ROUTE:
        (r / f"route_{fam}").symlink_to(b / f"route_{fam}")
    if cell == "id":                                       # the id root's final students = the reference lp root's (no GPU)
        for fam in P1:
            (r / f"p1_{fam}").symlink_to(REF_LP / f"p1_{fam}")
    print(f"[d638 splice] lp root {r.name}: links (route_* -> {b.name})", flush=True)


def same_npz(a, b):
    za, zb = np.load(a), np.load(b)
    if sorted(za.files) != sorted(zb.files):
        return False, f"keys {sorted(za.files)} vs {sorted(zb.files)}"
    for k in za.files:
        x, y = za[k], zb[k]
        if x.dtype.kind == "f":
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
                return False, k
        elif not np.array_equal(x, y):
            return False, k
    return True, ""


def cmd_check_teachers():
    bad = []
    for mine, ref in ((root("id") / "teacher_p1", SRC / "teacher_p1"), (root("id", "fb") / "teacher_p1", REF_FB / "teacher_p1"),
                      (root("id", "lp") / "teacher_p1", REF_LP / "teacher_p1")):
        files = sorted(p.name for p in ref.glob("lc9_*__s*.npz"))
        assert len(files) == 30 and sorted(p.name for p in mine.glob("lc9_*__s*.npz")) == files, (mine, ref)
        for fn in files:
            ok, why = same_npz(mine / fn, ref / fn)
            if not ok:
                bad.append(f"{mine.parent.name}/{fn}: {why}")
        print(f"[d638 identity] {mine.parent.name}/teacher_p1 vs {ref.parent.name}/teacher_p1: "
              f"{'BITWISE' if not [b_ for b_ in bad if b_.startswith(mine.parent.name)] else 'DIFFERS'} (30 files)", flush=True)
    if bad:
        raise SystemExit("STOP: identity teachers differ: " + "; ".join(bad[:6]))
    print("[d638 identity] base / fb / lp teachers of the no-op root == the references BITWISE", flush=True)


def cmd_check_ii():
    """identity (ii), AMENDED (the first run stopped at 3.83e-4 vs 1e-5: IFNet's POOLED net is not bitwise-deterministic on the GPU, plain /
    is0 already 1.2e-4 apart, ShallowNet bitwise on every form): ShallowNet's isSpr BITWISE AND IFNet's isSpr within 1e-3 with 100 % argmax
    agreement (its own pooled-net nondeterminism; the student path is the same code for both families)."""
    res = {}
    for fam, stem in P1.items():
        a = np.load(D638 / "p1" / "id_ii" / fam / "runs" / f"lc9_0__{stem}isSpr__s860.npz")
        b = np.load(REF_LP / f"p1_{fam}" / "runs" / f"lc9_0__{stem}isSpr__s860.npz")
        assert np.array_equal(a["test_idx"], b["test_idx"]), fam
        pa, pb = a["prob_last"].astype(np.float64), b["prob_last"].astype(np.float64)
        res[fam] = (float(np.abs(pa - pb).max()), float(np.mean(pa.argmax(1) == pb.argmax(1))))
    print(f"[d638 identity (ii)] the P1-student path on lc9_0 / 860 vs the reference isSpr: " +
          ", ".join(f"{k} max |dp| {v[0]:.2e} argmax agree {v[1]:.4f}" for k, v in res.items()), flush=True)
    if not (res["sh"][0] == 0.0 and res["ifn"][0] <= 1e-3 and res["ifn"][1] == 1.0):
        raise SystemExit(f"STOP: identity (ii) fails the amended rule (sh bitwise; ifn <= 1e-3 with 100 % argmax): {res}")


if __name__ == "__main__":
    c = sys.argv[1]
    {"base": lambda: cmd_base(sys.argv[2]), "p1t": lambda: cmd_p1t(sys.argv[2], sys.argv[3]),
     "students": lambda: cmd_students(sys.argv[2], sys.argv[3]), "lproot": lambda: cmd_lproot(sys.argv[2]), "linkp1": lambda: cmd_linkp1(sys.argv[2], sys.argv[3]),
     "check-teachers": cmd_check_teachers, "check-ii": cmd_check_ii}[c]()

#!/usr/bin/env python
"""Track 2: re-pick the package's family weights AND the user-blend r on the SEALED-MIX objective, cross-fitted, from the
per-seed bench result files on disk (T2-D145, keep-busy T2 seed 3). Zero compute.

Every family weight so far was picked on unseen-user val; the sealed mix weighs returning users equally (T2-D124c). This
script scores a grid of packages in the t2p_pkgshape shape (one seed per member per index, p = sum_m w_m p_m, then the causal
prior correction (1, 3) per stream) and picks on the sealed mix with an honest hold-out:
  grid    w = (FBCNet, ShallowNet, ST-L, MOR) on the simplex in steps of --step, x r in --rs; the FBCNet member at blend r is
          fbc + 3 r (fbcb - fbc) (fbcb = the D124c blend at r = 1/3 of the same seed, so 3 fbcb - 2 fbc = the routed per-user
          copy; linear, exact; asserted to be a probability)
  groups  (t2p_pkgshape / T2-D124c) kit test = P04 + P09, seen = seen1 P04 + P10 sessions 2-3, seen1 val = P05 + P08;
          new = (kit test + seen1 val) / 2, mix = (seen + new) / 2; a group score = the mean of its cells' balanced accuracy
  folds   participant p in {P04, P05, P08, P09, P10} x seed half h in {0-5, 6-11}: the pick for (p, h) maximises the mean mix
          over the OTHER half's indices with p's cells left out of every group (ties -> the grid point nearest the anchor in
          L1, then the first); the cross-fitted package scores p's cells at index i in half h with the (p, h) pick
  read    cross-fitted package vs the anchor, paired by index: mix / seen / new / kit test (mean, up / down), bar from
          t2p_pkgshape.BARS; the FULL pick (all participants, all 12 indices) is what would ship, reported with its in-sample
          numbers beside (optimistic, never deciding)
  --check reproduces t2p_pkgshape's numbers for the anchor (every group, every index) before anything else
  prior   (T2-D152) spec "alphas" / "n0s" add the causal prior correction's (alpha, n0) to the grid (default (1, 3) only);
          without "step" / "rs" the weights / r stay at the anchor's; ties -> nearest to the anchor in L1 (n0 on log10)

  python t2p_repick.py --export E --spec spec.json [--out res.json] [--jobs 16]
spec = t2p_pkgshape's members (fbc, fbcb, sh, stl, mor) + {"n": 12, "anchor": {"w": [0.5, 0.2, 0.2, 0.1], "r": 0.3333333333},
        "step": 0.05, "rs": [...], "bar": "d141"}
"""

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from ni26 import metrics  # noqa: E402
from t2p_pkgshape import BARS  # noqa: E402
from t2p_report import causal_prior_correction  # noqa: E402

PROTOS = ("kit", "seen1")
FAMS = ("fbc", "sh", "stl", "mor")
USERS = ("P04", "P05", "P08", "P09", "P10")
G = {}                                   # worker globals (fork)


def simplex(step):
    k = int(round(1 / step))
    return np.array([(a, b, c, k - a - b - c) for a in range(k + 1) for b in range(k + 1 - a) for c in range(k + 1 - a - b)],
                    np.float64) / k


def cell_ba(y, pr, cid, n_cells, K=3):
    """per-cell balanced accuracy = metrics.balanced_accuracy within each cell (mean recall over union(y, pred))."""
    hit = np.zeros((n_cells, K))
    sup = np.zeros((n_cells, K))
    prd = np.zeros((n_cells, K))
    np.add.at(sup, (cid, y), 1)
    np.add.at(prd, (cid, pr), 1)
    np.add.at(hit, (cid, y), (pr == y).astype(float))
    present = (sup > 0) | (prd > 0)
    rec = np.where(sup > 0, hit / np.maximum(sup, 1), 0.0)
    return (rec * present).sum(1) / present.sum(1)


def score_points(job):
    """job = list of (grid point index, w (4,), r, alpha, n0) -> {point: {proto: (n, n_cells) per-cell BA}}"""
    out = {}
    for gi, w, r, a, n0 in job:
        res = {}
        for proto in PROTOS:
            d = G[proto]
            rows = np.empty((d["n"], d["n_cells"]))
            for i in range(d["n"]):
                P = d["P"][i]
                fb = P["fbc"] + 3.0 * r * (P["fbcb"] - P["fbc"])
                mix = w[0] * fb + w[1] * P["sh"] + w[2] * P["stl"] + w[3] * P["mor"]
                pc = causal_prior_correction(mix, d["keys"], a, n0)
                rows[i] = cell_ba(d["y"], pc.argmax(1), d["cid"], d["n_cells"])
            res[proto] = rows
        out[gi] = res
    return out


def load(args, spec):
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y_all = idx["y"].to_numpy().astype(int)
    subj, ses = idx["subject"].to_numpy(), idx["session"].to_numpy()
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    mems, n = spec["members"], int(spec["n"])
    data, files = {}, {}
    for proto in PROTOS:
        te = va = None
        Ps = []
        for i in range(n):
            P = {}
            for m in ("fbc", "fbcb", "sh", "stl", "mor"):
                d = mems[m]
                f = d[proto].format(s=int(d.get("offset", 0)) + i)
                z = np.load(f)
                files.setdefault(m, {}).setdefault(proto, []).append(f)
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), f"{f}: windows differ"
                P[m] = np.concatenate([z["prob_last"], z["prob_val_last"]]).astype(np.float64)
            pu = 3.0 * P["fbcb"] - 2.0 * P["fbc"]
            assert pu.min() > -1e-5 and np.abs(pu.sum(1) - 1).max() < 1e-4, (proto, i, pu.min())
            Ps.append(P)
        rows = np.concatenate([te, va])
        is_test = np.r_[np.ones(len(te), bool), np.zeros(len(va), bool)]
        cl = cells[rows]
        ucell, cid = np.unique(cl, return_inverse=True)
        cs = pd.Series(ucell)
        c_subj = cs.str.split("|").str[0].to_numpy()
        c_ses = cs.str.split("|").str[1].to_numpy()
        c_test = np.array([is_test[cid == c].all() for c in range(len(ucell))])
        assert all(is_test[cid == c].all() or (~is_test[cid == c]).all() for c in range(len(ucell))), "a cell spans test and val"
        if proto == "kit":
            groups = dict(kit_test=c_test, kit_val=~c_test)
        else:
            groups = dict(seen=c_test & np.isin(c_subj, ["P04", "P10"]), p09=c_test & (c_subj == "P09"), seen1_val=~c_test)
            assert set(c_ses[groups["seen"]]) <= {"2", "3"}
        data[proto] = dict(P=Ps, y=y_all[rows], cid=cid, n_cells=len(ucell), keys=(strm[rows], ordr[rows]), n=n,
                           cells=ucell, c_subj=c_subj, groups=groups, rows=rows, is_test=is_test)
    return data, files


def group_scores(ba, data, keep_users=None):
    """ba = {proto: (n, n_cells)} -> {group: (n,)} with cells of users not in keep_users left out (None = all)."""
    out = {}
    for proto in PROTOS:
        d = data[proto]
        for g, mk in d["groups"].items():
            m = mk if keep_users is None else mk & np.isin(d["c_subj"], keep_users)
            out[g] = ba[proto][:, m].mean(1)
    out["new"] = (out["kit_test"] + out["seen1_val"]) / 2
    out["mix"] = (out["seen"] + out["new"]) / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--check-only", action="store_true", help="the anchor check against t2p_pkgshape, then stop")
    args = ap.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    data, files = load(args, spec)
    G.update(data)
    n = int(spec["n"])
    aw, ar = np.asarray(spec["anchor"]["w"], np.float64), float(spec["anchor"]["r"])
    aa, an = float(spec["anchor"].get("alpha", 1.0)), float(spec["anchor"].get("n0", 3.0))
    assert (aa, an) == (1.0, 3.0), "the anchor's prior = the shipped one (1, 3), the one t2p_pkgshape checks"
    # T2-D152: "step" / "rs" absent = weights / r held at the anchor; "alphas" / "n0s" absent = the shipped prior (1, 3) only
    # (so a D145 spec gives D145's grid in D145's order)
    W = simplex(float(spec["step"])) if spec.get("step") else aw[None]
    rs = [float(r) for r in spec["rs"]] if spec.get("rs") else [ar]
    pri = [(float(a), float(n0)) for a in spec.get("alphas", [1.0]) for n0 in spec.get("n0s", [3.0])]
    pts = [(w, r, a, n0) for a, n0 in pri for r in rs for w in W]
    ia = [k for k, (w, r, a, n0) in enumerate(pts) if np.abs(w - aw).max() < 1e-9 and abs(r - ar) < 1e-9 and (a, n0) == (aa, an)]
    assert len(ia) == 1, f"the anchor {aw} r {ar} prior ({aa}, {an}) is not on the grid"
    ia = ia[0]

    # --check: the anchor through this code = t2p_pkgshape's arm on the same files (every index, every group)
    import t2p_pkgshape as PS
    a_ba = score_points([(ia, aw, ar, aa, an)])[ia]
    a_gs = group_scores(a_ba, data)
    idx = pd.read_csv(Path(args.export) / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    strm = (idx["subject"] + "|" + idx["session"]).to_numpy()
    ordr = idx["order"].to_numpy()
    worst = 0.0
    for proto, gmap in (("kit", dict(kit_test="test", kit_val="val")), ("seen1", dict(seen="seen", seen1_val="val", p09="p09"))):
        d = data[proto]
        for i in (0, n - 1):
            P = d["P"][i]
            fb = P["fbc"] + 3.0 * ar * (P["fbcb"] - P["fbc"])
            mix = aw[0] * fb + aw[1] * P["sh"] + aw[2] * P["stl"] + aw[3] * P["mor"]
            full = np.full((len(idx), 3), np.nan)
            full[d["rows"]] = mix
            ok = np.zeros(len(idx), bool)
            ok[d["rows"]] = True
            Pc = np.zeros_like(full)
            Pc[ok] = causal_prior_correction(full[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
            pr = Pc.argmax(1)
            tm = np.zeros(len(idx), bool)
            tm[d["rows"][d["is_test"]]] = True
            vm = ok & ~tm
            subj = idx["subject"].to_numpy()
            masks = dict(test=tm, val=vm) if proto == "kit" else dict(seen=tm & np.isin(subj, ["P04", "P10"]), p09=tm & (subj == "P09"), val=vm)
            for g, pg in gmap.items():
                ref = metrics.cell_balanced_accuracy(y[masks[pg]], pr[masks[pg]], cells[masks[pg]])[0]
                worst = max(worst, abs(ref - a_gs[g][i]))
    lv = {g: float(a_gs[g].mean()) for g in ("kit_test", "kit_val", "seen", "seen1_val")}
    print(f"anchor check vs the kit metric path: max |diff| {worst:.2e}; anchor levels {lv}", flush=True)
    assert worst < 1e-12, worst
    if spec.get("anchor_levels"):
        dl = max(abs(lv[k] - v) for k, v in spec["anchor_levels"].items())
        print(f"anchor levels vs the registered ones: max |diff| {dl:.2e}", flush=True)
        assert dl < 1e-9, dl
    if args.check_only:
        return

    # every grid point: per-cell BA for every index, both protocols
    jobs = [list(c) for c in np.array_split(np.array([(k, *pts[k][0], *pts[k][1:]) for k in range(len(pts))], dtype=object),
                                            min(len(pts), args.jobs * 4))]
    jobs = [[(int(t[0]), np.asarray(t[1:5], np.float64), float(t[5]), float(t[6]), float(t[7])) for t in job] for job in jobs if len(job)]
    BA = {p: np.empty((len(pts), n, data[p]["n_cells"])) for p in PROTOS}
    with ProcessPoolExecutor(args.jobs) as ex:
        for res in ex.map(score_points, jobs):
            for k, v in res.items():
                for p in PROTOS:
                    BA[p][k] = v[p]
    print(f"scored {len(pts)} grid points x {n} indices x 2 protocols", flush=True)

    def gs_all(keep):
        """{group: (n_points, n)} for cells of `keep` users."""
        out = {}
        for proto in PROTOS:
            d = data[proto]
            for g, mk in d["groups"].items():
                m = mk if keep is None else mk & np.isin(d["c_subj"], keep)
                out[g] = BA[proto][:, :, m].mean(2)
        out["new"] = (out["kit_test"] + out["seen1_val"]) / 2
        out["mix"] = (out["seen"] + out["new"]) / 2
        return out

    dist_a = np.array([np.abs(w - aw).sum() + abs(r - ar) + abs(a - aa) + abs(np.log10(n0 / an)) for w, r, a, n0 in pts])

    def pick(obj):
        best = obj.max()
        cand = np.flatnonzero(obj >= best - 1e-12)
        return int(cand[np.argmin(dist_a[cand])])

    halves = [np.arange(0, n // 2), np.arange(n // 2, n)]
    picks = {}
    for p in USERS:
        keep = [u for u in USERS if u != p]
        gs = gs_all(keep)
        for h, ix in enumerate(halves):
            other = halves[1 - h]
            picks[(p, h)] = pick(gs["mix"][:, other].mean(1))
    # the cross-fitted package: user p's cells at index i (half h) scored with picks[(p, h)]
    cf = {p: np.empty((n, data[p]["n_cells"])) for p in PROTOS}
    for proto in PROTOS:
        d = data[proto]
        for h, ix in enumerate(halves):
            for p in USERS:
                cm = d["c_subj"] == p
                if not cm.any():
                    continue
                k = picks[(p, h)]
                cf[proto][np.ix_(ix, np.flatnonzero(cm))] = BA[proto][k][np.ix_(ix, np.flatnonzero(cm))]
        other_users = sorted(set(d["c_subj"]) - set(USERS))
        assert not other_users, other_users
    cf_gs = group_scores(cf, data)
    an_gs = group_scores({p: BA[p][ia] for p in PROTOS}, data)
    full_k = pick(gs_all(None)["mix"].mean(1))
    full_gs = group_scores({p: BA[p][full_k] for p in PROTOS}, data)

    def delta(gs):
        r = {}
        for g in ("mix", "seen", "new", "kit_test", "kit_val", "seen1_val", "p09"):
            v = gs[g] - an_gs[g]
            r[g] = dict(d=float(v.mean()), up=int((v > 0).sum()), down=int((v < 0).sum()), sd=float(v.std(ddof=1)))
        return r

    def desc(k):
        w, r, a, n0 = pts[k]
        return dict(w=dict(zip(FAMS, [round(float(x), 4) for x in w])), r=round(r, 4), alpha=a, n0=n0)

    res = dict(spec=spec, files=files, n_points=len(pts), anchor=desc(ia),
               anchor_levels={g: float(an_gs[g].mean()) for g in ("kit_test", "kit_val", "seen", "seen1_val", "mix")},
               picks={f"{p}|h{h}": desc(k) for (p, h), k in picks.items()},
               crossfit=delta(cf_gs), full_pick=desc(full_k), full_in_sample=delta(full_gs))
    bar = BARS.get(spec.get("bar", ""))
    if bar:
        r = res["crossfit"]
        res["bar"] = {k: r[k]["d"] >= v and r[k]["up"] >= bar.get(f"{k}_up", 0) for k, v in bar.items() if not k.endswith("_up")}
        res["met"] = all(res["bar"].values())
    print(f"anchor {res['anchor']}: levels {res['anchor_levels']}")
    print("fold picks: " + "; ".join(f"{k} {v['w']} r {v['r']} prior ({v['alpha']:g}, {v['n0']:g})" for k, v in res["picks"].items()))
    for name in ("crossfit", "full_in_sample"):
        print(f"\n{name} vs anchor" + (f"  (full pick {res['full_pick']})" if name == "full_in_sample" else ""))
        for g, v in res[name].items():
            print(f"  {g:9s} {v['d']:+.4f}  ({v['up']} up / {v['down']} down, sd {v['sd']:.4f})")
    if bar:
        print(f"\nbar {spec['bar']} (cross-fitted): " + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in res["bar"].items())
              + f" -> {'MET' if res['met'] else 'not met'}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

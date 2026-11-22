#!/usr/bin/env python
"""T2-D317 (R6, lead/FIRST_PRINCIPLES_2026-11-01_T2.md): does a stream's fingerprint tell a returning PROTEUS user from a
new one? Zero GPU, descriptive, report only (no build, no upload; pre-registered in DECISIONS_T2 T2-D317).
  - T2-D124's fingerprints of all 14 public streams (outputs/t2_d124b/fp_proteus.npy, row i = export index row i);
  - 10 fits: leave out user v, fit the package's space (t2p_fpcalib.fit_space: StandardScaler + shrinkage LDA on user
    ids) on the other 9 users' session-1 windows;
  - NEW = every session of v (14 instances); SEEN = P04 / P10 sessions 2 / 3 when the owner is among the 9 (36 instances
    of 4 distinct streams);
  - per stream, the running mean of the projected windows at t windows: S1 = distance to the nearest centroid (D124b's
    s_t), S2 = nearest / second-nearest distance; k* = the nearest centroid;
  - AUC (seen vs new; smaller = seen) at every t, 90 % CI by a user-cluster bootstrap (and a stream-cluster one).
  mem-guard python tracks/t2_bci/scripts/t2_r6_louo.py  ->  outputs/t2_r6/{read.json, r6_louo.png, curves.npz}
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_fpcalib as FC  # noqa: E402

OUTS = Path.home() / "workspace/outputs/neural_interface_challenge"
FP = OUTS / "t2_d124b/fp_proteus.npy"
EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = OUTS / "t2_r6"
T_READ = [10, 30, 60, 120]
T_PRIMARY = 60
N_BOOT = 2000


def auc(pos, neg):
    """P(score_seen > score_new) + ties / 2; scores are NEGATED distances, so smaller distance = more 'seen'."""
    pos, neg = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def main():
    # T2-D489: --fp / --out read another fingerprint (the ST-L encoder embeddings) through the same protocol; defaults = T2-D317's
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", default=str(FP))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    OUT = Path(a.out)
    Fp = np.load(a.fp)
    idx = pd.read_csv(EXP / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    assert len(idx) == len(Fp) and (idx["i"].to_numpy() == np.arange(len(idx))).all()
    subj, ses = idx["subject"].to_numpy(), idx["session"].to_numpy()
    users = sorted(set(subj))
    assert len(users) == 10
    streams = {k: g.sort_values("order")["i"].to_numpy() for k, g in idx.groupby(["subject", "session"], sort=True)}
    multi = sorted({u for u, s in streams if s != "1"})
    assert multi == ["P04", "P10"], multi
    t_min = min(len(r) for r in streams.values())

    inst = []                                                     # one row per (fit, scored stream)
    for v in users:
        trm = (subj != v) & (ses == "1")
        proj, mu, cls = FC.fit_space(Fp[trm], subj[trm])
        for (u, s), rows in streams.items():
            if u == v:
                lab = 0
            elif s != "1":
                lab = 1
            else:
                continue                                          # a training stream of this fit
            Z = proj(Fp[rows])
            zb = np.cumsum(Z, 0) / np.arange(1, len(Z) + 1)[:, None]
            d = np.sort(np.linalg.norm(zb[:, None, :] - mu[None], axis=-1), 1)
            k = np.linalg.norm(zb[:, None, :] - mu[None], axis=-1).argmin(1)
            inst.append(dict(fit=v, user=u, session=s, seen=lab, S1=d[:, 0], S2=d[:, 0] / d[:, 1],
                             kstar=[cls[j] for j in k]))
    seen = [r for r in inst if r["seen"]]
    new = [r for r in inst if not r["seen"]]
    assert len(seen) == 36 and len(new) == 14, (len(seen), len(new))

    def at(r, stat, t):
        x = r[stat]
        return x[-1] if t is None else x[t - 1]

    def auc_t(stat, t, S=None, N=None):
        S, N = seen if S is None else S, new if N is None else N
        return auc([-at(r, stat, t) for r in S], [-at(r, stat, t) for r in N])

    rng = np.random.default_rng(20261101)
    seen_users = sorted({r["user"] for r in seen})
    seen_streams = sorted({(r["user"], r["session"]) for r in seen})
    new_users = sorted({r["user"] for r in new})

    def boot(stat, t, by):
        out = []
        for _ in range(N_BOOT):
            if by == "user":
                S = [r for u in rng.choice(seen_users, len(seen_users)) for r in seen if r["user"] == u]
            else:
                S = [r for st in rng.choice(len(seen_streams), len(seen_streams)) for r in seen
                     if (r["user"], r["session"]) == seen_streams[st]]
            N = [r for u in rng.choice(new_users, len(new_users)) for r in new if r["user"] == u]
            out.append(auc_t(stat, t, S, N))
        return [float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))]

    read = dict(n_seen_instances=len(seen), n_seen_streams=len(seen_streams), n_seen_users=len(seen_users),
                n_new_instances=len(new), n_new_users=len(new_users), t_min_stream=t_min, table={})
    for stat in ("S1", "S2"):
        for t in T_READ + [None]:
            key = f"{stat}@{'end' if t is None else t}"
            read["table"][key] = dict(auc=auc_t(stat, t), ci90_user=boot(stat, t, "user"), ci90_stream=boot(stat, t, "stream"),
                                      seen_median=float(np.median([at(r, stat, t) for r in seen])),
                                      new_min=float(min(at(r, stat, t) for r in new)),
                                      new_median=float(np.median([at(r, stat, t) for r in new])))
    read["primary"] = dict(key=f"S1@{T_PRIMARY}", **read["table"][f"S1@{T_PRIMARY}"])
    read["kstar_own_end"] = f"{sum(r['kstar'][-1] == r['user'] for r in seen)} / {len(seen)}"
    read["kstar_own_t10plus"] = float(np.mean([np.mean([k == r["user"] for k in r["kstar"][9:]]) for r in seen]))
    ts = np.arange(10, t_min + 1)
    curves = {stat: np.array([auc_t(stat, int(t)) for t in ts]) for stat in ("S1", "S2")}
    bands = {stat: np.array([boot(stat, int(t), "user") if t in (10, 20, 30, 45, 60, 90, 120, 150, t_min) else [np.nan, np.nan]
                             for t in ts]) for stat in ("S1", "S2")}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "curves.npz", t=ts, **{f"auc_{k}": v for k, v in curves.items()},
                        **{f"ci_{k}": v for k, v in bands.items()})
    (OUT / "read.json").write_text(json.dumps(read, indent=1))
    plot(seen, new, ts, curves, bands, t_min)
    print(json.dumps(dict(primary=read["primary"], kstar_own_end=read["kstar_own_end"],
                          table={k: (round(v["auc"], 3), [round(x, 3) for x in v["ci90_user"]]) for k, v in read["table"].items()})))


def plot(seen, new, ts, curves, bands, t_min):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    HUE = {"P04": "#2a78d6", "P10": "#eb6834"}                    # categorical slots 1 / 2 (validated, light)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURF)
    for ax in (a, b):
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    t_all = np.arange(1, t_min + 1)
    for r in new:
        a.plot(t_all, r["S1"][:t_min], color="#a9a8a2", lw=1, zorder=1)
    ends = []
    for (u, s) in sorted({(r["user"], r["session"]) for r in seen}):
        med = np.median([r["S1"][:t_min] for r in seen if (r["user"], r["session"]) == (u, s)], 0)
        a.plot(t_all, med, color=HUE[u], lw=2, zorder=3, solid_capstyle="round")
        ends.append([med[-1], f"{u} s{s}"])
    ends.sort()
    for i in range(1, len(ends)):                                 # dodge the direct labels (>= 0.45 units apart)
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.45)
    for y, lab in ends:
        a.text(t_min + 3, y, lab, va="center", fontsize=8, color=INK2)
    a.plot([], [], color="#a9a8a2", lw=1, label="new user (left out; 14 streams)")
    for u in ("P04", "P10"):
        a.plot([], [], color=HUE[u], lw=2, label=f"{u} later session (seen; median of 9 fits)")
    a.axvline(T_PRIMARY, color=INK2, lw=0.8, ls=":")
    a.set_xlim(1, t_min + 22)
    a.set_ylim(None, a.get_ylim()[1] + 1.8)                      # room for the legend above the highest stream
    a.set_xlabel("windows seen in the stream (t)")
    a.set_ylabel("distance to nearest training-user centroid")
    a.set_title("A  Running-mean fingerprint distance (S1)", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8, loc="upper right")

    b.axhspan(0.8, 1.0, color="#eef4fc", zorder=0)
    b.axhline(0.5, color=INK2, lw=0.8, ls=":")
    b.axhline(0.7, color=INK2, lw=0.8, ls="--")
    xr = t_min + 12                                               # reference labels live in the right margin
    b.text(xr, 0.81, "viable (≥ 0.8)", fontsize=8, color=INK2, va="bottom")
    b.text(xr, 0.705, "closed (< 0.7)", fontsize=8, color=INK2, va="bottom")
    b.text(xr, 0.49, "chance", fontsize=8, color=INK2, va="top")
    for stat, ls, lab, dx in (("S1", "-", "S1 nearest distance", 0), ("S2", "--", "S2 nearest / 2nd-nearest", 3)):
        b.plot(ts, curves[stat], color=INK if stat == "S1" else INK2, lw=2, ls=ls, label=lab)
        ok = ~np.isnan(bands[stat][:, 0])
        b.errorbar(ts[ok] + dx, curves[stat][ok], yerr=np.abs(bands[stat][ok].T - curves[stat][ok]), fmt="o", ms=3,
                   color=INK if stat == "S1" else INK2, lw=0.8, capsize=2)
    b.plot([xr + 3, xr + 3], [0.51, 0.56], ls="none", marker="D", ms=4.5, color=INK2)
    b.text(xr + 9, 0.535, "D124b external\nK 6: 0.56, K 7: 0.51", va="center", fontsize=7.5, color=INK2)
    b.axvline(T_PRIMARY, color=INK2, lw=0.8, ls=":")
    b.set_xlim(ts[0], t_min + 80)
    b.set_xticks([t for t in (10, 50, 100, 150) if t <= t_min] + [t_min])
    b.set_ylim(0.0, 1.0)
    b.set_xlabel("windows seen in the stream (t)")
    b.set_ylabel("AUC, seen vs new (90 % CI, user bootstrap)")
    b.set_title("B  Seen / new separation (2 seen users, 4 streams)", loc="left", fontsize=10)
    b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "r6_louo.png", dpi=200, facecolor=SURF)


if __name__ == "__main__":
    main()

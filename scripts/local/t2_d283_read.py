#!/usr/bin/env python
"""T2-D283 read (zero GPU; written before the jobs ended, as pre-registered in docs/DECISIONS_T2.md): the TTT panel's Step 1.
After `s1 fetch ni26 ni26-t2p-d283-ttaln-<kit|seen1>-<jobid>`: the in-job adopted ST-L (seeds 0-11, plain) and its LayerNorm-affine
test-time MAE twins mae1e-05P / mae1e-05K, beside T2-D253's fbc_rc75 (204+i), sh_vr (168+i), mor (i); w4321.
  anchor c_inj = w4321 with the in-job plain ST-L; arms c_P / c_K; members alone; c_disk (T2-D247's on-disk stl_rcll) beside.
Per arm: CANDIDATE iff d141 (mix >= +0.005, >= 9 / 12 up, kit test >= -0.005) AND new >= 0 AND >= 3 of the 4 kit participants up.
KILL iff member-alone kit val delta (seeds 0-5 mean) < -0.01, OR a collapse (a stream where the adapted member gives one class to
>= 90 % of its windows and the plain member does not), OR 13.26 ms + the median TTA overhead per window (kit) > 18.1 ms.
Descriptive: LN drift per stream, argmax flip rate vs plain, the T2-D252 hard / easy split.
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python scripts/local/t2_d283_read.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
from t2p_report import causal_prior_correction  # noqa: E402

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
REC = "fmE30b64le0.0001lh0.001wd0.05ll0.75ema0dp0mx0.4cgrc0.75"
CPT = {"kit": "cptstlarge-d89b", "seen1": "cptpt-d111-with"}
OFFS = {"fbc_rc75": 204, "sh_vr": 168, "mor": 0}
W = {"fbc_rc75": 0.4, "sh_vr": 0.3, "mor": 0.1}
ARMS = ("P", "K")
N = 12


def main():
    out = O / "t2_d283"
    out.mkdir(exist_ok=True)
    runs = {}
    for p in ("kit", "seen1"):
        d = sorted(S.glob(f"ni26-t2p-d283-ttaln-{p}-[0-9]*/runs"))
        assert len(d) == 1, f"need one fetched dir for {p}: {d}"
        runs[p] = d[0]
    s253 = json.load(open(O / "t2_d253/spec.json"))
    mem = {m: s253["members"][m] for m in OFFS}
    assert {m: d["offset"] for m, d in mem.items()} == OFFS
    mem["stl_disk"] = s253["members"]["stl_rcll"]
    for k, suf in (("stl_inj", ""), ("stl_P", "mae1e-05P"), ("stl_K", "mae1e-05K")):
        mem[k] = dict(offset=0, **{p: str(runs[p] / f"{p}__fm-st_large-mean__sz__n00__{REC}{CPT[p]}{suf}__s{{s}}.npz") for p in runs})
    arms = {"c_inj": dict(W, stl_inj=0.2), "c_P": dict(W, stl_P=0.2), "c_K": dict(W, stl_K=0.2), "c_disk": dict(W, stl_disk=0.2),
            "a_inj": {"stl_inj": 1.0}, "a_P": {"stl_P": 1.0}, "a_K": {"stl_K": 1.0}, "c_noST": {"fbc_rc75": 0.5, "sh_vr": 0.375, "mor": 0.125}}
    sp = dict(n=N, anchor="c_inj", bar="d141", members=mem, arms=arms)
    (out / "spec.json").write_text(json.dumps(sp, indent=1) + "\n")
    with open(out / "read.txt", "w") as fh:
        rc = subprocess.run([sys.executable, "-u", str(ROOT / "tracks/t2_bci/scripts/t2p_pkgshape.py"), "--export", str(E), "--spec", str(out / "spec.json"),
                             "--out", str(out / "read.json")], stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f"t2p_pkgshape rc {rc} (see {out / 'read.txt'})"
    rj = json.load(open(out / "read.json"))
    r = rj["arms"]
    rows = {(x["proto"], x["i"], x["arm"]): x for x in rj["rows"]}
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    subj = idx["subject"].str.split("/").str[-1].to_numpy()
    y = idx["y"].to_numpy().astype(int)
    strm = (subj + "|" + idx["session"].to_numpy()).astype(object)
    ordr = idx["order"].to_numpy()
    desc = {a: dict(flip=[], hard=[], easy=[], drift=[], over_ms=[], collapse=[]) for a in ARMS}
    for proto in ("kit", "seen1"):
        for i in range(N):
            P, te, Z = {}, None, {}
            for m, d in mem.items():
                z = np.load(d[proto].format(s=int(d["offset"]) + i))
                if te is None:
                    te, va = z["test_idx"], z["val_idx"]
                assert np.array_equal(z["test_idx"], te) and np.array_equal(z["val_idx"], va), (m, proto, i)
                Pm = np.full((len(idx), 3), np.nan)
                Pm[te], Pm[va] = z["prob_last"], z["prob_val_last"]
                P[m], Z[m] = Pm, z
            ok = np.zeros(len(idx), bool)
            ok[te] = ok[va] = True
            mix = np.zeros(len(idx), bool)
            if proto == "kit":
                mix[te] = True
            else:
                mix[va] = True
                mix[te[np.isin(subj[te], ["P04", "P10"])]] = True

            def pred(p):
                q = np.zeros_like(p)
                q[ok] = causal_prior_correction(p[ok], (strm[ok], ordr[ok]), 1.0, 3.0)
                return q.argmax(1)
            hard = mix & (pred(sum(w * P[m] for m, w in arms["c_noST"].items())) != y)
            ez = mix & ~hard
            base = pred(P["stl_inj"])
            raw0 = P["stl_inj"].argmax(1)
            for a in ARMS:
                pa, rawa = pred(P[f"stl_{a}"]), P[f"stl_{a}"].argmax(1)
                desc[a]["flip"].append(float((rawa[ok] != raw0[ok]).mean()))
                desc[a]["hard"].append(float((pa[hard] == y[hard]).mean() - (base[hard] == y[hard]).mean()))
                desc[a]["easy"].append(float((pa[ez] == y[ez]).mean() - (base[ez] == y[ez]).mean()))
                z = Z[f"stl_{a}"]
                desc[a]["drift"] += [float(v) for v in np.concatenate([z["ln_drift_test"], z["ln_drift_val"]])]
                if proto == "kit":
                    tp, ta = z["tta_s_per_window"]
                    desc[a]["over_ms"].append(1e3 * float(ta - tp))
                for s_ in np.unique(strm[ok]):
                    w_ = ok & (strm == s_)
                    fa = np.bincount(rawa[w_], minlength=3).max() / w_.sum()
                    f0 = np.bincount(raw0[w_], minlength=3).max() / w_.sum()
                    if fa >= 0.9 and f0 < 0.9:
                        desc[a]["collapse"].append(f"{proto}:{i}:{s_} ({fa:.2f})")
    res = {}
    print(f"in-job plain vs on-disk stl_rcll (c_disk vs c_inj): mix {r['c_disk']['mix']['d']:+.4f}, kit test {r['c_disk']['kit_test']['d']:+.4f} (the re-run's own draw noise)")
    for a in ARMS:
        c, dd = r[f"c_{a}"], desc[a]
        kv = float(np.mean([rows[("kit", i, f"a_{a}")]["val"] - rows[("kit", i, "a_inj")]["val"] for i in range(6)]))
        over = float(np.median(dd["over_ms"]))
        kparts = {k: v for k, v in c["participants"].items() if k.startswith("kit:")}
        n_up = sum(v > 0 for v in kparts.values())
        kill = []
        if kv < -0.01:
            kill.append(f"member kit val {kv:+.4f} < -0.01 (seeds 0-5)")
        if dd["collapse"]:
            kill.append(f"collapse {dd['collapse'][:3]}")
        if 13.26 + over > 18.1:
            kill.append(f"time 13.26 + {over:.2f} ms > 18.1")
        cand = (not kill and c["mix"]["d"] >= 0.005 and c["mix"]["up"] >= 9 and c["kit_test"]["d"] >= -0.005 and c["new"]["d"] >= 0 and n_up >= 3)
        v = "KILLED: " + "; ".join(kill) if kill else ("CANDIDATE -> T2-D283r (seeds 12-23)" if cand else "NOT ADOPTED")
        res[a] = dict(verdict=v, mix=c["mix"], kit_test=c["kit_test"], new=c["new"], seen=c["seen"], kit_val=c["kit_val"], kit_parts=kparts,
                      member_kit_val_0_5=kv, overhead_ms=over, flip=float(np.mean(dd["flip"])), hard=float(np.mean(dd["hard"])),
                      easy=float(np.mean(dd["easy"])), drift_median=float(np.median(dd["drift"])), drift_max=float(np.max(dd["drift"])),
                      collapse=dd["collapse"])
        x = res[a]
        print(f"arm {a}: c_{a} vs c_inj mix {c['mix']['d']:+.4f} ({c['mix']['up']} / {N} up), kit test {c['kit_test']['d']:+.4f}, new {c['new']['d']:+.4f}, "
              f"seen {c['seen']['d']:+.4f}, kit val {c['kit_val']['d']:+.4f}; kit participants up {n_up} / 4 ("
              + ", ".join(f"{k[4:]} {v_:+.4f}" for k, v_ in kparts.items()) + ")")
        print(f"   member kit val (0-5) {kv:+.4f}; TTA overhead {over:.2f} ms / window; flips {100 * x['flip']:.2f} %; hard {x['hard']:+.4f}, easy {x['easy']:+.4f}; "
              f"LN drift median {x['drift_median']:.2e} (max {x['drift_max']:.2e}); collapses {len(dd['collapse'])} -> {v}")
    overall = "CANDIDATE(S): " + ", ".join(a for a in ARMS if res[a]["verdict"].startswith("CANDIDATE")) if any(res[a]["verdict"].startswith("CANDIDATE") for a in ARMS) \
        else "no candidate -> the TTT line closes on T2"
    print("VERDICT:", overall)
    (out / "verdict.json").write_text(json.dumps(dict(arms=res, overall=overall), indent=1) + "\n")


if __name__ == "__main__":
    main()

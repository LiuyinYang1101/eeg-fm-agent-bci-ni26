#!/usr/bin/env python
"""T2-D281 / D281b (zero compute, a gate; pre-registered in docs/DECISIONS_T2.md): member audit of every ZIP T2 may ship + the warm-up twins.

From each ZIP's BUILD log (sofia_s1/logs/<job>.out): every member's kit-test cell BA line and the ST-L / MOR per-epoch losses; the
member count against the ZIP's config.json. (CNN loss curves are not in any build log: ni26.train.fit_eval never prints them.)
Sealed-shippable ZIPs (all-data builds: the kit-test line is IN-SAMPLE), T2-D281 — a member FAILS iff
  (a) in-sample < 0.50; (b) < its family's median over the audited builds - 0.10 (family = member kind + recipe);
  (c) ST-L / MOR: last-epoch loss > 0.85 x epoch-1 loss, or last > min + 0.10; (d) members / seeds in the log != the config's.
Warm-up twins (--train kit: the kit-test line is HELD OUT), T2-D281b — a CNN seed fails iff < 0.40 or < its twin family's median - 0.10;
  the ST-L / MOR iff < 0.40 or < the min - 0.05 of the 12 package-shape kit files of its recipe (T2-D253 spec stl_rcll / mor); (c), (d) as above.
All 13, T2-D281b (c'): an ST-L / MOR curve fails iff last - min > 3 x its kind's tail noise (median over the 13 of the sd of the last 5 losses).
  mem-guard ~/miniconda3/envs/neural_interface_challenge/bin/python -I scripts/local/t2_d281_audit.py [--add LABEL:ZIP:LOG ...] [--out audit_<row>.json]
"""

import json
import re
import statistics
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

O = Path.home() / "workspace/outputs/neural_interface_challenge"
S = O / "sofia_s1"
E = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"
OUT = O / "t2_d281"
ZIPS = {  # sha8: (repacked ZIP, build log); all-data builds that T2 may ship
    "5f538614": ("ni26-t2p-repack-d248-u5st-1535074/t2p_d248_u5st_r.zip", "ni26-t2p-pkg-d248-u5st-1535073"),
    "3a716ad4": ("ni26-t2p-repack-d276-vp2r15-1538347/t2p_d276_vp2r15_r.zip", "ni26-t2p-pkg-d276-vp2r15-1538346"),
    "9eb2e084": ("ni26-t2p-repack-d276-vp2r15d1-1538349/t2p_d276_vp2r15d1_r.zip", "ni26-t2p-pkg-d276-vp2r15d1-1538348"),
    "54188b5a": ("ni26-t2p-repack-d276-nv15d1-1538351/t2p_d276_nv15d1_r.zip", "ni26-t2p-pkg-d276-nv15d1-1538350"),
    "5acacd71": ("ni26-t2p-repack-d210-u1-1533012/t2p_d210_u1.zip", "ni26-t2p-pkg-d162-line-1523376"),
    "f8238563": ("ni26-t2p-repack-d210-nv1-1533014/t2p_d210_nv1.zip", "ni26-t2p-pkg-d162-draw1-1523484"),
    "5dfa7e82": ("ni26-t2p-repack-d210-nv2-1533015/t2p_d210_nv2.zip", "ni26-t2p-pkg-d162-draw2-1523486"),
    "c1d30428": ("ni26-t2p-repack-d210-nv3-1533016/t2p_d210_nv3.zip", "ni26-t2p-pkg-d162-draw3-1523488"),
    "f9002dc7": ("ni26-t2p-repack-d210-nv4-1533017/t2p_d210_nv4.zip", "ni26-t2p-pkg-d162-draw4-1523490"),
    "191c922d": ("ni26-t2p-repack-d211-nv15-1533211/t2p_d211_nv15_r.zip", "ni26-t2p-pkg-d211-nv15-1533210"),
    "5816509e": ("ni26-t2p-repack-d232-u5rc-1534340/t2p_d232_u5rc_r.zip", "ni26-t2p-pkg-d232-u5rc-1534339"),
}
TWINS = {  # T2-D281b: the kit-split warm-up twins (T2-W5 / T2-W6)
    "09e14f30": ("ni26-t2p-repack-d279-kit10-1537585/t2p_d279_kit10_r.zip", "ni26-t2p-pkg-d279-kit10-1537584"),
    "0777788c": ("ni26-t2p-repack-d279-kit15-1537587/t2p_d279_kit15_r.zip", "ni26-t2p-pkg-d279-kit15-1537586"),
}
R_CNN1 = re.compile(r"^\[t2p_final\] seed (\d+): kit-test cellBA ([\d.]+)")
R_OTHER = re.compile(r"^\[t2p_final\] (\w+) seed (\d+): kit-test cellBA ([\d.]+)")
R_EP = re.compile(r"^\[t2p_final (fm|mor) s(\d+)\]\s+ep (\d+) loss ([\d.]+)")


def seeds(v):
    return sorted(int(x) for x in str(v).split(","))


def parse(sha, zp, log, rows, curves, problems):
    with zipfile.ZipFile(S / zp) as z:
        cfg = json.loads(z.read("config.json"))
    fams = {"cnn1": f"{cfg['model']}|{cfg['recipe']}",
            "cnn2": f"{cfg['cnn2']['model']}|{cfg['cnn2'].get('recipe', cfg['recipe'])}"}
    st = [m for m in cfg["fm_members"] if m["family"] == "st"]
    mor = [m for m in cfg["fm_members"] if m["family"] == "mor"]
    fams["st"] = "st|" + "|".join(f"{k}{st[0].get(k)}" for k in ("epochs", "mixup", "llrd", "rc")) if st else None
    fams["mor"] = "mor|" + "|".join(f"{k}{mor[0].get(k)}" for k in ("epochs", "lr_enc", "lora_in")) if mor else None
    want = {"cnn1": seeds(cfg["seeds"]), "cnn2": seeds(cfg["cnn2"]["seeds"]),
            "st": sorted(int(m["seed"]) for m in st), "mor": sorted(int(m["seed"]) for m in mor)}
    got = {k: [] for k in want}
    ep = {}
    for ln in (S / "logs" / f"{log}.out").read_text(errors="replace").splitlines():
        if m := R_CNN1.match(ln):
            got["cnn1"].append((int(m[1]), float(m[2])))
        elif m := R_OTHER.match(ln):
            k = {"FM": "st", "MOR": "mor", cfg["cnn2"]["model"]: "cnn2"}.get(m[1])
            assert k, ln
            got[k].append((int(m[2]), float(m[3])))
        elif m := R_EP.match(ln):
            ep.setdefault(("st" if m[1] == "fm" else "mor", int(m[2])), []).append((int(m[3]), float(m[4])))
    for k in want:
        if sorted(s for s, _ in got[k]) != want[k]:
            problems.append(f"{sha} (d) {k}: log seeds {sorted(s for s, _ in got[k])} != config {want[k]}")
        for s, v in got[k]:
            rows.append(dict(zip=sha, log=log, kind=k, family=fams[k], seed=s, score=v))
    for (k, s), c in ep.items():
        n_ep = int((st if k == "st" else mor)[0]["epochs"])
        if [e for e, _ in c] != list(range(1, n_ep + 1)):
            problems.append(f"{sha} (d) {k} s{s}: epochs logged {[e for e, _ in c][:3]}...{len(c)} != 1..{n_ep}")
        lo = [v for _, v in c]
        curves[f"{sha}:{k}:s{s}"] = dict(kind=k, first=lo[0], last=lo[-1], min=min(lo), n=len(lo), tail_sd=float(np.std(lo[-5:], ddof=1)),
                                         fail_c=lo[-1] > 0.85 * lo[0] or lo[-1] > min(lo) + 0.10)
    for k in ("st", "mor"):
        for s in want[k]:
            if (k, s) not in ep:
                problems.append(f"{sha} (d) {k} s{s}: no epoch lines")


def reference():
    """T2-D281b: the 12 package-shape kit files of the twins' ST-L / MOR recipe -> raw argmax cell BA on the kit test (the log's metric)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from ni26 import metrics
    idx = pd.read_csv(E / "index.csv", dtype={"subject": str, "session": str})
    idx["subject"] = idx["subject"].str.split("/").str[-1]
    y = idx["y"].to_numpy().astype(int)
    cells = (idx["subject"] + "|" + idx["session"] + "|" + idx["interface"]).to_numpy()
    spec = json.load(open(O / "t2_d253/spec.json"))
    ref = {}
    for k, m in (("st", "stl_rcll"), ("mor", "mor")):
        d = spec["members"][m]
        v = []
        for i in range(12):
            z = np.load(d["kit"].format(s=int(d["offset"]) + i))
            te = z["test_idx"]
            v.append(metrics.cell_balanced_accuracy(y[te], z["prob_last"].argmax(1), cells[te])[0])
        ref[k] = dict(member=m, scores=[float(x) for x in v], min=float(min(v)), median=float(np.median(v)))
    return ref


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], metavar="LABEL:ZIP:LOG",
                    help="T2-D284 on: a new all-data build to audit with the sealed-shippable rules (ZIP relative to sofia_s1, LOG = the build job name-id)")
    ap.add_argument("--out", default="audit.json")
    args = ap.parse_args()
    for spec in args.add:
        lab, zp, log = spec.split(":")
        assert lab not in ZIPS and (S / zp).is_file() and (S / "logs" / f"{log}.out").is_file(), spec
        ZIPS[lab] = (zp, log)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves, problems = [], {}, []
    for sha, (zp, log) in ZIPS.items():
        parse(sha, zp, log, rows, curves, problems)
    trows, tcurves, tproblems = [], {}, []
    for sha, (zp, log) in TWINS.items():
        parse(sha, zp, log, trows, tcurves, tproblems)
    bad = {}
    # ---- T2-D281 on the 11 sealed-shippable ZIPs (in-sample)
    med = {f: statistics.median(r["score"] for r in rows if r["family"] == f) for f in {r["family"] for r in rows}}
    for r in rows:
        if r["score"] < 0.50 or r["score"] < med[r["family"]] - 0.10:
            bad.setdefault(r["zip"], []).append(f"{r['kind']} s{r['seed']} {r['score']:.4f} (median {med[r['family']]:.4f})")
    # ---- T2-D281b on the twins (held out)
    ref = reference()
    tmed = {}
    for r in trows:
        if r["kind"] in ("cnn1", "cnn2"):
            key = (r["zip"], r["kind"])
            tmed.setdefault(key, statistics.median(x["score"] for x in trows if (x["zip"], x["kind"]) == key))
            fail = r["score"] < 0.40 or r["score"] < tmed[key] - 0.10
            why = f"(twin median {tmed[key]:.4f})"
        else:
            fail = r["score"] < 0.40 or r["score"] < ref[r["kind"]]["min"] - 0.05
            why = f"(reference min {ref[r['kind']]['min']:.4f}, median {ref[r['kind']]['median']:.4f})"
        r["fail"] = fail
        if fail:
            bad.setdefault(r["zip"], []).append(f"{r['kind']} s{r['seed']} held-out {r['score']:.4f} {why}")
    # ---- curves: (c) and the D281b tail-noise rule (c') over all 13
    allc = {**curves, **tcurves}
    tail = {k: float(np.median([c["tail_sd"] for c in allc.values() if c["kind"] == k])) for k in ("st", "mor")}
    for key, c in allc.items():
        c["fail_c2"] = c["last"] - c["min"] > 3 * tail[c["kind"]]
        if c["fail_c"] or c["fail_c2"]:
            bad.setdefault(key.split(":")[0], []).append(f"{key} loss {c['first']:.3f} -> {c['last']:.3f} (min {c['min']:.3f}; tail noise {tail[c['kind']]:.3f})")
    for p in problems + tproblems:
        bad.setdefault(p.split()[0], []).append(p)
    print("sealed-shippable family medians (in-sample): " + "; ".join(f"{f} {v:.4f} (n {sum(r['family'] == f for r in rows)})" for f, v in sorted(med.items())))
    print("twin references (12 package-shape kit files, held-out raw cell BA): " + "; ".join(f"{k} = {v['member']} min {v['min']:.4f} median {v['median']:.4f}" for k, v in ref.items()))
    print(f"tail noise (median sd of the last 5 epoch losses over the 13 ZIPs): ST-L {tail['st']:.4f}, MOR {tail['mor']:.4f} -> (c') bars {3 * tail['st']:.3f} / {3 * tail['mor']:.3f}")
    for sha in list(ZIPS) + list(TWINS):
        rs = [r for r in rows + trows if r["zip"] == sha]
        lo = {k: min((r["score"] for r in rs if r["kind"] == k), default=float("nan")) for k in ("cnn1", "cnn2", "st", "mor")}
        cv = {k: v for k, v in allc.items() if k.startswith(sha)}
        tag = "held-out" if sha in TWINS else "in-sample"
        print(f"{sha} ({tag}): members {len(rs)}; min cnn1 {lo['cnn1']:.4f} cnn2 {lo['cnn2']:.4f} st {lo['st']:.4f} mor {lo['mor']:.4f}; "
              + ", ".join(f"{k.split(':', 1)[1]} loss {c['first']:.3f}->{c['last']:.3f} (min {c['min']:.3f})" for k, c in cv.items())
              + f" -> {'FAIL: ' + '; '.join(bad[sha]) if sha in bad else 'PASS'}")
    verdict = "ALL PASS" if not bad else f"FAIL: {sorted(bad)}"
    print("VERDICT:", verdict)
    (OUT / args.out).write_text(json.dumps(dict(rows=rows, twin_rows=trows, curves=allc, medians=med, reference=ref, tail_noise=tail,
                                                    problems=problems + tproblems, bad=bad, verdict=verdict), indent=1))


if __name__ == "__main__":
    main()

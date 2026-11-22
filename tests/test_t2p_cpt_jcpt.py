#!/usr/bin/env python
"""T2-D350 (J-CPT arm A): t2p_cpt --jcpt-spec / --save-at / --schedule const on tiny synthetic pools, CPU, st_small, 4 steps:
  1. the holdouts: PROTEUS 'jcpt' keeps P01 P02 P03 P06 P07 + P10 session 1 only; a T1-like pool drops split test and xuser4 fold 0's
     held-out + inner users (seeds 12, 13); a T3-like pool keeps split train only; each recorded in the summary;
  2. any-rate / short pools: 1 s at 120 Hz -> 128 samples, 5 s at 128 Hz -> a 4 s crop -> 512; Muse's 4 channels pass;
  3. the draw p = group weight x sqrt(n) within the group;
  4. --save-at 2,4 writes ck_<tag>_s00002 / s00004 with a status-ok summary (args.steps = the step) that t2p_fm.use_cpt accepts;
  5. the PROTEUS-derived streams of the pool hold no typical-eval / P04 stream (cpt_leak on the kit and seen1 test streams = []).
  python tests/test_t2p_cpt_jcpt.py  ->  "OK"
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
PROTEUS_CH = ("C2 CP2 P2 Pz CPz Cz C1 CP1 P1 P3 PO7 P5 TP7 CP5 CP3 C3 PO8 P6 P4 TP8 CP6 CP4 C6 C4 FC6 FC4 FC2 F6 F4 F2 AF4 AF3 F5 F3 F1 Fz "
              "FC5 FC3 FC1 FCz C5").split()
T1_CH = ['Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
         'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2', 'F6', 'Fp2']


def export(d, X, idx, ch, fs):
    d.mkdir(parents=True)
    np.save(d / "X.npy", X.astype(np.float32))
    idx.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps(dict(sfreq=fs, n_times=X.shape[-1], ch_names=ch)))


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        rows = [(f"Dreyer2026Proteus/P{u:02d}", str(s)) for u in range(1, 11) for s in ((1, 2, 3) if u in (4, 10) else (1,)) for _ in range(6)]
        ip = pd.DataFrame(rows, columns=["subject", "session"])
        ip["split"] = np.where(ip.subject.str.endswith(("P04", "P09")), "test", np.where(ip.subject.str.endswith(("P05", "P08")), "val", "train"))
        export(T / "export_proteus_v1", rng.normal(size=(len(ip), 41, 480)), ip, PROTEUS_CH, 120.0)
        i1 = pd.DataFrame(dict(subject=[f"Alljoined/{u}" for u in range(1, 21) for _ in range(10)]))
        i1["split"] = np.tile(["train"] * 6 + ["val"] * 2 + ["test"] * 2, 20)
        export(T / "t1_like", rng.normal(size=(len(i1), 32, 120)), i1, T1_CH, 120.0)
        i3 = pd.DataFrame(dict(subject=[f"s{k}" for k in range(60)], split=["train"] * 40 + ["val"] * 10 + ["test"] * 10))
        export(T / "t3_like", rng.normal(size=(len(i3), 4, 640)), i3, ["TP9", "AF7", "AF8", "TP10"], 128.0)
        spec = dict(proteus_group="T2", group_weights=dict(T1=1 / 3, T2=1 / 3, T3=1 / 3), pools=[
            dict(path=str(T / "t1_like"), group="T1", keep=dict(split=["train", "val"]), xuser4=dict(fold=0, seeds=[12, 13])),
            dict(path=str(T / "t3_like"), group="T3", keep=dict(split=["train"]))])
        (T / "spec.json").write_text(json.dumps(spec))
        out = T / "out"
        cmd = [sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_cpt.py"), "--fm", "st_small", "--proteus", str(T / "export_proteus_v1"),
               "--proteus-pool", "jcpt", "--jcpt-spec", str(T / "spec.json"), "--out", str(out), "--steps", "4", "--batch", "4",
               "--accum", "1", "--eval-every", "100", "--save-at", "2,4", "--save-tag", "jT", "--schedule", "const", "--token-scale"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
        for st in (2, 4):
            d = out / f"ck_jT_s{st:05d}"
            sm = json.loads((d / "cpt_summary.json").read_text())
            assert sm["status"] == "ok" and sm["args"]["steps"] == st and (d / "checkpoint-cpt.pth").exists(), (st, sm["status"])
        sm = json.loads((out / "ck_jT_s00002" / "cpt_summary.json").read_text())
        P = {p["name"]: p for p in sm["pools"]}
        pr = P["export_proteus_v1"]
        assert pr["n_train"] + pr["n_val"] == 6 * 6, pr["n_train"]                      # P01 P02 P03 P06 P07 + P10 s1, 6 windows each
        assert sorted(sm["proteus_streams"]) == sorted(f"{u}|1" for u in ("P01", "P02", "P03", "P06", "P07", "P10")), sm["proteus_streams"]
        t1p = P["t1_like"]
        out_u = set(t1p["holdout"]["xuser4_out"])
        assert {"Alljoined/1", "Alljoined/5", "Alljoined/9", "Alljoined/13", "Alljoined/17"} <= out_u and len(out_u) >= 7, out_u
        assert not out_u & set(t1p["holdout"]["subjects_seen"])
        assert t1p["n_train"] + t1p["n_val"] == 8 * (20 - len(out_u)), t1p["n_train"]   # test split out, fold-0 users out
        assert (t1p["t_crop"], t1p["n_out"]) == (120, 128) and (P["t3_like"]["t_crop"], P["t3_like"]["n_out"]) == (512, 512)
        assert P["t3_like"]["n_train"] + P["t3_like"]["n_val"] == 40
        g = {}
        for p in sm["pools"]:
            g[p["group"]] = g.get(p["group"], 0) + p["p_draw"]
        assert all(abs(v - 1 / 3) < 1e-9 for v in g.values()), g
        import t2p_fm
        t2p_fm.use_cpt("st_small", out / "ck_jT_s00002" / "checkpoint-cpt.pth", lambda m: None)
        from t2p_bench import load_export  # noqa: F401  (import check only)
        streams_kit_seen1 = {("P04", "1"), ("P04", "2"), ("P04", "3"), ("P09", "1"), ("P10", "2"), ("P10", "3"), ("P05", "1"), ("P08", "1")}
        assert t2p_fm.cpt_leak(out / "ck_jT_s00002" / "checkpoint-cpt.pth", streams_kit_seen1) == []
    print("OK")


if __name__ == "__main__":
    main()

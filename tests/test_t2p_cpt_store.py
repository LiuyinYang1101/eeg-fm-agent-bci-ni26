#!/usr/bin/env python
"""T2-D356 / T2-FM-D2 (the ST-base from-scratch insurance run): t2p_cpt --store-split / --random-init / --state-every / --resume
on a tiny synthetic the in-house FM project store, read through the in-house FM project's REAL loader (INHOUSE_FM_SRC, default the ni26-fm-tree worktree), CPU, st_small:
  1. pool = the split's train pool only (a store recording outside the split never loads); the val curve = its tier-1 recordings;
  2. channel rule: exact names (case-insensitive), 'EEG X-M1' / 'X_M2' -> X, bipolar 'Fpz-Cz' dropped, unknown names dropped, EGI
     E<n> through egi_to_st.json (the reference Cz dropped); a recording with < 4 mapped channels is out (Sleep-EDF's 2 bipolar);
  3. the in-house FM project's masks drop a bad channel (a new montage group) and the windows of a bad span; indexed hours = split hours - the span;
  4. PROTEUS streams from the ni26_proteus rows ('P01|1', 'P10|1'; val 'P06|1'); a J-CPT holdout row (P04) refuses the pool;
  5. --random-init: summary random_init, no base sha; t2p_fm.use_cpt accepts it (file sha only) and cpt_leak(kit / seen1) = [];
  6. --state-every 2 + --resume continues at step 2 (resumed_from recorded); a changed recipe refuses to resume.
  python tests/test_t2p_cpt_store.py  ->  "OK" (or "SKIP: ..." without an the in-house FM project checkout)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
HOME = Path.home() / "workspace/projects"
SRC = Path(os.environ.get("INHOUSE_FM_SRC") or next((p for p in (HOME / "_wt_ni26fm/src", HOME / "inhouse_fm/src")
                                                    if (p / "inhouse_fm/data/continuous_dataset.py").exists()), Path("/none")))
FS, DUR = 256, 40.0                                    # 40 s = 10 windows of 4 s at hop 4 s
EGI129 = [f"E{k}" for k in range(1, 129)] + ["Cz"]


def shard(path, recs, rng):
    """an the in-house FM project consolidated-v1 shard: one group per recording (signal fp16 (C, T), sfreq, channel_names, portable_key)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = "consolidated-v1"
        for k, (pk, names) in enumerate(recs):
            g = f.create_group(f"r{k:03d}")
            x = rng.normal(size=(len(names), int(FS * DUR))).astype(np.float16)
            g.create_dataset("signal", data=x, chunks=(len(names), 1280))
            g.attrs["sfreq"] = float(FS)
            g.attrs["channel_names"] = json.dumps(names)
            g.attrs["portable_key"] = pk


def build(T, holdout_row=False):
    rng = np.random.default_rng(0)
    A, B = T / "preproc_tA", T / "ni26_only"
    bci = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2", "XX9"]
    sleep = ["EEG F3-M2", "EEG C4-M1", "O1_M2", "f4-m1", "EEG Fpz-Cz"]
    recs = {"bcix": [(f"bcix/bcix_{u}/{u}_s000.h5", bci) for u in range(1, 8)],             # u=7: in the store, not in the split
            "sleepx": [(f"sleepx/sleepx_{u}/{u}_s000.h5", sleep) for u in range(1, 5)],
            "sleepedf_ext": [(f"sleepedf_ext/sleepedf_ext_{u}/{u}_s000.h5", ["EEG Fpz-Cz", "EEG Pz-Oz"]) for u in range(1, 3)],
            "egix": [(f"egix/egix_{u}/{u}_s000.h5", EGI129) for u in range(1, 3)]}
    for ds, r in recs.items():
        shard(A / ds / f"{ds}.shard100.h5", r, rng)
    pro = [("P01", "1", "train"), ("P10", "1", "train"), ("P06", "1", "val")] + ([("P04", "1", "train")] if holdout_row else [])
    rp = [(f"ni26_proteus/ni26_proteus_{u}/Dreyer2026Proteus-{u}-{s}-mi-1_g000_s000.h5", u, s, sp) for u, s, sp in pro]
    shard(B / "ni26_proteus" / "ni26_proteus.shard100.h5", [(pk, ["C3", "C4", "Cz", "CPz", "FCz"]) for pk, *_ in rp], rng)
    rm = [(f"ni26_muse/ni26_muse_{u}/{u}_s000.h5", ["TP9", "AF7", "AF8", "TP10"]) for u in range(1, 4)]
    shard(B / "ni26_muse" / "ni26_muse.shard100.h5", rm, rng)
    rows = []

    def row(root, pk, ds, pid, ses, tier, src=None):
        rows.append(dict(dataset=ds, patient_id=pid, session_id=ses, out_path=str(root / pk), _pk=pk, in_pretrain=tier == 0, tier=tier,
                         duration_s=DUR, _source=src or ds, exclude_reason="", n_channels=0, n_samples=int(FS * DUR)))
    for ds, r in recs.items():
        for k, (pk, _) in enumerate(r):
            if ds == "bcix" and k == 6:
                continue                                                                       # not in the split: must not load
            row(A, pk, ds, f"{ds}:{k}", "s0", 1 if (ds == "bcix" and k == 5) else 0, src=("scalp_bids:ds005505" if ds == "egix" else None))
    for pk, u, s, sp in rp:
        row(B, pk, "ni26_proteus", f"ni26_proteus:{u}", f"Dreyer2026Proteus-{u}-{s}-mi-1_g000", 0 if sp == "train" else 1)
    for k, (pk, _) in enumerate(rm):
        row(B, pk, "ni26_muse", f"ni26_muse:{k}", "s0", 0)
    pd.DataFrame(rows).to_parquet(T / "split.parquet")
    masks = [dict(key="bcix/bcix_1/1_s000.h5", usable=True, bad_channels=[1], bad_spans_s=[[8.0, 16.0]])]
    (T / "masks.json").write_text(json.dumps(masks))
    return A, B


def run(T, A, B, out, *extra):
    cmd = [sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_cpt.py"), "--fm", "st_small", "--random-init", "--lr", "1e-4", "--wd",
           "1e-5", "--mask-ratio", "0.75", "--store-split", str(T / "split.parquet"), "--store-roots", f"{A},{B}", "--store-src", str(SRC),
           "--store-masks", str(T / "masks.json"), "--store-workers", "2", "--store-index-workers", "1", "--out", str(out), "--steps", "4",
           "--batch", "4", "--accum", "2", "--eval-every", "100", "--save-at", "2,4", "--save-tag", "stb", "--token-scale",
           "--state-every", "2", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))


def main():
    if not SRC.exists():
        print(f"SKIP: no the in-house FM project src (set INHOUSE_FM_SRC; looked for {SRC})")
        return
    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        A, B = build(T)
        out = T / "out"
        r = run(T, A, B, out)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        sm = json.loads((out / "ck_stb_s00002" / "cpt_summary.json").read_text())
        assert sm["random_init"] is True and sm["base_sha256"] is None and sm["base_ckpt"] is None and sm["status"] == "ok"
        st = sm["store"]
        import hashlib
        assert st["split_sha256"] == hashlib.sha256((T / "split.parquet").read_bytes()).hexdigest()
        S = st["sources"]
        assert S["bcix"]["recordings"] == 5 and S["bcix"]["n_val"] > 0, S["bcix"]                     # 5 train (u=6 val, u=7 not in split)
        assert S["sleepedf_ext"]["recordings"] == 2 and S["sleepedf_ext"]["recordings_kept"] == 0      # 2 bipolar channels: out
        P = {p["name"]: p for p in sm["pools"]}
        assert set(P) == {"bcix", "sleepx", "hbn", "ni26_proteus", "ni26_muse"}, sorted(P)          # an HBN release -> the source hbn
        assert [g["ch"] for g in P["sleepx"]["groups"]] == [["C4", "F3", "F4", "O1"]], P["sleepx"]["groups"]
        bg = sorted(tuple(g["ch"]) for g in P["bcix"]["groups"])
        assert bg == [("Cz", "Fp1", "Fz", "O1", "O2", "Pz"), ("Cz", "Fp1", "O1", "O2", "Pz")], bg      # the masked Fz -> its own group
        egi = json.loads((ROOT / "tracks/t2_bci/fm/egi_to_st.json").read_text())["GSN-HydroCel-129"]
        assert P["hbn"]["groups"][0]["ch"] == sorted(egi.values()), "EGI names not through egi_to_st.json"
        assert S["hbn"]["channel_rules"].get("egi_ref") == 2                              # the stored reference Cz, x 2
        assert abs(S["bcix"]["split_train_h"] - 5 * DUR / 3600) < 1e-5
        assert abs(S["bcix"]["indexed_h"] - (5 * DUR - 8.0) / 3600) < 1e-5, S["bcix"]                 # the 8-16 s span: windows at 8 and 12 s
        assert abs(S["ni26_muse"]["indexed_h"] - S["ni26_muse"]["split_train_h"]) < 1e-5
        assert sorted(sm["proteus_streams"]) == ["P01|1", "P06|1", "P10|1"], sm["proteus_streams"]
        assert set(sm["val_curve"][0]["val"]) >= {"bcix", "ni26_proteus"}, sm["val_curve"][0]["val"]
        assert len(st["shards"]) == 6 and all(sh["size"] > 0 for sh in st["shards"])
        assert abs(sum(p["p_draw"] for p in sm["pools"]) - 1) < 1e-9 and abs(sum(st["token_share"].values()) - 1) < 1e-9
        import t2p_fm
        t2p_fm.use_cpt("st_small", out / "ck_stb_s00004" / "checkpoint-cpt.pth", lambda m: None)
        assert t2p_fm.t1.CPT_INFO["st_small"]["random_init"] is True and t2p_fm.t1.CPT_INFO["st_small"]["base_sha256"] is None
        test_streams = {("P04", "1"), ("P04", "2"), ("P04", "3"), ("P09", "1"), ("P10", "2"), ("P10", "3"), ("P05", "1"), ("P08", "1")}
        assert t2p_fm.cpt_leak(out / "ck_stb_s00004" / "checkpoint-cpt.pth", test_streams) == []
        assert (out / "state-cpt.pth").exists()
        r2 = run(T, A, B, out, "--resume")                                                              # continues at step 2 -> 4
        assert r2.returncode == 0 and "RESUMED at step 2" in r2.stdout, r2.stdout[-2000:] + r2.stderr[-2000:]
        assert json.loads((out / "cpt_summary.json").read_text())["resumed_from"]["step"] == 2
        r3 = run(T, A, B, out, "--resume", "--min-lr", "1e-7")                                          # another recipe: refused
        assert r3.returncode != 0 and "recipe differs" in (r3.stdout + r3.stderr), r3.stdout[-1500:] + r3.stderr[-1500:]
    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        A, B = build(T, holdout_row=True)
        r = run(T, A, B, T / "out")
        assert r.returncode != 0 and "J-CPT holdout PROTEUS streams" in (r.stdout + r.stderr), r.stdout[-1500:] + r.stderr[-1500:]
    print("OK")


if __name__ == "__main__":
    main()

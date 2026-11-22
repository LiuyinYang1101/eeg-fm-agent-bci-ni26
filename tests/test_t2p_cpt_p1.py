#!/usr/bin/env python
"""T2-FM-D6 (ST-L + the in-house FM project's P1 v2_shared): t2p_cpt --init-ckpt / --store-sources / --sup-* on a synthetic store + two synthetic
P1 source files, through the in-house FM project's REAL SupSource / SupSamplerV2 / SupStream / SupHeadsV2 (INHOUSE_FM_SRC, default the ni26-fm-tree
worktree), CPU, st_small:
  1. --init-ckpt: a status-ok t2p_cpt checkpoint is loaded (sha + base checked), its PROTEUS streams join the run's, the lineage is recorded;
  2. --store-sources keeps only the named sources in the MAE pool;
  3. P1: v2_shared heads = shared 'motor' (MI) + 'erp' with its shift head (the wide ERP file); a channel name outside ST's table is
     dropped and recorded; the supervised loss, the per-term clip scale and the person-holdout curve (steps 0 / 2 / 4) are recorded;
  4. the checkpoint passes t2p_fm.use_cpt (base = the pinned file) and cpt_leak(kit / seen1) = [];
  5. --state-every + --resume restores the heads (RESUMED, the run completes).
  python tests/test_t2p_cpt_p1.py  ->  "OK" (or "SKIP: ...")
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import test_t2p_cpt_store as ts  # noqa: E402  (the synthetic the in-house FM project store)

MI_CH = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "XYZ"]
ERP_CH = ["Fz", "Cz", "Pz", "Oz", "P3", "P4"]


def p1_source(path, family, classes, ch, T_wide, T_eval=None, crop_on=0, n_person=10, per=12, rng=None):
    """a P1 source file in the in-house FM project's SupSource schema (one recording per person)."""
    N = n_person * per
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=rng.normal(size=(N, len(ch), T_wide)).astype(np.float16))
        f.create_dataset("y", data=np.tile(np.arange(len(classes)), N // len(classes) + 1)[:N].astype(np.int64))
        f.create_dataset("person", data=np.array([f"s{k // per}".encode() for k in range(N)], dtype=object), dtype=h5py.string_dtype())
        f.create_dataset("rec_idx", data=np.array([k // per for k in range(N)], np.int32))
        g = f.create_group("rec")
        g.create_dataset("n_ch", data=np.full(n_person, len(ch), np.int64))
        g.create_dataset("coords", data=np.zeros((n_person, len(ch), 3), np.float32))
        g.create_dataset("anat", data=np.full((n_person, len(ch), 3), -1, np.int64))
        g.create_dataset("present", data=np.ones((n_person, len(ch)), bool))
        g.create_dataset("key", data=np.array([f"rec{k}".encode() for k in range(n_person)], dtype=object), dtype=h5py.string_dtype())
        g.create_dataset("ch_names", data=np.array([[c.encode() for c in ch]] * n_person, dtype=object), dtype=h5py.string_dtype())
        f.attrs["class_names"] = json.dumps(classes)
        f.attrs["modality_id"] = 3
        f.attrs["modality"] = "eeg"
        f.attrs["family"] = family
        f.attrs["front_end"] = "session"
        f.attrs["t_pre_s"] = 0.0
        if T_eval is not None:
            f.attrs["T_eval"] = T_eval
            f.attrs["crop_on"] = crop_on


def run(T, A, B, out, *extra):
    cmd = [sys.executable, str(ROOT / "tracks/t2_bci/scripts/t2p_cpt.py"), "--fm", "st_small", "--lr", "2e-5", "--wd", "1e-5",
           "--mask-ratio", "0.75", "--store-split", str(T / "split.parquet"), "--store-roots", f"{A},{B}", "--store-src", str(ts.SRC),
           "--store-masks", str(T / "masks.json"), "--store-workers", "1", "--store-index-workers", "1", "--out", str(out),
           "--batch", "4", "--accum", "2", "--eval-every", "100", "--token-scale", "--schedule", "const", *extra]
    return subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))


def main():
    if not ts.SRC.exists():
        print(f"SKIP: no the in-house FM project src (set INHOUSE_FM_SRC; looked for {ts.SRC})")
        return
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        T = Path(tmp)
        A, B = ts.build(T)
        init = T / "init"
        r = run(T, A, B, init, "--steps", "2", "--save-at", "2", "--save-tag", "i0", "--store-sources", "ni26_proteus,bcix")
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
        ick = init / "ck_i0_s00002" / "checkpoint-cpt.pth"
        sd = T / "p1"
        sd.mkdir()
        p1_source(sd / "lee2019_mi.h5", "mi", ["left_hand", "right_hand"], MI_CH, 768, rng=rng)
        p1_source(sd / "lee2019_erp.h5", "erp", ["NonTarget", "Target"], ERP_CH, 192 + 2 * 77, T_eval=192, crop_on=77, rng=rng)
        out = T / "out"
        sup = ["--init-ckpt", str(ick), "--store-sources", "ni26_proteus", "--sup-dir", str(sd), "--sup-sources", "lee2019_mi,lee2019_erp",
               "--sup-groups", "mi+me:0.6,erp:0.4", "--sup-token-budget", "2048", "--sup-holdout-every", "2", "--sup-holdout-windows", "8",
               "--sup-no-hash", "--save-at", "2,4", "--save-tag", "p1", "--state-every", "2", "--steps", "4"]
        r = run(T, A, B, out, *sup)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        sm = json.loads((out / "ck_p1_s00004" / "cpt_summary.json").read_text())
        assert sm["init_ckpt"]["path"] == str(ick) and len(sm["init_ckpt"]["sha256"]) == 64
        assert {p["name"] for p in sm["pools"]} == {"ni26_proteus"}, [p["name"] for p in sm["pools"]]
        assert sorted(sm["proteus_streams"]) == ["P01|1", "P06|1", "P10|1"], sm["proteus_streams"]
        S = sm["sup"]
        assert S["heads"]["ce"] == ["erp", "motor"] and S["heads"]["shift"] == ["erp"], (S["heads"]["ce"], S["heads"]["shift"])
        assert S["channels_dropped"] == {"lee2019_mi": ["XYZ"]}, S["channels_dropped"]
        assert [h["step"] for h in S["holdout"]] == [0, 2, 4] and all("balacc" in v for v in S["holdout"][-1]["holdout"].values())
        assert S["curve"] and 0 < S["curve"][0]["scale"] <= 1.0 and S["weight"] == 0.02 and S["term_clip"] == 2.0
        assert S["sampler"]["per_source"]["lee2019_erp"]["shift"] is True and S["sampler"]["per_source"]["lee2019_mi"]["shift"] is False
        import t2p_fm
        t2p_fm.use_cpt("st_small", out / "ck_p1_s00004" / "checkpoint-cpt.pth", lambda m: None)
        streams = {("P04", "1"), ("P04", "2"), ("P04", "3"), ("P09", "1"), ("P10", "2"), ("P10", "3"), ("P05", "1"), ("P08", "1")}
        assert t2p_fm.cpt_leak(out / "ck_p1_s00004" / "checkpoint-cpt.pth", streams) == []
        r2 = run(T, A, B, out, *sup, "--resume")
        assert r2.returncode == 0 and "RESUMED at step 2" in r2.stdout, r2.stdout[-2500:] + r2.stderr[-2500:]
    print("OK")


if __name__ == "__main__":
    main()

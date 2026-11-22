#!/usr/bin/env python
"""Self-test (backlog 69): FM -> CNN distillation plumbing in t2p_bench (CPU, the real export + synthetic external pools).

  1. kd_select: the same windows on every call (rng KD_RNG), an equal share per pool and per subject, rows sorted within a subject;
     kd_aligned: (N, 41, 480) float32, finite, keys 'pool:row' in kd_select order, for the aligns none / ea / sz; a pool's 500-sample windows
     are cut to 480;
  2. student (t2p_bench end to end, FBCNet B3, kit, 1 epoch): a fake teacher dump (train_idx = the kit training windows, Dirichlet p_train,
     the kd_select keys, Dirichlet p_ext) -> the kd and kdx recipes run and record res["kd"]; a teacher whose train_idx is off by one window,
     or whose kd_key differs, is refused;
  3. teacher (unless --skip-fm; t2p_bench end to end, fm:st_small:mean, kit, 1 epoch): --fm-kd-dump writes kdteacher__kit__s0.npz with
     train_idx = the kit training windows, p_train / p_ext rows summing to 1, kd_key = kd_select's.
  python t2p_kd_selftest.py [--skip-fm]  ->  "OK"
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
import t2p_bench as B  # noqa: E402

EXPORT = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def pool(d, names, n_sub=3, n_ses=2, per=20, L=500, seed=0):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    C = len(names)
    X, rows = [], []
    for s in range(n_sub):
        A = np.eye(C) + 0.2 * rng.standard_normal((C, C))
        for ss in range(n_ses):
            X.append(np.einsum("cd,ndt->nct", A, rng.standard_normal((per, C, L))) * 10.0)
            rows += [dict(subject=f"S{s}", session=str(ss + 1), run="1", onset_s=float(k)) for k in range(per)]
    np.save(d / "X.npy", np.concatenate(X).astype(np.float16))
    df = pd.DataFrame(rows)
    df.insert(0, "i", range(len(df)))
    df.to_csv(d / "index.csv", index=False)
    (d / "meta.json").write_text(json.dumps(dict(ch_names=names, sfreq=120)))


def bench(out, *args):
    r = subprocess.run([sys.executable, str(HERE / "t2p_bench.py"), "--export", str(EXPORT), "--out", str(out), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout[-3000:] + r.stderr[-3000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fm", action="store_true")
    a = ap.parse_args()
    import mne
    meta = json.loads((EXPORT / "meta.json").read_text())
    dst = list(meta["ch_names"])
    std = [c for c in mne.channels.make_standard_montage("standard_1005").ch_names if c not in dst]
    tmp = Path(tempfile.mkdtemp(prefix="t2p_kd_"))
    pool(tmp / "poolA", dst + std[:21], seed=1)
    pool(tmp / "poolB", dst[:35] + std[21:41], n_sub=2, seed=2)
    pools = [str(tmp / "poolA"), str(tmp / "poolB")]
    s1, s2 = B.kd_select(pools, 60), B.kd_select(pools, 60)
    assert all(np.array_equal(p["rows"], q["rows"]) for p, q in zip(s1, s2)), "kd_select must be deterministic"
    for p in s1:
        subj = p["idx"]["subject"].to_numpy()[p["rows"]]
        cnt = pd.Series(subj).value_counts()
        assert cnt.nunique() == 1 and cnt.iloc[0] == (60 // 2) // len(cnt), (p["name"], cnt.to_dict())
        for u in pd.unique(subj):
            r = p["rows"][subj == u]
            assert np.all(np.diff(r) > 0), "rows sorted within a subject"
    keys = None
    for mode in ("none", "ea", "sz"):
        Xk, kk = B.kd_aligned(s1, mode, 0.05, dst)
        assert Xk.shape == (60, 41, 480) and Xk.dtype == np.float32 and np.isfinite(Xk).all(), (mode, Xk.shape)
        assert keys is None or np.array_equal(kk, keys)
        keys = kk
    assert list(keys[:2]) == [f"poolA:{s1[0]['rows'][0]}", f"poolA:{s1[0]['rows'][1]}"], keys[:2]
    print(f"kd_select / kd_aligned: deterministic, 30 / 30 per pool (equal per subject), (60, 41, 480) for none / ea / sz")

    X, idx, _ = B.load_export(str(EXPORT))
    (fname, trm, vam, tem), = [f for f in B.folds(idx, "kit")]
    tr = np.flatnonzero(trm)
    rng = np.random.default_rng(0)
    good = tmp / "kdteacher__kit__s0.npz"
    np.savez(good, train_idx=tr, p_train=rng.dirichlet(np.ones(3), len(tr)).astype(np.float32), kd_key=keys,
             p_ext=rng.dirichlet(np.ones(3), len(keys)).astype(np.float32))
    np.savez(tmp / "badidx__kit__s0.npz", train_idx=tr[1:], p_train=np.ones((len(tr) - 1, 3), np.float32) / 3, kd_key=keys,
             p_ext=np.ones((len(keys), 3), np.float32) / 3)
    np.savez(tmp / "badkey__kit__s0.npz", train_idx=tr, p_train=np.ones((len(tr), 3), np.float32) / 3, kd_key=keys[::-1],
             p_ext=np.ones((len(keys), 3), np.float32) / 3)
    common = ["--protocols", "kit", "--models", "fbcnet_bboth", "--aligns", "none", "--seeds", "204", "--epochs", "1",
              "--kd-pools", ",".join(pools), "--kd-n", "60"]
    for rec in ("r1_pulxx_vp2_e180_vr_rc75_kd", "r1_pulxx_vp2_e180_vr_rc75_kdx"):
        o = tmp / f"out_{rec}"
        rc, log = bench(o, *common, "--recipe", rec, "--kd-teacher", str(tmp / "kdteacher__{protocol}__s0.npz"))
        assert rc == 0, log
        f, = list(o.glob("kit__*.npz"))
        res = json.loads(str(np.load(f)["res"]))
        assert res["kd"]["alpha"] == 0.5 and res["kd"]["ext"] == int(rec.endswith("kdx")) and \
            res["kd"]["n_ext"] == (60 if rec.endswith("kdx") else 0), res.get("kd")
        print(f"student {rec}: ran, res['kd'] = {res['kd']}")
    for bad, msg in (("badidx", "training windows"), ("badkey", "distillation windows")):
        rc, log = bench(tmp / f"out_{bad}", *common, "--recipe", "r1_pulxx_vp2_e180_vr_rc75_kdx", "--kd-teacher",
                        str(tmp / (bad + "__{protocol}__s0.npz")))
        assert rc != 0 and msg in log, (bad, log[-800:])
    print("student: a teacher with other training windows / other distillation windows is refused")

    if not a.skip_fm:
        o = tmp / "out_fm"
        rc, log = bench(o, "--protocols", "kit", "--models", "fm:st_small:mean", "--aligns", "sz", "--seeds", "0", "--fm-epochs", "1",
                        "--fm-kd-dump", "--kd-pools", ",".join(pools), "--kd-n", "60")
        assert rc == 0, log
        t = np.load(o / "kdteacher__kit__s0.npz")
        assert np.array_equal(t["train_idx"], tr) and np.array_equal(t["kd_key"], keys)
        assert np.allclose(t["p_train"].sum(1), 1, atol=1e-3) and np.allclose(t["p_ext"].sum(1), 1, atol=1e-3)
        print(f"teacher: kdteacher__kit__s0.npz = {len(t['train_idx'])} training + {len(t['kd_key'])} external windows, rows sum to 1")
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("OK")


if __name__ == "__main__":
    main()

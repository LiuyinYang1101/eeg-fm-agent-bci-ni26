#!/usr/bin/env python
"""T2-D367: the SSL-pretrained CNN init path, CPU:
  1. ni26.train.CNN_INIT empty -> fit_eval bitwise reproducible (the shipped path unchanged);
  2. CNN_INIT = another net's state_dict, lr 0 -> after training every tensor but final_layer.* equals the init (floating parameters;
     BatchNorm running statistics may move), final_layer = this seed's own fresh draw (= the no-init run's);
  3. t2p_bench's CNN stem gains 'ini<tag>' only with --cnn-init;
  4. t2p_cnn_ssl.py end to end on the real export + a fake crop pool (kit train / val users' streams + one kit TEST-user crop that the
     kit pool must drop), 2 steps -> a state_dict that loads through CNN_INIT.
  python tests/test_cnn_init.py  ->  "OK"
"""
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(R / "tracks" / "t2_bci" / "scripts"))
import ni26.train as T  # noqa: E402
from test_rc import data  # noqa: E402

EXP = Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1"


def main():
    d = data()
    model = "fbcnet_bboth"
    plain = replace(T.RECIPES["r1"], epochs=1)
    T.CNN_INIT.clear()
    a0 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
    a1 = T.fit_eval(model, d, plain, 0, device="cpu", return_state=True)
    assert all(torch.equal(a0["state_last"][k], a1["state_last"][k]) for k in a0["state_last"]), "no init must be bitwise reproducible"
    print("no init: bitwise reproducible")

    torch.manual_seed(123)
    other = T.build_model(model, d["Xtr"].shape[1], 3, d["Xtr"].shape[2], 120.0).state_dict()
    T.CNN_INIT["sd"] = other
    z = T.fit_eval(model, d, replace(plain, lr=0.0, wd=0.0), 0, device="cpu", return_state=True)["state_last"]
    T.CNN_INIT.clear()
    z0 = T.fit_eval(model, d, replace(plain, lr=0.0, wd=0.0), 0, device="cpu", return_state=True)["state_last"]
    n_eq = 0
    for k, v in z.items():
        if not v.is_floating_point() or "running_" in k:
            continue
        if k.startswith("final_layer."):
            assert torch.equal(v, z0[k]), f"{k}: final_layer must be the seed's own draw"
        else:
            assert torch.allclose(v, other[k]), f"{k}: not the init"
            n_eq += 1
    assert n_eq > 0
    print(f"init: {n_eq} non-final tensors = the init at lr 0; final_layer = the seed's own draw")

    import t2p_bench as B
    ns = B.argparse.Namespace(recipe="r1_pulxx_vp2_e180_vr_rc75", epochs=0, bag="", crop="", recolour_own="", cnn_init="", cnn_init_tag="")
    assert B.task_rec(ns, model) == "r1_pulxx_vp2_e180_vr_rc75"
    ns.cnn_init, ns.cnn_init_tag = "x_{protocol}.pt", "ssl1"
    assert B.task_rec(ns, model) == "r1_pulxx_vp2_e180_vr_rc75inissl1"
    print("stems: 'ini<tag>' only with --cnn-init")

    X, idx, meta = B.load_export(EXP)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (_, trm, vam, tem), = B.folds(idx, "kit")
        pick = list(np.flatnonzero(trm | vam)[:40]) + [int(np.flatnonzero(tem)[0])]      # the last one is a kit TEST user's window
        Xc = np.concatenate([np.asarray(X[pick], np.float32), np.zeros((len(pick), X.shape[1], 120), np.float32)], 2)   # 600 samples
        cd = td / "crops"
        cd.mkdir()
        np.save(cd / "X.npy", Xc)
        sub = idx["subject"].to_numpy()[pick]
        pd.DataFrame(dict(subject=["Dreyer2026Proteus/" + s for s in sub], session=idx["session"].to_numpy()[pick])).to_csv(cd / "index.csv", index=False)
        (cd / "meta.json").write_text(json.dumps(dict(ch_names=meta["ch_names"], sfreq=meta["sfreq"])))
        out = td / "ssl.pt"
        cmd = [sys.executable, str(R / "tracks/t2_bci/scripts/t2p_cnn_ssl.py"), "--export", str(EXP), "--crops", str(cd), "--protocol", "kit",
               "--out", str(out), "--device", "cpu", "--bs", "64", "--epochs", "1", "--max-steps", "2"]
        p = subprocess.run(cmd, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        line = [x for x in p.stdout.splitlines() if "export windows" in x][0]
        assert " + 40 crops" in line, line                  # the kit TEST user's crop was dropped
        summ = json.loads(Path(str(out) + ".json").read_text())
        assert summ["steps"] == 2 and summ["protocol"] == "kit", summ
        sd = torch.load(out, map_location="cpu", weights_only=True)
        T.CNN_INIT["sd"] = sd
        T.fit_eval(model, d, plain, 0, device="cpu")
        T.CNN_INIT.clear()
        print(f"t2p_cnn_ssl: {line.split('] ')[1][:80]} ...; 2 steps, the saved state loads through CNN_INIT")
    print("OK")


if __name__ == "__main__":
    main()

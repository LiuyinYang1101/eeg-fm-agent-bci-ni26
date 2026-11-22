#!/usr/bin/env python
"""T2-D649 (W29, the FM subject-adversarial GRL), CPU, a random-init ST-base, 1 layer, 2 epochs, user-balanced + mixup (= W25's C form):
  1. grl absent == the pre-D649 t2p_fm.py (pinned commit fb968525) BITWISE (prob_last);
  2. grl 0 (the lambda-0 twin: the adversary trains, the GRL passes 0) == the plain C run BITWISE (the adversary's init is RNG-isolated);
  3. grl 1: runs; the engagement curve is recorded every epoch (adv_hold_acc in [0, 1], lambda ramped, chance = 1 / n_users); the output moves;
  4. refused with micro > 1 and with FM-DRO; 5. the bench tag 'grl<lambda>' only when --fm-grl is given.
  6. the continuation path (init_state from a grl run, as t2p_bench --fm-cont-*) runs with a fresh adversary; no adversary key in the state;
     the curves split train_loss = train_task_loss + train_adv_loss (Track3's LOW).
  python tests/test_t2_d649_grl.py  ->  "OK"
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tracks/t2_bci/scripts"))
import t2p_fm as FM  # noqa: E402

REF_COMMIT = "fb968525"
CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(649)
    nn_, nt = 48, 12
    yy, yt = (np.arange(nn_) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    utr = (np.arange(nn_) % 4).astype(np.int64)
    w = np.where(utr == 0, 1 / 3, 1.0).astype(np.float32)            # user 0 = a 3-session user (C's user-balance weights)
    d = dict(Xtr=rng.standard_normal((nn_, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=utr, wtr=w)
    kw = dict(epochs=2, bs=24, lr_enc=1e-4, lr_head=3e-2, wd=0.05, llrd=0.75, device="cpu", layer=1, calib="gain", ls=0.1, mixup=0.4)
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_fm_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_fm.py"], capture_output=True, text=True, check=True).stdout)
        assert "T2-D649" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_fm_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_fm_ref"] = OLD
        spec.loader.exec_module(OLD)
        r_c = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
        r_old = OLD.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    finally:
        ref.unlink(missing_ok=True)
    assert np.array_equal(r_c["prob_last"], r_old["prob_last"]), float(np.abs(r_c["prob_last"] - r_old["prob_last"]).max())
    assert "grl_curve" not in r_c
    print(f"1. grl absent == the pre-D649 t2p_fm.py ({REF_COMMIT}) bitwise (prob_last; user-balanced + mixup)")
    r0 = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, grl=0.0, **kw)
    assert np.array_equal(r0["prob_last"], r_c["prob_last"]), float(np.abs(r0["prob_last"] - r_c["prob_last"]).max())
    assert len(r0["grl_curve"]) == 2 and all(c["lam"] == 0.0 for c in r0["grl_curve"])
    print(f"2. grl 0 (the twin) == C bitwise; its engagement curve {[round(c['adv_hold_acc'], 3) for c in r0['grl_curve']]} (chance {r0['grl_chance']:.3f})")
    logs = []
    r1 = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, grl=1.0, grl_ramp=0.5, **kw)
    cv = r1["grl_curve"]
    assert len(cv) == 2 and all(0.0 <= c["adv_hold_acc"] <= 1.0 for c in cv) and cv[-1]["lam"] == 1.0 and abs(r1["grl_chance"] - 0.25) < 1e-12
    assert any(m.startswith("T2-D649 GRL: lambda max 1") for m in logs) and any(m.startswith("T2-D649 GRL engagement") for m in logs)
    dp = float(np.abs(r1["prob_last"] - r_c["prob_last"]).max())
    assert dp > 0, dp
    print(f"3. grl 1 (ramp 0.5): curve {[(c['epoch'], round(c['lam'], 3), round(c['adv_hold_acc'], 3)) for c in cv]}; output vs C max |d| {dp:.2e}")
    for bad, why in ((dict(micro=2), "micro"), (dict(dro_eta=0.1), "dro")):
        dd = dict(d) if why != "dro" else {k: v for k, v in d.items() if k != "wtr"}
        try:
            FM.fit_eval_fm("rand_st_base", dd, 0, CH, log=lambda m: None, grl=1.0, **bad, **kw)
            raise SystemExit(f"FAIL: grl + {why} accepted")
        except AssertionError as e:
            assert "T2-D649" in str(e) or "T2-D645" in str(e) or "T2-D642" in str(e), (why, e)
    print("4. grl refused with micro > 1 and with FM-DRO")
    src = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert '+ (f"grl{args.fm_grl:g}" if getattr(args, "fm_grl", None) is not None else ""))' in src and '"--fm-grl", type=float, default=None' in src
    print("5. t2p_bench: tag 'grl<lambda>' only when --fm-grl is given")
    rs = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, grl=0.3, return_state=True, **kw)   # the cont slots' path (t2p_bench --fm-cont-*)
    assert rs.get("state_last") is not None and not any(k.startswith("adv") for k in rs["state_last"]), "the adversary leaked into the member state"
    kc = dict(kw, epochs=1, lr_enc=1e-5, lr_head=3e-3)
    rc = FM.fit_eval_fm("rand_st_base", d, 50000, CH, log=lambda m: None, grl=0.3, warm=0.0, init_state=rs["state_last"], cont_reinit=False, **kc)
    assert len(rc["grl_curve"]) == 1 and np.isfinite(rc["prob_last"]).all()
    row = rc["curves"][-1]
    assert abs(row["train_task_loss"] + row["train_adv_loss"] - row["train_loss"]) < 1e-6, row
    print(f"6. the continuation (init_state from a grl run, as --fm-cont-*) runs with a fresh adversary (curve {rc['grl_curve']}); "
          f"no adversary key in state_last; curves split train_loss = task {row['train_task_loss']:.4f} + adversary {row['train_adv_loss']:.4f} (Track3 LOW)")
    print("OK")


if __name__ == "__main__":
    main()

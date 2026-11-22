#!/usr/bin/env python
"""T2-D644 (W27, astra: the ST channel-embedding table released from LLRD's depth-0 lr), CPU, a random-init ST-base, 1 layer, 1 epoch, LLRD 0.75:
  1. chemb_lr absent == the pre-D644 t2p_fm.py (pinned commit d6b7566a) BITWISE (prob_last);
  2. chemb_lr 1e-4: the log names the table's own group (peak lr 1e-4, weight decay 0; it left enc_L0's ~3.2e-6); the table's INACTIVE rows
     (never indexed) are bitwise equal to the off run's (zero grad + wd 0 -> unchanged), the ACTIVE rows (enc.chan_idx) moved differently, and
     the output differs; a LoRA-only encoder is refused;
  3. t2p_bench: --fm-chemb-lr absent leaves the result-file tag unchanged; on, the tag gains 'ce<lr>'.
  python tests/test_t2_d644_chemb.py  ->  "OK"
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

REF_COMMIT = "d6b7566a"
CH = json.loads((Path.home() / "workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1/meta.json").read_text())["ch_names"]


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(644)
    nn_, nt = 48, 12
    yy, yt = (np.arange(nn_) % 3).astype(np.int64), (np.arange(nt) % 3).astype(np.int64)
    d = dict(Xtr=rng.standard_normal((nn_, len(CH), 480)).astype(np.float32), ytr=yy, Xva=np.zeros((0, len(CH), 480), np.float32), yva=yy[:0],
             cva=np.array([], dtype=object), Xte=rng.standard_normal((nt, len(CH), 480)).astype(np.float32), yte=yt,
             cte=np.array(["a"] * nt, dtype=object), ext=[], utr=(np.arange(nn_) % 2).astype(np.int64))
    kw = dict(epochs=1, bs=24, lr_enc=1e-4, lr_head=3e-2, wd=0.05, llrd=0.75, device="cpu", layer=1, calib="gain", ls=0.1, return_model=True)
    fd = tempfile.NamedTemporaryFile(dir=ROOT / "tracks/t2_bci/scripts", prefix="_t2p_fm_ref_", suffix=".py", delete=False)
    ref = Path(fd.name)
    fd.close()
    try:
        ref.write_text(subprocess.run(["git", "-C", str(ROOT), "show", f"{REF_COMMIT}:tracks/t2_bci/scripts/t2p_fm.py"], capture_output=True, text=True, check=True).stdout)
        assert "T2-D644" not in ref.read_text()
        spec = importlib.util.spec_from_file_location("t2p_fm_ref", ref)
        OLD = importlib.util.module_from_spec(spec)
        sys.modules["t2p_fm_ref"] = OLD
        spec.loader.exec_module(OLD)
        r_off = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
        r_old = OLD.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, **kw)
    finally:
        ref.unlink(missing_ok=True)
    assert np.array_equal(r_off["prob_last"], r_old["prob_last"]), float(np.abs(r_off["prob_last"] - r_old["prob_last"]).max())
    print(f"1. chemb_lr absent == the pre-D644 t2p_fm.py ({REF_COMMIT}) bitwise (prob_last; random-init ST-base, 1 layer, 1 epoch, LLRD 0.75)")
    logs = []
    r_on = FM.fit_eval_fm("rand_st_base", d, 0, CH, log=logs.append, chemb_lr=1e-4, **kw)
    line = [m for m in logs if m.startswith("T2-D644 channel-embedding table")]
    assert len(line) == 1 and "peak lr 0.0001, weight decay 0" in line[0] and "was enc_L0_nowd lr 3.17e-06 wd 0" in line[0], line
    tab = lambda r: r["net_last"].enc.vit.enc_channel_emd.channel_transformation.weight.detach().numpy()   # noqa: E731
    act = np.zeros(tab(r_on).shape[0], bool)
    act[np.asarray(r_on["chan_idx"])] = True
    assert act.sum() == len(CH) and r_on["chan_idx"] == r_off["chan_idx"], (act.sum(), len(CH))
    assert np.array_equal(tab(r_on)[~act], tab(r_off)[~act]), "inactive rows moved"
    da = float(np.abs(tab(r_on)[act] - tab(r_off)[act]).max())
    dp = float(np.abs(r_on["prob_last"] - r_off["prob_last"]).max())
    assert da > 0 and dp > 0, (da, dp)
    try:
        FM.fit_eval_fm("rand_st_base", d, 0, CH, log=lambda m: None, chemb_lr=1e-4, lora=4, **kw)
        raise SystemExit("FAIL: chemb_lr with a LoRA encoder was accepted")
    except AssertionError as e:
        assert "T2-D644" in str(e), e
    wv = np.ones(nn_, np.float32); wv[::3] = 1 / 3                     # the combination rule's U = C (user-balanced, data['wtr']) + chemb
    r_u = FM.fit_eval_fm("rand_st_base", dict(d, wtr=wv), 0, CH, log=lambda m: None, chemb_lr=1e-4, **kw)
    assert np.isfinite(r_u["prob_last"]).all() and np.array_equal(tab(r_u)[~act], tab(r_off)[~act])
    print(f"2. chemb_lr 1e-4: {line[0]}; inactive rows ({(~act).sum()}) bitwise == the off run, active rows ({act.sum()}) moved "
          f"(max |d| {da:.2e}), output differs ({dp:.2e}); LoRA refused; U = wtr + chemb_lr runs (inactive rows unchanged)")
    src = (ROOT / "tracks/t2_bci/scripts/t2p_bench.py").read_text()
    assert '+ (f"ce{args.fm_chemb_lr:g}" if getattr(args, "fm_chemb_lr", None) else ""))' in src and '"--fm-chemb-lr", type=float, default=None' in src
    print("3. t2p_bench: tag 'ce<lr>' only when --fm-chemb-lr is given (default None = the old tag)")
    print("OK")


if __name__ == "__main__":
    main()

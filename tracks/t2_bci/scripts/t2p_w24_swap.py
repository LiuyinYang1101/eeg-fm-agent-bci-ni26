#!/usr/bin/env python
"""T2-D638 W24' (PREPARED; only if rule (b) passes, and a ship use needs the owner's typed line): the CNN-member replacement of a gated base
package (W21, or W23 when rule (b) picks the session-1 router) by a user-balanced CNN-only donor (t2p_final --train all --user-balance, W21's
CNN specs). EXACTLY the six CNN files are replaced: weights.pt, weights2.pt, weights3.pt (the pooled nets) and weights_pu.pt, weights2_pu.pt,
weights3_pu.pt (the per-user copies, fine-tuned unweighted from the balanced nets); every other file (config.json, submission.py, the FM members,
side heads, dcfam, tta_vp.pt, ...) is the base's byte for byte. Asserts first: the donor's config says user_balance; its CNN specs (models,
model_kwargs, recipes, epochs, seeds, align modes and scales, the copies' n_seeds / users / recipe_pu / pu_lr / pu_epochs / files) == the base's;
each replaced file has the base's structure (lists, state-dict keys and shapes) but other values. The new zip keeps the base's member order and
per-member compression. Writes <out>.swap.json (per-file sha256, replaced / kept, max |dθ| per file).
  python tracks/t2_bci/scripts/t2p_w24_swap.py --base BASE.zip --donor DONOR.zip --out NEW.zip
"""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

SWAP = ("weights.pt", "weights2.pt", "weights3.pt", "weights_pu.pt", "weights2_pu.pt", "weights3_pu.pt")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def leaves(obj, path=""):
    """flatten nested lists / dicts of tensors -> [(path, tensor)] in a fixed order."""
    if isinstance(obj, torch.Tensor):
        return [(path, obj)]
    if isinstance(obj, dict):
        return [x for k in obj for x in leaves(obj[k], f"{path}/{k}")]
    if isinstance(obj, (list, tuple)):
        return [x for i, v in enumerate(obj) for x in leaves(v, f"{path}[{i}]")]
    return [(path, obj)]


def spec(cfg):
    a = lambda d: dict(mode=d["mode"], scale=round(float(d["scale"]), 6))   # noqa: E731
    out = dict(model=cfg["model"], model_kwargs=cfg["model_kwargs"], recipe=cfg["recipe"], epochs=cfg["epochs"], seeds=cfg["seeds"],
               align=a(cfg["align"]), train=cfg["train"], train_subjects=cfg["train_subjects"])
    for k in ("cnn2", "cnn3"):
        c = cfg[k]
        out[k] = dict(model=c["model"], model_kwargs=c["model_kwargs"], recipe=c["recipe"], epochs=c["epochs"], seeds=c["seeds"], file=c["file"],
                      align=a(c["align"]))
    for k in ("user_blend", "user_blend2", "user_blend3"):
        u = cfg[k]
        out[k] = {f: u.get(f) for f in ("users", "file", "n_seeds", "recipe_pu", "pu_lr", "pu_epochs")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zb, zd = zipfile.ZipFile(a.base), zipfile.ZipFile(a.donor)
    cb, cd = json.loads(zb.read("config.json")), json.loads(zd.read("config.json"))
    assert cd.get("user_balance") is True and "user_balance" not in cb, "the donor must be a --user-balance build and the base must not"
    assert cd.get("export_sha") == cb.get("export_sha") and cd.get("kit_sha") == cb.get("kit_sha"), \
        ("the donor was trained on another export / kit (Track3 LOW)", cd.get("export_sha"), cb.get("export_sha"), cd.get("kit_sha"), cb.get("kit_sha"))
    sb, sd = spec(cb), spec(cd)
    diff = {k: (sb[k], sd[k]) for k in sb if sb[k] != sd[k]}
    assert not diff, f"the donor's CNN specs differ from the base's: {json.dumps(diff, default=str)[:2000]}"
    names_b = [i.filename for i in zb.infolist()]
    assert all(f in names_b for f in SWAP) and all(f in zd.namelist() for f in SWAP), "a CNN file is missing in the base or the donor"
    rep = dict(base=a.base, base_sha256=sha(Path(a.base).read_bytes()), donor=a.donor, donor_sha256=sha(Path(a.donor).read_bytes()), files={},
               versions_base=cb.get("versions"), versions_donor=cd.get("versions"),
               versions_note=("the CNN members (the six replaced files) were trained with versions_donor; the package config.json is the base's "
                              "byte for byte and records versions_base (the FM / side / DC members' training env); load compatibility under the "
                              "worker's torch is proven by the inert repack, K1 and K2"))
    if cb.get("versions") != cd.get("versions"):
        print(f"[w24 swap] NOTE: the donor was trained with {cd.get('versions')}; the base config records {cb.get('versions')}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        for f in SWAP:                                  # structure equal, values different
            pb, pd_ = Path(td) / f"b_{f}", Path(td) / f"d_{f}"
            pb.write_bytes(zb.read(f)); pd_.write_bytes(zd.read(f))
            lb, ld = leaves(torch.load(pb, map_location="cpu")), leaves(torch.load(pd_, map_location="cpu"))
            assert [p for p, _ in lb] == [p for p, _ in ld], f"{f}: another structure"
            assert all(not isinstance(x, torch.Tensor) or (x.shape == y.shape and x.dtype == y.dtype) for (_, x), (_, y) in zip(lb, ld)), f"{f}: shapes / dtypes"
            dmax = max(float((x.double() - y.double()).abs().max()) for (_, x), (_, y) in zip(lb, ld) if isinstance(x, torch.Tensor) and x.is_floating_point())
            assert dmax > 0, f"{f}: identical to the base's (nothing replaced)"
            rep["files"][f] = dict(action="replaced (donor)", tensors=len(lb), max_abs_dtheta=dmax, sha256=sha(zd.read(f)), base_sha256=sha(zb.read(f)))
        out = Path(a.out)
        assert not out.exists(), out
        with zipfile.ZipFile(out, "w") as zo:           # the base's member order and per-member compression
            for info in zb.infolist():
                data = zd.read(info.filename) if info.filename in SWAP else zb.read(info.filename)
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.compress_type, zi.external_attr = info.compress_type, info.external_attr
                zo.writestr(zi, data)
                if info.filename not in SWAP:
                    rep["files"][info.filename] = dict(action="kept (base, byte-identical)", sha256=sha(data))
    with zipfile.ZipFile(out) as zc:                    # read back
        for f, r in rep["files"].items():
            assert sha(zc.read(f)) == r["sha256"], f
        assert [i.filename for i in zc.infolist()] == names_b
    rep["out"], rep["out_sha256"] = str(out), sha(out.read_bytes())
    Path(str(out) + ".swap.json").write_text(json.dumps(rep, indent=1))
    print(f"[w24 swap] {out}: replaced {list(SWAP)} from the donor (max |dθ| " +
          ", ".join(f"{f} {rep['files'][f]['max_abs_dtheta']:.3g}" for f in SWAP) + f"); {len(names_b) - len(SWAP)} files kept byte-identical; sha256 {rep['out_sha256']}", flush=True)


if __name__ == "__main__":
    main()

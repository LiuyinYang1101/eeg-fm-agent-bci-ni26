#!/usr/bin/env python3
"""Track 2: finish a t2p_preft.py run that was stopped (walltime / scancel) after its `--save-at` checkpoints were written
but before its own end-of-run selection, by applying the SAME selection from the job log (stdlib only; login node).

  python3 t2p_preft_finalize.py --out <runs/ni26/ni26-t2p-preft-X> --log <logs/...out> --fm mor_p1v2 --note "..."

Reads the log's `saved checkpoint-stepN.pth (held-out BA mean M, sha256 S12)` lines, checks every file's sha256 against the
logged prefix, picks the best mean held-out BA (ties -> the earliest step; t2p_preft.py's rule), copies it to
checkpoint-cpt.pth and writes cpt_summary.json in t2p_preft's format (what t2p_fm.use_cpt checks: status, fm, base_sha256 =
the init file's, ckpt_sha256 = the copied file's, args.steps, pools[].name), marked `finalized_from_log`. Never overwrites an
existing checkpoint-cpt.pth / cpt_summary.json.
"""

import argparse
import ast
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="the t2p_preft --out directory")
    ap.add_argument("--log", required=True, help="the job's log")
    ap.add_argument("--fm", required=True)
    ap.add_argument("--decision", default="")
    ap.add_argument("--steps", type=int, default=0, help="the run's configured --steps (checked against the log's step lines)")
    ap.add_argument("--note", required=True, help="why the run was finalized from its log")
    args = ap.parse_args()
    out, text = Path(args.out), Path(args.log).read_text()
    ck, sj = out / "checkpoint-cpt.pth", out / "cpt_summary.json"
    assert not ck.exists() and not sj.exists(), f"{out} already has checkpoint-cpt.pth / cpt_summary.json"
    assert re.search(rf"\b{re.escape(args.fm)}\b", text), f"--fm {args.fm} not in the log"

    m = re.search(r"init = the pinned root checkpoint (\S+) \(sha256 ([0-9a-f]{12})\)", text)
    assert m, "no 'init = the pinned root checkpoint' line"
    init_path, init12 = m.group(1), m.group(2)
    base = sha256_file(init_path)
    assert base.startswith(init12), f"init file sha {base[:12]} != logged {init12}"

    steps = [int(s) for s in re.findall(r"\] step \d+/(\d+) loss", text)]
    done = [int(s) for s in re.findall(r"\] step (\d+)/\d+ loss", text)]
    assert len(set(steps)) <= 1 and (not steps or not args.steps or steps[0] == args.steps), (set(steps), args.steps)
    n_steps = steps[0] if steps else args.steps
    assert n_steps, "no step lines in the log: pass --steps"
    pools = []
    for pm in re.finditer(r"\] pool (\S+): (\d+) FM-known ch \(dropped (\[.*?\])\), (\d+) train / (\d+) val "
                          r"\(holdout (\w+), subjects (\[.*?\])\)", text):
        pools.append(dict(name=pm.group(1), ch=int(pm.group(2)), dropped=ast.literal_eval(pm.group(3)),
                          n_train=int(pm.group(4)), n_val=int(pm.group(5)), holdout=pm.group(6),
                          held_subjects=ast.literal_eval(pm.group(7))))
    assert pools, "no pool lines"

    saved = []
    for sm in re.finditer(r"\] saved (checkpoint-step(\d+)\.pth) \(held-out BA mean ([0-9.]+), sha256 ([0-9a-f]{12})\)", text):
        p = out / sm.group(1)
        assert p.exists(), p
        full = sha256_file(p)
        assert full.startswith(sm.group(4)), f"{p.name}: sha {full[:12]} != logged {sm.group(4)}"
        saved.append(dict(step=int(sm.group(2)), mean_ba=float(sm.group(3)), path=str(p), sha256=full))
    assert saved, "no saved checkpoints in the log"
    selected = max(saved, key=lambda r: (r["mean_ba"], -r["step"]))   # t2p_preft.py's rule
    shutil.copyfile(selected["path"], ck)
    ck_sha = sha256_file(ck)
    assert ck_sha == selected["sha256"]
    cal = [ln.split("] ", 1)[1] for ln in text.splitlines() if "calibrat" in ln and ln.startswith("[t2p_preft")]
    summ = dict(status="ok", fm=args.fm, kind=f"supervised pre-fine-tune ({args.decision}), finalized from the job log",
                base_sha256=base, ckpt_sha256=ck_sha, parent=dict(ckpt=init_path, sha256=base, kind="pinned root"),
                args=dict(steps=n_steps, steps_run=max(done) if done else None, decision=args.decision),
                pools=pools, saved=saved, selected=selected, mae_calibration_log=cal,
                finalized_from_log=dict(log=str(args.log), note=args.note,
                                        at=datetime.datetime.now().isoformat(timespec="seconds")))
    sj.write_text(json.dumps(summ, indent=1))
    print("selection (pools' held-out BA only): " + ", ".join(f"{r['step']} {r['mean_ba']}" for r in saved) +
          f" -> step {selected['step']}; {ck} sha256 {ck_sha[:12]}; base {base[:12]}; {len(pools)} pools; "
          f"steps run {summ['args']['steps_run']} / {summ['args']['steps']}")


if __name__ == "__main__":
    main()

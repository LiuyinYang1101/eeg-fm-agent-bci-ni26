#!/usr/bin/env python
"""Write the worker-image distribution allowlist `dists_<kit sha7>.txt` read by sitecustomize.py (the import guard).

The Codabench image is python:3.12-slim + torch/torchvision/torchaudio + `pip install -r requirements.txt` of the kit
(tools/Dockerfile). Its distribution set is reproduced by pip's resolver without installing anything:

    python make_dists.py --kit third_party/2026-competition --sha 442edcd

Needs PyPI access (resolution only). Re-run whenever the kit pin moves; commit the new file.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--sha", required=True)
    args = ap.parse_args()
    req = subprocess.run(["git", "-C", args.kit, "show", f"{args.sha}:requirements.txt"], check=True,
                         capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as td:
        rf, rep = Path(td) / "req.txt", Path(td) / "report.json"
        rf.write_text(req + "\ntorch\ntorchvision\ntorchaudio\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
                        "--report", str(rep), "-r", str(rf)], check=True)
        r = json.loads(rep.read_text())
    names = sorted({i["metadata"]["name"] for i in r["install"]}, key=str.lower)
    out = HERE / f"dists_{args.sha[:7]}.txt"
    head = (f"# Codabench worker image distributions, kit {args.sha[:7]} requirements.txt + torch trio, resolved by pip "
            f"{r.get('pip_version', '?')} on python {r['environment']['python_full_version']} (make_dists.py)\n")
    out.write_text(head + "\n".join(names) + "\n")
    print(f"{out}: {len(names)} distributions")


if __name__ == "__main__":
    main()

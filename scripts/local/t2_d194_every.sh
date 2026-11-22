#!/usr/bin/env bash
# T2-D194 step 1 (local CPU, zero compute): the package-shape cost of routing on every k-th window. T2-D159's FBCNet r1_pulxx copies (r 1/3,
# seeds 48-59) and T2-D164's ShallowNet r1_pulx copies (r 1/2, seeds 24-35) re-blended with `--every k` (k = 4, 8); cA_k vs cA (every 1).
#   mem-guard bash scripts/local/t2_d194_every.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniconda3/envs/neural_interface_challenge/bin/python
E=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
O=$HOME/workspace/outputs/neural_interface_challenge
S=$O/sofia_s1
blend() { $PY -u tracks/t2_bci/scripts/t2p_fpcalib.py blend --export "$E" --fp "$O/t2_d124b/fp_proteus.npy" "$@"; }
for k in 4 8; do
  for p in kit seen1; do
    d=( "$S"/ni26-t2p-d159-fbc-r1_pulxx-$p-*/ ); blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d194/fbc_e$k" --model fbcnet_bboth --recipe r1_pulxx --every $k --tag 3e$k --seeds 48,49,50,51,52,53,54,55,56,57,58,59
  done
  d=( "$S"/ni26-t2p-d164-sh-r1_pulx-[0-9]*/ )
  for p in kit seen1; do
    blend --runs "${d[0]}runs" --protocols $p --out "$O/t2_d194/sh_e$k" --model shallow --align ea --recipe r1_pulx --r 0.5 --every $k --tag 2e$k --seeds 24,25,26,27,28,29,30,31,32,33,34,35
  done
done
$PY -I - <<'PYEOF'
import json
O = "~/workspace/outputs/neural_interface_challenge"
s = json.load(open(f"{O}/t2_d164/spec.json")); m = {k: s["members"][k] for k in ("fbcb_pulxxh", "shb_pulx", "stl", "mor")}
arms = {"cA": {"fbcb_pulxxh": 0.5, "shb_pulx": 0.2, "stl": 0.2, "mor": 0.1}}
for k in (4, 8):
    m[f"fbc_e{k}"] = {"offset": 48, **{p: f"{O}/t2_d194/fbc_e{k}/{p}__fbcnet_bboth__none__n00__r1_pulxxfb3e{k}__s{{s}}.npz" for p in ("kit", "seen1")}}
    m[f"sh_e{k}"] = {"offset": 24, **{p: f"{O}/t2_d194/sh_e{k}/{p}__shallow__ea__n00__r1_pulxfb2e{k}__s{{s}}.npz" for p in ("kit", "seen1")}}
    arms[f"cA_e{k}"] = {f"fbc_e{k}": 0.5, f"sh_e{k}": 0.2, "stl": 0.2, "mor": 0.1}
open(f"{O}/t2_d194/spec_every.json", "w").write(json.dumps({"n": 12, "anchor": "cA", "bar": "", "members": m, "arms": arms}, indent=2) + "\n")
PYEOF
$PY -u tracks/t2_bci/scripts/t2p_pkgshape.py --export "$E" --spec "$O/t2_d194/spec_every.json" --out "$O/t2_d194/read_every.json" > "$O/t2_d194/read_every.txt" 2>&1
echo "rc $?"

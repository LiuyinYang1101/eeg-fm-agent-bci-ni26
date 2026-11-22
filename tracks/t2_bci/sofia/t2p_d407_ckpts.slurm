#!/usr/bin/env bash
# T2-D407: stage the in-house FM project's C / C + P1 checkpoints for the bench, free CPU (t2p_mor_ckpts.slurm's pattern, sources from the environment so the
# code needs no edit when the in-house FM project sends the paths). In THIS job's own runs/ni26 dir (never in the in-house FM project's runs/fm dirs): a symlink <name>.pt per
# checkpoint + its tracks/fm/fm_contract.py record beside it (<name>.pt.contract.json, family mor_p1v2: sha256, key set = the pinned root's,
# native rate 256), which t2p_fm.use_cpt requires for a contract-only checkpoint. Any source missing -> rc 2; any contract not ok -> rc 4.
# NAMES = '+'-separated link names; SRC_<name> = the absolute sofia path of that checkpoint. Re-running with the same RUN_NAME adds links.
# STRIP = '+'-separated names whose checkpoint carries extra (P1 supervised-head) keys: written as a head-less copy with exactly the pinned
# mor_p1v2 root's key set (+ <name>.pt.strip.json: source sha256, kept / dropped counts) instead of a symlink.
# Never `s1 fetch` this dir (the links point at ≈ 1.2 GB files).
#   S1_EXPORT=RUN_NAME=ni26-t2p-d407-ckpts,NAMES=C150k,SRC_C150k=/sofia/... S1_PURPOSE="..." \
#     s1 submit ni26 tracks/t2_bci/sofia/t2p_d407_ckpts.slurm --time=00:30:00 --job-name=ni26-t2p-d407-ckpts
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-}")
rc=0; n=0
for k in ${NAMES//+/ }; do
  n=$((n + 1))
  v="SRC_$k"; s="${!v:-}"
  [ -n "$s" ] && [ -f "$s" ] || { echo "missing source for $k: '$s'"; rc=2; continue; }
  if [[ "+${STRIP:-}+" == *"+$k+"* ]]; then
    # T2-D407 amendment (12:5x, before any CP1 number): a C + P1 checkpoint carries P1's supervised heads on top of the MOR layout; the
    # strict loaders and fm_contract demand the pinned mor_p1v2 root's key set (the pinned P1 v2 is itself saved head-less), and the T2
    # fine-tune builds its own head. So keep EXACTLY the root's keys (all present, same shapes, else rc 4), every other top-level field
    # unchanged, and write a real file <name>.pt here (not a symlink); the source sha256 is recorded beside it.
    rm -f "$OUT/$k.pt"
    python - "$s" "$OUT/$k.pt" "$S1_CODE" <<'PY' || { rc=4; continue; }
import hashlib, json, sys
import torch
src, dst, code = sys.argv[1:4]
sys.path.insert(0, f"{code}/tracks/t1_image/scripts")
import t1_fm as t1
def state_of(ck):
    return ck["model"] if isinstance(ck, dict) and "model" in ck and isinstance(ck["model"], dict) else ck
ck = torch.load(src, map_location="cpu", weights_only=False)
st = state_of(ck); root = state_of(torch.load(t1.CKPT["mor_p1v2"], map_location="cpu", weights_only=False, mmap=True))
miss = [k for k in root if k not in st]; bad = [k for k in root if k in st and tuple(st[k].shape) != tuple(root[k].shape)]
assert not miss and not bad, ("missing", miss[:5], "shape", bad[:5])
extra = sorted(k for k in st if k not in root)
new = {k: st[k] for k in root}
out = dict(ck, model=new) if st is not ck else new
torch.save(out, dst)
h = hashlib.sha256()
with open(src, "rb") as f:
    while b := f.read(1 << 24):
        h.update(b)
json.dump(dict(source=src, source_sha256=h.hexdigest(), kept=len(new), dropped=len(extra), dropped_prefixes=sorted({k.split(".")[0] for k in extra})),
          open(dst + ".strip.json", "w"), indent=1)
print(f"strip {src.rsplit('/', 2)[-2]}: kept {len(new)} root keys, dropped {len(extra)} ({sorted({k.split('.')[0] for k in extra})}); source sha {h.hexdigest()[:12]}")
PY
  else
    ln -sfn "$s" "$OUT/$k.pt"
  fi
  python -u "$S1_CODE/tracks/fm/fm_contract.py" "$OUT/$k.pt" --fm mor_p1v2 --out "$OUT/$k.pt.contract.json" > "$OUT/$k.contract.log" 2>&1 || rc=4
  python - "$OUT/$k.pt.contract.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(sys.argv[1].rsplit("/", 1)[-1], c.get("status"), "keyset", c.get("keyset_scope"), c.get("n_tensors"), "rate", c.get("native_rate_hz"), "sha", (c.get("sha256") or "")[:12], c.get("problems"))
PY
done
[ "$n" -ge 1 ] || { echo "NAMES is empty"; rc=2; n=1; }
s1_finish $rc "$OUT" '*.pt.contract.json' "$n"

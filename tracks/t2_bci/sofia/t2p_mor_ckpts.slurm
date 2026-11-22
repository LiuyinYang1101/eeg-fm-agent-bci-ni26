#!/usr/bin/env bash
# T2-D386: stage the MOR dose checkpoints for the bench, free CPU. In THIS job's own runs/ni26 dir (never in the in-house FM project's runs/fm dirs):
# a symlink per checkpoint (mor100 = the pinned inhouse_fm MOR ck150k = B's init = dose 0; B5k / B10k / B20k = the in-house FM project's annealed
# continuations) + its tracks/fm/fm_contract.py record beside the symlink (<link>.contract.json, family mor_p1v2: sha256, key set =
# the pinned root's, native rate 256), which t2p_fm.use_cpt requires for a contract-only checkpoint. Any contract not ok -> rc 4.
# Never `s1 fetch` this dir (the links point at ≈ 1.2 GB files).
#   S1_EXPORT=RUN_NAME=ni26-t2p-d386-mor-ckpts S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2p_mor_ckpts.slurm --time=00:30:00 --job-name=ni26-t2p-d386-ckpts
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
declare -A SRC=(
  [mor100]="$S1_ROOT/models/inhouse_100m_mor_ck150k.pt"
  [B5k]="$S1_ROOT/runs/fm/ni26fm_B_cont5k/checkpoint-00005000.pt"
  [B10k]="$S1_ROOT/runs/fm/ni26fm_B_cont10k/checkpoint-00010000.pt"
  [B20k]="$S1_ROOT/runs/fm/ni26fm_B_cont20k/checkpoint-00020000.pt"
)
# T2-D463: CKPT_PATH (S1_EXPORT) = ONE checkpoint path under S1_ROOT, linked as c2l.pt, replaces the D386 list (a .sha256 sidecar is hash-checked first)
KEYS=(mor100 B5k B10k B20k)
[ -n "${CKPT_PATH:-}" ] && { SRC=([c2l]="$S1_ROOT/$CKPT_PATH"); KEYS=(c2l); }
rc=0
for k in "${KEYS[@]}"; do
  s=${SRC[$k]}
  [ -f "$s" ] || { echo "missing $s"; rc=2; continue; }
  if [ -f "$s.sha256" ]; then [ "$(sha256sum "$s" | cut -d" " -f1)" = "$(cut -d" " -f1 "$s.sha256")" ] || { echo "sha mismatch $s"; rc=3; continue; }; fi
  ln -sfn "$s" "$OUT/$k.pt"
  python -u "$S1_CODE/tracks/fm/fm_contract.py" "$OUT/$k.pt" --fm mor_p1v2 --out "$OUT/$k.pt.contract.json" > "$OUT/$k.contract.log" 2>&1 || rc=4
  python - "$OUT/$k.pt.contract.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(sys.argv[1].rsplit("/", 1)[-1], c.get("status"), "keyset", c.get("keyset_scope"), c.get("n_tensors"), "rate", c.get("native_rate_hz"), "sha", (c.get("sha256") or "")[:12], c.get("problems"))
PY
done
s1_finish $rc "$OUT" '*.pt.contract.json' "${#KEYS[@]}"

#!/usr/bin/env bash
# T2-FM-D6: copy the two shipped CPT v1 ST-L checkpoints (+ their cpt_summary.json) from runs/ni26 into runs/fm/t2fm-p1-init/<proto>/
# for the fm-tag ST-L + P1 runs (the manager's ruling 2026-11-06 ≈ 21:0x: an ni26-tag job copies them, sha-checked; an fm job never
# cross-reads runs/ni26). Models, not data; 4 files. CPU only -> zen5_vis (free).
#   S1_PURPOSE="..." s1 submit ni26 tracks/t2_bci/sofia/t2fm_copy_init.slurm --partition=zen5_vis --time=00:30:00 --job-name=ni26-t2fm-copyinit \
#       [-- <proto>:<runs/ni26 dir> ...]   (default kit:ni26-t2p-cpt-stlarge-d89b seen1:ni26-t2p-cpt-d111-with; the ship arm: all:ni26-t2p-cpt-d111-all)
#SBATCH --partition=zen5_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
set -uo pipefail
source "$S1_ROOT/hub/remote/lib/s1-job.sh"
s1_banner
[ "${S1_TAG:-}" = ni26 ] || { echo "!! [copyinit] tag ${S1_TAG:-?} != ni26" >&2; exit 2; }
s1_env ni26b || exit 2
OUT=$(s1_outdir "${RUN_NAME:-ni26-t2fm-copyinit}")
DST="$S1_ROOT/runs/fm/t2fm-p1-init"
rc=0
PAIRS=("$@"); [ ${#PAIRS[@]} -gt 0 ] || PAIRS=("kit:ni26-t2p-cpt-stlarge-d89b" "seen1:ni26-t2p-cpt-d111-with")
for pair in "${PAIRS[@]}"; do
  p=${pair%%:*}; src="$S1_ROOT/runs/ni26/${pair#*:}"
  mkdir -p "$DST/$p" || { rc=1; continue; }
  python - "$src" "$DST/$p" <<'PY' || rc=1
import hashlib, json, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
sm = json.loads((src / "cpt_summary.json").read_text())
assert sm.get("status") == "ok", sm.get("status")
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 24):
            h.update(b)
    return h.hexdigest()
s0 = sha(src / "checkpoint-cpt.pth")
assert s0 == sm["ckpt_sha256"], f"{src}: file sha {s0[:12]} != summary {sm['ckpt_sha256'][:12]}"
shutil.copy2(src / "checkpoint-cpt.pth", dst / "checkpoint-cpt.pth.tmp")
s1 = sha(dst / "checkpoint-cpt.pth.tmp")
assert s1 == s0, f"copy sha {s1[:12]} != source {s0[:12]}"
(dst / "checkpoint-cpt.pth.tmp").rename(dst / "checkpoint-cpt.pth")
shutil.copy2(src / "cpt_summary.json", dst / "cpt_summary.json")
(dst / "COPIED_FROM.json").write_text(json.dumps(dict(src=str(src), ckpt_sha256=s0, base_sha256=sm.get("base_sha256"),
                                                      steps=sm.get("steps_done"), proteus_streams=sm.get("proteus_streams")), indent=1))
print(f"[copyinit] {src.name} -> {dst} sha256 {s0}", flush=True)
PY
done
ls -la "$DST"/*/ | tee -a "$OUT/copied.txt"
[ $rc = 0 ] && touch "$OUT/copied.ok"
s1_finish "$rc" "$OUT" 'copied.ok' 1

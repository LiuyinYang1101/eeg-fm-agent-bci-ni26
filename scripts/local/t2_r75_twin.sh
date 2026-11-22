#!/usr/bin/env bash
# Row 75 (APPROVED by the user 2026-10-29: "one combined warm-up twin ... built Oct 30, I upload it Oct 31"): the warm-up twin = T2-W7's
# package (34d79b8d, kit15 SK10) repacked with the clean MOR-free template and tv8c ON: one `s1 sync ni26` (commit first), then ONE
# t2p_repack.slurm job (repack + clean checks NEW0 = REF + robustness + timing, then the kit replay under the worker-image guard).
# ≈ 0.3 GPU-h. Read: `s1 fetch ni26 ni26-t2p-repack-r75-twin-<id>`; the .repack.json (NEW0 vs REF 0 differ; NEW ms / window <= 18.1) and the
# kit_sim / kit_warmup json; the ZIP + sha256 go to SUBMISSIONS for the user's Oct 31 upload.
#   bash scripts/local/t2_r75_twin.sh            (refuses before 2026-10-30)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ "$(date +%Y%m%d)" -ge 20261029 ] || { echo "row 75: the twin is built from Oct 29 (the user, 2026-10-29: \"do it asap\")"; exit 2; }
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the row 75 code first"; exit 2; }
W7=$S1_ROOT/runs/ni26/ni26-t2p-repack-d286-kit15sk-1540552/t2p_d286_kit15sk_r.zip
s1 sync ni26
S1_PURPOSE="T2 row 75 (user-approved 2026-10-29): the combined warm-up twin, W7's members, MOR-free SK10 template + tv8c ON, ~0.3 GPU-h; feeds the user's Oct 31 upload" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 --job-name=ni26-t2p-repack-r75-twin -- "$W7" t2p_r75_twin \
  --drop-family mor --tta-vp 8 --tta-copies --tta-vp-recipe r1_pulxx_vp2_e180_vr_rc75 --gpu-graph

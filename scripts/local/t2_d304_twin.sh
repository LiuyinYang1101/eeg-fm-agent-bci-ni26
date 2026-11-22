#!/usr/bin/env bash
# T2-D304 SKλ twin (APPROVED by the user 2026-10-30: "I approve the SKλ twin (W8 + tta.lam 0.95, ≈ 0.3 GPU-h), build today; I upload it Oct 31"):
# T2-W8 (dc43b392, the row 75 template) repacked with the SKλ template and config tta.lam = 0.95 (everything else W8's): one `s1 sync ni26`, ONE
# t2p_repack.slurm job (T2-D210 reconfig checks: OLD vs REF inert; parts + the SKλ recombine vs t2p_report.causal_sinkhorn(lam); robustness; timing;
# the kit replay under the worker guard). ≈ 0.3 GPU-h. Read the .repack.json + kit_warmup.json, fetch the zip, re-hash -> SUBMISSIONS T2-W9.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_report.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the D304 code first"; exit 2; }
W8=$S1_ROOT/runs/ni26/ni26-t2p-repack-r75-twin-1547928/t2p_r75_twin.zip
s1 sync ni26
S1_PURPOSE="T2-D304 (user-approved 2026-10-30): the SK-lambda warm-up twin = W8 + tta.lam 0.95, ~0.3 GPU-h; feeds the user's Oct 31 upload" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 --job-name=ni26-t2p-repack-d304-skl-twin -- "$W8" t2p_d304_skl_twin \
  --set tta.lam=0.95

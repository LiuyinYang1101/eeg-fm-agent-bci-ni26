#!/usr/bin/env bash
# Row 76 W10 twin (the user, 2026-10-30: "Run the MIRepNet screen after W9 / W10 are built ..."): the MOR + SKλ warm-up twin:
# T2-W7 (34d79b8d, MOR, SK10) repacked with the current template (MOR branch) and config tta.lam = 0.95 (everything else W7's): one sync, ONE
# t2p_repack.slurm job (T2-D210 reconfig checks: OLD vs REF inert; parts + the SKλ recombine vs t2p_report.causal_sinkhorn(lam); robustness; timing;
# the kit replay under the worker guard). ≈ 0.16 GPU-h. Read the .repack.json + kit_warmup.json, fetch the zip, re-hash -> SUBMISSIONS T2-W10.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_report.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the D304 code first"; exit 2; }
W7=$S1_ROOT/runs/ni26/ni26-t2p-repack-d286-kit15sk-1540552/t2p_d286_kit15sk_r.zip
s1 sync ni26
S1_PURPOSE="T2 row 76 (the user's go 2026-10-30): the W10 warm-up twin = W7 + tta.lam 0.95 (MOR + SK-lambda), ~0.16 GPU-h; feeds the user's Oct 31 upload" \
  s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 --job-name=ni26-t2p-repack-r76-w10-twin -- "$W7" t2p_r76_w10_twin \
  --set tta.lam=0.95

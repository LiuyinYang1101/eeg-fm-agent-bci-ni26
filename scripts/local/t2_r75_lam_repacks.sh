#!/usr/bin/env bash
# Row 75 / 76 after T2-W9 / W10 Finished (2026-10-30; the user's approvals: "after it Finishes, the Nov 5 ZIPs get tta.lam by config" and the A4 MOR
# ZIP of row 76): config-only repacks with tta.lam = 0.95 (t2p_repack's T2-D210 reconfig checks: the template inert with lam off; parts + the SK-lambda
# recombine vs t2p_report.causal_sinkhorn(lam); robustness; timing; the kit replay under the worker guard):
#   the four MOR-free SK10 Nov 5 ZIPs (P1 / F1 / P2 / F2 from the row 75 repacks; P3 / F3 are the prior hedge: no lam)  +  A4 = 36f29f82 (P1's draw
#   WITH the MOR, SK10) + lam.  ≈ 5 x 0.16 GPU-h.
#   bash scripts/local/t2_r75_lam_repacks.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_report.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the code first"; exit 2; }
R=$S1_ROOT/runs/ni26
s1 sync ni26
while read -r name zip; do
  S1_PURPOSE="T2 row 75 / 76 (user-approved): SK-lambda (tta.lam 0.95) config repack of the Nov 5 ZIP $name, ~0.16 GPU-h; feeds the Nov 5 uploads" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 --job-name=ni26-t2p-repack-lam-$name -- "$R/$zip" t2p_lam_$name --set tta.lam=0.95
done <<'EOF'
p1-lv-d0 ni26-t2p-repack-r75-p1-lv-d0-sk-1548776/t2p_r75_p1-lv-d0-sk.zip
f1-lv-d1 ni26-t2p-repack-r75-f1-lv-d1-sk-1548777/t2p_r75_f1-lv-d1-sk.zip
p2-ln-d0 ni26-t2p-repack-r75-p2-ln-d0-sk-1548778/t2p_r75_p2-ln-d0-sk.zip
f2-ln-d1 ni26-t2p-repack-r75-f2-ln-d1-sk-1548779/t2p_r75_f2-ln-d1-sk.zip
a4-lv-d0-mor ni26-t2p-repack-d288-vp2r15sk-1541403/t2p_d288_vp2r15sk_r.zip
EOF

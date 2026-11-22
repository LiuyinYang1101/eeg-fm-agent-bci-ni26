#!/usr/bin/env bash
# Row 75 after-Finish step (APPROVED by the user 2026-10-29: "after a Finish, MOR-free repacks of the Nov 5 picks + fallbacks (≈ 0.9 GPU-h, Nov 2–4)
# with the kit gate; tv8c shipping is decided after the Nov 5 reads"; T2-W8 Finished 0.57080 on 2026-10-30): the six Nov 5 ZIPs
# (P1 / P2 / P3 + their fallbacks) repacked MOR-free (the proportional rule) into the warm-up-verified row 75 template, tv OFF (cnn_tta_vp absent),
# gpu_graph on; each job = t2p_repack.py's row 75 checks (NEW0 = REF exactly) + robustness + timing + the kit replay under the worker guard.
# ≈ 6 x 0.15 GPU-h. Read: fetch each dir's .repack.json / kit_warmup.json + the zip; the new sha256s go to the user.
#   bash scripts/local/t2_r75_repacks.sh            (refuses before 2026-11-02)
set -euo pipefail
cd "$(dirname "$0")/../.."
[ "$(date +%Y%m%d)" -ge 20261030 ] || { echo "row 75: the MOR-free repacks run from Oct 30 (the user, 2026-10-30: \"Build the MOR-free Nov 5 repacks today instead of Nov 2.\")"; exit 2; }
[ -z "$(git status --porcelain -- src/ni26/train.py tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_bench.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the row 75 code first"; exit 2; }
R=$S1_ROOT/runs/ni26
s1 sync ni26
while read -r name zip; do
  S1_PURPOSE="T2 row 75 (user-approved): MOR-free repack of the Nov 5 ZIP $name (tv off), ~0.15 GPU-h; feeds the Nov 5 uploads" \
    s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 --job-name=ni26-t2p-repack-r75-$name -- "$R/$zip" t2p_r75_$name \
    --drop-family mor --gpu-graph
done <<'EOF'
p1-lv-d0-sk ni26-t2p-repack-d288-vp2r15sk-1541403/t2p_d288_vp2r15sk_r.zip
f1-lv-d1-sk ni26-t2p-repack-d288-vp2r15d1sk-1541404/t2p_d288_vp2r15d1sk_r.zip
p2-ln-d0-sk ni26-t2p-repack-d288-nv15sk-1541406/t2p_d288_nv15sk_r.zip
f2-ln-d1-sk ni26-t2p-repack-d288-nv15d1sk-1541407/t2p_d288_nv15d1sk_r.zip
p3-lv-d0-prior ni26-t2p-repack-d276-vp2r15-1538347/t2p_d276_vp2r15_r.zip
f3-lv-d1-prior ni26-t2p-repack-d276-vp2r15d1-1538349/t2p_d276_vp2r15d1_r.zip
EOF

#!/usr/bin/env bash
# T2-D638 W24' donor on the LOCAL 5090 (the manager's speed option, ONLY after rule (b) passes; the same GPU / env as D638's cells and the
# builder golden): t2p_final CNN-only, W21's CNN specs (fbc / sh / ifn x seeds 200-214, the copies at r 0.3333 / 0.5 / 0.5 with W21's pu
# recipes, --train all), --user-balance, template W11 (three families) = launch_w24_d638.sh's DONOR_ARGS verbatim. Refuses unless the builder
# golden PASSED (outputs/t2_w24_golden/check.log). -> outputs/t2_w24_donor/<DN>.zip + .sha256; then
#   ROUTER=w21|w23 DONOR_LOCAL=<that zip> PREBUILD_CHOICE_AT=... bash tracks/t2_bci/sofia/launch_w24_d638.sh PREBUILD   (s1 put + sha pin, (2)-(5))
#   mem-guard bash scripts/local/t2_w24_donor_local.sh [--dry]
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -z "$(git status --porcelain -- src/ni26 tracks/t2_bci/scripts tracks/t2_bci/submission tracks/t2_bci/sofia/launch_w24_d638.sh scripts/local/t2_w24_donor_local.sh)" ] || { echo "commit first"; exit 2; }
grep -q "^GOLDEN PASS" "$HOME/workspace/outputs/neural_interface_challenge/t2_w24_golden/check.log" 2>/dev/null || { echo "the W24' builder golden has not PASSED"; exit 3; }
b_=$(python3 -c "import json; v = json.load(open('$HOME/workspace/outputs/neural_interface_challenge/t2_d638/result.json'))['rules']['verdict']; print(v['b_best'] or '')" 2>/dev/null || true)
[ -n "$b_" ] || { echo "rule (b) has not passed (outputs/t2_d638/result.json: no b_best): no donor"; exit 3; }   # the manager: golden AND rule (b)
echo "### rule (b) passed: best cell $b_ (ROUTER=${b_#c} for launch_w24_d638.sh)"
PY=$HOME/miniconda3/envs/neural_interface_challenge/bin/python
EXP=$HOME/workspace/datasets/neural_interface_challenge/t2_proteus/export_proteus_v1
D=$HOME/workspace/outputs/neural_interface_challenge/t2_w24_donor
DN=t2p_w24donor_PREBUILD
[ ! -e "$D/$DN.zip" ] || { echo "$D/$DN.zip exists"; exit 3; }
mapfile -t ARGS < <(sed -n '/^DONOR_ARGS=(/,/)$/p' tracks/t2_bci/sofia/launch_w24_d638.sh | sed 's/^DONOR_ARGS=(//; s/)$//' | tr ' ' '\n' | grep -v '^$' | sed "s/\$SEEDS/200,201,202,203,204,205,206,207,208,209,210,211,212,213,214/")
printf '%s\n' "${ARGS[@]}" | grep -q -- "--user-balance" || { echo "DONOR_ARGS not parsed"; exit 4; }
job=$(mktemp "$HOME/workspace/outputs/neural_interface_challenge/t2_w24_donor_job.XXXX.sh")
{
  echo "set -uo pipefail; cd '$(pwd)'; mkdir -p '$D'; echo \"### \$(date +%T) w24 donor start (git \$(git rev-parse --short HEAD))\" >> '$D/donor.log'"
  echo "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "$PY -u tracks/t2_bci/scripts/t2p_final.py --export '$EXP' --out '$D/$DN' ${ARGS[*]} >> '$D/donor.log' 2>&1 || { echo '!! donor failed' >> '$D/donor.log'; exit 5; }"
  echo "[ -f '$D/$DN.zip' ] || { echo '!! no donor zip' >> '$D/donor.log'; exit 6; }"
  echo "(cd '$D' && sha256sum '$DN.zip' > '$DN.zip.sha256'); echo \"### done w24 donor \$(date +%T): \$(cat '$D/$DN.zip.sha256')\" >> '$D/donor.log'"
} > "$job"
if [ "${1:-}" = --dry ]; then cat "$job"; rm -f "$job"; exit 0; fi
gpu-submit -L "neural_interface_challenge:t2-w24-donor" bash "$job"

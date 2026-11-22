#!/usr/bin/env bash
# T2 heavy plan H1 / backlog row 82 (fresh draws): config-only repacks of the BUILT ZIPs of docs/T2_SEALED_MANIFEST.md (MOR + SK10,
# members byte-identical to their prior fallbacks) into the Nov 5 read's state, one draw at a time (row 82: no pre-building).
# Exactly the Nov 5 pipelines (row 75):
#   w10 (the MOR stays; row 84 says >= +0.009):  ONE repack  --set tta.lam=0.95                          (as A4 d03a2707)
#   w9  (MOR-free):                              repack 1  --drop-family mor --gpu-graph  ->  repack 2 (afterok)  --set tta.lam=0.95   (as P1 81628a48)
# Each repack = t2p_repack.slurm's checks (reconfig parts, robustness, timing, the kit replay under the worker guard), ≈ 0.16 GPU-h.
# Draw order (row 82, fixed 2026-10-31 before any sealed score): L_V lv-d2 lv-d3 lv-d4 lv-d5 lv-d6 lv-d7; L_N ln-d5 ln-d6 ln-d2 ln-d3 ln-d4.
#   bash scripts/local/t2_h1_draws.sh dry w9 lv-d2 ln-d5      (prints the submissions; S1_DRYRUN for each; nothing runs)
#   bash scripts/local/t2_h1_draws.sh GO  w9 lv-d2 ln-d5      (submits; only after the Nov 5 read and the user's word)
# Then: s1 fetch, re-hash, the MOR file check, a docs/SUBMISSIONS.md row per ZIP (command, seeds, snapshot) before any upload.
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-}"; tpl="${2:-}"; shift 2 || true
[[ "$mode" == dry || "$mode" == GO ]] && [[ "$tpl" == w9 || "$tpl" == w10 ]] && [ $# -ge 1 ] \
  || { echo "usage: $0 dry|GO w9|w10 <draw> [<draw> ...]   (draws: lv-d2..lv-d7, ln-d2..ln-d6)"; exit 2; }
[ -z "$(git status --porcelain -- tracks/t2_bci/scripts/t2p_repack.py tracks/t2_bci/scripts/t2p_report.py tracks/t2_bci/submission/t2p_submission.py tracks/t2_bci/sofia/t2p_repack.slurm)" ] \
  || { echo "commit the repack code first"; exit 2; }
R=$S1_ROOT/runs/ni26
declare -A SRC=(
  [lv-d2]=ni26-t2p-repack-d288-vp2r15d2sk-1541405/t2p_d288_vp2r15d2sk_r.zip
  [lv-d3]=ni26-t2p-repack-d288-vp2r15d3sk-1541408/t2p_d288_vp2r15d3sk_r.zip
  [lv-d4]=ni26-t2p-repack-d288-vp2r15d4sk-1541409/t2p_d288_vp2r15d4sk_r.zip
  [lv-d5]=ni26-t2p-repack-d288-vp2r15d5sk-1541410/t2p_d288_vp2r15d5sk_r.zip
  [lv-d6]=ni26-t2p-repack-d288-vp2r15d6sk-1541411/t2p_d288_vp2r15d6sk_r.zip
  [lv-d7]=ni26-t2p-repack-d288-vp2r15d7sk-1541412/t2p_d288_vp2r15d7sk_r.zip
  [ln-d5]=ni26-t2p-repack-d288-nv15d5sk-1541413/t2p_d288_nv15d5sk_r.zip
  [ln-d6]=ni26-t2p-repack-d288-nv15d6sk-1541414/t2p_d288_nv15d6sk_r.zip
  [ln-d2]=ni26-t2p-repack-d288-nv2sk-1541415/t2p_d288_nv2sk_r.zip
  [ln-d3]=ni26-t2p-repack-d288-nv3sk-1541416/t2p_d288_nv3sk_r.zip
  [ln-d4]=ni26-t2p-repack-d288-nv4sk-1541417/t2p_d288_nv4sk_r.zip
)
for d in "$@"; do [ -n "${SRC[$d]:-}" ] || { echo "unknown draw $d"; exit 2; }; done
dry=(); [ "$mode" == dry ] && dry=(S1_DRYRUN=1)
[ "$mode" == GO ] && s1 sync ni26
sub() {   # name, dependency ('' or afterok:<id>), source zip (sofia abs), t2p_repack options...
  local name="$1" dep="$2" zip="$3"; shift 3
  local purpose="T2 H1 / row 82 draw $name (config repack of a built ZIP into the Nov 5 read's state, ~0.16 GPU-h)"
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="$dep")
  env "${dry[@]}" S1_PURPOSE="$purpose" s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:30:00 "${depflag[@]}" \
    --job-name="ni26-t2p-repack-h1-$name" -- "$zip" "t2p_h1_$name" "$@" 2>&1 | grep -E "SUBMITTED|DRYRUN|test-only|refus|error|ERROR" || true
}
for d in "$@"; do
  if [ "$tpl" == w10 ]; then
    sub "$d-mor-lam" "" "$R/${SRC[$d]}" --set tta.lam=0.95
  else
    out=$(sub "$d-mf" "" "$R/${SRC[$d]}" --drop-family mor --gpu-graph); echo "$out"
    if [ "$mode" == GO ]; then
      j=$(echo "$out" | sed -n 's/^SUBMITTED \([0-9]*\).*/\1/p'); [ -n "$j" ] || { echo "step 1 of $d not submitted"; exit 2; }
      sub "$d-mf-lam" "afterok:$j" "$R/ni26-t2p-repack-h1-$d-mf-$j/t2p_h1_$d-mf.zip" --set tta.lam=0.95
    else
      echo "  (GO would then submit $d-mf-lam afterok:<step-1 id> on \$R/ni26-t2p-repack-h1-$d-mf-<id>/t2p_h1_$d-mf.zip --set tta.lam=0.95)"
    fi
  fi
done

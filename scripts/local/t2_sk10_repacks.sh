#!/usr/bin/env bash
# T2 backlog row 62 (gated on T2-W7 Finished on the warm-up; the user's rule 2026-10-27): every ZIP of docs/T2_SEALED_MANIFEST.md repacked
# with the SK10 decoder, exactly as T2-D286 made T2-W7 (t2p_repack.slurm --allow-differ 0 --set tta=sinkhorn). Members stay byte-identical.
#   bash scripts/local/t2_sk10_repacks.sh <row>          # print the 15 submit commands (no connection)
#   bash scripts/local/t2_sk10_repacks.sh <row> --check  # one s1 sh: do the 15 input ZIPs still exist on sofia?
#   bash scripts/local/t2_sk10_repacks.sh <row> --go     # s1 sync ni26 + the 15 submits (only after the row is pre-registered)
# Output names: t2p_<row>_<tag>sk_r (tag = the manifest ZIP's stem without t2p_dNNN_ and _r); fetch with t2_fetch_repacks.sh.
set -uo pipefail
cd "$(dirname "$0")/../.."
row=${1:?row, e.g. d288}; mode=${2:-print}
RUNS=$S1_ROOT/runs/ni26
SET='tta={"kind":"sinkhorn","iters":10,"n0":3}'
mapfile -t paths < <(grep -E '^\| (Nov 5|L_[VN] next)' docs/T2_SEALED_MANIFEST.md | awk -F'|' '{gsub(/[ `]/, "", $9); print $9}')
[ ${#paths[@]} -eq 15 ] || { echo "manifest rows: ${#paths[@]} != 15"; exit 2; }
case $mode in
  --check)
    s1 sh "for z in $(printf '%s ' "${paths[@]#sofia_s1/}"); do [ -f $RUNS/\$z ] && echo ok \$z || echo MISSING \$z; done"
    exit ;;
  --go)
    s1 sync ni26 2>&1 | tail -2 ;;
esac
for p in "${paths[@]}"; do
  rel=${p#sofia_s1/}; z=$(basename "$rel" .zip)
  tag=${z#t2p_d[0-9][0-9][0-9]_}; tag=${tag%_r}
  name=t2p_${row}_${tag}sk_r; job=ni26-t2p-repack-$row-${tag}sk
  purpose="T2 row 62 / T2-${row^^}: SK10 repack of the sealed manifest ZIP $z (members unchanged; feeds the Nov 5-17 sealed uploads)"
  cmd=(s1 submit ni26 tracks/t2_bci/sofia/t2p_repack.slurm --time=01:00:00 --job-name="$job" -- "$RUNS/$rel" "$name" --allow-differ 0 --set "$SET")
  if [ "$mode" = --go ]; then
    S1_PURPOSE="$purpose" "${cmd[@]}" 2>&1 | grep -E "SUBMITTED|REFUSED|ERROR|COOLDOWN"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "stop: submit failed for $z"; exit 3; }
  else
    printf '%q ' "${cmd[@]}"; echo
  fi
done

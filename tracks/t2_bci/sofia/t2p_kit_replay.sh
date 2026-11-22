#!/usr/bin/env bash
# The OFFICIAL kit on a packaged T2 zip, inference-only like Codabench, from a READ-ONLY unzip, under the worker-image
# import guard (src/ni26/worker_guard): (a) Simulated 41 x 480 @ 120 Hz, (b) the PROTEUS warm-up
# (BCI[study=dreyer2026proteus], test P04 + P09). Writes <base>.kit_{sim,warmup}.{log,json}, <base>.guard_refused.txt.
#   bash t2p_kit_replay.sh <zip> <out base>        (needs S1_CODE, S1_JOBTMP, NI26_KIT, BENCHOPT_DATA_HOME; exit 0 = both ran)
set -uo pipefail
ZIP="$1"; BASE="$2"
KIT="$S1_JOBTMP/kit"; rm -rf "$KIT"; cp -r "$NI26_KIT" "$KIT"
RO="$S1_JOBTMP/pkg_ro"; chmod -R u+w "$RO" 2>/dev/null; rm -rf "$RO"; mkdir -p "$RO"
(cd "$RO" && unzip -q "$ZIP"); chmod -R a-w "$RO"
echo "### package $(sha256sum "$ZIP" | cut -c1-16) $(du -m "$ZIP" | cut -f1) MB: $(ls "$RO" | tr '\n' ' ')"
cd "$KIT/tracks/bci_decoding" || exit 3
rc=0
G="$S1_CODE/src/ni26/worker_guard"
GENV=(env "PYTHONPATH=$G${PYTHONPATH:+:$PYTHONPATH}" "NI26_WORKER_GUARD=$G/dists_442edcd.txt"
      "NI26_WORKER_GUARD_ROOTS=$RO:$KIT" "NI26_WORKER_GUARD_LOG=$BASE.guard_refused.txt")
"${GENV[@]}" python -c "import torch, braindecode, benchopt
try:
    import timm; raise SystemExit('!! guard inactive: timm imported')
except ModuleNotFoundError: print('### worker guard active: torch', torch.__version__, 'ok, timm refused')" 2>"$BASE.guard_selftest.err" \
  || { echo "!! worker guard self-test failed:"; grep -v Warning "$BASE.guard_selftest.err" | tail -25; exit 3; }
: > "$BASE.guard_refused.txt"
for spec in "sim|Simulated['n_chans, n_times'=[(41,480)],sfreq=120.0]" "warmup|BCI[study=dreyer2026proteus,subset=full]"; do
  nm=${spec%%|*}; ds=${spec#*|}
  echo "### kit run $nm: $ds ($(date +%T))"
  t0=$(date +%s)
  COMPET_SUBMISSION_DIR="$RO" "${GENV[@]}" benchopt run . -d "$ds" -s "$RO/submission.py" -r 1 --no-plot --no-html --no-display \
      --output "kit_$nm" > "$BASE.kit_$nm.log" 2>&1 || { echo "!! kit run $nm failed"; tail -30 "$BASE.kit_$nm.log"; rc=3; }
  echo "### kit run $nm took $(( $(date +%s) - t0 )) s"
  python - "$nm" "$BASE" "$(( $(date +%s) - t0 ))" <<'PY'
import glob, json, os, sys
import pandas as pd
nm, base, secs = sys.argv[1], sys.argv[2], int(sys.argv[3])
fs = sorted(glob.glob(f"outputs/kit_{nm}*.parquet"), key=os.path.getmtime)
if not fs:
    print(f"!! no parquet for {nm}"); sys.exit(0)
df = pd.read_parquet(fs[-1])
sc = {c: float(df[c].iloc[-1]) for c in df.columns if c.startswith("objective_") and df[c].dtype.kind in "fi"}
sc["wall_seconds"] = secs
print(f"### kit {nm}:", json.dumps(sc))
json.dump(sc, open(f"{base}.kit_{nm}.json", "w"), indent=1)
PY
done
echo "### package load lines (versions, load fallbacks; review T2 P1a):"; grep -h "^\[ni26\]" "$BASE".kit_*.log | sort -u | head -8
echo "### worker guard refused (optional imports degrade as on Codabench):"; sort "$BASE.guard_refused.txt" | uniq -c
chmod -R u+w "$RO"
exit $rc

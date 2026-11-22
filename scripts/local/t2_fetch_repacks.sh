#!/usr/bin/env bash
# Fetch T2 repack dirs, check each ZIP's sha256 against its .zip.sha256, check the unpacked copy file-by-file against the ZIP's CRC-32s,
# then delete the unpacked copy (memory: keep only the repacked ZIP locally). Prints name, sha256 (16), the D140-bar fields of repack.json.
#   bash scripts/local/t2_fetch_repacks.sh <repack dir name> ...
set -uo pipefail
S=$HOME/workspace/outputs/neural_interface_challenge/sofia_s1
for d in "$@"; do
  s1 fetch ni26 "$d" >/dev/null 2>&1 || { echo "$d: fetch failed"; continue; }
  z=$(ls "$S/$d"/t2p_*.zip); want=$(cut -d' ' -f1 "$z.sha256"); got=$(sha256sum "$z" | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "$d: SHA MISMATCH"; continue; }
  u="${z%.zip}"
  if [ -d "$u" ]; then
    python3 - "$z" "$u" <<'PY' && rm -rf -- "$u"
import sys, zipfile, zlib
from pathlib import Path
z, u = sys.argv[1], Path(sys.argv[2])
infos = zipfile.ZipFile(z).infolist()
assert sorted(i.filename for i in infos) == sorted(p.name for p in u.iterdir()), "file sets differ"
for i in infos:
    c = 0
    with open(u / i.filename, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            c = zlib.crc32(b, c)
    assert c == i.CRC, i.filename
PY
  fi
  python3 -c "
import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[2], sys.argv[3][:16], 'clean_differ', r['clean_differ'], 'ms', r['new_ms_per_window'], 'robust', r['all_pass'], 'kit_test', round(r['kit_test_cellba'],4), 'unpacked', 'removed' if not __import__('os').path.isdir(sys.argv[4]) else 'KEPT')" "$S/$d"/t2p_*.repack.json "$d" "$got" "$u"
done

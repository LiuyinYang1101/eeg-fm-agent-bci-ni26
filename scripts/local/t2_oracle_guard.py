"""T2-D550 (the manager ≈ 02:xx, Nov 17): a HARD GUARD against TRUE-LABEL ORACLE outputs in any T2 pool / stack / nomination reader.

t2p_bench writes an in-stream copy's outputs as a twin whose suffix is is<N><code> (T2-D509) or isS<code> (T2-D510), r<k> for a repeat; the code is
'' for the label selection 'true' (the stream's own TRUE labels, the ceiling arm), 'pa' / 'pc' / 'ps' / 'pr' for the teacher's labels. A code-less
suffix is therefore an ORACLE: never a deployable member, never an input of a decision read. is0 (the unadapted base) and isT (the base with the
fine-tune times) are not oracles. D550 run 1 found 42 such files in the pool, topping its ranking.
  oracle_reason(path, z=None) -> a reason string, or None; refuse_oracle(path, z=None) raises SystemExit on an oracle.
"""
import json
import re
from pathlib import Path

ORACLE_SUFFIX = re.compile(r"^is(S|[1-9]\d*)(r\d+)?$")
ORACLE_IN_NAME = re.compile(r"is(S|[1-9]\d*)(r\d+)?__")


def oracle_reason(path, z=None):
    """the file name, and when the npz is given (or readable) its manifest (res.twin_suffix), checked against the oracle forms."""
    name = Path(path).name
    if ORACLE_IN_NAME.search(name):
        return f"name carries a true-label in-stream suffix ({ORACLE_IN_NAME.search(name).group(0)[:-2]})"
    if z is not None and "res" in getattr(z, "files", []):
        try:
            suf = json.loads(str(z["res"])).get("twin_suffix")
        except Exception:
            suf = None
        if suf is not None and ORACLE_SUFFIX.match(str(suf)):
            return f"manifest twin_suffix {suf!r} (a true-label in-stream copy)"
    return None


def refuse_oracle(path, z=None):
    r = oracle_reason(path, z)
    if r is not None:
        raise SystemExit(f"!! T2 oracle guard: refused {path}: {r}")

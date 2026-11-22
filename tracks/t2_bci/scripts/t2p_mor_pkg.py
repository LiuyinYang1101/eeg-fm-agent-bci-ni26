#!/usr/bin/env python
"""Track 2 T2-D140: packaging helpers for the T2 submission template.

template_source(st, mor) = the template with src/ni26/fm/steeg_pt.py inlined at its marker (what t2p_final writes as
submission.py).

Public release note: this file also held the code that packaged the in-house MOR model as a member (merging its adapters
into the weights and inlining its encoder source). That model is unpublished, so those parts are not part of this release;
`mor=True` raises.
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

TEMPLATE = REPO / "tracks" / "t2_bci" / "submission" / "t2p_submission.py"
STEEG_PT = REPO / "src" / "ni26" / "fm" / "steeg_pt.py"
MOR_SRC = None                               # the in-house MOR source: not part of this public release


def _inline(src, marker_prefix, path, label):
    marker = next(ln for ln in src.splitlines() if ln.startswith(marker_prefix))
    code = path.read_text()
    return src.replace(marker, f"# ---- inlined by t2p_final.py: {label} (sha256 {hashlib.sha256(code.encode()).hexdigest()[:12]})\n"
                       + code + f"# ---- end inlined {Path(label).name}")


def template_source(st, mor):
    if mor:
        raise NotImplementedError("the in-house MOR model is not part of this public release")
    src = TEMPLATE.read_text()
    if st:
        src = _inline(src, "# @@INLINE_STEEG_PT@@", STEEG_PT, "src/ni26/fm/steeg_pt.py")
    return src


def merge(net):
    """Folded the in-house MOR member's adapters into its weights for packaging (not part of this public release)."""
    raise NotImplementedError("the in-house MOR model is not part of this public release")

# T2 gap follow-up (descriptive): is the returning sessions' decline with stream position caused by the in-stream copies (R1 = 0.5 from window 60)?
import numpy as np, pandas as pd
SRC36 = open("scripts/local/t2_d536_read.py").read()
exec(compile(SRC36.split('\nW18 = ["D524", "D529"]\n')[0], "d536_head(gappos)", "exec"))
W18 = ["D524", "D529"]; SIDE_MODE = "unc"
def bypos(A):
    out = {}
    for (f, d), (ix, mx) in A.items():
        kind = np.where(U[ix] == TRU[f], "stranger", np.where(np.isin(SES[ix], ("2", "3")), "returning", "other"))
        pos = np.zeros(len(ix), int)
        for s in np.unique(STREAM[ix]):
            j = np.flatnonzero(STREAM[ix] == s); j = j[np.argsort(ORD[ix][j], kind="stable")]; pos[j] = np.arange(len(j))
        ok = mx.argmax(1) == Y[ix]
        for kd in ("stranger", "returning"):
            for b, (lo, hi) in {"0-59": (0, 60), "60-99": (60, 100), "100-149": (100, 150), "150+": (150, 10**6)}.items():
                m = (kind == kd) & (pos >= lo) & (pos < hi)
                out.setdefault((kd, b), []).append(ok[m].mean())
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}
AW = bypos(mix(W18))
R1 = 0.0
BW = bypos(mix(W18))
for k in AW: print(k, "W18", AW[k], "noP1", BW[k], "diff", round(AW[k] - BW[k], 4))

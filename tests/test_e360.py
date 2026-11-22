"""T2-D382: the e360 lines differ from M6's 180-epoch CNN lines in name + epochs only; the per-user copies are unchanged."""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402


def test_e360():
    for b, n in (("r1_pulxx_vp2_e180_vr_rc75_tl15", "r1_pulxx_vp2_e360_vr_rc75_tl15"), ("r1_pulx_vp2_e180_vr", "r1_pulx_vp2_e360_vr")):
        rb, rn = T.RECIPES[b], T.RECIPES[n]
        assert rn.name == n and {k for k, v in asdict(rn).items() if v != asdict(rb)[k]} == {"name", "epochs"} and rn.epochs == 360 and rb.epochs == 180, n
        assert T.peruser_for(rn) == T.peruser_for(rb), f"{n}: the copies must be unchanged"


if __name__ == "__main__":
    test_e360()
    print("OK")

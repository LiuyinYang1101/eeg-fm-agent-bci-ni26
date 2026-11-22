"""T2-D388: r1_pulxx_vp2_e180_vr_rc75_tl15_pv10 = M6's tilt FBCNet line with S1's pv10 copies: the POOLED recipe differs from the tl15 line in
name + pu_vp only (pu_vp never reaches the pooled fit), and its copies = the rc75_pv10 line's copies (S1's) exactly."""
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ni26.train as T  # noqa: E402


def test_tl15_pv10():
    n, b, s1 = T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15_pv10"], T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_tl15"], T.RECIPES["r1_pulxx_vp2_e180_vr_rc75_pv10"]
    assert {k for k, v in asdict(n).items() if v != asdict(b)[k]} == {"name", "pu_vp"} and n.pu_vp == 1.0, "pooled = the tl15 line"
    assert {k for k, v in asdict(n).items() if v != asdict(s1)[k]} == {"name", "tilt"}, "= S1's pv10 line + the tilt"
    assert T.peruser_for(n) == T.peruser_for(s1) and T.peruser_for(n).vp == 1 and T.peruser_for(n).vp_a == 1.0, "copies = S1's pv10 copies"
    assert T.peruser_for(n) != T.peruser_for(b), "the copies must differ from M6's plain copies"


if __name__ == "__main__":
    test_tl15_pv10()
    print("OK")

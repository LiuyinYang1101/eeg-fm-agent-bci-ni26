"""T2-D381: t2p_time_zip.patch — no edits leave config.json byte for byte; an edit lands at its dotted key and is reported."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracks/t2_bci/scripts"))
import t2p_time_zip as TZ  # noqa: E402


def test_patch():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "config.json"
        raw = '{"cnn_tta_vp": {"k": 8, "copies": true, "file": "tv.pt"}, "fm_weight": 0.35}'
        f.write_text(raw)
        assert TZ.patch(f, []) == [] and f.read_text() == raw, "no edits must leave the file untouched"
        out = TZ.patch(f, ["cnn_tta_vp.copies=false"])
        assert out == [("cnn_tta_vp.copies", True, False)], out
        c = json.loads(f.read_text())
        assert c["cnn_tta_vp"] == {"k": 8, "copies": False, "file": "tv.pt"} and c["fm_weight"] == 0.35, c
        out = TZ.patch(f, ["cnn_tta_vp.k=4", "cnn_tta_vp.copies=true"])
        assert out == [("cnn_tta_vp.k", 8, 4), ("cnn_tta_vp.copies", False, True)], out
        assert json.loads(f.read_text())["cnn_tta_vp"]["k"] == 4


if __name__ == "__main__":
    test_patch()
    print("OK")

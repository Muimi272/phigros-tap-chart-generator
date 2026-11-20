import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen.infer import (  # noqa: E402
    _sanitize_chart_id,
    group_ranges_from_tap,
    tap_to_dataset_row,
)
from phigros_gen.schema import TYPE_IDS  # noqa: E402


def _time_sec(tick: int, bpm: float) -> float:
    return tick * 1.875 / bpm


def make_tap(bpm: float = 120.0) -> dict:
    return {
        "schema": "phigros-tap/v1",
        "meta": {
            "title": "Test Song",
            "difficulty": "IN",
            "bpm": bpm,
            "offsetSec": 0.0,
            "durationSec": 10.0,
            "tickRate": 32,
            "source": "official",
            "audio": "song.mp3",
        },
        "groups": [
            {
                "groupId": 0,
                "timeTick": 128,
                "timeSec": _time_sec(128, bpm),
                "notes": [{"type": "tap", "holdTicks": 0, "holdSec": 0.0}],
            },
            {
                "groupId": 1,
                "timeTick": 256,
                "timeSec": _time_sec(256, bpm),
                "notes": [
                    {"type": "drag", "holdTicks": 0, "holdSec": 0.0},
                    {"type": "hold", "holdTicks": 64, "holdSec": _time_sec(64, bpm)},
                ],
            },
        ],
        "numNotes": 3,
    }


def test_group_ranges_from_tap():
    assert group_ranges_from_tap(make_tap()) == [(0, 1), (1, 3)]


def test_tap_to_dataset_row_reuses_dataset_schema_fields():
    row = tap_to_dataset_row(make_tap(), "chart_id")
    assert row["chart_id"] == "chart_id"
    assert row["num_notes"] == 3
    assert row["num_groups"] == 2
    assert row["groups"][0]["notes"][0]["slot_id"] == 0
    assert row["groups"][1]["notes"][0]["slot_id"] == 0
    assert row["groups"][1]["notes"][1]["slot_id"] == 1
    assert row["groups"][1]["notes"][1]["type_id"] == TYPE_IDS["hold"]
    assert row["groups"][1]["notes"][1]["hold_sec"] == pytest.approx(_time_sec(64, 120.0))
    assert row["groups"][0]["time_sec"] == pytest.approx(_time_sec(128, 120.0))
    assert all(n["x_abs"] == 0.5 for g in row["groups"] for n in g["notes"])


def test_sanitize_chart_id():
    assert _sanitize_chart_id("A Song / 01") == "A_Song_01"
    assert _sanitize_chart_id("...") == "chart"


def test_full_inference_roundtrip_without_audio():
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    from phigros_gen.model import build_model, default_config
    from phigros_gen.infer import predict_tap_chart

    tap = make_tap()
    config = default_config(use_audio=False, n_difficulties=2, n_sources=2)
    model = build_model(config)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "fold": 0,
        "seed": 20261120,
        "epoch": 1,
        "uses_audio": False,
    }
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        predictions, ranges, meta = predict_tap_chart(
            tap, checkpoint, tmpdir, device=torch.device("cpu"), chart_id="test_chart"
        )
    assert meta["num_notes"] == 3
    assert meta["uses_audio"] is False
    assert len(predictions) == 3
    assert ranges == [(0, 1), (1, 3)]
    for prediction in predictions:
        assert 0.0 <= prediction["x_abs"] <= 1.0
        assert prediction["side"] in ("above", "below")

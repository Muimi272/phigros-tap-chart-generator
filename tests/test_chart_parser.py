import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen.chart_parser import (
    build_event_interpolator,
    decode_v1_move,
    interpolate_event,
    parse_chart,
    project_x_abs,
)


def test_v1_move_decode_packed_value():
    assert decode_v1_move(440260) == (0.5, 0.5)
    assert decode_v1_move(440520) == (0.5, 1.0)
    assert decode_v1_move(880260) == (1.0, 0.5)


def test_interpolation_and_cross_boundary():
    events = [
        {"startTime": 0, "endTime": 10, "start": 0.0, "end": 10.0},
        {"startTime": 10, "endTime": 20, "start": 10.0, "end": 20.0},
    ]
    assert interpolate_event(events, 5, "start", "end") == 5.0
    # Exact boundary resolves to the next event's start value.
    assert interpolate_event(events, 10, "start", "end") == 10.0
    assert interpolate_event(events, 25, "start", "end") == 20.0
    assert interpolate_event(events, -5, "start", "end") == 0.0


def test_zero_width_event_interpolation():
    interp = build_event_interpolator(
        [{"startTime": 3, "endTime": 3, "start": 7.0, "end": 9.0}],
        "start",
        "end",
    )
    assert interp(3) == 7.0


def test_rotation_projection_uses_cosine_degrees():
    assert project_x_abs(0.5, 1.0, 0.0) == 0.55625
    assert project_x_abs(0.5, 1.0, 90.0) == 0.5
    assert project_x_abs(0.5, 1.0, 180.0) == 0.44375


def _write_chart(tmp_path: Path, chart: dict) -> Path:
    path = tmp_path / "chart.json"
    path.write_text(json.dumps(chart), encoding="utf-8")
    return path


def test_parse_v3_interpolation(tmp_path):
    chart = {
        "formatVersion": 3,
        "offset": 0.0,
        "judgeLineList": [
            {
                "bpm": 200.0,
                "judgeLineMoveEvents": [
                    {"startTime": 0, "endTime": 1000, "start": 0.5, "end": 1.0,
                     "start2": 0.5, "end2": 0.5}
                ],
                "judgeLineRotateEvents": [
                    {"startTime": 0, "endTime": 1000, "start": 0.0, "end": 90.0}
                ],
                "notesAbove": [
                    {"type": 1, "time": 500, "positionX": 1.0, "holdTime": 0.0,
                     "speed": 1.0, "floorPosition": 1.0}
                ],
                "notesBelow": [],
            }
        ],
    }
    parsed = parse_chart(_write_chart(tmp_path, chart))
    note = parsed.notes[0]
    assert parsed.format_version == 3
    assert note.time_tick == 500
    assert note.line_x == 0.75
    assert note.theta_deg == 45.0
    expected_x = 0.75 + 1.0 * 0.05625 * math.cos(math.radians(45.0))
    assert note.x_abs == pytest.approx(expected_x)
    assert note.side == "above"


def test_parse_v1_decode_and_hold(tmp_path):
    chart = {
        "formatVersion": 1,
        "offset": 0.0,
        "judgeLineList": [
            {
                "bpm": 100.0,
                "judgeLineMoveEvents": [
                    {"startTime": 0, "endTime": 100, "start": 440260, "end": 440520}
                ],
                "judgeLineRotateEvents": [
                    {"startTime": 0, "endTime": 100, "start": 0, "end": 180}
                ],
                "notesAbove": [
                    {"type": 3, "time": 50, "positionX": 2.0, "holdTime": 32,
                     "speed": 1.0, "floorPosition": 1.0}
                ],
                "notesBelow": [],
            }
        ],
    }
    parsed = parse_chart(_write_chart(tmp_path, chart))
    note = parsed.notes[0]
    assert parsed.format_version == 1
    assert note.type == "hold"
    assert note.hold_ticks == 32
    assert note.hold_sec == pytest.approx(32 * 1.875 / 100.0)
    assert note.line_x == 0.5
    assert note.theta_deg == 90.0
    assert note.x_abs == pytest.approx(0.5)

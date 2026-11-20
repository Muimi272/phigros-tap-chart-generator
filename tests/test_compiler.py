import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen.compiler import (  # noqa: E402
    END_TIME,
    CompileError,
    compile_chart,
    compiled_chart_parseable,
    validate_compiled_chart,
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
                    {"type": "tap", "holdTicks": 0, "holdSec": 0.0},
                    {"type": "drag", "holdTicks": 0, "holdSec": 0.0},
                ],
            },
            {
                "groupId": 2,
                "timeTick": 384,
                "timeSec": _time_sec(384, bpm),
                "notes": [
                    {"type": "hold", "holdTicks": 64, "holdSec": _time_sec(64, bpm)}
                ],
            },
        ],
        "numNotes": 4,
    }


def predictions() -> list:
    return [
        {"x_abs": 0.5, "side": "above"},
        {"x_abs": 0.7, "side": "above"},
        {"x_abs": 0.3, "side": "below"},
        {"x_abs": 0.5, "side": "above"},
    ]


def test_coordinate_transform_and_floor_position():
    chart = compile_chart(make_tap(), predictions())
    line = chart["judgeLineList"][0]
    assert chart["formatVersion"] == 3
    assert chart["numOfNotes"] == 4

    above = line["notesAbove"]
    below = line["notesBelow"]
    assert len(above) == 3
    assert len(below) == 1

    # First above note (tap, x=0.5) -> positionX 0, floorPosition tick*1.875/bpm.
    assert above[0]["positionX"] == pytest.approx(0.0)
    assert above[0]["floorPosition"] == pytest.approx(128 * 1.875 / 120.0)
    # Second above note (tap, x=0.7).
    assert above[1]["positionX"] == pytest.approx((0.7 - 0.5) / 0.05625)
    # Below note (drag, x=0.3) -> negative local x.
    assert below[0]["positionX"] == pytest.approx((0.3 - 0.5) / 0.05625)
    # All speeds are 1.0.
    for note in above + below:
        assert note["speed"] == 1.0


def test_hold_time_inherited_and_non_hold_zero():
    chart = compile_chart(make_tap(), predictions())
    line = chart["judgeLineList"][0]
    hold_notes = [n for n in line["notesAbove"] + line["notesBelow"] if n["type"] == TYPE_IDS["hold"]]
    assert len(hold_notes) == 1
    assert hold_notes[0]["holdTime"] == pytest.approx(64.0)
    assert hold_notes[0]["time"] == 384
    for note in line["notesAbove"] + line["notesBelow"]:
        if note["type"] != TYPE_IDS["hold"]:
            assert note["holdTime"] == 0.0


def test_same_press_group_preserved_and_sorted_by_x():
    chart = compile_chart(make_tap(), predictions())
    line = chart["judgeLineList"][0]
    group1_notes = [
        n for n in line["notesAbove"] + line["notesBelow"] if n["time"] == 256
    ]
    assert len(group1_notes) == 2
    # Same-press group is preserved by exact tick, even across above/below.
    assert {n["time"] for n in group1_notes} == {256}
    assert [n["positionX"] for n in line["notesAbove"] if n["time"] == 256] == pytest.approx(
        [(0.7 - 0.5) / 0.05625]
    )
    assert [n["positionX"] for n in line["notesBelow"] if n["time"] == 256] == pytest.approx(
        [(0.3 - 0.5) / 0.05625]
    )


def test_same_side_group_notes_sorted_by_x():
    tap = make_tap()
    # Rewrite group 1 so both notes are on the same side.
    tap["groups"][1]["notes"] = [
        {"type": "tap", "holdTicks": 0, "holdSec": 0.0},
        {"type": "drag", "holdTicks": 0, "holdSec": 0.0},
    ]
    preds = [
        {"x_abs": 0.5, "side": "above"},
        {"x_abs": 0.7, "side": "above"},
        {"x_abs": 0.3, "side": "above"},
        {"x_abs": 0.5, "side": "above"},
    ]
    chart = compile_chart(tap, preds)
    above = chart["judgeLineList"][0]["notesAbove"]
    group1 = [n for n in above if n["time"] == 256]
    assert len(group1) == 2
    assert group1[0]["positionX"] < group1[1]["positionX"]


def test_event_endpoints_are_end_time():
    chart = compile_chart(make_tap(), predictions())
    line = chart["judgeLineList"][0]
    for events in (
        line["speedEvents"],
        line["judgeLineDisappearEvents"],
        line["judgeLineMoveEvents"],
        line["judgeLineRotateEvents"],
    ):
        assert len(events) == 1
        assert events[0]["endTime"] == pytest.approx(END_TIME)
    assert line["judgeLineMoveEvents"][0]["start"] == 0.5
    assert line["judgeLineMoveEvents"][0]["end"] == 0.5
    assert line["judgeLineRotateEvents"][0]["start"] == 0.0
    assert line["judgeLineRotateEvents"][0]["end"] == 0.0


def test_counts_are_consistent():
    chart = compile_chart(make_tap(), predictions())
    line = chart["judgeLineList"][0]
    assert line["numOfNotes"] == 4
    assert line["numOfNotesAbove"] == 3
    assert line["numOfNotesBelow"] == 1
    assert chart["numOfNotes"] == line["numOfNotes"]


def test_compiled_chart_is_schema_valid_and_parser_loadable():
    chart = compile_chart(make_tap(), predictions())
    assert validate_compiled_chart(chart) == []
    assert compiled_chart_parseable(chart) == []


def test_x_out_of_bounds_raises_without_clamping():
    preds = predictions()
    preds[0] = {"x_abs": 1.01, "side": "above"}
    with pytest.raises(CompileError, match="outside"):
        compile_chart(make_tap(), preds)


def test_prediction_count_mismatch_raises():
    with pytest.raises(CompileError, match="expected 4 predictions"):
        compile_chart(make_tap(), predictions()[:3])


def test_nonsensical_bpm_raises():
    tap = make_tap()
    tap["meta"]["bpm"] = 0.0
    with pytest.raises(CompileError, match="bpm"):
        compile_chart(tap, predictions())


def test_nonsensical_type_raises():
    tap = make_tap()
    tap["groups"][0]["notes"][0]["type"] = "bogus"
    with pytest.raises(CompileError, match="not an allowed note type"):
        compile_chart(tap, predictions())


def test_nonsensical_hold_raises():
    tap = make_tap()
    tap["groups"][0]["notes"][0]["holdTicks"] = 10
    tap["groups"][0]["notes"][0]["holdSec"] = _time_sec(10, 120.0)
    with pytest.raises(CompileError, match="holdTicks"):
        compile_chart(tap, predictions())

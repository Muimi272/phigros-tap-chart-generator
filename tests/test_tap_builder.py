import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen import schema
from phigros_gen.chart_parser import ParsedChart, ParsedNote
from phigros_gen.tap_builder import ChartMeta, build_dataset_row, build_tap, chart_duration_sec


def make_note(
    type_id: int,
    hold_ticks: int,
    source_seq: int,
    x_abs: float,
    side: str,
    time_tick: int = 0,
    bpm: float = 200.0,
) -> ParsedNote:
    from phigros_gen.schema import TYPE_NAMES, hold_sec_from_ticks

    return ParsedNote(
        note_id=source_seq,
        source_seq=source_seq,
        line_index=0,
        side=side,
        type=TYPE_NAMES[type_id],
        type_id=type_id,
        time_tick=time_tick,
        hold_ticks=hold_ticks,
        hold_sec=hold_sec_from_ticks(hold_ticks, bpm),
        position_x=0.0,
        line_x=0.5,
        theta_deg=0.0,
        x_abs=x_abs,
        outside_range=False,
    )


def make_chart(notes, bpm=200.0, format_version=3) -> ParsedChart:
    return ParsedChart(
        format_version=format_version,
        bpm=bpm,
        offset_sec=0.0,
        line_count=1,
        notes=notes,
        num_notes=len(notes),
    )


def make_meta() -> ChartMeta:
    return ChartMeta(
        chart_id="Test_Test_HD",
        song_id="Test",
        source="official",
        title="Test",
        difficulty="HD",
        chart_path="data/raw/phigros-charts-repo/Test/Test_HD.json",
        audio_path="data/raw/phigros-charts-repo/Test/Test.mp3",
        tap_path="data/processed/taps/Test_Test_HD.json",
    )


def test_slot_id_is_independent_of_x_and_side():
    notes = [
        make_note(4, 0, 5, 0.9, "below", time_tick=8),  # flick, seq 5
        make_note(1, 0, 1, 0.1, "above", time_tick=8),  # tap, seq 1
        make_note(3, 32, 3, 0.5, "above", time_tick=8),  # hold, seq 3
    ]
    chart = make_chart(notes)
    row = build_dataset_row(chart, make_meta())
    group = row["groups"][0]
    slots = [n["slot_id"] for n in group["notes"]]
    # Stable rule: (type_id, hold_ticks, source_seq)
    assert slots == [0, 1, 2]
    assert [n["type"] for n in group["notes"]] == ["tap", "hold", "flick"]
    assert group["notes"][0]["x_abs"] == pytest.approx(0.1)
    assert group["notes"][1]["hold_ticks"] == 32


def test_groups_by_exact_tick_and_num_notes():
    notes = [
        make_note(1, 0, 0, 0.2, "above", time_tick=0),
        make_note(2, 0, 1, 0.4, "below", time_tick=0),
        make_note(1, 0, 2, 0.6, "above", time_tick=16),
    ]
    chart = make_chart(notes)
    tap = build_tap(chart, make_meta())
    assert tap["numNotes"] == 3
    assert [g["timeTick"] for g in tap["groups"]] == [0, 16]
    assert len(tap["groups"][0]["notes"]) == 2
    assert len(tap["groups"][1]["notes"]) == 1
    assert tap["groups"][1]["timeSec"] == pytest.approx(16 * 1.875 / 200.0)


def test_hold_duration_and_chart_duration():
    notes = [make_note(3, 32, 0, 0.5, "above", time_tick=64, bpm=200.0)]
    chart = make_chart(notes, bpm=200.0)
    assert notes[0].hold_sec == pytest.approx(32 * 1.875 / 200.0)
    # Last hold ends at tick 96.
    assert chart_duration_sec(chart) == pytest.approx(96 * 1.875 / 200.0)


def test_tap_and_dataset_row_satisfy_validators():
    notes = [
        make_note(1, 0, 0, 0.25, "above", time_tick=0),
        make_note(3, 48, 1, 0.75, "below", time_tick=0),
        make_note(2, 0, 2, 0.5, "above", time_tick=32),
    ]
    chart = make_chart(notes)
    meta = make_meta()
    tap = build_tap(chart, meta)
    row = build_dataset_row(chart, meta)
    assert schema.validate_tap(tap) == []
    assert schema.validate_dataset_row(row) == []
    # tap JSON must not leak x_abs/side labels.
    assert all("x_abs" not in n for g in tap["groups"] for n in g["notes"])

"""Build the ``phigros-tap/v1`` tap JSON and ``dataset.jsonl`` rows."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .chart_parser import ParsedChart, ParsedNote
from .schema import SIDE_IDS, TICK_RATE, TYPE_IDS, hold_sec_from_ticks, time_sec_from_tick


@dataclass
class ChartMeta:
    chart_id: str
    song_id: str
    source: str
    title: str
    difficulty: str
    chart_path: str
    audio_path: str
    tap_path: str


def chart_duration_sec(chart: ParsedChart) -> float:
    """Duration covering at least the last note's hold end time."""

    if not chart.notes:
        return 0.0
    max_tick_end = max((n.time_tick + n.hold_ticks) for n in chart.notes)
    return time_sec_from_tick(max_tick_end, chart.bpm)


def _ordered_groups(chart: ParsedChart) -> list[tuple[int, list[ParsedNote]]]:
    """Group notes by exact ``time_tick`` in canonical order.

    The note list is already sorted by ``(time_tick, source_seq)`` from the
    parser; this function re-sorts each group by ``(type_id, hold_ticks,
    source_seq)`` and assigns the stable ``slot_id``.  Neither ``x_abs`` nor
    ``side`` participates in the slot rule.
    """

    groups: list[tuple[int, list[ParsedNote]]] = []
    for time_tick, batch in itertools.groupby(chart.notes, key=lambda n: n.time_tick):
        ordered = sorted(batch, key=lambda n: (n.type_id, n.hold_ticks, n.source_seq))
        for slot_id, note in enumerate(ordered):
            note.slot_id = slot_id
        groups.append((time_tick, ordered))
    return groups


def build_tap(chart: ParsedChart, meta: ChartMeta) -> dict:
    """Build the ``phigros-tap/v1`` document (no x_abs/side labels)."""

    groups = []
    for time_tick, notes in _ordered_groups(chart):
        groups.append(
            {
                "groupId": len(groups),
                "timeTick": time_tick,
                "timeSec": time_sec_from_tick(time_tick, chart.bpm),
                "notes": [
                    {
                        "type": note.type,
                        "holdTicks": note.hold_ticks,
                        "holdSec": note.hold_sec,
                    }
                    for note in notes
                ],
            }
        )

    return {
        "schema": "phigros-tap/v1",
        "meta": {
            "title": meta.title,
            "difficulty": meta.difficulty,
            "bpm": chart.bpm,
            "offsetSec": chart.offset_sec,
            "durationSec": chart_duration_sec(chart),
            "tickRate": TICK_RATE,
            "source": meta.source,
            "audio": meta.audio_path,
        },
        "groups": groups,
        "numNotes": chart.num_notes,
    }


def build_dataset_row(chart: ParsedChart, meta: ChartMeta) -> dict:
    """Build one ``dataset.jsonl`` row including supervised labels."""

    groups = []
    for time_tick, notes in _ordered_groups(chart):
        groups.append(
            {
                "group_id": len(groups),
                "time_tick": time_tick,
                "time_sec": time_sec_from_tick(time_tick, chart.bpm),
                "notes": [
                    {
                        "note_id": note.note_id,
                        "slot_id": note.slot_id,
                        "type": note.type,
                        "type_id": note.type_id,
                        "hold_ticks": note.hold_ticks,
                        "hold_sec": note.hold_sec,
                        "x_abs": note.x_abs,
                        "side": note.side,
                        "side_id": SIDE_IDS[note.side],
                    }
                    for note in notes
                ],
            }
        )

    return {
        "chart_id": meta.chart_id,
        "song_id": meta.song_id,
        "source": meta.source,
        "title": meta.title,
        "difficulty": meta.difficulty,
        "chart_path": meta.chart_path,
        "audio_path": meta.audio_path,
        "tap_path": meta.tap_path,
        "format_version": chart.format_version,
        "bpm": chart.bpm,
        "offset_sec": chart.offset_sec,
        "duration_sec": chart_duration_sec(chart),
        "num_groups": len(groups),
        "num_notes": chart.num_notes,
        "groups": groups,
    }

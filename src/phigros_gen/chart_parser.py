"""Parser for Phigros ``formatVersion`` 1 and 3 chart JSON.

The parser normalises v1/v3 semantic differences and computes the screen-space
absolute x coordinate at each note time using the formula frozen in
``TRAINING_PLAN.md``:

    xAbs = lineX + noteX * 0.05625 * cos(theta)

``lineX`` and ``theta`` are linearly interpolated from the judge-line move and
rotate events at the exact note tick.  Rotation events are stored in degrees.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from .schema import SIDE_IDS, TYPE_IDS, TYPE_NAMES, hold_sec_from_ticks

NOTE_X_SCALE = 0.05625


def decode_v1_move(value: int) -> tuple[float, float]:
    """Decode a v1 packed move value ``1000*x + y``.

    Per the plan: ``x = (v // 1000) / 880`` and ``y = (v % 1000) / 520``.
    """

    v = int(value)
    return (v // 1000) / 880.0, (v % 1000) / 520.0


def lerp(t0: float, v0: float, t1: float, v1: float, t: float) -> float:
    """Linear interpolation; zero-width segments return the start value."""

    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


def project_x_abs(line_x: float, note_x: float, theta_deg: float) -> float:
    """Forward projection of a note's local x to screen-normalised x."""

    return line_x + note_x * NOTE_X_SCALE * math.cos(math.radians(theta_deg))


def build_event_interpolator(events: list[dict], start_key: str, end_key: str):
    """Build a fast callable interpolating ``events`` at a single time.

    Events are expected to be contiguous (as move/rotate events are in the
    sample repository).  The returned callable uses half-open intervals
    ``[startTime, endTime)`` so an exact boundary resolves to the next event,
    matching the intended cross-boundary behaviour.
    """

    ordered = sorted(events, key=lambda e: (e["startTime"], e["endTime"]))
    starts = [float(e["startTime"]) for e in ordered]
    last = len(ordered) - 1

    def interpolate(t: float) -> float:
        idx = bisect_right(starts, t) - 1
        if idx < 0:
            return float(ordered[0][start_key])
        event = ordered[idx]
        start_time = float(event["startTime"])
        end_time = float(event["endTime"])
        if start_time == end_time and t == start_time:
            return float(event[start_key])
        if start_time <= t < end_time:
            return lerp(
                start_time,
                float(event[start_key]),
                end_time,
                float(event[end_key]),
                t,
            )
        if idx >= last or t >= end_time:
            return float(event[end_key])
        return float(event[end_key])

    return interpolate


def interpolate_event(events: list[dict], t: float, start_key: str, end_key: str) -> float:
    """One-shot convenience wrapper around :func:`build_event_interpolator`."""

    return build_event_interpolator(events, start_key, end_key)(t)


def _prepare_move_events(moves: list[dict], format_version: int) -> list[dict]:
    """Decode move events into scalar start/end components for interpolation."""

    prepared = []
    for event in moves:
        if format_version == 1:
            start_x, start_y = decode_v1_move(event["start"])
            end_x, end_y = decode_v1_move(event["end"])
        else:
            start_x, start_y = float(event["start"]), float(event["start2"])
            end_x, end_y = float(event["end"]), float(event["end2"])
        prepared.append(
            {
                "startTime": float(event["startTime"]),
                "endTime": float(event["endTime"]),
                "start_x": start_x,
                "end_x": end_x,
                "start_y": start_y,
                "end_y": end_y,
            }
        )
    return prepared


@dataclass
class ParsedNote:
    note_id: int
    source_seq: int
    line_index: int
    side: str
    type: str
    type_id: int
    time_tick: int
    hold_ticks: int
    hold_sec: float
    position_x: float
    line_x: float
    theta_deg: float
    x_abs: float
    outside_range: bool


@dataclass
class DroppedNote:
    line_index: int
    side: str
    reason: str
    raw: dict


@dataclass
class ParsedChart:
    format_version: int
    bpm: float
    offset_sec: float
    line_count: int
    notes: list[ParsedNote] = field(default_factory=list)
    dropped_notes: list[DroppedNote] = field(default_factory=list)
    num_notes: int = 0


def _is_finite_number(value) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def parse_chart(path: Path) -> ParsedChart:
    """Parse one chart JSON file into a :class:`ParsedChart`."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "judgeLineList" not in data:
        raise ValueError(f"{path} is not a chart JSON (missing judgeLineList)")

    format_version = int(data.get("formatVersion", 0))
    if format_version not in (1, 3):
        raise ValueError(f"{path} has unsupported formatVersion {format_version}")

    lines = data.get("judgeLineList", [])
    if not isinstance(lines, list) or not lines:
        raise ValueError(f"{path} has no judge lines")

    bpm = None
    for line in lines:
        candidate = line.get("bpm")
        if _is_finite_number(candidate) and float(candidate) > 0:
            bpm = float(candidate)
            break
    if bpm is None:
        raise ValueError(f"{path} has no valid bpm")

    offset_sec = float(data.get("offset", 0.0))
    if not math.isfinite(offset_sec):
        offset_sec = 0.0

    notes: list[ParsedNote] = []
    dropped: list[DroppedNote] = []
    source_seq = 0

    for line_index, line in enumerate(lines):
        move_events = _prepare_move_events(line.get("judgeLineMoveEvents", []), format_version)
        if move_events:
            x_interp = build_event_interpolator(move_events, "start_x", "end_x")
        else:
            x_interp = lambda _t: 0.5

        rotate_events = line.get("judgeLineRotateEvents", [])
        if rotate_events:
            theta_interp = build_event_interpolator(rotate_events, "start", "end")
        else:
            theta_interp = lambda _t: 0.0

        for side in ("above", "below"):
            raw_notes = line.get("notesAbove" if side == "above" else "notesBelow", [])
            for raw in raw_notes:
                note_type_id = raw.get("type")
                if note_type_id not in TYPE_NAMES:
                    dropped.append(
                        DroppedNote(line_index, side, "unknown_type", dict(raw))
                    )
                    source_seq += 1
                    continue
                if not _is_finite_number(raw.get("time")):
                    dropped.append(
                        DroppedNote(line_index, side, "invalid_time", dict(raw))
                    )
                    source_seq += 1
                    continue
                time_tick = int(round(float(raw["time"])))
                if time_tick < 0:
                    dropped.append(
                        DroppedNote(line_index, side, "negative_time", dict(raw))
                    )
                    source_seq += 1
                    continue
                if not _is_finite_number(raw.get("positionX")):
                    dropped.append(
                        DroppedNote(line_index, side, "invalid_position_x", dict(raw))
                    )
                    source_seq += 1
                    continue

                position_x = float(raw["positionX"])
                hold_ticks = 0
                if note_type_id == TYPE_IDS["hold"]:
                    if not _is_finite_number(raw.get("holdTime")):
                        dropped.append(
                            DroppedNote(line_index, side, "invalid_hold_time", dict(raw))
                        )
                        source_seq += 1
                        continue
                    hold_ticks = int(round(float(raw["holdTime"])))
                    if hold_ticks <= 0:
                        dropped.append(
                            DroppedNote(line_index, side, "zero_length_hold", dict(raw))
                        )
                        source_seq += 1
                        continue

                line_x = x_interp(time_tick)
                theta_deg = theta_interp(time_tick)
                x_abs = project_x_abs(line_x, position_x, theta_deg)
                outside_range = not (0.0 <= x_abs <= 1.0)
                notes.append(
                    ParsedNote(
                        note_id=0,
                        source_seq=source_seq,
                        line_index=line_index,
                        side=side,
                        type=TYPE_NAMES[note_type_id],
                        type_id=note_type_id,
                        time_tick=time_tick,
                        hold_ticks=hold_ticks,
                        hold_sec=hold_sec_from_ticks(hold_ticks, bpm),
                        position_x=position_x,
                        line_x=line_x,
                        theta_deg=theta_deg,
                        x_abs=x_abs,
                        outside_range=outside_range,
                    )
                )
                source_seq += 1

    # Canonical order: exact time tick, then original file order.
    notes.sort(key=lambda n: (n.time_tick, n.source_seq))
    for index, note in enumerate(notes):
        note.note_id = index

    return ParsedChart(
        format_version=format_version,
        bpm=bpm,
        offset_sec=offset_sec,
        line_count=len(lines),
        notes=notes,
        dropped_notes=dropped,
        num_notes=len(notes),
    )

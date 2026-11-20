"""Compile a ``phigros-tap/v1`` document and per-note predictions into a chart.

The compiler is intentionally independent of torch/numpy so it can be used as
a standalone CLI and validated even before the model checkpoint is available.

Output contract (``TRAINING_PLAN.md`` section 9):

* one centred, stationary judge line;
* move ``(0.5, 0.5)``, rotate ``0``, alpha ``1``, speed events constant ``1.0``;
* every generated event has ``endTime == 1e9``;
* local ``positionX = (x_abs - 0.5) / 0.05625``;
* ``time``/``holdTime`` inherit the tap tick values;
* note ``speed = 1.0`` and ``floorPosition = time_tick * 1.875 / bpm``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .chart_parser import parse_chart
from .schema import (
    SIDE_IDS,
    SIDE_NAMES,
    TYPE_IDS,
    TYPE_NAMES,
    validate_tap,
)


X_SCALE = 0.05625
BEAT_SEC = 1.875
END_TIME = 1_000_000_000.0
LINE_EVENT_START = -999999.0
SPEED_EVENT_START = 0.0
LINE_X = 0.5
LINE_Y = 0.5
LINE_ROTATION = 0.0
NOTE_SPEED = 1.0
SPEED_EVENT_VALUE = 1.0


class CompileError(ValueError):
    """Raised when tap JSON or predictions cannot be compiled."""


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _normalize_side(value: Any) -> str:
    """Accept ``"above"/"below"`` or ``0/1`` and reject everything else."""
    if isinstance(value, bool):
        raise CompileError("side must be 'above'/'below' or 0/1, not a boolean")
    if isinstance(value, str):
        if value not in SIDE_IDS:
            raise CompileError(f"unknown side {value!r}; expected 'above' or 'below'")
        return value
    if isinstance(value, (int, float)):
        side_id = int(value)
        if side_id not in SIDE_NAMES:
            raise CompileError(f"unknown side id {value!r}; expected 0 or 1")
        return SIDE_NAMES[side_id]
    raise CompileError("side must be 'above'/'below' or 0/1")


def validate_tap_document(tap: Dict[str, Any]) -> List[str]:
    """Structural and semantic validation of a tap document.

    Uses :func:`phigros_gen.schema.validate_tap` for semantic checks and, when
    available, the frozen ``TAP_JSON_SCHEMA`` for structural checks.
    """
    if not isinstance(tap, dict):
        return ["tap document must be an object"]
    errors: List[str] = []
    meta = tap.get("meta")
    bpm_ok = (
        isinstance(meta, dict)
        and _is_finite_number(meta.get("bpm"))
        and float(meta.get("bpm")) > 0.0
    )
    if not bpm_ok:
        # Guard before ``schema.validate_tap``, which derives ``timeSec`` and can
        # divide by a nonsensical bpm (for example zero).
        errors.append("tap.meta.bpm must be a positive finite number")
    else:
        errors.extend(validate_tap(tap))
    try:
        import jsonschema
    except Exception:  # pragma: no cover - optional dependency
        jsonschema = None
    if jsonschema is not None:
        from .schema import TAP_JSON_SCHEMA

        try:
            jsonschema.validate(tap, TAP_JSON_SCHEMA)
        except Exception as exc:  # jsonschema.ValidationError
            errors.append(f"jsonschema: {exc}")
    return errors


def _flatten_tap_notes(tap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten tap groups into per-note records in stable order."""
    flat: List[Dict[str, Any]] = []
    for group_index, group in enumerate(tap["groups"]):
        tick = int(group["timeTick"])
        for note_index, note in enumerate(group["notes"]):
            flat.append(
                {
                    "group_index": group_index,
                    "note_index": note_index,
                    "time_tick": tick,
                    "type": note["type"],
                    "hold_ticks": float(note["holdTicks"]),
                }
            )
    return flat


def compile_chart(
    tap: Dict[str, Any], predictions: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compile ``tap`` plus one prediction per note into a formatVersion 3 chart.

    ``predictions`` must be in flattened tap order (group by group, and inside
    each group the same order as ``group.notes``).  Each item must be a dict
    with ``x_abs`` (finite, in ``[0, 1]``) and ``side`` (``"above"``/``"below"``
    or ``0``/``1``).  Nothing is silently clamped.
    """
    errors = validate_tap_document(tap)
    if errors:
        raise CompileError("invalid tap JSON: " + "; ".join(errors))

    meta = tap["meta"]
    bpm = float(meta["bpm"])
    if not (math.isfinite(bpm) and bpm > 0.0):
        raise CompileError("tap.meta.bpm must be a positive finite number")
    offset_sec = float(meta.get("offsetSec", 0.0))
    if not math.isfinite(offset_sec):
        raise CompileError("tap.meta.offsetSec must be finite")

    flat_notes = _flatten_tap_notes(tap)
    n_notes = len(flat_notes)
    declared = int(tap.get("numNotes", -1))
    if declared != n_notes:
        raise CompileError(
            f"tap.numNotes ({declared}) does not match flattened notes ({n_notes})"
        )

    if len(predictions) != n_notes:
        raise CompileError(
            f"expected {n_notes} predictions for {n_notes} notes, got {len(predictions)}"
        )

    parsed_predictions: List[Dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise CompileError(f"prediction[{index}] must be an object")
        x_abs = prediction.get("x_abs")
        if not _is_finite_number(x_abs):
            raise CompileError(f"prediction[{index}].x_abs must be a finite number")
        x_abs = float(x_abs)
        if not (0.0 <= x_abs <= 1.0):
            raise CompileError(
                f"prediction[{index}].x_abs={x_abs} is outside [0, 1]; refusing to clamp"
            )
        side = _normalize_side(prediction.get("side"))
        parsed_predictions.append({"x_abs": x_abs, "side": side})

    notes_above: List[Dict[str, Any]] = []
    notes_below: List[Dict[str, Any]] = []
    cursor = 0

    for group in tap["groups"]:
        group_notes = group["notes"]
        tick = int(group["timeTick"])
        group_predictions = parsed_predictions[cursor : cursor + len(group_notes)]
        cursor += len(group_notes)

        # Deterministic write order: sort within a same-press group by predicted
        # x.  Each note keeps its own type, hold duration and side.
        order = sorted(
            range(len(group_predictions)),
            key=lambda i: group_predictions[i]["x_abs"],
        )

        for slot_index in order:
            note = group_notes[slot_index]
            prediction = group_predictions[slot_index]
            note_type = note["type"]
            hold_ticks = float(note["holdTicks"])
            position_x = (prediction["x_abs"] - 0.5) / X_SCALE
            floor_position = tick * BEAT_SEC / bpm
            entry = {
                "type": TYPE_IDS[note_type],
                "time": tick,
                "positionX": position_x,
                "holdTime": hold_ticks if note_type == "hold" else 0.0,
                "speed": NOTE_SPEED,
                "floorPosition": floor_position,
            }
            if prediction["side"] == "above":
                notes_above.append(entry)
            else:
                notes_below.append(entry)

    above_count = len(notes_above)
    below_count = len(notes_below)
    line = {
        "numOfNotes": n_notes,
        "numOfNotesAbove": above_count,
        "numOfNotesBelow": below_count,
        "bpm": bpm,
        "speedEvents": [
            {
                "startTime": SPEED_EVENT_START,
                "endTime": END_TIME,
                "floorPosition": 0.0,
                "value": SPEED_EVENT_VALUE,
            }
        ],
        "notesAbove": notes_above,
        "notesBelow": notes_below,
        "judgeLineDisappearEvents": [
            {
                "startTime": LINE_EVENT_START,
                "endTime": END_TIME,
                "start": 0.0,
                "end": 0.0,
                "start2": 0.0,
                "end2": 0.0,
            }
        ],
        "judgeLineMoveEvents": [
            {
                "startTime": LINE_EVENT_START,
                "endTime": END_TIME,
                "start": LINE_X,
                "end": LINE_X,
                "start2": LINE_Y,
                "end2": LINE_Y,
            }
        ],
        "judgeLineRotateEvents": [
            {
                "startTime": LINE_EVENT_START,
                "endTime": END_TIME,
                "start": LINE_ROTATION,
                "end": LINE_ROTATION,
                "start2": 0.0,
                "end2": 0.0,
            }
        ],
    }

    chart = {
        "formatVersion": 3,
        "offset": offset_sec,
        "numOfNotes": n_notes,
        "judgeLineList": [line],
    }

    post_errors = validate_compiled_chart(chart)
    if post_errors:
        raise CompileError(
            "compiled chart failed post-validation: " + "; ".join(post_errors)
        )
    return chart


CHART_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Phigros formatVersion 3 chart",
    "type": "object",
    "required": ["formatVersion", "offset", "numOfNotes", "judgeLineList"],
    "properties": {
        "formatVersion": {"const": 3},
        "offset": {"type": "number"},
        "numOfNotes": {"type": "integer", "minimum": 0},
        "judgeLineList": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": [
                    "numOfNotes",
                    "numOfNotesAbove",
                    "numOfNotesBelow",
                    "bpm",
                    "speedEvents",
                    "notesAbove",
                    "notesBelow",
                    "judgeLineDisappearEvents",
                    "judgeLineMoveEvents",
                    "judgeLineRotateEvents",
                ],
                "properties": {
                    "numOfNotes": {"type": "integer", "minimum": 0},
                    "numOfNotesAbove": {"type": "integer", "minimum": 0},
                    "numOfNotesBelow": {"type": "integer", "minimum": 0},
                    "bpm": {"type": "number", "exclusiveMinimum": 0},
                    "speedEvents": {"type": "array", "minItems": 1},
                    "notesAbove": {"type": "array"},
                    "notesBelow": {"type": "array"},
                    "judgeLineDisappearEvents": {"type": "array", "minItems": 1},
                    "judgeLineMoveEvents": {"type": "array", "minItems": 1},
                    "judgeLineRotateEvents": {"type": "array", "minItems": 1},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}


def validate_compiled_chart(chart: Dict[str, Any]) -> List[str]:
    """Semantic checks plus (when available) JSON Schema for a compiled chart."""
    errors: List[str] = []
    if not isinstance(chart, dict):
        return ["chart must be an object"]

    try:
        import jsonschema
    except Exception:  # pragma: no cover - optional dependency
        jsonschema = None
    if jsonschema is not None:
        try:
            jsonschema.validate(chart, CHART_JSON_SCHEMA)
        except Exception as exc:  # jsonschema.ValidationError
            errors.append(f"jsonschema: {exc}")

    if chart.get("formatVersion") != 3:
        errors.append("chart.formatVersion must be 3")
    if not _is_finite_number(chart.get("offset")):
        errors.append("chart.offset must be finite")

    declared_total = chart.get("numOfNotes")
    if not isinstance(declared_total, int) or isinstance(declared_total, bool) or declared_total < 0:
        errors.append("chart.numOfNotes must be a non-negative integer")
    else:
        declared_total = int(declared_total)

    lines = chart.get("judgeLineList")
    if not isinstance(lines, list) or not lines:
        errors.append("chart.judgeLineList must be a non-empty array")
        return errors

    for line_index, line in enumerate(lines):
        prefix = f"chart.judgeLineList[{line_index}]"
        if not isinstance(line, dict):
            errors.append(f"{prefix} must be an object")
            continue

        bpm = line.get("bpm")
        if not (_is_finite_number(bpm) and float(bpm) > 0.0):
            errors.append(f"{prefix}.bpm must be a positive finite number")

        notes_above = line.get("notesAbove")
        notes_below = line.get("notesBelow")
        if not isinstance(notes_above, list):
            errors.append(f"{prefix}.notesAbove must be an array")
            notes_above = []
        if not isinstance(notes_below, list):
            errors.append(f"{prefix}.notesBelow must be an array")
            notes_below = []

        for side_name, side_notes in (("notesAbove", notes_above), ("notesBelow", notes_below)):
            for note_index, note in enumerate(side_notes):
                nprefix = f"{prefix}.{side_name}[{note_index}]"
                if not isinstance(note, dict):
                    errors.append(f"{nprefix} must be an object")
                    continue
                note_type = note.get("type")
                if note_type not in TYPE_NAMES:
                    errors.append(f"{nprefix}.type must be 1..4")
                time_tick = note.get("time")
                if not (_is_finite_number(time_tick) and float(time_tick) >= 0.0):
                    errors.append(f"{nprefix}.time must be a non-negative finite number")
                if not _is_finite_number(note.get("positionX")):
                    errors.append(f"{nprefix}.positionX must be finite")
                hold_time = note.get("holdTime")
                if not (_is_finite_number(hold_time) and float(hold_time) >= 0.0):
                    errors.append(f"{nprefix}.holdTime must be a non-negative finite number")
                if note_type == TYPE_IDS["hold"] and float(hold_time) <= 0.0:
                    errors.append(f"{nprefix}.holdTime must be positive for hold notes")
                if note_type != TYPE_IDS["hold"] and float(hold_time) != 0.0:
                    errors.append(f"{nprefix}.holdTime must be 0 for non-hold notes")
                if float(note.get("speed", 0.0)) != NOTE_SPEED:
                    errors.append(f"{nprefix}.speed must equal {NOTE_SPEED}")
                if not _is_finite_number(note.get("floorPosition")):
                    errors.append(f"{nprefix}.floorPosition must be finite")
                elif _is_finite_number(time_tick) and _is_finite_number(bpm):
                    expected_floor = float(time_tick) * BEAT_SEC / float(bpm)
                    if abs(float(note["floorPosition"]) - expected_floor) > 1e-6:
                        errors.append(
                            f"{nprefix}.floorPosition must equal time * 1.875 / bpm "
                            f"(expected {expected_floor}, got {note['floorPosition']})"
                        )

        above_count = len(notes_above)
        below_count = len(notes_below)
        if not isinstance(line.get("numOfNotesAbove"), int) or line.get("numOfNotesAbove") != above_count:
            errors.append(f"{prefix}.numOfNotesAbove must equal len(notesAbove) ({above_count})")
        if not isinstance(line.get("numOfNotesBelow"), int) or line.get("numOfNotesBelow") != below_count:
            errors.append(f"{prefix}.numOfNotesBelow must equal len(notesBelow) ({below_count})")
        if not isinstance(line.get("numOfNotes"), int) or line.get("numOfNotes") != above_count + below_count:
            errors.append(
                f"{prefix}.numOfNotes must equal notesAbove + notesBelow ({above_count + below_count})"
            )

        for event_key in (
            "speedEvents",
            "judgeLineDisappearEvents",
            "judgeLineMoveEvents",
            "judgeLineRotateEvents",
        ):
            events = line.get(event_key)
            if not isinstance(events, list) or not events:
                errors.append(f"{prefix}.{event_key} must be a non-empty array")
                continue
            for event_index, event in enumerate(events):
                eprefix = f"{prefix}.{event_key}[{event_index}]"
                if not isinstance(event, dict):
                    errors.append(f"{eprefix} must be an object")
                    continue
                end_time = event.get("endTime")
                if not _is_finite_number(end_time):
                    errors.append(f"{eprefix}.endTime must be finite")
                elif abs(float(end_time) - END_TIME) > 1e-6:
                    errors.append(f"{eprefix}.endTime must equal {END_TIME}")
                if not _is_finite_number(event.get("startTime")):
                    errors.append(f"{eprefix}.startTime must be finite")

    if isinstance(declared_total, int):
        line_total = 0
        for line in lines:
            if isinstance(line, dict) and isinstance(line.get("numOfNotes"), int):
                line_total += int(line["numOfNotes"])
        if declared_total != line_total:
            errors.append(
                f"chart.numOfNotes ({declared_total}) must equal sum of line numOfNotes ({line_total})"
            )
    return errors


def compiled_chart_parseable(chart: Dict[str, Any]) -> List[str]:
    """Write ``chart`` to a temporary file and parse it with ``chart_parser``."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "compiled_chart.json"
            path.write_text(json.dumps(chart), encoding="utf-8")
            parsed = parse_chart(path)
        if parsed.num_notes != int(chart.get("numOfNotes", -1)):
            return [
                f"parser saw {parsed.num_notes} notes, chart declares {chart.get('numOfNotes')}"
            ]
        return []
    except Exception as exc:  # pragma: no cover - depends on parser edge cases
        return [f"chart_parser failed: {exc}"]


def compile_chart_with_report(
    tap: Dict[str, Any], predictions: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compile and run every post-validation step, returning a report dict."""
    chart = compile_chart(tap, predictions)
    validation_errors = validate_compiled_chart(chart)
    parse_errors = compiled_chart_parseable(chart)
    return {
        "num_notes": int(tap.get("numNotes", 0)),
        "num_notes_above": int(chart["judgeLineList"][0]["numOfNotesAbove"]),
        "num_notes_below": int(chart["judgeLineList"][0]["numOfNotesBelow"]),
        "bpm": float(tap["meta"]["bpm"]),
        "offset_sec": float(chart["offset"]),
        "valid_semantic": not validation_errors,
        "valid_parser": not parse_errors,
        "validation_errors": validation_errors,
        "parse_errors": parse_errors,
        "chart": chart,
    }


def load_tap(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list):
        raise CompileError("predictions file must contain a JSON array")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile a tap JSON and predictions into a formatVersion 3 chart."
    )
    parser.add_argument("--tap", required=True, help="path to phigros-tap/v1 JSON")
    parser.add_argument(
        "--predictions", required=True, help="path to per-note prediction JSON array"
    )
    parser.add_argument("--out", required=True, help="output chart JSON path")
    parser.add_argument(
        "--report",
        default=None,
        help="optional JSON report path (default: <out>.report.json)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tap = load_tap(args.tap)
    predictions = load_predictions(args.predictions)
    report = compile_chart_with_report(tap, predictions)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report["chart"], handle, ensure_ascii=False, indent=2)

    report_path = args.report or (out_path + ".report.json")
    os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {k: v for k, v in report.items() if k != "chart"},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Compiled {report['num_notes']} notes -> {out_path}; "
        f"valid_semantic={report['valid_semantic']}, valid_parser={report['valid_parser']}"
    )
    return 0 if (report["valid_semantic"] and report["valid_parser"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

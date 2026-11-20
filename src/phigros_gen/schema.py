"""Schema and semantic validators for the internal data contract.

The contract is frozen in ``docs/DATA_CONTRACT.md``.  This module intentionally
does not depend on ``jsonschema`` so the data pipeline can run in the minimal
environment described by ``TRAINING_PLAN.md``.  It provides the JSON Schema
dictionaries required by the plan plus pure-Python semantic validators that
cover the constraints which a structural JSON Schema cannot express (for
example, that ``time_sec`` is derived from ``time_tick`` and ``bpm``).
"""

from __future__ import annotations

import math

TICK_RATE = 32
BEAT_SEC_NUMERATOR = 1.875

TYPE_IDS = {"tap": 1, "drag": 2, "hold": 3, "flick": 4}
TYPE_NAMES = {1: "tap", 2: "drag", 3: "hold", 4: "flick"}
SIDE_IDS = {"above": 0, "below": 1}
SIDE_NAMES = {0: "above", 1: "below"}
ALLOWED_SOURCES = {"official", "special", "community"}


def time_sec_from_tick(time_tick: int, bpm: float) -> float:
    """Derive seconds from a 1/32-beat tick using the contract formula."""

    return time_tick * BEAT_SEC_NUMERATOR / bpm


def hold_sec_from_ticks(hold_ticks: int, bpm: float) -> float:
    """Derive hold duration in seconds from 1/32-beat ticks."""

    return hold_ticks * BEAT_SEC_NUMERATOR / bpm


TAP_NOTE_SCHEMA = {
    "type": "object",
    "required": ["type", "holdTicks", "holdSec"],
    "properties": {
        "type": {"enum": ["tap", "drag", "hold", "flick"]},
        "holdTicks": {"type": "number", "minimum": 0},
        "holdSec": {"type": "number", "minimum": 0},
    },
    "additionalProperties": False,
}

TAP_META_SCHEMA = {
    "type": "object",
    "required": [
        "title",
        "difficulty",
        "bpm",
        "offsetSec",
        "durationSec",
        "tickRate",
        "source",
        "audio",
    ],
    "properties": {
        "title": {"type": "string"},
        "difficulty": {"type": "string"},
        "bpm": {"type": "number", "exclusiveMinimum": 0},
        "offsetSec": {"type": "number"},
        "durationSec": {"type": "number", "minimum": 0},
        "tickRate": {"type": "integer", "minimum": 1},
        "source": {"enum": sorted(ALLOWED_SOURCES)},
        "audio": {"type": "string"},
    },
    "additionalProperties": False,
}

TAP_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "phigros-tap/v1",
    "type": "object",
    "required": ["schema", "meta", "groups", "numNotes"],
    "properties": {
        "schema": {"const": "phigros-tap/v1"},
        "meta": TAP_META_SCHEMA,
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["groupId", "timeTick", "timeSec", "notes"],
                "properties": {
                    "groupId": {"type": "integer", "minimum": 0},
                    "timeTick": {"type": "integer", "minimum": 0},
                    "timeSec": {"type": "number", "minimum": 0},
                    "notes": {"type": "array", "items": TAP_NOTE_SCHEMA},
                },
                "additionalProperties": False,
            },
        },
        "numNotes": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}


DATASET_NOTE_SCHEMA = {
    "type": "object",
    "required": [
        "note_id",
        "slot_id",
        "type",
        "type_id",
        "hold_ticks",
        "hold_sec",
        "x_abs",
        "side",
        "side_id",
    ],
    "properties": {
        "note_id": {"type": "integer", "minimum": 0},
        "slot_id": {"type": "integer", "minimum": 0},
        "type": {"enum": ["tap", "drag", "hold", "flick"]},
        "type_id": {"enum": [1, 2, 3, 4]},
        "hold_ticks": {"type": "number", "minimum": 0},
        "hold_sec": {"type": "number", "minimum": 0},
        "x_abs": {"type": "number"},
        "side": {"enum": ["above", "below"]},
        "side_id": {"enum": [0, 1]},
    },
    "additionalProperties": False,
}

DATASET_GROUP_SCHEMA = {
    "type": "object",
    "required": ["group_id", "time_tick", "time_sec", "notes"],
    "properties": {
        "group_id": {"type": "integer", "minimum": 0},
        "time_tick": {"type": "integer", "minimum": 0},
        "time_sec": {"type": "number", "minimum": 0},
        "notes": {"type": "array", "items": DATASET_NOTE_SCHEMA},
    },
    "additionalProperties": False,
}

DATASET_ROW_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "dataset row",
    "type": "object",
    "required": [
        "chart_id",
        "song_id",
        "source",
        "title",
        "difficulty",
        "chart_path",
        "audio_path",
        "tap_path",
        "format_version",
        "bpm",
        "offset_sec",
        "duration_sec",
        "num_groups",
        "num_notes",
        "groups",
    ],
    "properties": {
        "chart_id": {"type": "string"},
        "song_id": {"type": "string"},
        "source": {"enum": sorted(ALLOWED_SOURCES)},
        "title": {"type": "string"},
        "difficulty": {"type": "string"},
        "chart_path": {"type": "string"},
        "audio_path": {"type": "string"},
        "tap_path": {"type": "string"},
        "format_version": {"enum": [1, 3]},
        "bpm": {"type": "number", "exclusiveMinimum": 0},
        "offset_sec": {"type": "number"},
        "duration_sec": {"type": "number", "minimum": 0},
        "num_groups": {"type": "integer", "minimum": 0},
        "num_notes": {"type": "integer", "minimum": 0},
        "groups": {"type": "array", "items": DATASET_GROUP_SCHEMA},
    },
    "additionalProperties": False,
}


SPLITS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "splits",
    "type": "object",
    "required": ["seed", "folds"],
    "properties": {
        "seed": {"type": "integer"},
        "folds": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "fold",
                    "test_song_ids",
                    "inner_val_song_ids",
                    "train_song_ids",
                ],
                "properties": {
                    "fold": {"type": "integer", "minimum": 0},
                    "test_song_ids": {"type": "array", "items": {"type": "string"}},
                    "inner_val_song_ids": {"type": "array", "items": {"type": "string"}},
                    "train_song_ids": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def validate_tap(tap: dict) -> list[str]:
    """Semantic validation for a ``phigros-tap/v1`` document.

    Returns a list of human-readable problems; an empty list means valid.
    """

    errors: list[str] = []
    if not isinstance(tap, dict):
        return ["tap document must be an object"]
    if tap.get("schema") != "phigros-tap/v1":
        errors.append("tap.schema must equal 'phigros-tap/v1'")

    meta = tap.get("meta")
    if not isinstance(meta, dict):
        errors.append("tap.meta must be an object")
        meta = {}
    else:
        bpm = meta.get("bpm")
        if not (_is_number(bpm) and float(bpm) > 0):
            errors.append("tap.meta.bpm must be a positive number")
        if meta.get("tickRate") != TICK_RATE:
            errors.append(f"tap.meta.tickRate must equal {TICK_RATE}")
        if meta.get("source") not in ALLOWED_SOURCES:
            errors.append("tap.meta.source must be official, special, or community")
        if meta.get("durationSec") is None or (
            _is_number(meta.get("durationSec")) and float(meta["durationSec"]) < 0
        ):
            errors.append("tap.meta.durationSec must be a non-negative number")

    groups = tap.get("groups")
    if not isinstance(groups, list):
        errors.append("tap.groups must be an array")
        return errors

    prev_tick: int | None = None
    total_notes = 0
    for idx, group in enumerate(groups):
        prefix = f"tap.groups[{idx}]"
        if not isinstance(group, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if group.get("groupId") != idx:
            errors.append(f"{prefix}.groupId must equal its index ({idx})")
        tick = group.get("timeTick")
        if not (_is_number(tick) and int(tick) == tick and int(tick) >= 0):
            errors.append(f"{prefix}.timeTick must be a non-negative integer")
        else:
            tick = int(tick)
            if prev_tick is not None and tick <= prev_tick:
                errors.append(f"{prefix}.timeTick must be strictly increasing")
            prev_tick = tick
            if _is_number(meta.get("bpm")):
                expected = time_sec_from_tick(tick, float(meta["bpm"]))
                actual = group.get("timeSec")
                if not _is_number(actual) or not _is_finite(actual):
                    errors.append(f"{prefix}.timeSec must be a finite number")
                elif abs(float(actual) - expected) > 1e-6:
                    errors.append(
                        f"{prefix}.timeSec must equal timeTick * 1.875 / bpm "
                        f"(expected {expected}, got {actual})"
                    )

        notes = group.get("notes")
        if not isinstance(notes, list):
            errors.append(f"{prefix}.notes must be an array")
            continue
        for nidx, note in enumerate(notes):
            nprefix = f"{prefix}.notes[{nidx}]"
            if not isinstance(note, dict):
                errors.append(f"{nprefix} must be an object")
                continue
            ntype = note.get("type")
            if ntype not in TYPE_IDS:
                errors.append(f"{nprefix}.type is not an allowed note type")
                continue
            hold_ticks = note.get("holdTicks")
            if not (_is_number(hold_ticks) and float(hold_ticks) >= 0):
                errors.append(f"{nprefix}.holdTicks must be a non-negative number")
            elif ntype == "hold":
                if float(hold_ticks) <= 0:
                    errors.append(f"{nprefix}.holdTicks must be positive for hold notes")
            else:
                if float(hold_ticks) != 0:
                    errors.append(f"{nprefix}.holdTicks must be 0 for non-hold notes")
            if _is_number(meta.get("bpm")) and _is_number(hold_ticks):
                expected_hold = hold_sec_from_ticks(float(hold_ticks), float(meta["bpm"]))
                actual_hold = note.get("holdSec")
                if not _is_number(actual_hold) or not _is_finite(actual_hold):
                    errors.append(f"{nprefix}.holdSec must be a finite number")
                elif abs(float(actual_hold) - expected_hold) > 1e-6:
                    errors.append(
                        f"{nprefix}.holdSec must equal holdTicks * 1.875 / bpm"
                    )
        total_notes += len(notes)

    declared = tap.get("numNotes")
    if not (_is_number(declared) and int(declared) == declared and int(declared) >= 0):
        errors.append("tap.numNotes must be a non-negative integer")
    elif int(declared) != total_notes:
        errors.append(f"tap.numNotes ({int(declared)}) must equal total notes ({total_notes})")
    return errors


def validate_dataset_row(row: dict) -> list[str]:
    """Semantic validation for one line of ``dataset.jsonl``."""

    errors: list[str] = []
    if not isinstance(row, dict):
        return ["dataset row must be an object"]

    for field in ("chart_id", "song_id", "title", "difficulty", "chart_path",
                  "audio_path", "tap_path"):
        if not isinstance(row.get(field), str) or not row.get(field):
            errors.append(f"row.{field} must be a non-empty string")
    if row.get("source") not in ALLOWED_SOURCES:
        errors.append("row.source must be official, special, or community")
    if row.get("format_version") not in (1, 3):
        errors.append("row.format_version must be 1 or 3")
    bpm = row.get("bpm")
    if not (_is_number(bpm) and float(bpm) > 0):
        errors.append("row.bpm must be a positive number")
    for field in ("offset_sec", "duration_sec"):
        if not (_is_number(row.get(field)) and _is_finite(row.get(field))):
            errors.append(f"row.{field} must be a finite number")

    groups = row.get("groups")
    if not isinstance(groups, list):
        errors.append("row.groups must be an array")
        return errors

    seen_note_ids: set[int] = set()
    prev_tick: int | None = None
    total_notes = 0
    for gidx, group in enumerate(groups):
        prefix = f"row.groups[{gidx}]"
        if not isinstance(group, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if group.get("group_id") != gidx:
            errors.append(f"{prefix}.group_id must equal its index ({gidx})")
        tick = group.get("time_tick")
        if not (_is_number(tick) and int(tick) == tick and int(tick) >= 0):
            errors.append(f"{prefix}.time_tick must be a non-negative integer")
        else:
            tick = int(tick)
            if prev_tick is not None and tick <= prev_tick:
                errors.append(f"{prefix}.time_tick must be strictly increasing")
            prev_tick = tick
            expected_sec = time_sec_from_tick(tick, float(bpm)) if _is_number(bpm) else None
            actual_sec = group.get("time_sec")
            if not (_is_number(actual_sec) and _is_finite(actual_sec)):
                errors.append(f"{prefix}.time_sec must be a finite number")
            elif expected_sec is not None and abs(float(actual_sec) - expected_sec) > 1e-6:
                errors.append(f"{prefix}.time_sec must equal time_tick * 1.875 / bpm")

        notes = group.get("notes")
        if not isinstance(notes, list):
            errors.append(f"{prefix}.notes must be an array")
            continue
        for nidx, note in enumerate(notes):
            nprefix = f"{prefix}.notes[{nidx}]"
            if not isinstance(note, dict):
                errors.append(f"{nprefix} must be an object")
                continue
            if not (_is_number(note.get("note_id")) and int(note["note_id"]) >= 0):
                errors.append(f"{nprefix}.note_id must be a non-negative integer")
            else:
                nid = int(note["note_id"])
                if nid in seen_note_ids:
                    errors.append(f"{nprefix}.note_id ({nid}) is duplicated")
                seen_note_ids.add(nid)
            if not (_is_number(note.get("slot_id")) and int(note["slot_id"]) >= 0):
                errors.append(f"{nprefix}.slot_id must be a non-negative integer")
            if note.get("type") not in TYPE_IDS:
                errors.append(f"{nprefix}.type is not allowed")
            elif note.get("type_id") != TYPE_IDS[note["type"]]:
                errors.append(f"{nprefix}.type_id does not match type")
            if note.get("side") not in SIDE_IDS:
                errors.append(f"{nprefix}.side is not allowed")
            elif note.get("side_id") != SIDE_IDS[note["side"]]:
                errors.append(f"{nprefix}.side_id does not match side")
            x_abs = note.get("x_abs")
            if not (_is_number(x_abs) and _is_finite(x_abs)):
                errors.append(f"{nprefix}.x_abs must be a finite number")
            hold_ticks = note.get("hold_ticks")
            if not (_is_number(hold_ticks) and float(hold_ticks) >= 0):
                errors.append(f"{nprefix}.hold_ticks must be a non-negative number")
            if _is_number(bpm) and _is_number(hold_ticks):
                expected_hold = hold_sec_from_ticks(float(hold_ticks), float(bpm))
                actual_hold = note.get("hold_sec")
                if not (_is_number(actual_hold) and _is_finite(actual_hold)):
                    errors.append(f"{nprefix}.hold_sec must be a finite number")
                elif abs(float(actual_hold) - expected_hold) > 1e-6:
                    errors.append(f"{nprefix}.hold_sec must equal hold_ticks * 1.875 / bpm")
        total_notes += len(notes)

        slot_ids = []
        for note in notes:
            if _is_number(note.get("slot_id")):
                slot_ids.append(int(note["slot_id"]))
        if slot_ids and sorted(slot_ids) != list(range(len(slot_ids))):
            errors.append(f"{prefix}.notes.slot_id must be a contiguous 0..N-1 permutation")

    if row.get("num_notes") != total_notes:
        errors.append(f"row.num_notes must equal total notes ({total_notes})")
    if row.get("num_groups") != len(groups):
        errors.append("row.num_groups must equal len(groups)")
    return errors


def validate_splits(splits: dict) -> list[str]:
    """Semantic validation for ``splits.json``."""

    errors: list[str] = []
    if not isinstance(splits, dict):
        return ["splits must be an object"]
    folds = splits.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        errors.append("splits.folds must contain exactly 4 folds")
        return errors

    all_songs: set[str] = set()
    test_seen: set[str] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            errors.append("each fold must be an object")
            continue
        fold_id = fold.get("fold")
        if fold_id is None or not (_is_number(fold_id) and int(fold_id) >= 0):
            errors.append("fold.fold must be a non-negative integer")
        test = fold.get("test_song_ids", [])
        inner = fold.get("inner_val_song_ids", [])
        train = fold.get("train_song_ids", [])
        for name in ("test_song_ids", "inner_val_song_ids", "train_song_ids"):
            value = fold.get(name)
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                errors.append(f"fold.{name} must be an array of strings")
        test_s = set(test)
        inner_s = set(inner)
        train_s = set(train)
        if test_s & inner_s or test_s & train_s or inner_s & train_s:
            errors.append(f"fold {fold_id} test/inner_val/train sets must be disjoint")
        if not inner_s:
            errors.append(f"fold {fold_id} must have at least one inner_val song")
        if test_s & test_seen:
            errors.append(f"fold {fold_id} test songs overlap another fold's test set")
        test_seen |= test_s
        all_songs |= test_s | inner_s | train_s

    if len(test_seen) != len(all_songs):
        errors.append("every song_id must appear in exactly one outer test fold")
    return errors

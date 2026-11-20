"""Build the processed dataset from ``data/raw/phigros-charts-repo``.

Outputs:

* ``data/processed/taps/{chart_id}.json``
* ``data/processed/dataset.jsonl``
* ``data/processed/splits.json``
* ``data/interim/audit.json``
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen import schema
from phigros_gen.chart_parser import parse_chart
from phigros_gen.tap_builder import ChartMeta, build_dataset_row, build_tap

RAW = REPO / "data" / "raw" / "phigros-charts-repo"
INTERIM = REPO / "data" / "interim"
PROCESSED = REPO / "data" / "processed"
TAPS_DIR = PROCESSED / "taps"
DATASET_PATH = PROCESSED / "dataset.jsonl"
SPLITS_PATH = PROCESSED / "splits.json"
AUDIT_PATH = INTERIM / "audit.json"

SEED = 20261120
NUM_FOLDS = 4

INDEX_SKIP = {"meta.json", "line.json", "line.copy.json", "meta.blocked.json"}
SONG_MERGE = {"Introduction_Legacy": "introduction"}


def read_index(name: str) -> list[str]:
    path = RAW / name
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return [str(x) for x in value]


def load_meta(folder: str) -> dict:
    path = RAW / folder / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def normalize_chart_id(folder: str, stem: str) -> str:
    raw = f"{folder}/{stem}"
    return re.sub(r"[\s/\\\.]+", "_", raw).strip("_")


def derive_difficulty(stem: str) -> str:
    for token in ("AT_Legacy", "IN_Legacy"):
        if re.search(rf"[._]{re.escape(token)}", stem, re.IGNORECASE):
            return token
    for token in ("EZ", "HD", "IN", "AT", "SP", "Legacy"):
        if re.search(rf"[._]{token}(?=[._]|$)", stem, re.IGNORECASE):
            return token
    return "UNKNOWN"


def discover_charts(folder_source: dict[str, str]) -> list[dict]:
    charts = []
    for folder_path in sorted(RAW.iterdir()):
        if not folder_path.is_dir() or folder_path.name.startswith("."):
            continue
        folder = folder_path.name
        source = folder_source.get(folder, "community")
        song_id = SONG_MERGE.get(folder, folder)
        meta = load_meta(folder)
        title = meta.get("name") or folder
        music_file = meta.get("musicFile")

        for chart_path in sorted(folder_path.glob("*.json")):
            if chart_path.name in INDEX_SKIP:
                continue
            audio_rel = None
            audio_exists = False
            if music_file:
                audio_rel = f"data/raw/phigros-charts-repo/{folder}/{music_file}"
                audio_exists = (folder_path / music_file).exists()
            charts.append(
                {
                    "path": chart_path,
                    "folder": folder,
                    "chart_id": normalize_chart_id(folder, chart_path.stem),
                    "source": source,
                    "song_id": song_id,
                    "title": title,
                    "difficulty": derive_difficulty(chart_path.stem),
                    "audio_rel": audio_rel,
                    "audio_exists": audio_exists,
                }
            )
    return charts


def build_splits(song_ids: list[str], source_by_song: dict[str, str], seed: int) -> dict:
    rng = random.Random(seed)
    by_source: dict[str, list[str]] = defaultdict(list)
    for song_id in song_ids:
        by_source[source_by_song[song_id]].append(song_id)
    for songs in by_source.values():
        songs.sort()

    folds: list[list[str]] = [[] for _ in range(NUM_FOLDS)]
    for source in sorted(by_source, key=lambda s: -len(by_source[s])):
        songs = by_source[source]
        rng.shuffle(songs)
        for song_id in songs:
            def fold_key(fold_index: int) -> tuple[int, int, int]:
                source_count = sum(
                    1 for s in folds[fold_index] if source_by_song[s] == source
                )
                return (len(folds[fold_index]), source_count, fold_index)

            best = min(range(NUM_FOLDS), key=fold_key)
            folds[best].append(song_id)

    split_folds = []
    for fold_index in range(NUM_FOLDS):
        test_songs = sorted(folds[fold_index])
        train_pool = [s for s in song_ids if s not in test_songs]
        rng.shuffle(train_pool)
        pool_order = {s: i for i, s in enumerate(train_pool)}
        test_sources = Counter(source_by_song[s] for s in test_songs)
        inner_val = min(
            train_pool,
            key=lambda s: (test_sources.get(source_by_song[s], 0), pool_order[s]),
        )
        train_songs = sorted(s for s in train_pool if s != inner_val)
        split_folds.append(
            {
                "fold": fold_index,
                "test_song_ids": test_songs,
                "inner_val_song_ids": [inner_val],
                "train_song_ids": train_songs,
            }
        )

    return {"seed": seed, "folds": split_folds}


def count_duplicates(charts_and_notes) -> tuple[int, int, int]:
    """Count exact duplicate notes and the groups that contain them.

    A duplicate note is an identical ``(time_tick, type_id, hold_ticks,
    position_x, side)`` tuple.  ``duplicate_notes`` counts every extra copy and
    ``duplicate_groups`` counts distinct time ticks with at least one extra copy.
    Duplicates are counted within each chart, never across different charts.
    """

    duplicate_notes = 0
    duplicate_groups = 0
    total_groups = 0
    for _chart_id, notes in charts_and_notes:
        by_group: dict[int, list[tuple]] = defaultdict(list)
        for note in notes:
            key = (
                note.type_id,
                note.hold_ticks,
                note.position_x,
                note.side,
            )
            by_group[note.time_tick].append(key)
        total_groups += len(by_group)
        for keys in by_group.values():
            counts = Counter(keys)
            extra = sum(v - 1 for v in counts.values())
            if extra > 0:
                duplicate_notes += extra
                duplicate_groups += 1
    return duplicate_notes, duplicate_groups, total_groups


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    TAPS_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    folder_source: dict[str, str] = {}
    for folder in read_index("official.json"):
        folder_source[folder] = "official"
    for folder in read_index("fanmade.json"):
        folder_source[folder] = "community"
    for folder in read_index("special.json"):
        folder_source[folder] = "special"
    folder_source["sample"] = "community"

    candidates = discover_charts(folder_source)

    dataset_rows = []
    kept_charts = []
    dropped_items = []
    note_drops = []
    missing_audio = []
    validation_errors = []

    for entry in candidates:
        chart_id = entry["chart_id"]
        if not entry["audio_exists"]:
            missing_audio.append(
                {
                    "chart_id": chart_id,
                    "folder": entry["folder"],
                    "audio_path": entry["audio_rel"],
                }
            )
            dropped_items.append(
                {"chart_id": chart_id, "folder": entry["folder"], "reason": "missing_audio"}
            )
            continue

        try:
            chart = parse_chart(entry["path"])
        except Exception as exc:  # keep pipeline going and report precisely
            dropped_items.append(
                {
                    "chart_id": chart_id,
                    "folder": entry["folder"],
                    "reason": "parse_error",
                    "detail": str(exc),
                }
            )
            continue

        if chart.num_notes == 0:
            dropped_items.append(
                {"chart_id": chart_id, "folder": entry["folder"], "reason": "no_notes"}
            )
            continue
        for dropped_note in chart.dropped_notes:
            note_drops.append(
                {
                    "chart_id": chart_id,
                    "folder": entry["folder"],
                    "line_index": dropped_note.line_index,
                    "side": dropped_note.side,
                    "reason": dropped_note.reason,
                }
            )

        tap_path_rel = f"data/processed/taps/{chart_id}.json"
        meta = ChartMeta(
            chart_id=chart_id,
            song_id=entry["song_id"],
            source=entry["source"],
            title=entry["title"],
            difficulty=entry["difficulty"],
            chart_path=f"data/raw/phigros-charts-repo/{entry['folder']}/{entry['path'].name}",
            audio_path=entry["audio_rel"],
            tap_path=tap_path_rel,
        )
        tap = build_tap(chart, meta)
        row = build_dataset_row(chart, meta)
        errors = schema.validate_tap(tap) + schema.validate_dataset_row(row)
        if errors:
            validation_errors.append({"chart_id": chart_id, "errors": errors})
            dropped_items.append(
                {"chart_id": chart_id, "folder": entry["folder"], "reason": "validation_failed"}
            )
            continue

        write_json(TAPS_DIR / f"{chart_id}.json", tap)
        dataset_rows.append(row)
        kept_charts.append((chart, meta))

    dataset_rows.sort(key=lambda r: r["chart_id"])
    with DATASET_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in dataset_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    song_ids = sorted({row["song_id"] for row in dataset_rows})
    source_by_song = {}
    for row in dataset_rows:
        source_by_song[row["song_id"]] = row["source"]

    splits = build_splits(song_ids, source_by_song, SEED)
    split_errors = schema.validate_splits(splits)
    if split_errors:
        validation_errors.append({"component": "splits", "errors": split_errors})
    write_json(SPLITS_PATH, splits)

    all_notes = [note for chart, _meta in kept_charts for note in chart.notes]
    duplicate_notes, duplicate_groups, total_groups = count_duplicates(
        [(meta.chart_id, chart.notes) for chart, meta in kept_charts]
    )

    dropped_reason_counts = Counter(item["reason"] for item in dropped_items)
    note_drop_reason_counts = Counter(item["reason"] for item in note_drops)
    type_counts = Counter(note.type for note in all_notes)
    side_counts = Counter(note.side for note in all_notes)
    source_chart_counts = Counter(row["source"] for row in dataset_rows)
    source_song_counts = Counter(source_by_song.values())
    version_counts = Counter(row["format_version"] for row in dataset_rows)
    outside_notes = [note for note in all_notes if note.outside_range]
    outside_by_chart = Counter(
        meta.chart_id for chart, meta in kept_charts for note in chart.notes if note.outside_range
    )
    x_values = [note.x_abs for note in all_notes]
    group_size_counts = Counter(
        len(row["groups"][idx]["notes"]) for row in dataset_rows for idx in range(len(row["groups"]))
    )

    # Build a reliable song -> folders map from discovery metadata.
    song_id_to_folders: dict[str, set[str]] = defaultdict(set)
    for row in dataset_rows:
        folder = row["chart_path"].split("/")[3]
        song_id_to_folders[row["song_id"]].add(folder)

    x_min = min(x_values) if x_values else 0.0
    x_max = max(x_values) if x_values else 0.0

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "schema": "phigros-data-audit/v1",
        "summary": {
            "charts_total": len(candidates),
            "charts_kept": len(dataset_rows),
            "charts_dropped": len(dropped_items),
            "songs_total": len(song_ids),
            "notes_total": len(all_notes),
            "groups_total": total_groups,
            "v1_charts": version_counts.get(1, 0),
            "v3_charts": version_counts.get(3, 0),
        },
        "type_counts": dict(type_counts),
        "side_counts": dict(side_counts),
        "source_counts": {
            "charts": dict(source_chart_counts),
            "songs": dict(source_song_counts),
        },
        "group_size_counts": {str(k): v for k, v in sorted(group_size_counts.items())},
        "x_abs": {
            "min": x_min,
            "max": x_max,
            "outside_count": len(outside_notes),
            "outside_by_chart": dict(outside_by_chart),
        },
        "duplicates": {
            "definition": "exact duplicates: identical (time_tick, type_id, hold_ticks, position_x, side) within a chart",
            "duplicate_notes": duplicate_notes,
            "duplicate_groups": duplicate_groups,
        },
        "dropped": {
            "total": len(dropped_items),
            "reasons": dict(dropped_reason_counts),
            "items": dropped_items,
        },
        "dropped_notes": {
            "total": len(note_drops),
            "reasons": dict(note_drop_reason_counts),
            "items": note_drops,
        },
        "missing_audio": missing_audio,
        "song_id_map": {
            song_id: sorted(folders) for song_id, folders in sorted(song_id_to_folders.items())
        },
        "source_map": {
            folder: folder_source.get(folder, "community") for folder in sorted(folder_source)
        },
        "validation_errors": validation_errors,
        "charts": [
            {
                "chart_id": row["chart_id"],
                "song_id": row["song_id"],
                "source": row["source"],
                "difficulty": row["difficulty"],
                "format_version": row["format_version"],
                "bpm": row["bpm"],
                "num_groups": row["num_groups"],
                "num_notes": row["num_notes"],
                "duration_sec": row["duration_sec"],
            }
            for row in dataset_rows
        ],
    }
    write_json(AUDIT_PATH, audit)

    print(f"charts total/kept/dropped: {len(candidates)}/{len(dataset_rows)}/{len(dropped_items)}")
    print(f"songs: {len(song_ids)}, notes: {len(all_notes)}, groups: {total_groups}")
    print(f"v1/v3: {version_counts.get(1, 0)}/{version_counts.get(3, 0)}")
    print(f"x_abs range: [{x_min:.6f}, {x_max:.6f}], outside: {len(outside_notes)}")
    print(f"duplicates: {duplicate_notes} notes across {duplicate_groups} groups")
    print(f"missing audio: {len(missing_audio)}")
    print(f"validation errors: {len(validation_errors)}")
    for fold in splits["folds"]:
        print(
            f"fold {fold['fold']}: test={len(fold['test_song_ids'])} "
            f"inner_val={len(fold['inner_val_song_ids'])} train={len(fold['train_song_ids'])}"
        )
    return 0 if not validation_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

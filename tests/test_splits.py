import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen import schema


def _load_build_dataset():
    spec = importlib.util.spec_from_file_location(
        "build_dataset", REPO / "scripts" / "build_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_splits_pass_validation():
    splits = {
        "seed": 20261120,
        "folds": [
            {
                "fold": 0,
                "test_song_ids": ["A"],
                "inner_val_song_ids": ["B"],
                "train_song_ids": ["C", "D"],
            },
            {
                "fold": 1,
                "test_song_ids": ["B"],
                "inner_val_song_ids": ["C"],
                "train_song_ids": ["A", "D"],
            },
            {
                "fold": 2,
                "test_song_ids": ["C"],
                "inner_val_song_ids": ["D"],
                "train_song_ids": ["A", "B"],
            },
            {
                "fold": 3,
                "test_song_ids": ["D"],
                "inner_val_song_ids": ["A"],
                "train_song_ids": ["B", "C"],
            },
        ],
    }
    assert schema.validate_splits(splits) == []


def test_overlapping_test_sets_are_rejected():
    base = {
        "seed": 1,
        "folds": [
            {"fold": 0, "test_song_ids": ["A"], "inner_val_song_ids": ["B"], "train_song_ids": ["C"]},
            {"fold": 1, "test_song_ids": ["A"], "inner_val_song_ids": ["C"], "train_song_ids": ["B"]},
            {"fold": 2, "test_song_ids": ["B"], "inner_val_song_ids": ["A"], "train_song_ids": ["C"]},
            {"fold": 3, "test_song_ids": ["C"], "inner_val_song_ids": ["A"], "train_song_ids": ["B"]},
        ],
    }
    errors = schema.validate_splits(base)
    assert any("overlap" in error for error in errors)


def test_incomplete_union_is_rejected():
    splits = {
        "seed": 1,
        "folds": [
            {"fold": 0, "test_song_ids": ["A"], "inner_val_song_ids": ["B"], "train_song_ids": ["C"]},
            {"fold": 1, "test_song_ids": ["B"], "inner_val_song_ids": ["C"], "train_song_ids": ["A"]},
            {"fold": 2, "test_song_ids": ["C"], "inner_val_song_ids": ["A"], "train_song_ids": ["B"]},
            {"fold": 3, "test_song_ids": ["D"], "inner_val_song_ids": ["E"], "train_song_ids": ["F"]},
        ],
    }
    errors = schema.validate_splits(splits)
    assert errors
    assert any("outer test fold" in error for error in errors)


def test_build_splits_balance_and_disjointness():
    module = _load_build_dataset()
    song_ids = [
        "official_a", "official_b", "official_c", "official_d", "official_e",
        "special_a", "special_b", "special_c", "special_d", "special_e", "special_f",
        "community_a",
    ]
    source_by_song = {}
    for song in song_ids:
        if song.startswith("official"):
            source_by_song[song] = "official"
        elif song.startswith("special"):
            source_by_song[song] = "special"
        else:
            source_by_song[song] = "community"

    splits = module.build_splits(song_ids, source_by_song, 20261120)
    assert schema.validate_splits(splits) == []
    for fold in splits["folds"]:
        assert len(fold["test_song_ids"]) == 3
        assert len(fold["inner_val_song_ids"]) >= 1
        assert len(fold["test_song_ids"]) + len(fold["inner_val_song_ids"]) + len(
            fold["train_song_ids"]
        ) == len(song_ids)


def test_real_splits_file_if_present():
    splits_path = REPO / "data" / "processed" / "splits.json"
    if not splits_path.exists():
        pytest.skip("data/processed/splits.json has not been generated yet")
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assert schema.validate_splits(splits) == []
    test_songs = [song for fold in splits["folds"] for song in fold["test_song_ids"]]
    assert len(set(test_songs)) == len(test_songs)
    all_songs = set(test_songs)
    for fold in splits["folds"]:
        all_songs.update(fold["inner_val_song_ids"])
        all_songs.update(fold["train_song_ids"])
    assert len(all_songs) == len(set(test_songs))

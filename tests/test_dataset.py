import json
import os
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from phigros_gen.audio_features import log_mel_spectrogram, sample_window  # noqa: E402
from phigros_gen.dataset import (  # noqa: E402
    TapDataset,
    Vocab,
    collate_charts,
    x_abs_to_bin,
)


def _write_features(path, n_frames=200, n_mels=64):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        frames=np.random.randn(n_frames, n_mels).astype(np.float32),
        frame_rate=np.float32(50.0),
        audio_seconds=np.float32(n_frames / 50.0),
    )


def _make_row(chart_id, song_id, difficulty="IN", source="official", groups=None):
    if groups is None:
        groups = [
            {
                "group_id": 0,
                "time_tick": 64,
                "time_sec": 0.5,
                "notes": [
                    {
                        "note_id": 0,
                        "slot_id": 0,
                        "type": "tap",
                        "type_id": 1,
                        "hold_ticks": 0,
                        "hold_sec": 0.0,
                        "x_abs": 0.2,
                        "side": "above",
                        "side_id": 0,
                    },
                    {
                        "note_id": 1,
                        "slot_id": 1,
                        "type": "drag",
                        "type_id": 2,
                        "hold_ticks": 0,
                        "hold_sec": 0.0,
                        "x_abs": 0.8,
                        "side": "below",
                        "side_id": 1,
                    },
                ],
            },
            {
                "group_id": 1,
                "time_tick": 128,
                "time_sec": 1.0,
                "notes": [
                    {
                        "note_id": 2,
                        "slot_id": 0,
                        "type": "hold",
                        "type_id": 3,
                        "hold_ticks": 32,
                        "hold_sec": 0.3,
                        "x_abs": 0.55,
                        "side": "above",
                        "side_id": 0,
                    }
                ],
            },
        ]
    return {
        "chart_id": chart_id,
        "song_id": song_id,
        "source": source,
        "title": chart_id,
        "difficulty": difficulty,
        "chart_path": f"data/raw/{chart_id}.json",
        "audio_path": f"data/raw/{chart_id}.mp3",
        "tap_path": f"data/processed/taps/{chart_id}.json",
        "format_version": 3,
        "bpm": 180.0,
        "offset_sec": 0.0,
        "duration_sec": 10.0,
        "num_groups": len(groups),
        "num_notes": sum(len(g["notes"]) for g in groups),
        "groups": groups,
    }


def test_x_abs_to_bin_bounds():
    assert x_abs_to_bin(0.0) == 0
    assert x_abs_to_bin(1.0) == 127
    assert x_abs_to_bin(0.5) == 64
    assert x_abs_to_bin(-0.1) == 0
    assert x_abs_to_bin(1.5) == 127
    assert x_abs_to_bin(float("nan")) == 0


def test_tap_dataset_builds_features_and_targets(tmp_path):
    rows = [_make_row("A_A_IN", "A"), _make_row("B_B_HD", "B", difficulty="HD")]
    for row in rows:
        _write_features(os.path.join(tmp_path, "features", f"{row['chart_id']}.npz"))
    vocab = Vocab.from_rows(rows)
    ds = TapDataset(rows, vocab, os.path.join(tmp_path, "features"), use_audio=True)
    assert len(ds) == 2
    item = ds[0]
    assert item["scalars"].shape == (3, 7)
    assert item["type_idx"].tolist() == [0, 1, 2]
    assert item["audio"].shape == (3, 9, 64)
    assert item["x_bin"].shape == (3,)
    assert item["side"].shape == (3,)
    assert item["group_ranges"] == [(0, 2), (2, 3)]
    assert item["num_notes"] == 3
    assert item["uses_audio"] is True
    assert item["x_abs"][0] == pytest.approx(0.2)
    assert item["scalars"][0, 2] == pytest.approx(0.0)  # hold_log for tap


def test_no_audio_ablation_zeros_window(tmp_path):
    rows = [_make_row("A_A_IN", "A")]
    vocab = Vocab.from_rows(rows)
    ds = TapDataset(rows, vocab, os.path.join(tmp_path, "features"), use_audio=False)
    item = ds[0]
    assert item["uses_audio"] is False
    assert torch.all(item["audio"] == 0).item()


def test_collate_pads_variable_length(tmp_path):
    row_long = _make_row("A_A_IN", "A")
    row_short = _make_row("B_B_HD", "B", difficulty="HD")
    row_short["groups"] = row_short["groups"][:1]
    row_short["num_groups"] = 1
    row_short["num_notes"] = 2
    rows = [row_long, row_short]
    vocab = Vocab.from_rows(rows)
    ds = TapDataset(rows, vocab, os.path.join(tmp_path, "features"), use_audio=True)
    batch = collate_charts([ds[0], ds[1]])
    assert batch["scalars"].shape == (2, 3, 7)
    assert batch["mask"][0].sum() == 3
    assert batch["mask"][1].sum() == 2
    assert batch["group_ranges"][0] == [(0, 2), (2, 3)]
    assert batch["group_ranges"][1] == [(0, 2)]


def test_log_mel_shape_and_window_sampling():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(16000).astype(np.float32)
    frames = log_mel_spectrogram(audio)
    assert frames.ndim == 2
    assert frames.shape[1] == 64
    assert frames.shape[0] > 0
    win = sample_window(frames, frame_rate=50.0, time_sec=1.0, n_frames=9)
    assert win.shape == (9, 64)
    edge = sample_window(frames, frame_rate=50.0, time_sec=0.0, n_frames=9)
    assert edge.shape == (9, 64)

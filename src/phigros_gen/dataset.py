"""Dataset and batching for PhiTapMicro.

Reads ``data/processed/dataset.jsonl`` (plus optional precomputed audio
features) and turns each chart into a variable-length sequence of per-note
feature vectors and supervision targets.  Groups stay contiguous so the
Hungarian loss can operate on slot sets with a shared assignment.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from phigros_gen.audio_features import (
    FRAME_RATE,
    WINDOW_FRAMES,
    read_features_npz,
    sample_window,
)


N_BINS = 128
NUM_TYPES = 4
NUM_SCALARS = 7
SCALAR_KEYS = (
    "dt_log1p",
    "abs_time_scaled",
    "hold_log_scaled",
    "group_size_scaled",
    "slot_rank_scaled",
    "density_log_scaled",
    "bpm_scaled",
)

# Fixed input scaling constants (no label statistics; no cross-fold leakage).
ABS_TIME_SCALE = 120.0
HOLD_LOG_SCALE = 3.0
GROUP_SIZE_SCALE = 8.0
DENSITY_LOG_SCALE = 3.0
BPM_SCALE = 200.0


def load_jsonl(path: str) -> "List[Dict[str, Any]]":
    rows: "List[Dict[str, Any]]" = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_rows(
    rows: Sequence[Dict[str, Any]],
    song_ids: Optional[Sequence[str]] = None,
    chart_ids: Optional[Sequence[str]] = None,
) -> "List[Dict[str, Any]]":
    song_set = None if song_ids is None else set(song_ids)
    chart_set = None if chart_ids is None else set(chart_ids)
    out: "List[Dict[str, Any]]" = []
    for row in rows:
        if song_set is not None and row.get("song_id") not in song_set:
            continue
        if chart_set is not None and row.get("chart_id") not in chart_set:
            continue
        out.append(row)
    return out


def split_rows(
    rows: Sequence[Dict[str, Any]],
    test_song_ids: Sequence[str],
    inner_val_song_ids: Sequence[str],
    train_song_ids: Sequence[str],
) -> "Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]":
    """Return ``(train_rows, val_rows, test_rows)`` by ``song_id`` membership."""
    test_set = set(test_song_ids)
    val_set = set(inner_val_song_ids)
    train_set = set(train_song_ids)
    train_rows: "List[Dict[str, Any]]" = []
    val_rows: "List[Dict[str, Any]]" = []
    test_rows: "List[Dict[str, Any]]" = []
    for row in rows:
        sid = row.get("song_id")
        if sid in test_set:
            test_rows.append(row)
        elif sid in val_set:
            val_rows.append(row)
        elif sid in train_set:
            train_rows.append(row)
    return train_rows, val_rows, test_rows


class Vocab:
    """Categorical vocabularies for difficulty and chart source."""

    def __init__(self, difficulties: Sequence[str], sources: Sequence[str]):
        self.difficulties = sorted(set(difficulties))
        self.sources = sorted(set(sources))
        self.difficulty_to_idx = {d: i for i, d in enumerate(self.difficulties)}
        self.source_to_idx = {s: i for i, s in enumerate(self.sources)}

    @property
    def num_difficulties(self) -> int:
        return len(self.difficulties)

    @property
    def num_sources(self) -> int:
        return len(self.sources)

    def difficulty_index(self, difficulty: str) -> int:
        return self.difficulty_to_idx.get(difficulty, 0)

    def source_index(self, source: str) -> int:
        return self.source_to_idx.get(source, 0)

    @classmethod
    def from_rows(cls, rows: Sequence[Dict[str, Any]]) -> "Vocab":
        difficulties = [str(r.get("difficulty", "unknown")) for r in rows]
        sources = [str(r.get("source", "community")) for r in rows]
        return cls(difficulties, sources)

    def to_config(self) -> Dict[str, Any]:
        return {
            "difficulty_vocab": self.difficulties,
            "source_vocab": self.sources,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Vocab":
        return cls(
            difficulties=config.get("difficulty_vocab", ["unknown"]),
            sources=config.get("source_vocab", ["community"]),
        )


def x_abs_to_bin(x_abs: float, n_bins: int = N_BINS) -> int:
    if not math.isfinite(float(x_abs)):
        return 0
    return int(min(max(math.floor(float(x_abs) * n_bins), 0), n_bins - 1))


class TapDataset(Dataset):
    """Variable-length chart dataset.

    Each item is a single chart (not a single note), which keeps group
    boundaries available for the Hungarian loss.  The collate function pads
    note sequences within a batch.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        vocab: Vocab,
        features_dir: str,
        use_audio: bool = True,
        n_bins: int = N_BINS,
    ):
        self.rows = list(rows)
        self.vocab = vocab
        self.features_dir = features_dir
        self.use_audio = use_audio
        self.n_bins = n_bins
        self.samples: "List[Dict[str, Any]]" = [
            self._build_sample(row) for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frames(self, chart_id: str) -> Optional[np.ndarray]:
        if not self.use_audio:
            return None
        path = os.path.join(self.features_dir, f"{chart_id}.npz")
        if not os.path.isfile(path):
            return None
        try:
            return read_features_npz(path)["frames"]
        except Exception:
            return None

    def _build_sample(self, row: Dict[str, Any]) -> Dict[str, Any]:
        chart_id = str(row["chart_id"])
        frames = self._load_frames(chart_id)
        has_audio = frames is not None and frames.shape[0] > 0
        if frames is None:
            frames = np.zeros((0, 64), dtype=np.float32)
        if frames.shape[1] != 64:
            # Contract guarantees 64; defensive re-check keeps callers safe.
            frames = np.zeros((frames.shape[0], 64), dtype=np.float32)

        diff_idx = self.vocab.difficulty_index(str(row.get("difficulty", "unknown")))
        src_idx = self.vocab.source_index(str(row.get("source", "community")))
        bpm = float(row.get("bpm", 120.0))

        note_times: "List[float]" = []
        note_rows: "List[Tuple[Dict[str, Any], int]]" = []
        group_ranges: "List[Tuple[int, int]]" = []
        group_sizes: "List[int]" = []

        cursor = 0
        for group in row.get("groups", []):
            notes = sorted(group.get("notes", []), key=lambda n: int(n.get("slot_id", 0)))
            start = cursor
            for note in notes:
                note_rows.append((note, diff_idx, src_idx))
                note_times.append(float(note.get("time_sec", group.get("time_sec", 0.0))))
                cursor += 1
            end = cursor
            if end > start:
                group_ranges.append((start, end))
                group_sizes.append(end - start)

        n = len(note_rows)
        times = np.asarray(note_times, dtype=np.float32) if n else np.zeros(0, dtype=np.float32)

        scalars = np.zeros((n, NUM_SCALARS), dtype=np.float32)
        type_idx = np.zeros((n,), dtype=np.int64)
        diff_idx_arr = np.full((n,), diff_idx, dtype=np.int64)
        src_idx_arr = np.full((n,), src_idx, dtype=np.int64)
        x_bin = np.zeros((n,), dtype=np.int64)
        side = np.zeros((n,), dtype=np.int64)
        x_abs = np.zeros((n,), dtype=np.float32)
        audio = np.zeros((n, WINDOW_FRAMES, 64), dtype=np.float32)
        slot_ids = np.zeros((n,), dtype=np.int64)
        group_ids = np.zeros((n,), dtype=np.int64)
        u_slot = np.zeros((n,), dtype=np.float32)

        # Local density: notes per second inside a +/-1 second window.
        densities = np.zeros((n,), dtype=np.float32)
        if n:
            for i, t in enumerate(times):
                lo, hi = t - 1.0, t + 1.0
                densities[i] = float(np.sum((times >= lo) & (times <= hi))) / 2.0

        prev_time = 0.0
        for i, (note, d_idx, s_idx) in enumerate(note_rows):
            t = times[i]
            dt = 0.0 if i == 0 else float(t - prev_time)
            hold_sec = float(note.get("hold_sec", 0.0))
            group_id_local = self._group_index_for(group_ranges, i)
            gsize = float(group_sizes[group_id_local])
            slot_id = float(int(note.get("slot_id", 0)))
            scalars[i] = (
                math.log1p(max(dt, 0.0)),
                t / ABS_TIME_SCALE,
                math.log1p(max(hold_sec, 0.0)) / HOLD_LOG_SCALE,
                gsize / GROUP_SIZE_SCALE,
                slot_id / max(1.0, gsize - 1.0),
                math.log1p(max(densities[i], 0.0)) / DENSITY_LOG_SCALE,
                bpm / BPM_SCALE,
            )
            type_idx[i] = max(0, min(int(note.get("type_id", 1)) - 1, NUM_TYPES - 1))
            diff_idx_arr[i] = d_idx
            src_idx_arr[i] = s_idx
            slot_ids[i] = int(note.get("slot_id", 0))
            group_ids[i] = group_id_local
            u_slot[i] = (slot_id + 1.0) / (gsize + 1.0) - 0.5
            xv = float(note.get("x_abs", 0.0))
            x_abs[i] = xv
            x_bin[i] = x_abs_to_bin(xv, self.n_bins)
            side[i] = int(note.get("side_id", 0)) % 2
            if has_audio:
                audio[i] = sample_window(
                    frames, FRAME_RATE, t, WINDOW_FRAMES
                ).astype(np.float32)
            prev_time = t

        return {
            "chart_id": chart_id,
            "song_id": str(row.get("song_id", chart_id)),
            "scalars": scalars,
            "type_idx": type_idx,
            "diff_idx": diff_idx_arr,
            "src_idx": src_idx_arr,
            "audio": audio,
            "times": times,
            "dts": self._compute_dts(times),
            "x_bin": x_bin,
            "side": side,
            "x_abs": x_abs,
            "slot_ids": slot_ids,
            "group_ids": group_ids,
            "u_slot": u_slot,
            "group_ranges": group_ranges,
            "group_sizes": np.asarray(group_sizes, dtype=np.int64),
            "num_notes": n,
            "uses_audio": bool(has_audio),
        }

    @staticmethod
    def _group_index_for(group_ranges: Sequence[Tuple[int, int]], note_idx: int) -> int:
        for gi, (start, end) in enumerate(group_ranges):
            if start <= note_idx < end:
                return gi
        return 0

    @staticmethod
    def _compute_dts(times: np.ndarray) -> np.ndarray:
        if times.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        dts = np.zeros_like(times)
        dts[1:] = times[1:] - times[:-1]
        return dts

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        return {
            "chart_id": sample["chart_id"],
            "song_id": sample["song_id"],
            "scalars": torch.as_tensor(sample["scalars"], dtype=torch.float32),
            "type_idx": torch.as_tensor(sample["type_idx"], dtype=torch.long),
            "diff_idx": torch.as_tensor(sample["diff_idx"], dtype=torch.long),
            "src_idx": torch.as_tensor(sample["src_idx"], dtype=torch.long),
            "audio": torch.as_tensor(sample["audio"], dtype=torch.float32),
            "times": torch.as_tensor(sample["times"], dtype=torch.float32),
            "dts": torch.as_tensor(sample["dts"], dtype=torch.float32),
            "x_bin": torch.as_tensor(sample["x_bin"], dtype=torch.long),
            "side": torch.as_tensor(sample["side"], dtype=torch.long),
            "x_abs": torch.as_tensor(sample["x_abs"], dtype=torch.float32),
            "slot_ids": torch.as_tensor(sample["slot_ids"], dtype=torch.long),
            "group_ids": torch.as_tensor(sample["group_ids"], dtype=torch.long),
            "u_slot": torch.as_tensor(sample["u_slot"], dtype=torch.float32),
            "group_ranges": list(sample["group_ranges"]),
            "group_sizes": torch.as_tensor(sample["group_sizes"], dtype=torch.long),
            "num_notes": int(sample["num_notes"]),
            "uses_audio": bool(sample["uses_audio"]),
        }

def collate_charts(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch of variable-length charts along the note axis."""
    max_notes = max(int(item["num_notes"]) for item in batch)
    n_mels = 64
    n_frames = WINDOW_FRAMES

    def pad_2d(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
        n = tensor.shape[0]
        if n == max_notes:
            return tensor
        out = torch.zeros((max_notes, tensor.shape[1]), dtype=tensor.dtype)
        out[:n] = tensor
        return out

    def pad_1d(tensor: torch.Tensor, value: int = 0) -> torch.Tensor:
        n = tensor.shape[0]
        if n == max_notes:
            return tensor
        out = torch.zeros((max_notes,), dtype=tensor.dtype)
        out[:n] = tensor
        return out

    def pad_audio(tensor: torch.Tensor) -> torch.Tensor:
        n = tensor.shape[0]
        if n == max_notes:
            return tensor
        out = torch.zeros((max_notes, n_frames, n_mels), dtype=tensor.dtype)
        out[:n] = tensor
        return out

    scalars = torch.stack([pad_2d(item["scalars"]) for item in batch])
    type_idx = torch.stack([pad_1d(item["type_idx"]) for item in batch])
    diff_idx = torch.stack([pad_1d(item["diff_idx"]) for item in batch])
    src_idx = torch.stack([pad_1d(item["src_idx"]) for item in batch])
    audio = torch.stack([pad_audio(item["audio"]) for item in batch])
    times = torch.stack([pad_1d(item["times"]) for item in batch])
    dts = torch.stack([pad_1d(item["dts"]) for item in batch])
    x_bin = torch.stack([pad_1d(item["x_bin"]) for item in batch])
    side = torch.stack([pad_1d(item["side"]) for item in batch])
    x_abs = torch.stack([pad_1d(item["x_abs"]) for item in batch])
    group_ids = torch.stack([pad_1d(item["group_ids"]) for item in batch])
    u_slot = torch.stack([pad_1d(item["u_slot"]) for item in batch])
    mask = torch.zeros((len(batch), max_notes), dtype=torch.bool)
    for i, item in enumerate(batch):
        mask[i, : int(item["num_notes"])] = True

    return {
        "chart_id": [item["chart_id"] for item in batch],
        "song_id": [item["song_id"] for item in batch],
        "scalars": scalars,
        "type_idx": type_idx,
        "diff_idx": diff_idx,
        "src_idx": src_idx,
        "audio": audio,
        "times": times,
        "dts": dts,
        "x_bin": x_bin,
        "side": side,
        "x_abs": x_abs,
        "group_ids": group_ids,
        "u_slot": u_slot,
        "mask": mask,
        "group_ranges": [item["group_ranges"] for item in batch],
        "num_notes": torch.as_tensor(
            [item["num_notes"] for item in batch], dtype=torch.long
        ),
        "uses_audio": torch.as_tensor(
            [item["uses_audio"] for item in batch], dtype=torch.bool
        ),
    }


class ChartBatchSampler(Sampler):
    """Greedy chart batching that respects a max-note token budget."""

    def __init__(
        self,
        dataset: TapDataset,
        batch_size: int,
        max_tokens: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterable[List[int]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        batch: List[int] = []
        batch_tokens = 0
        for idx in indices:
            n_notes = int(self.dataset.samples[idx]["num_notes"])
            would_exceed = (
                self.max_tokens is not None
                and batch_tokens + n_notes > self.max_tokens
                and batch
            )
            if (len(batch) >= self.batch_size) or would_exceed:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += n_notes
        if batch:
            yield batch

    def __len__(self) -> int:
        # Upper bound; the exact count depends on token packing.
        return max(1, math.ceil(len(self.dataset) / max(1, self.batch_size)))


def make_dataloader(
    dataset: TapDataset,
    batch_size: int,
    max_tokens: Optional[int] = 8192,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    sampler = ChartBatchSampler(
        dataset,
        batch_size=batch_size,
        max_tokens=max_tokens,
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_charts,
        num_workers=num_workers,
    )

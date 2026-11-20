"""Checkpoint loading and tap-chart inference for PhiTapMicro.

Inference reuses the exact feature construction path from
:class:`phigros_gen.dataset.TapDataset` and the model interface from
:mod:`phigros_gen.model`, so there is no second, drift-prone feature
implementation.  Torch is imported lazily so the pure helpers in this module
(for example :func:`tap_to_dataset_row`) remain importable before torch is
installed.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import SIDE_NAMES, TYPE_IDS


def load_tap(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_predictions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list):
        raise ValueError("predictions file must contain a JSON array")
    return data


def group_ranges_from_tap(tap: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return contiguous note ranges for each tap group."""
    ranges: List[Tuple[int, int]] = []
    cursor = 0
    for group in tap["groups"]:
        start = cursor
        cursor += len(group["notes"])
        ranges.append((start, cursor))
    return ranges


def _sanitize_chart_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "chart"


def tap_to_dataset_row(tap: Dict[str, Any], chart_id: str) -> Dict[str, Any]:
    """Convert a tap document into a synthetic dataset row.

    Labels (``x_abs``/``side``) are dummy values; the row only exists so that
    :class:`phigros_gen.dataset.TapDataset` can build the *identical* feature
    tensors used during training.  Slot ids are assigned from the original
    note enumeration order, which is stable and independent of ``x_abs``.
    """
    meta = tap["meta"]
    groups: List[Dict[str, Any]] = []
    note_id = 0
    for group in tap["groups"]:
        notes: List[Dict[str, Any]] = []
        for slot_id, note in enumerate(group["notes"]):
            notes.append(
                {
                    "note_id": note_id,
                    "slot_id": slot_id,
                    "type": note["type"],
                    "type_id": TYPE_IDS[note["type"]],
                    "hold_ticks": note["holdTicks"],
                    "hold_sec": note["holdSec"],
                    "x_abs": 0.5,
                    "side": "above",
                    "side_id": 0,
                    "time_sec": group["timeSec"],
                }
            )
            note_id += 1
        groups.append(
            {
                "group_id": len(groups),
                "time_tick": group["timeTick"],
                "time_sec": group["timeSec"],
                "notes": notes,
            }
        )
    return {
        "chart_id": chart_id,
        "song_id": chart_id,
        "source": meta.get("source", "community"),
        "title": meta.get("title", chart_id),
        "difficulty": meta.get("difficulty", "unknown"),
        "chart_path": "",
        "audio_path": meta.get("audio", ""),
        "tap_path": "",
        "format_version": 3,
        "bpm": meta["bpm"],
        "offset_sec": meta.get("offsetSec", 0.0),
        "duration_sec": meta.get("durationSec", 0.0),
        "num_groups": len(groups),
        "num_notes": note_id,
        "groups": groups,
    }


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location=map_location)
    for key in ("model_state", "config"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint {path!r} is missing {key!r}")
    return checkpoint


def build_model_from_checkpoint(checkpoint: Dict[str, Any]):
    from phigros_gen.model import build_model

    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def is_smoke_checkpoint(checkpoint: Dict[str, Any]) -> bool:
    """True for smoke checkpoints (frozen config epochs=20 and patience=8)."""
    config = checkpoint.get("config", {})
    return int(config.get("epochs", 200)) == 20 and int(config.get("patience", 30)) == 8


def pick_device(prefer: Optional[str] = None):
    import torch

    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_audio_frames(
    chart_id: str,
    tap: Dict[str, Any],
    features_dir: str,
    repo_root: str = ".",
) -> str:
    """Reuse ``{chart_id}.npz`` when present, otherwise extract it from audio."""
    path = os.path.join(features_dir, f"{chart_id}.npz")
    if os.path.isfile(path):
        return path

    from phigros_gen.audio_features import FRAME_RATE, extract_features, write_features_npz

    audio_rel = tap.get("meta", {}).get("audio", "")
    audio_path = audio_rel if os.path.isabs(audio_rel) else os.path.join(repo_root, audio_rel)
    if not audio_rel or not os.path.isfile(audio_path):
        raise FileNotFoundError(
            f"audio file not found for chart {chart_id!r}: {audio_path!r}"
        )
    frames, audio_seconds = extract_features(audio_path)
    os.makedirs(features_dir, exist_ok=True)
    write_features_npz(path, frames, FRAME_RATE, audio_seconds)
    return path


def predict_tap_chart(
    tap: Dict[str, Any],
    checkpoint: Dict[str, Any],
    features_dir: str,
    device=None,
    chart_id: Optional[str] = None,
    repo_root: str = ".",
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int]], Dict[str, Any]]:
    """Run the checkpoint on a tap document and return per-note predictions.

    Returns ``(predictions, group_ranges, meta)`` where ``predictions`` follows
    the tap's flattened note order and each entry is ``{"x_abs", "side"}``.
    """
    from phigros_gen.dataset import TapDataset, Vocab, collate_charts
    from phigros_gen.model import infer_chart

    config = checkpoint["config"]
    uses_audio = bool(checkpoint.get("uses_audio", config.get("use_audio", True)))
    if chart_id is None:
        chart_id = _sanitize_chart_id(tap.get("meta", {}).get("title", "chart"))

    if uses_audio:
        ensure_audio_frames(chart_id, tap, features_dir, repo_root)

    vocab = Vocab.from_config(config)
    row = tap_to_dataset_row(tap, chart_id)
    dataset = TapDataset([row], vocab, features_dir, use_audio=uses_audio)
    sample = dataset[0]
    batch = collate_charts([sample])

    model = build_model_from_checkpoint(checkpoint)
    if device is None:
        device = pick_device()
    model.to(device)

    out = infer_chart(model, batch, device)
    num_notes = int(sample["num_notes"])
    x_abs = out["x_abs"][0, :num_notes].tolist()
    side_id = out["side_id"][0, :num_notes].tolist()
    predictions = [
        {"x_abs": float(x), "side": SIDE_NAMES[int(s)]}
        for x, s in zip(x_abs, side_id)
    ]
    meta = {
        "chart_id": chart_id,
        "uses_audio": bool(sample["uses_audio"]),
        "num_notes": num_notes,
    }
    return predictions, group_ranges_from_tap(tap), meta


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run PhiTapMicro inference from a tap JSON."
    )
    parser.add_argument("--tap", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output chart JSON path")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="optional prediction JSON path (default: <out>.predictions.json)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .compiler import compile_chart_with_report

    args = build_parser().parse_args(argv)
    tap = load_tap(args.tap)
    checkpoint = load_checkpoint(args.checkpoint)
    if is_smoke_checkpoint(checkpoint):
        raise ValueError(
            "refusing to generate held-out artifacts from a smoke checkpoint; "
            "use a real fold checkpoint (epochs=200, patience=30)"
        )
    predictions, _, meta = predict_tap_chart(
        tap,
        checkpoint,
        args.features_dir,
        device=args.device,
        chart_id=_sanitize_chart_id(os.path.splitext(os.path.basename(args.tap))[0]),
        repo_root=args.repo_root,
    )

    predictions_out = args.predictions_out or (args.out + ".predictions.json")
    os.makedirs(os.path.dirname(os.path.abspath(predictions_out)) or ".", exist_ok=True)
    with open(predictions_out, "w", encoding="utf-8") as handle:
        json.dump(
            {"chart_id": meta["chart_id"], "uses_audio": meta["uses_audio"], "predictions": predictions},
            handle,
            ensure_ascii=False,
            indent=2,
        )

    report = compile_chart_with_report(tap, predictions)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report["chart"], handle, ensure_ascii=False, indent=2)

    print(
        f"Inferred {meta['num_notes']} notes (audio={meta['uses_audio']}) -> {args.out}; "
        f"valid_semantic={report['valid_semantic']}, valid_parser={report['valid_parser']}"
    )
    return 0 if (report["valid_semantic"] and report["valid_parser"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

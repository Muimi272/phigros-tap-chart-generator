"""Independent outer-test evaluation for PhiTapMicro fold checkpoints.

The module reads ``data/processed/dataset.jsonl`` and ``data/processed/splits.json``
and evaluates each fold checkpoint strictly on its held-out ``test_song_ids``.
It never uses outer-test metrics for early stopping or hyperparameter selection.

Torch/numpy/scipy are imported lazily so the pure metric helpers (JS distance,
bootstrap, unexpected-duplicate rate) can be tested in isolation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FROZEN_SPLITS_SHA256 = "772fd2eac22a32581191f5da366bce178f5319354c05fb674cc0b0c3db2e7aa0"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits_file(
    splits: Dict[str, Any], rows: Sequence[Dict[str, Any]], splits_path: str
) -> Dict[str, Any]:
    """Structural + data-derived validation of ``splits.json`` before evaluation."""
    from phigros_gen.schema import validate_splits

    semantic_errors = validate_splits(splits)
    splits_sha = file_sha256(splits_path)
    available_song_ids = {row["song_id"] for row in rows}
    per_fold: List[Dict[str, Any]] = []
    for fold in splits.get("folds", []):
        test_songs = list(fold.get("test_song_ids", []))
        train_songs = list(fold.get("train_song_ids", []))
        val_songs = list(fold.get("inner_val_song_ids", []))
        unknown_test_songs = [s for s in test_songs if s not in available_song_ids]
        per_fold.append(
            {
                "fold": fold.get("fold"),
                "test_song_ids": test_songs,
                "test_charts": len(select_rows(rows, song_ids=test_songs)),
                "train_charts": len(select_rows(rows, song_ids=train_songs)),
                "inner_val_charts": len(select_rows(rows, song_ids=val_songs)),
                "test_song_ids_nonempty": bool(test_songs),
                "unknown_test_song_ids": unknown_test_songs,
                "ok": bool(test_songs) and not unknown_test_songs,
            }
        )
    return {
        "splits_path": splits_path,
        "splits_sha256": splits_sha,
        "splits_sha256_ok": splits_sha == FROZEN_SPLITS_SHA256,
        "semantic_errors": semantic_errors,
        "per_fold": per_fold,
        "all_folds_ok": all(p["ok"] for p in per_fold),
    }


def select_rows(
    rows: Sequence[Dict[str, Any]],
    song_ids: Optional[Sequence[str]] = None,
    chart_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    song_set = None if song_ids is None else set(song_ids)
    chart_set = None if chart_ids is None else set(chart_ids)
    out: List[Dict[str, Any]] = []
    for row in rows:
        if song_set is not None and row.get("song_id") not in song_set:
            continue
        if chart_set is not None and row.get("chart_id") not in chart_set:
            continue
        out.append(row)
    return out


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def js_divergence(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_bins: int = 128,
    low: float = 0.0,
    high: float = 1.0,
) -> float:
    """Jensen-Shannon divergence between two distributions over ``[low, high]``."""
    import numpy as np

    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    bins = np.linspace(low, high, int(n_bins) + 1)
    hist_a = np.histogram(a, bins=bins)[0].astype(np.float64)
    hist_b = np.histogram(b, bins=bins)[0].astype(np.float64)
    eps = 1e-9
    hist_a = (hist_a + eps) / (hist_a.sum() + eps * len(hist_a))
    hist_b = (hist_b + eps) / (hist_b.sum() + eps * len(hist_b))
    mix = 0.5 * (hist_a + hist_b)

    def kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    return 0.5 * (kl(hist_a, mix) + kl(hist_b, mix))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    seed: int = 20261120,
    alpha: float = 0.05,
) -> List[Optional[float]]:
    """Bootstrap 95% CI (percentile) for the mean of ``values``."""
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.array(
        [arr[rng.integers(0, arr.size, size=arr.size)].mean() for _ in range(int(n_boot))]
    )
    return [
        float(np.percentile(means, 100.0 * alpha / 2.0)),
        float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
    ]


def bootstrap_paired_ci(
    differences: Sequence[float],
    n_boot: int = 2000,
    seed: int = 20261120,
    alpha: float = 0.05,
) -> List[Optional[float]]:
    """Per-chart paired bootstrap CI for the mean of ``differences``."""
    import numpy as np

    diffs = np.asarray(list(differences), dtype=np.float64)
    if diffs.size == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.array(
        [diffs[rng.integers(0, diffs.size, size=diffs.size)].mean() for _ in range(int(n_boot))]
    )
    return [
        float(np.percentile(means, 100.0 * alpha / 2.0)),
        float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0))),
    ]


def unexpected_duplicate_rate(
    pred_bins: Sequence[int],
    true_bins: Sequence[int],
    group_ranges: Sequence[Tuple[int, int]],
) -> float:
    """Fraction of notes in unexpected same-tick, same-bin landing duplicates.

    A note is a duplicate when another note in the same group shares its bin.
    The ground-truth duplicate count is subtracted so only *unexpected*
    duplicates (beyond those already present in the reference chart) count.
    """
    import numpy as np

    pred = np.asarray(list(pred_bins), dtype=np.int64)
    true = np.asarray(list(true_bins), dtype=np.int64)
    n = int(pred.shape[0])
    if n == 0:
        return 0.0
    pred_dup = np.zeros(n, dtype=bool)
    true_dup = np.zeros(n, dtype=bool)
    for start, end in group_ranges:
        start, end = int(start), int(end)
        if end - start < 2:
            continue
        p = pred[start:end]
        t = true[start:end]
        for i in range(end - start):
            pred_dup[start + i] = bool(np.sum(p == p[i]) > 1)
            true_dup[start + i] = bool(np.sum(t == t[i]) > 1)
    unexpected = max(0, int(pred_dup.sum()) - int(true_dup.sum()))
    return unexpected / n


def out_of_bounds_rate(values: Sequence[float]) -> float:
    import numpy as np

    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean((arr < 0.0) | (arr > 1.0)))


def discover_checkpoints(
    checkpoint_dir: str, seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Scan ``checkpoint_dir`` for fold checkpoints and read their metadata."""
    import torch

    found: List[Dict[str, Any]] = []
    if not os.path.isdir(checkpoint_dir):
        return found
    for name in sorted(os.listdir(checkpoint_dir)):
        if not name.endswith(".pt"):
            continue
        if re.fullmatch(r"model_fold(\d+)_seed(\d+)\.pt", name) is None:
            # Consume only the canonical fold checkpoint names; preserved
            # variants such as *_v1 or *_v2a are not used for evaluation.
            continue
        path = os.path.join(checkpoint_dir, name)
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if "config" not in checkpoint or "model_state" not in checkpoint:
            continue
        ckpt_seed = int(checkpoint.get("seed", -1))
        if seed is not None and ckpt_seed != int(seed):
            continue
        fold = checkpoint.get("fold")
        if fold is None:
            continue
        if _looks_like_smoke(checkpoint, checkpoint_dir, name):
            continue
        found.append(
            {
                "path": path,
                "name": name,
                "fold": int(fold),
                "seed": ckpt_seed,
                "epoch": int(checkpoint.get("epoch", -1)),
                "uses_audio": bool(
                    checkpoint.get("uses_audio", checkpoint.get("config", {}).get("use_audio", True))
                ),
                "config": checkpoint["config"],
            }
        )
    return found


def _looks_like_smoke(
    checkpoint: Dict[str, Any], checkpoint_dir: str, name: str
) -> bool:
    """Reject smoke checkpoints using frozen config and fold-report evidence."""
    config = checkpoint.get("config", {})
    if int(config.get("epochs", 200)) == 20 and int(config.get("patience", 30)) == 8:
        return True
    fold = checkpoint.get("fold")
    seed = checkpoint.get("seed")
    if fold is not None and seed is not None:
        report_name = name.replace("model_", "fold_", 1).replace(".pt", ".json")
        report_path = os.path.join(checkpoint_dir, report_name)
        if os.path.isfile(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as handle:
                    report = json.load(handle)
                if int(report.get("train_charts", 0)) <= 2:
                    return True
            except Exception:
                pass
    return False


def _collect_fold_predictions(model, dataloader, device) -> Dict[str, Dict[str, Any]]:
    """Run a checkpoint and collect Hungarian-aligned per-note predictions."""
    import torch

    from phigros_gen.metrics import hungarian_assignments
    from phigros_gen.model import bin_centers

    model.eval()
    per_chart: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        for batch in dataloader:
            out = model.forward_full(
                batch["scalars"].to(device),
                batch["type_idx"].to(device),
                batch["diff_idx"].to(device),
                batch["src_idx"].to(device),
                batch["audio"].to(device),
                batch["times"].to(device),
                batch["dts"].to(device),
                u_slot=batch["u_slot"].to(device),
                group_ids=batch["group_ids"].to(device),
                src_key_padding_mask=(~batch["mask"]).to(device),
            )
            n_charts = int(batch["scalars"].shape[0])
            for b in range(n_charts):
                n = int(batch["num_notes"][b])
                if n == 0:
                    continue
                chart_id = batch["chart_id"][b]
                group_ranges = batch["group_ranges"][b]
                x_logits = out["x_bin_logits"][b, :n]
                side_logits = out["side_logits"][b, :n]
                x_bin_targets = batch["x_bin"][b, :n]
                assignments = hungarian_assignments(x_logits, x_bin_targets, group_ranges)
                centers = bin_centers(int(x_logits.shape[-1])).to(device)
                pred_bin = x_logits.argmax(dim=-1)
                pred_x = centers[pred_bin]
                pred_side = side_logits.argmax(dim=-1)

                true_x = batch["x_abs"][b, :n]
                true_side = batch["side"][b, :n]
                true_bin = x_bin_targets

                pred_x_list: List[float] = []
                pred_side_list: List[int] = []
                true_x_list: List[float] = []
                true_side_list: List[int] = []
                true_bin_list: List[int] = []
                pred_bin_list: List[int] = []
                for gi, (start, end) in enumerate(group_ranges):
                    col = assignments[gi]
                    for slot_local, target_local in enumerate(col):
                        slot = int(start) + int(slot_local)
                        target = int(start) + int(target_local)
                        pred_x_list.append(float(pred_x[slot].item()))
                        pred_side_list.append(int(pred_side[slot].item()))
                        pred_bin_list.append(int(pred_bin[slot].item()))
                        true_x_list.append(float(true_x[target].item()))
                        true_side_list.append(int(true_side[target].item()))
                        true_bin_list.append(int(true_bin[target].item()))

                per_chart[chart_id] = {
                    "chart_id": chart_id,
                    "pred_x": pred_x_list,
                    "pred_side": pred_side_list,
                    "pred_bin": pred_bin_list,
                    "true_x": true_x_list,
                    "true_side": true_side_list,
                    "true_bin": true_bin_list,
                    "group_ranges": [tuple(r) for r in group_ranges],
                }
    return per_chart


def evaluate_fold_checkpoint(
    rows: Sequence[Dict[str, Any]],
    splits: Dict[str, Any],
    checkpoint_info: Dict[str, Any],
    features_dir: str,
    device=None,
) -> Dict[str, Any]:
    """Evaluate one fold checkpoint on its outer-test charts."""
    from phigros_gen.dataset import TapDataset, Vocab, make_dataloader
    from phigros_gen.infer import pick_device
    from phigros_gen.metrics import evaluate_baselines
    from phigros_gen.train import evaluate_model

    fold = int(checkpoint_info["fold"])
    config = checkpoint_info["config"]
    uses_audio = bool(checkpoint_info["uses_audio"])
    fold_info = next((f for f in splits["folds"] if int(f["fold"]) == fold), None)
    if fold_info is None:
        raise ValueError(f"no split metadata for fold {fold}")

    test_rows = select_rows(rows, song_ids=fold_info["test_song_ids"])
    train_rows = select_rows(rows, song_ids=fold_info["train_song_ids"])
    val_rows = select_rows(rows, song_ids=fold_info["inner_val_song_ids"])
    if not test_rows:
        raise ValueError(f"fold {fold} has no test charts")
    _verify_fold_report(
        checkpoint_info,
        expected_train_charts=len(train_rows),
        expected_val_charts=len(val_rows),
        expected_test_charts=len(test_rows),
    )

    vocab = Vocab.from_config(config)
    test_ds = TapDataset(test_rows, vocab, features_dir, use_audio=uses_audio)
    loader = make_dataloader(
        test_ds,
        batch_size=4,
        max_tokens=8192,
        shuffle=False,
        seed=0,
        num_workers=0,
    )

    model = _load_model_from_info(checkpoint_info)
    if device is None:
        device = pick_device()
    model.to(device)

    model_result = evaluate_model(model, loader, device)
    predictions = _collect_fold_predictions(model, loader, device)
    baselines = evaluate_baselines(train_rows, test_rows)

    per_chart_model = model_result["per_chart"]
    per_chart_baseline: Dict[str, Dict[str, float]] = {}
    for row in test_rows:
        chart_id = row["chart_id"]
        one_baseline = evaluate_baselines(train_rows, [row])
        per_chart_baseline[chart_id] = one_baseline["baseline_x_mae"]

    strongest_baseline = baselines.get("strongest_baseline", "global_median")
    paired_differences: List[float] = []
    for chart_id in sorted(per_chart_model):
        model_mae = float(per_chart_model[chart_id].get("x_mae", 0.0))
        baseline_mae = float(per_chart_baseline.get(chart_id, {}).get(strongest_baseline, 0.0))
        paired_differences.append(model_mae - baseline_mae)

    all_pred_x: List[float] = []
    all_pred_bin: List[int] = []
    all_true_x: List[float] = []
    all_true_bin: List[int] = []
    all_group_ranges: List[Tuple[int, int]] = []
    range_offset = 0
    for chart in predictions.values():
        all_pred_x.extend(chart["pred_x"])
        all_pred_bin.extend(chart["pred_bin"])
        all_true_x.extend(chart["true_x"])
        all_true_bin.extend(chart["true_bin"])
        all_group_ranges.extend(
            (start + range_offset, end + range_offset)
            for start, end in chart["group_ranges"]
        )
        range_offset += len(chart["true_x"])

    note_metrics = {
        "out_of_bounds_rate": out_of_bounds_rate(all_pred_x),
        "unexpected_duplicate_rate": unexpected_duplicate_rate(
            all_pred_bin, all_true_bin, all_group_ranges
        ),
        "js_distance": js_divergence(all_pred_x, all_true_x),
    }

    return {
        "fold": fold,
        "seed": checkpoint_info["seed"],
        "epoch": checkpoint_info["epoch"],
        "uses_audio": uses_audio,
        "checkpoint": checkpoint_info["path"],
        "num_test_charts": len(test_rows),
        "num_test_notes": int(model_result["overall"].get("num_notes", 0)),
        "model": {
            "x_mae": float(model_result["overall"].get("x_mae", 0.0)),
            "x_rmse": float(model_result["overall"].get("x_rmse", 0.0)),
            "top1": float(model_result["overall"].get("top1", 0.0)),
            "top5": float(model_result["overall"].get("top5", 0.0)),
            "side_macro_f1": float(model_result["overall"].get("side_macro_f1", 0.0)),
            **note_metrics,
        },
        "baselines": baselines,
        "strongest_baseline": strongest_baseline,
        "paired_differences_vs_strongest": paired_differences,
        "bootstrap": {
            "model_vs_strongest_baseline_mae": bootstrap_paired_ci(paired_differences),
            "model_x_mae": bootstrap_mean_ci(
                [float(per_chart_model[c].get("x_mae", 0.0)) for c in sorted(per_chart_model)]
            ),
        },
        "per_chart_model": {
            c: float(per_chart_model[c].get("x_mae", 0.0)) for c in sorted(per_chart_model)
        },
        "per_chart_baseline": {
            c: per_chart_baseline.get(c, {}) for c in sorted(per_chart_model)
        },
    }


def _verify_fold_report(
    checkpoint_info: Dict[str, Any],
    expected_train_charts: int,
    expected_val_charts: int,
    expected_test_charts: int,
) -> None:
    """Reject a checkpoint whose fold report disagrees with the frozen splits."""
    name = checkpoint_info.get("name", "")
    report_name = name.replace("model_", "fold_", 1).replace(".pt", ".json")
    report_path = os.path.join(
        os.path.dirname(checkpoint_info["path"]),
        report_name,
    )
    if not os.path.isfile(report_path):
        return
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except Exception:
        return
    train_charts = int(report.get("train_charts", -1))
    val_charts = int(report.get("val_charts", -1))
    test_charts = int(report.get("test_charts", -1))
    if train_charts != -1 and train_charts != expected_train_charts:
        raise ValueError(
            f"fold report train_charts={train_charts} disagrees with splits "
            f"(expected {expected_train_charts})"
        )
    if test_charts != -1 and test_charts != expected_test_charts:
        raise ValueError(
            f"fold report test_charts={test_charts} disagrees with splits "
            f"(expected {expected_test_charts})"
        )
    if val_charts != -1 and val_charts != expected_val_charts:
        raise ValueError(
            f"fold report val_charts={val_charts} disagrees with splits "
            f"(expected {expected_val_charts})"
        )


def _load_model_from_info(checkpoint_info: Dict[str, Any]):
    from phigros_gen.infer import load_checkpoint

    checkpoint = load_checkpoint(checkpoint_info["path"])
    from phigros_gen.model import build_model

    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def aggregate_folds(
    fold_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Weighted aggregation of per-fold outer-test results."""
    if not fold_results:
        return {}

    def weighted(values: Sequence[float], weights: Sequence[float]) -> float:
        total = float(sum(weights))
        if total <= 0:
            return 0.0
        return float(sum(v * w for v, w in zip(values, weights))) / total

    weights = [float(r.get("num_test_notes", 0)) for r in fold_results]
    metric_keys = [
        "x_mae",
        "x_rmse",
        "top1",
        "top5",
        "side_macro_f1",
        "out_of_bounds_rate",
        "unexpected_duplicate_rate",
        "js_distance",
    ]
    model_agg = {
        key: weighted([float(r["model"].get(key, 0.0)) for r in fold_results], weights)
        for key in metric_keys
    }
    model_agg["num_folds"] = len(fold_results)

    baseline_names = sorted(
        {
            name
            for r in fold_results
            for name in r.get("baselines", {}).get("baseline_x_mae", {}).keys()
        }
    )
    baseline_agg = {
        name: weighted(
            [float(r["baselines"]["baseline_x_mae"].get(name, 0.0)) for r in fold_results],
            weights,
        )
        for name in baseline_names
    }
    side_majority_values = [
        float(r.get("baselines", {}).get("baseline_side_macro_f1", {}).get("majority", 0.0))
        for r in fold_results
    ]
    baseline_side_f1 = weighted(side_majority_values, weights)
    model_side_f1 = float(model_agg.get("side_macro_f1", 0.0))
    strongest = min(baseline_agg, key=baseline_agg.get) if baseline_agg else "global_median"
    strongest_mae = float(baseline_agg.get(strongest, 0.0))
    model_mae = float(model_agg.get("x_mae", 0.0))

    all_paired: List[float] = []
    for r in fold_results:
        all_paired.extend(r.get("paired_differences_vs_strongest", []))

    return {
        "num_folds": len(fold_results),
        "model": model_agg,
        "baselines": baseline_agg,
        "baseline_side_macro_f1": baseline_side_f1,
        "side_macro_f1_improvement_vs_majority": model_side_f1 - baseline_side_f1,
        "strongest_baseline": strongest,
        "strongest_baseline_x_mae": strongest_mae,
        "x_mae_improvement_vs_baseline": _safe_divide(strongest_mae - model_mae, strongest_mae),
        "bootstrap_paired_ci_model_vs_strongest_baseline": bootstrap_paired_ci(all_paired),
        "per_fold": fold_results,
    }


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate fold checkpoints on held-out outer-test charts."
    )
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints")
    parser.add_argument("--out", default="artifacts/reports/eval_report.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import datetime as _dt

    args = build_parser().parse_args(argv)
    rows = load_jsonl(args.dataset)
    splits = load_json(args.splits)
    splits_validation = validate_splits_file(splits, rows, os.path.abspath(args.splits))
    if (
        splits_validation["semantic_errors"]
        or not splits_validation["splits_sha256_ok"]
        or not splits_validation["all_folds_ok"]
    ):
        report = {
            "status": "blocked_invalid_splits",
            "reason": "splits.json failed pre-evaluation validation",
            "generated_at": _dt.datetime.now().isoformat(),
            "splits_validation": splits_validation,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print("splits.json failed pre-evaluation validation", flush=True)
        return 1

    checkpoints = discover_checkpoints(args.checkpoint_dir, args.seed)
    if not checkpoints:
        print(
            f"No fold checkpoints found in {args.checkpoint_dir}; writing a blocked report.",
            flush=True,
        )
        report = {
            "status": "blocked",
            "reason": "no fold checkpoints available yet",
            "generated_at": _dt.datetime.now().isoformat(),
            "checkpoint_dir": args.checkpoint_dir,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return 1

    results_by_audio: Dict[bool, List[Dict[str, Any]]] = {True: [], False: []}
    for checkpoint_info in checkpoints:
        try:
            result = evaluate_fold_checkpoint(
                rows, splits, checkpoint_info, args.features_dir, device=args.device
            )
            results_by_audio[bool(checkpoint_info["uses_audio"])].append(result)
        except Exception as exc:  # keep other folds going; report error honestly
            results_by_audio[bool(checkpoint_info["uses_audio"])].append(
                {
                    "fold": checkpoint_info["fold"],
                    "checkpoint": checkpoint_info["path"],
                    "error": str(exc),
                }
            )

    report: Dict[str, Any] = {
        "generated_at": _dt.datetime.now().isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.seed,
        "splits_validation": splits_validation,
        "audio": None,
        "no_audio": None,
        "comparison": None,
    }
    for uses_audio, results in results_by_audio.items():
        clean = [r for r in results if "error" not in r]
        key = "audio" if uses_audio else "no_audio"
        if clean:
            report[key] = aggregate_folds(clean)
        else:
            report[key] = {"status": "no_valid_folds", "fold_results": results}

    audio_report = report.get("audio")
    no_audio_report = report.get("no_audio")
    if (
        isinstance(audio_report, dict)
        and isinstance(no_audio_report, dict)
        and "model" in audio_report
        and "model" in no_audio_report
    ):
        audio_mae = audio_report["model"].get("x_mae", 0.0)
        no_audio_mae = no_audio_report["model"].get("x_mae", 0.0)
        report["comparison"] = {
            "audio_x_mae": audio_mae,
            "no_audio_x_mae": no_audio_mae,
            "no_audio_minus_audio_x_mae": no_audio_mae - audio_mae,
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Nested 4-fold cross-validation entry point and report aggregation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from phigros_gen.dataset import Vocab, load_jsonl  # noqa: E402
from phigros_gen.train import (  # noqa: E402
    check_gpu,
    get_device,
    gpu_forward_backward_smoke,
    train_and_evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nested cross-validation.")
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    parser.add_argument("--splits", default="data/processed/splits.json")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-kind",
        default=None,
        choices=["transformer", "group_mlp"],
        help="Model architecture to train.",
    )
    return parser.parse_args()


def _weighted_mean(values: List[float], weights: List[float]) -> float:
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


def aggregate(
    fold_reports: List[Dict[str, Any]],
    seed: int,
    uses_audio: bool,
) -> Dict[str, Any]:
    model_metrics = ["x_mae", "x_rmse", "top1", "top5", "side_macro_f1", "side_acc"]
    weights = []
    per_fold = []
    per_chart: Dict[str, Dict[str, Any]] = {}
    baseline_names = set()
    baseline_values: Dict[str, List[float]] = {}
    side_majority_values: List[float] = []

    for report in fold_reports:
        test = report.get("test", {}).get("overall", {})
        w = float(test.get("num_notes", 0))
        weights.append(w)
        per_fold.append(
            {
                "fold": report["fold"],
                "seed": report["seed"],
                "best_epoch": report["best_epoch"],
                "elapsed_seconds": report["elapsed_seconds"],
                "peak_memory_bytes": report["peak_memory_bytes"],
                "num_parameters": report["num_parameters"],
                "test_charts": report["test_charts"],
                "test": test,
            }
        )
        for chart_id, metrics in report.get("test", {}).get("per_chart", {}).items():
            per_chart[chart_id] = metrics
        for name, value in report.get("baselines", {}).get("baseline_x_mae", {}).items():
            baseline_names.add(name)
            baseline_values.setdefault(name, []).append(float(value))
        side_majority = report.get("baselines", {}).get(
            "baseline_side_macro_f1", {}
        ).get("majority", 0.0)
        side_majority_values.append(float(side_majority))

    model_aggregate: Dict[str, Any] = {"num_folds": len(fold_reports)}
    for key in model_metrics:
        vals = [r["test"]["overall"].get(key, 0.0) for r in fold_reports]
        model_aggregate[key] = _weighted_mean(vals, weights)

    baseline_aggregate = {
        name: _weighted_mean(baseline_values[name], weights)
        for name in sorted(baseline_names)
    }
    strongest = min(baseline_aggregate.items(), key=lambda kv: kv[1]) if baseline_aggregate else ("none", None)
    baseline_side_f1 = _weighted_mean(side_majority_values, weights)

    return {
        "seed": seed,
        "uses_audio": uses_audio,
        "num_folds": len(fold_reports),
        "model": model_aggregate,
        "baselines": baseline_aggregate,
        "baseline_side_macro_f1": baseline_side_f1,
        "side_macro_f1_improvement_vs_majority": (
            model_aggregate["side_macro_f1"] - baseline_side_f1
        ),
        "strongest_baseline": strongest[0],
        "strongest_baseline_x_mae": strongest[1],
        "x_mae_improvement_vs_baseline": (
            (strongest[1] - model_aggregate["x_mae"]) / strongest[1]
            if strongest[1]
            else None
        ),
        "total_elapsed_seconds": sum(r["elapsed_seconds"] for r in fold_reports),
        "peak_memory_bytes": max(
            (r["peak_memory_bytes"] for r in fold_reports if r["peak_memory_bytes"]),
            default=None,
        ),
        "per_fold": per_fold,
        "per_chart": per_chart,
    }


def main() -> int:
    args = parse_args()
    rows = load_jsonl(args.dataset)
    vocab = Vocab.from_rows(rows)
    with open(args.splits, "r", encoding="utf-8") as handle:
        splits = json.load(handle)

    device = get_device(args.device)
    gpu_info = check_gpu()
    smoke_result = None
    if device.type == "cuda" or args.smoke:
        smoke_result = gpu_forward_backward_smoke(
            device,
            use_audio=not args.no_audio,
            vocab=vocab,
            model_kind=args.model_kind or "transformer",
        )

    folds = splits["folds"]
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]

    hyperparams: Dict[str, Any] = {}
    if args.epochs is not None:
        hyperparams["epochs"] = args.epochs
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size
    if args.model_kind is not None:
        hyperparams["model_kind"] = args.model_kind

    fold_reports: List[Dict[str, Any]] = []
    checkpoint_dir = os.path.join(args.output, "checkpoints")
    for fold_info in folds:
        report = train_and_evaluate(
            rows,
            vocab,
            args.features_dir,
            checkpoint_dir,
            fold_info["train_song_ids"],
            fold_info["inner_val_song_ids"],
            fold_info["test_song_ids"],
            int(fold_info["fold"]),
            args.seed,
            not args.no_audio,
            smoke=args.smoke,
            hyperparams=hyperparams,
            device=device,
            num_workers=args.num_workers,
        )
        fold_reports.append(report)

    report = aggregate(fold_reports, args.seed, not args.no_audio)
    report["gpu"] = gpu_info
    report["gpu_forward_backward_smoke"] = smoke_result
    report["splits"] = splits

    reports_dir = os.path.join(args.output, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "train_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

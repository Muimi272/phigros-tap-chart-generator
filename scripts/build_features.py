"""Batch feature extraction CLI for chart audio."""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from phigros_gen.audio_features import main as audio_features_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build log-mel features for charts.")
    parser.add_argument("--dataset", default="data/processed/dataset.jsonl")
    parser.add_argument("--output", default="data/processed/features")
    parser.add_argument(
        "--report",
        default="artifacts/reports/feature_report.json",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    argv = [
        "--dataset",
        args.dataset,
        "--output",
        args.output,
        "--repo-root",
        REPO_ROOT,
        "--report",
        args.report,
    ]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    return audio_features_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

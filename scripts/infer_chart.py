"""CLI entry point for tap-chart inference."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from phigros_gen.infer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

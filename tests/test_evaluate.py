import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phigros_gen.evaluate import (  # noqa: E402
    bootstrap_mean_ci,
    bootstrap_paired_ci,
    js_divergence,
    out_of_bounds_rate,
    unexpected_duplicate_rate,
)


def test_js_divergence_identical_distributions():
    pytest.importorskip("numpy")
    values = [0.1, 0.2, 0.3, 0.4]
    assert js_divergence(values, values) == pytest.approx(0.0, abs=1e-6)


def test_js_divergence_separated_distributions_positive():
    pytest.importorskip("numpy")
    a = [0.05, 0.06, 0.07]
    b = [0.95, 0.96, 0.97]
    assert js_divergence(a, b) > 0.5


def test_bootstrap_mean_ci_basic_behaviour():
    pytest.importorskip("numpy")
    lo, hi = bootstrap_mean_ci([1.0, 2.0, 3.0, 4.0], n_boot=500, seed=7)
    assert lo is not None and hi is not None
    assert lo <= 2.5 <= hi


def test_bootstrap_paired_ci_basic_behaviour():
    pytest.importorskip("numpy")
    lo, hi = bootstrap_paired_ci([-1.0, -0.5, 0.0, 0.5, 1.0], n_boot=500, seed=7)
    assert lo is not None and hi is not None
    assert lo <= hi


def test_out_of_bounds_rate():
    pytest.importorskip("numpy")
    assert out_of_bounds_rate([0.5, 0.6, 0.7]) == pytest.approx(0.0)
    assert out_of_bounds_rate([0.5, 1.1, -0.1]) == pytest.approx(2.0 / 3.0)


def test_unexpected_duplicate_rate_subtracts_true_duplicates():
    pytest.importorskip("numpy")
    # Two groups. In the second group both predicted and true bins are all
    # identical, so no unexpected duplicates remain.
    pred = [1, 2, 2]
    true = [1, 2, 2]
    ranges = [(0, 1), (1, 3)]
    assert unexpected_duplicate_rate(pred, true, ranges) == pytest.approx(0.0)


def test_unexpected_duplicate_rate_counts_new_duplicates():
    pytest.importorskip("numpy")
    pred = [1, 2, 2]
    true = [1, 3, 2]
    ranges = [(0, 1), (1, 3)]
    # Predicted group 2 has a duplicate pair; true group 2 does not.
    assert unexpected_duplicate_rate(pred, true, ranges) == pytest.approx(2.0 / 3.0)

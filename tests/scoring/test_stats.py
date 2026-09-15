from __future__ import annotations

import pytest

from toolchoice.scoring.stats import cluster_bootstrap, mcnemar_exact, wilson


def test_wilson_matches_a_hand_computed_interval() -> None:
    interval = wilson(40, 40)
    assert interval.point == pytest.approx(1.0)
    assert interval.low == pytest.approx(0.9124, abs=1e-3)
    assert interval.high == pytest.approx(1.0)


def test_wilson_on_an_empty_denominator_is_null_not_zero() -> None:
    """0.0 for an absent class is how a missing ANSWER population stayed invisible upstream."""
    interval = wilson(0, 0)
    assert interval.point is None
    assert interval.method.endswith("[UNMEASURABLE]")


def test_mcnemar_exact_matches_hand_computation() -> None:
    # b=5, c=0 -> two-sided exact binomial = 2 * 0.5**5
    assert mcnemar_exact(5, 0)["p_value"] == pytest.approx(2 * 0.5**5)
    assert mcnemar_exact(3, 3)["p_value"] == pytest.approx(1.0)


def test_mcnemar_reports_b_and_c_so_cancellation_is_visible() -> None:
    """A model that breaks twenty stems and fixes twenty has a net of zero and is not healthy."""
    result = mcnemar_exact(20, 20)
    assert result["b"] == 20 and result["c"] == 20
    assert result["discordant"] == 40


def test_no_discordant_pairs_is_null_not_one() -> None:
    assert mcnemar_exact(0, 0)["p_value"] is None


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    clusters = [1] * 50 + [0] * 50

    def mean(sample):
        return sum(sample) / len(sample) if sample else None

    interval = cluster_bootstrap(clusters, mean, resamples=500, seed=7)
    assert interval.point == pytest.approx(0.5)
    assert interval.low < 0.5 < interval.high

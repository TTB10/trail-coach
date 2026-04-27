"""Tests for TRIMP, CTL, ATL, TSB and ACWR calculations."""

from __future__ import annotations

import math

import pytest
from app.domain.training_load import (
    ATL_ALPHA,
    CTL_ALPHA,
    TrainingLoadSeries,
    calculate_trimp_karvonen,
    compute_acwr,
    compute_training_load_series,
)

# ----------------------------------------------------------------------------
# TRIMP — per-session score
# ----------------------------------------------------------------------------


class TestTrimpKarvonen:
    """Verify the per-session TRIMP formula."""

    def test_known_session(self) -> None:
        # 60 min @ HR_avg=150 (HR_max=190, HR_rest=54).
        # ratio = 96/136 ≈ 0.7059
        # TRIMP = 60 * 0.7059 * 0.64 * exp(1.92 * 0.7059) ≈ 105.11
        result = calculate_trimp_karvonen(duration_min=60, hr_avg=150, hr_max=190, hr_rest=54)
        assert math.isclose(result, 105.11, rel_tol=1e-3)

    def test_zero_duration_yields_zero(self) -> None:
        # No effort → no TRIMP.
        assert calculate_trimp_karvonen(0, hr_avg=150, hr_max=190, hr_rest=54) == 0.0

    def test_hr_at_resting_yields_zero(self) -> None:
        # When HR_avg <= HR_rest, the session is sub-aerobic.
        assert calculate_trimp_karvonen(60, hr_avg=54, hr_max=190, hr_rest=54) == 0.0
        assert calculate_trimp_karvonen(60, hr_avg=40, hr_max=190, hr_rest=54) == 0.0

    def test_hr_at_max_uses_clamped_ratio(self) -> None:
        # If HR_avg = HR_max, ratio = 1; TRIMP = duration * 0.64 * exp(1.92).
        result = calculate_trimp_karvonen(60, hr_avg=190, hr_max=190, hr_rest=54)
        expected = 60 * 1.0 * 0.64 * math.exp(1.92)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_higher_intensity_yields_higher_trimp(self) -> None:
        # Monotonicity: harder session => more TRIMP.
        easy = calculate_trimp_karvonen(60, hr_avg=130, hr_max=190, hr_rest=54)
        hard = calculate_trimp_karvonen(60, hr_avg=170, hr_max=190, hr_rest=54)
        assert hard > easy

    def test_longer_duration_yields_proportionally_more_trimp(self) -> None:
        # TRIMP is linear in duration at constant intensity.
        short = calculate_trimp_karvonen(30, hr_avg=150, hr_max=190, hr_rest=54)
        long_ = calculate_trimp_karvonen(60, hr_avg=150, hr_max=190, hr_rest=54)
        assert math.isclose(long_, 2 * short, rel_tol=1e-9)

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="duration_min"):
            calculate_trimp_karvonen(-10, hr_avg=150, hr_max=190, hr_rest=54)

    def test_invalid_hr_bounds_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            calculate_trimp_karvonen(60, hr_avg=150, hr_max=100, hr_rest=120)


# ----------------------------------------------------------------------------
# CTL / ATL — EWMA convergence
# ----------------------------------------------------------------------------


class TestTrainingLoadSeries:
    """Verify the EWMA-based fitness/fatigue model."""

    def test_empty_series_yields_no_points(self) -> None:
        series = compute_training_load_series([])
        assert series.points == []

    def test_first_day_starts_at_alpha_times_input(self) -> None:
        # Day 0 with TRIMP=100: CTL = alpha * 100, ATL = alpha * 100, TSB = 0.
        series = compute_training_load_series([100.0])
        p = series.points[0]
        assert math.isclose(p.ctl, CTL_ALPHA * 100, rel_tol=1e-9)
        assert math.isclose(p.atl, ATL_ALPHA * 100, rel_tol=1e-9)
        assert p.tsb == 0.0  # No previous day yet.

    def test_zero_input_yields_zero_everywhere(self) -> None:
        # 30 rest days → nothing accumulates.
        series = compute_training_load_series([0.0] * 30)
        for p in series.points:
            assert p.ctl == 0.0
            assert p.atl == 0.0
            assert p.tsb == 0.0

    def test_constant_input_atl_converges_quickly(self) -> None:
        # ATL has alpha=2/8, so after ~30 days it should be very close to input.
        series = compute_training_load_series([100.0] * 30)
        atl_30 = series.points[-1].atl
        assert math.isclose(atl_30, 100.0, rel_tol=1e-3)

    def test_constant_input_ctl_converges_more_slowly_than_atl(self) -> None:
        # After 30 days, CTL is still well below input (slower window).
        series = compute_training_load_series([100.0] * 30)
        last = series.points[-1]
        assert last.ctl < last.atl
        # Sanity: CTL is somewhere reasonable but not converged.
        assert 50 < last.ctl < 90

    def test_tsb_uses_previous_day_values(self) -> None:
        # TSB[i] should be CTL[i-1] - ATL[i-1].
        series = compute_training_load_series([100.0] * 10)
        for i in range(1, len(series.points)):
            expected_tsb = series.points[i - 1].ctl - series.points[i - 1].atl
            assert math.isclose(series.points[i].tsb, expected_tsb, rel_tol=1e-9)

    def test_tsb_is_negative_during_buildup(self) -> None:
        # When ramping up training, ATL outruns CTL → TSB < 0.
        series = compute_training_load_series([100.0] * 20)
        # After at least a few days, TSB should be clearly negative.
        assert series.points[10].tsb < 0
        assert series.points[19].tsb < 0

    def test_tsb_becomes_positive_after_taper(self) -> None:
        # 30 days of training, then 14 rest days → TSB should turn positive.
        days = [100.0] * 30 + [0.0] * 14
        series = compute_training_load_series(days)
        # On the last day of taper, fatigue (ATL) has dropped faster than fitness (CTL).
        assert series.points[-1].tsb > 0


# ----------------------------------------------------------------------------
# ACWR — Gabbett injury-risk ratio
# ----------------------------------------------------------------------------


class TestACWR:
    """Verify the acute-to-chronic workload ratio."""

    def test_empty_series_yields_no_acwr(self) -> None:
        assert compute_acwr([]) == []

    def test_zero_load_yields_zero_acwr(self) -> None:
        # No training → no ratio (and no division-by-zero noise).
        result = compute_acwr([0.0] * 30)
        assert all(r == 0.0 for r in result)

    def test_constant_load_converges_to_one(self) -> None:
        # If you train the same every day, acute and chronic loads match.
        # ACWR_t = sum(7d) / (sum(28d) / 4) = 7L / (28L / 4) = 7L / 7L = 1.0.
        result = compute_acwr([50.0] * 35)
        # On day 28 onwards, both windows are full → ACWR should be exactly 1.
        for i in range(27, len(result)):
            assert math.isclose(result[i], 1.0, rel_tol=1e-9)

    def test_sudden_spike_pushes_acwr_above_one(self) -> None:
        # 28 days at 50 TRIMP, then 7 days at 100 TRIMP → spike.
        days = [50.0] * 28 + [100.0] * 7
        result = compute_acwr(days)
        # On the last day, acute window is full of 100s, chronic mixes both.
        last = result[-1]
        assert last > 1.3  # Clearly in injury-risk zone.

    def test_sudden_drop_pushes_acwr_below_one(self) -> None:
        # 28 days at 100, then 7 days at 0 → big drop in acute load.
        days = [100.0] * 28 + [0.0] * 7
        result = compute_acwr(days)
        # On the last day, acute = 0, chronic > 0, so ACWR = 0.
        assert result[-1] == 0.0

    def test_partial_history_does_not_crash(self) -> None:
        # Only 5 days of data → both windows are partial. Should still work.
        result = compute_acwr([50.0] * 5)
        assert len(result) == 5
        # With 5 days at the same load, ACWR should be 1.0.
        for r in result:
            assert math.isclose(r, 1.0, rel_tol=1e-9)


# ----------------------------------------------------------------------------
# Integration sanity check
# ----------------------------------------------------------------------------


def test_realistic_training_block_yields_sensible_metrics() -> None:
    """Smoke test: a realistic 8-week block produces plausible values."""
    # 8 weeks: alternating hard (120) and easy (60) days, with a rest day every Sunday.
    days: list[float] = []
    for week in range(8):
        days.extend([120.0, 60.0, 120.0, 60.0, 120.0, 60.0, 0.0])

    series = compute_training_load_series(days)
    assert isinstance(series, TrainingLoadSeries)
    assert len(series.points) == 56

    # End-of-block values should be in a sensible physiological range.
    last = series.points[-1]
    assert 50 < last.ctl < 100
    assert 50 < last.atl < 100
    # TSB shouldn't be wildly extreme.
    assert -30 < last.tsb < 30

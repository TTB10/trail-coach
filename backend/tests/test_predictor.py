"""Tests for the Riegel V4.1 race-time prediction model."""

from __future__ import annotations

import math

import pytest
from app.domain.predictor import (
    BASELINE_EXPONENT_BY_EXPERIENCE,
    ERROR_MAX,
    GAP_THRESHOLD,
    MAX_EXPONENT,
    MIN_EXPONENT,
    SPEED_RATIO_FLOOR,
    ExperienceLevel,
    RacePrediction,
    StrengthTrainingFrequency,
    adjust_riegel_exponent,
    baseline_exponent_for_experience,
    calculate_error_margin,
    calculate_speed_degradation_factor,
    extreme_effort_adjustment,
    fitness_deficit_adjustment,
    gap_adjustment,
    predict_finish_time,
    strength_training_adjustment,
)

# ----------------------------------------------------------------------------
# Layer 1 — Speed degradation
# ----------------------------------------------------------------------------


class TestSpeedDegradation:
    """Verify the VMA degradation factor."""

    def test_same_ctl_yields_no_degradation(self) -> None:
        # When current shape matches record shape, ratio is 1.
        assert calculate_speed_degradation_factor(current_ctl=80, record_ctl=80) == 1.0

    def test_higher_ctl_caps_at_one(self) -> None:
        # Even if you're fitter than at the record, no bonus speed.
        assert calculate_speed_degradation_factor(current_ctl=100, record_ctl=80) == 1.0

    def test_low_ctl_yields_degradation(self) -> None:
        # Half the CTL → degradation, but never below the floor.
        ratio = calculate_speed_degradation_factor(current_ctl=40, record_ctl=80)
        assert SPEED_RATIO_FLOOR <= ratio < 1.0

    def test_very_low_ctl_clamps_to_floor(self) -> None:
        # Severe detraining → ratio clamped to the floor (no infinite degradation).
        ratio = calculate_speed_degradation_factor(current_ctl=1, record_ctl=100)
        assert ratio == SPEED_RATIO_FLOOR

    def test_zero_record_ctl_returns_one(self) -> None:
        # No reference CTL → no degradation, conservative default.
        assert calculate_speed_degradation_factor(current_ctl=50, record_ctl=0) == 1.0


# ----------------------------------------------------------------------------
# Layer 2 — Baseline exponent
# ----------------------------------------------------------------------------


class TestBaselineExponent:
    """Verify the experience-based baseline."""

    def test_elite_is_lowest(self) -> None:
        assert baseline_exponent_for_experience(ExperienceLevel.ELITE) == 1.04

    def test_beginner_is_highest(self) -> None:
        assert baseline_exponent_for_experience(ExperienceLevel.BEGINNER) == 1.12

    def test_baselines_are_strictly_ordered(self) -> None:
        # Elite < Confirmed < Amateur < Beginner.
        levels = [
            ExperienceLevel.ELITE,
            ExperienceLevel.CONFIRMED,
            ExperienceLevel.AMATEUR,
            ExperienceLevel.BEGINNER,
        ]
        values = [baseline_exponent_for_experience(level) for level in levels]
        assert values == sorted(values)


# ----------------------------------------------------------------------------
# Layer 3 — Fitness deficit
# ----------------------------------------------------------------------------


class TestFitnessDeficit:
    """Verify the fitness-deficit penalty."""

    def test_well_prepared_yields_zero(self) -> None:
        # CTL well above what the race demands → no penalty.
        assert fitness_deficit_adjustment(projected_ctl=80, target_effort=100) == 0.0

    def test_exact_match_yields_zero(self) -> None:
        # CTL exactly at the demand threshold → no penalty.
        # demand = 100 * 0.35 = 35.
        assert fitness_deficit_adjustment(projected_ctl=35, target_effort=100) == 0.0

    def test_underprepared_yields_positive(self) -> None:
        # CTL below demand → positive bump.
        bump = fitness_deficit_adjustment(projected_ctl=10, target_effort=100)
        assert bump > 0

    def test_severely_underprepared_bumps_more(self) -> None:
        mild = fitness_deficit_adjustment(projected_ctl=30, target_effort=100)
        severe = fitness_deficit_adjustment(projected_ctl=5, target_effort=100)
        assert severe > mild


# ----------------------------------------------------------------------------
# Layer 4 — Extreme effort penalty
# ----------------------------------------------------------------------------


class TestExtremeEffort:
    """Verify the per-km penalty for very long efforts."""

    def test_below_threshold_yields_zero(self) -> None:
        # 80 effort-km is below the 100 threshold → no penalty.
        assert extreme_effort_adjustment(80) == 0.0

    def test_just_above_hard_threshold(self) -> None:
        # 110 effort-km → +10 * 0.0003 = 0.003.
        assert math.isclose(extreme_effort_adjustment(110), 0.003, rel_tol=1e-6)

    def test_extreme_threshold_stacks(self) -> None:
        # 220 effort-km → 120*0.0003 + 20*0.0005 = 0.036 + 0.01 = 0.046.
        result = extreme_effort_adjustment(220)
        expected = 120 * 0.0003 + 20 * 0.0005
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_monotonic_in_effort(self) -> None:
        # More effort → more penalty.
        assert extreme_effort_adjustment(150) < extreme_effort_adjustment(180)


# ----------------------------------------------------------------------------
# Layer 5 — Gap penalty
# ----------------------------------------------------------------------------


class TestGapAdjustment:
    """Verify the penalty when target is much harder than the record."""

    def test_small_gap_yields_zero(self) -> None:
        # target/record = 2 → below threshold (3) → no penalty.
        assert gap_adjustment(target_effort=100, record_effort=50) == 0.0

    def test_at_threshold_yields_zero(self) -> None:
        # target/record = exactly 3 → no penalty (strict inequality).
        assert gap_adjustment(target_effort=150, record_effort=50) == 0.0

    def test_above_threshold_yields_positive(self) -> None:
        # target/record = 5 → (5 - 3) * 0.015 = 0.030.
        result = gap_adjustment(target_effort=250, record_effort=50)
        assert math.isclose(result, 0.030, rel_tol=1e-6)

    def test_zero_record_returns_zero(self) -> None:
        # Pathological input: shouldn't crash.
        assert gap_adjustment(target_effort=100, record_effort=0) == 0.0


# ----------------------------------------------------------------------------
# Layer 6 — Strength training bonus
# ----------------------------------------------------------------------------


class TestStrengthTraining:
    """Verify the strength-training bonus (negative bump)."""

    def test_never_yields_zero(self) -> None:
        assert strength_training_adjustment(StrengthTrainingFrequency.NEVER) == 0.0

    def test_once_per_week_is_negative(self) -> None:
        bonus = strength_training_adjustment(StrengthTrainingFrequency.ONCE_PER_WEEK)
        assert bonus == -0.01

    def test_more_is_more_bonus(self) -> None:
        once = strength_training_adjustment(StrengthTrainingFrequency.ONCE_PER_WEEK)
        twice = strength_training_adjustment(StrengthTrainingFrequency.TWICE_OR_MORE)
        assert twice < once  # More negative = more bonus.


# ----------------------------------------------------------------------------
# Combined exponent — clamping + stacking
# ----------------------------------------------------------------------------


class TestAdjustedExponent:
    """Verify the orchestration of all exponent layers."""

    def test_well_prepared_amateur(self) -> None:
        # No deficit, moderate effort, no gap, with strength training.
        # Expected: 1.09 (baseline) + 0 + 0 + 0 - 0.01 = 1.08.
        k = adjust_riegel_exponent(
            experience=ExperienceLevel.AMATEUR,
            projected_ctl=100,
            target_effort=80,
            record_effort=50,
            strength_training=StrengthTrainingFrequency.ONCE_PER_WEEK,
        )
        assert math.isclose(k, 1.08, rel_tol=1e-6)

    def test_clamped_to_max(self) -> None:
        # Pathological scenario: huge deficit, extreme effort, big gap.
        # k should hit the safety ceiling.
        k = adjust_riegel_exponent(
            experience=ExperienceLevel.BEGINNER,
            projected_ctl=1,
            target_effort=300,
            record_effort=10,
            strength_training=StrengthTrainingFrequency.NEVER,
        )
        assert k == MAX_EXPONENT

    def test_clamped_to_min(self) -> None:
        # Elite + huge strength bonus shouldn't push k below the floor.
        k = adjust_riegel_exponent(
            experience=ExperienceLevel.ELITE,
            projected_ctl=1000,
            target_effort=20,
            record_effort=20,
            strength_training=StrengthTrainingFrequency.TWICE_OR_MORE,
        )
        assert k >= MIN_EXPONENT


# ----------------------------------------------------------------------------
# Error margin
# ----------------------------------------------------------------------------


class TestErrorMargin:
    """Verify the error-margin calculation."""

    def test_short_race_for_elite(self) -> None:
        # 50 effort-km elite: base = 0.04 + 50/2000 = 0.065. * 1.0 = 0.065.
        result = calculate_error_margin(50, ExperienceLevel.ELITE)
        assert math.isclose(result, 0.065, rel_tol=1e-6)

    def test_beginner_has_more_uncertainty(self) -> None:
        elite = calculate_error_margin(100, ExperienceLevel.ELITE)
        beginner = calculate_error_margin(100, ExperienceLevel.BEGINNER)
        assert beginner > elite

    def test_capped_at_max(self) -> None:
        # Very long effort → error capped at ERROR_MAX.
        result = calculate_error_margin(1000, ExperienceLevel.BEGINNER)
        assert result == ERROR_MAX


# ----------------------------------------------------------------------------
# Main predict_finish_time — input validation + output schema
# ----------------------------------------------------------------------------


class TestPredictFinishTime:
    """Verify the orchestration of the full prediction."""

    def test_returns_a_race_prediction(self) -> None:
        result = predict_finish_time(
            record_time_min=330,
            record_effort=52,
            target_effort=140,
            experience=ExperienceLevel.AMATEUR,
            strength_training=StrengthTrainingFrequency.ONCE_PER_WEEK,
            current_ctl=70,
            record_ctl=80,
            projected_ctl=90,
        )
        assert isinstance(result, RacePrediction)
        assert result.predicted_minutes > 0

    def test_pessimistic_is_above_predicted(self) -> None:
        result = predict_finish_time(
            record_time_min=300,
            record_effort=50,
            target_effort=80,
            experience=ExperienceLevel.AMATEUR,
            strength_training=StrengthTrainingFrequency.NEVER,
            current_ctl=60,
            record_ctl=60,
            projected_ctl=60,
        )
        assert result.optimistic_minutes < result.predicted_minutes < result.pessimistic_minutes

    def test_riegel_exponent_is_within_bounds(self) -> None:
        result = predict_finish_time(
            record_time_min=300,
            record_effort=50,
            target_effort=200,
            experience=ExperienceLevel.BEGINNER,
            strength_training=StrengthTrainingFrequency.NEVER,
            current_ctl=30,
            record_ctl=50,
            projected_ctl=40,
        )
        assert MIN_EXPONENT <= result.riegel_exponent <= MAX_EXPONENT

    def test_negative_record_time_raises(self) -> None:
        with pytest.raises(ValueError, match="record_time_min"):
            predict_finish_time(
                record_time_min=-1,
                record_effort=50,
                target_effort=100,
                experience=ExperienceLevel.AMATEUR,
                strength_training=StrengthTrainingFrequency.NEVER,
                current_ctl=50,
                record_ctl=50,
                projected_ctl=50,
            )

    def test_zero_record_effort_raises(self) -> None:
        with pytest.raises(ValueError, match="record_effort"):
            predict_finish_time(
                record_time_min=300,
                record_effort=0,
                target_effort=100,
                experience=ExperienceLevel.AMATEUR,
                strength_training=StrengthTrainingFrequency.NEVER,
                current_ctl=50,
                record_ctl=50,
                projected_ctl=50,
            )


# ----------------------------------------------------------------------------
# Realistic scenarios — sanity checks
# ----------------------------------------------------------------------------


class TestRealisticScenarios:
    """Smoke tests with realistic athlete profiles."""

    def test_marathon_pr_to_100k_amateur(self) -> None:
        # 5h30 on a 52 effort-km marathon → ~14-19h on a 140 effort-km 100k.
        result = predict_finish_time(
            record_time_min=330,
            record_effort=52,
            target_effort=140,
            experience=ExperienceLevel.AMATEUR,
            strength_training=StrengthTrainingFrequency.ONCE_PER_WEEK,
            current_ctl=70,
            record_ctl=80,
            projected_ctl=90,
        )
        # Predicted between 14h and 20h (840 - 1200 minutes).
        assert 840 < result.predicted_minutes < 1200

    def test_elite_predictions_are_faster(self) -> None:
        # Same race effort, same record effort, only experience changes.
        kwargs = {
            "record_time_min": 300,
            "record_effort": 50,
            "target_effort": 100,
            "strength_training": StrengthTrainingFrequency.ONCE_PER_WEEK,
            "current_ctl": 70,
            "record_ctl": 70,
            "projected_ctl": 70,
        }
        elite = predict_finish_time(experience=ExperienceLevel.ELITE, **kwargs)
        beginner = predict_finish_time(experience=ExperienceLevel.BEGINNER, **kwargs)
        # Beginner extrapolates worse (higher k) → predicted longer.
        assert beginner.predicted_minutes > elite.predicted_minutes


# ----------------------------------------------------------------------------
# Sanity check on the lookup tables
# ----------------------------------------------------------------------------


def test_baseline_table_has_all_levels() -> None:
    """If anyone forgets to add a level to the table, this test catches it."""
    for level in ExperienceLevel:
        assert level in BASELINE_EXPONENT_BY_EXPERIENCE


def test_gap_threshold_constant() -> None:
    assert GAP_THRESHOLD == 3.0

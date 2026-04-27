"""Tests for the corrected effort calculation."""

from __future__ import annotations

import math

from app.domain.effort import (
    DMINUS_BONUS_AT_REFERENCE,
    DMINUS_REFERENCE_M,
    METERS_OF_DPLUS_PER_KM_FLAT,
    calculate_corrected_effort,
)

# ----------------------------------------------------------------------------
# Base formula: distance + D+ / 100
# ----------------------------------------------------------------------------


class TestBaseFormula:
    """Verify the core distance + elevation conversion."""

    def test_pure_flat_returns_distance(self) -> None:
        # 10 km flat = 10 km equivalent.
        assert calculate_corrected_effort(distance_km=10, elevation_gain_m=0) == 10.0

    def test_zero_distance_with_elevation(self) -> None:
        # 0 km but 500 m of D+ = 5 km equivalent.
        assert calculate_corrected_effort(distance_km=0, elevation_gain_m=500) == 5.0

    def test_marathon_with_1000m_dplus(self) -> None:
        # Classic trail: 42 km + 1000 m D+ → 52 km equivalent.
        assert calculate_corrected_effort(distance_km=42, elevation_gain_m=1000) == 52.0

    def test_100m_dplus_equals_1km_flat(self) -> None:
        # The defining rule: 100 m of D+ adds 1 km of equivalent flat.
        assert calculate_corrected_effort(distance_km=0, elevation_gain_m=100) == 1.0


# ----------------------------------------------------------------------------
# Negative values: must be clamped to zero
# ----------------------------------------------------------------------------


class TestInputClamping:
    """Verify that physically meaningless negative inputs become zero."""

    def test_negative_distance_is_clamped(self) -> None:
        # -5 km treated as 0 km, so only D+ counts.
        assert calculate_corrected_effort(distance_km=-5, elevation_gain_m=100) == 1.0

    def test_negative_elevation_gain_is_clamped(self) -> None:
        # -200 m of D+ treated as 0 m of D+.
        assert calculate_corrected_effort(distance_km=10, elevation_gain_m=-200) == 10.0

    def test_negative_elevation_loss_is_clamped(self) -> None:
        # -500 m of D- treated as 0 m, so no descent bonus is applied.
        result = calculate_corrected_effort(
            distance_km=10, elevation_gain_m=0, elevation_loss_m=-500
        )
        assert result == 10.0


# ----------------------------------------------------------------------------
# Descent bonus
# ----------------------------------------------------------------------------


class TestDescentBonus:
    """Verify the bonus added for steep descents."""

    def test_no_descent_means_no_bonus(self) -> None:
        # D- = 0 → effort unchanged.
        without_dminus = calculate_corrected_effort(distance_km=10, elevation_gain_m=500)
        with_zero_dminus = calculate_corrected_effort(
            distance_km=10, elevation_gain_m=500, elevation_loss_m=0
        )
        assert without_dminus == with_zero_dminus

    def test_descent_bonus_at_reference_value(self) -> None:
        # 10000 m of D- → factor = 1 + 1 * 0.04 = 1.04.
        # 10 km + 0 D+ = 10 km, then * 1.04 = 10.4.
        result = calculate_corrected_effort(
            distance_km=10,
            elevation_gain_m=0,
            elevation_loss_m=DMINUS_REFERENCE_M,
        )
        expected = 10.0 * (1.0 + DMINUS_BONUS_AT_REFERENCE)
        assert math.isclose(result, expected)

    def test_small_descent_yields_small_bonus(self) -> None:
        # 1000 m of D- → factor = 1 + 0.1 * 0.04 = 1.004.
        result = calculate_corrected_effort(
            distance_km=10, elevation_gain_m=0, elevation_loss_m=1000
        )
        expected = 10.0 * 1.004
        assert math.isclose(result, expected)


# ----------------------------------------------------------------------------
# Type tolerance: int inputs should also work
# ----------------------------------------------------------------------------


class TestTypeTolerance:
    """The function accepts ints as well as floats (cast internally)."""

    def test_int_inputs(self) -> None:
        assert calculate_corrected_effort(distance_km=10, elevation_gain_m=500) == 15.0

    def test_mixed_int_and_float(self) -> None:
        assert calculate_corrected_effort(distance_km=10.5, elevation_gain_m=500) == 15.5


# ----------------------------------------------------------------------------
# Sanity check on module constants
# ----------------------------------------------------------------------------


def test_constants_are_what_we_expect() -> None:
    """If anyone changes the constants by mistake, this test catches it."""
    assert METERS_OF_DPLUS_PER_KM_FLAT == 100.0
    assert DMINUS_REFERENCE_M == 10_000.0
    assert DMINUS_BONUS_AT_REFERENCE == 0.04

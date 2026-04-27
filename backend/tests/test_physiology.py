"""Tests for heart rate zone calculations."""

from __future__ import annotations

import pytest
from app.domain.physiology import (
    HeartRateZone,
    HeartRateZones,
    calculate_heart_rate_zones,
)

# ----------------------------------------------------------------------------
# Reference athlete: HR_max = 190, HR_rest = 54
# Heart rate reserve = 136 bpm
# ----------------------------------------------------------------------------


class TestReferenceAthlete:
    """Verify exact zone bounds for a known reference athlete."""

    @pytest.fixture
    def zones(self) -> HeartRateZones:
        return calculate_heart_rate_zones(hr_max=190, hr_rest=54)

    def test_z1_starts_at_resting_hr(self, zones: HeartRateZones) -> None:
        # Z1 starts at the resting HR by convention.
        assert zones.z1.lower_bpm == 54

    def test_z1_upper_at_60_percent_reserve(self, zones: HeartRateZones) -> None:
        # 54 + 0.60 * 136 = 135.6 → int(135.6) = 135.
        assert zones.z1.upper_bpm == 135

    def test_z2_bounds(self, zones: HeartRateZones) -> None:
        # Z2 = 60% to 75% of the reserve.
        assert zones.z2.lower_bpm == 135
        assert zones.z2.upper_bpm == 156

    def test_z3_bounds(self, zones: HeartRateZones) -> None:
        # Z3 = 75% to 85% of the reserve.
        assert zones.z3.lower_bpm == 156
        assert zones.z3.upper_bpm == 169

    def test_z4_bounds(self, zones: HeartRateZones) -> None:
        # Z4 = 85% to 95% of the reserve.
        assert zones.z4.lower_bpm == 169
        assert zones.z4.upper_bpm == 183

    def test_z5_bounds(self, zones: HeartRateZones) -> None:
        # Z5 starts at 95% of the reserve and goes up to HR_max.
        assert zones.z5.lower_bpm == 183
        assert zones.z5.upper_bpm == 190

    def test_walking_ceiling(self, zones: HeartRateZones) -> None:
        # 54 + 0.70 * 136 = 149.2 → int(149.2) = 149.
        assert zones.walking_ceiling_bpm == 149


# ----------------------------------------------------------------------------
# Zone continuity: each zone ends where the next one starts
# ----------------------------------------------------------------------------


class TestZoneContinuity:
    """Verify that zones are contiguous — no gaps, no overlaps."""

    def test_zones_are_contiguous(self) -> None:
        zones = calculate_heart_rate_zones(hr_max=190, hr_rest=54)
        assert zones.z1.upper_bpm == zones.z2.lower_bpm
        assert zones.z2.upper_bpm == zones.z3.lower_bpm
        assert zones.z3.upper_bpm == zones.z4.lower_bpm
        assert zones.z4.upper_bpm == zones.z5.lower_bpm

    def test_continuity_holds_for_other_athletes(self) -> None:
        # Same property should hold whatever the inputs.
        for hr_max, hr_rest in [(180, 50), (200, 60), (175, 65)]:
            zones = calculate_heart_rate_zones(hr_max=hr_max, hr_rest=hr_rest)
            assert zones.z1.upper_bpm == zones.z2.lower_bpm
            assert zones.z2.upper_bpm == zones.z3.lower_bpm
            assert zones.z3.upper_bpm == zones.z4.lower_bpm
            assert zones.z4.upper_bpm == zones.z5.lower_bpm


# ----------------------------------------------------------------------------
# Input validation: invalid arguments must raise
# ----------------------------------------------------------------------------


class TestInputValidation:
    """Verify that invalid inputs are rejected with a clear error."""

    def test_zero_hr_max_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            calculate_heart_rate_zones(hr_max=0, hr_rest=54)

    def test_negative_hr_rest_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            calculate_heart_rate_zones(hr_max=190, hr_rest=-5)

    def test_hr_max_equals_hr_rest_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            calculate_heart_rate_zones(hr_max=120, hr_rest=120)

    def test_hr_max_below_hr_rest_raises(self) -> None:
        # An invalid configuration: HR_rest cannot exceed HR_max.
        with pytest.raises(ValueError, match="strictly greater"):
            calculate_heart_rate_zones(hr_max=80, hr_rest=120)


# ----------------------------------------------------------------------------
# HeartRateZone — direct schema validation
# ----------------------------------------------------------------------------


class TestHeartRateZoneSchema:
    """Verify that the HeartRateZone model rejects malformed data."""

    def test_negative_lower_bpm_is_rejected(self) -> None:
        # Pydantic should refuse negative bpm via the Field(ge=0) constraint.
        with pytest.raises(ValueError):
            HeartRateZone(name="Z1", lower_bpm=-10, upper_bpm=100)

    def test_upper_below_lower_is_rejected(self) -> None:
        # Cross-field validator should reject inverted bounds.
        with pytest.raises(ValueError, match="upper_bpm"):
            HeartRateZone(name="Z3", lower_bpm=180, upper_bpm=150)

    def test_valid_zone_can_be_created(self) -> None:
        # Sanity check: a normal zone constructs fine.
        zone = HeartRateZone(name="Z2", lower_bpm=130, upper_bpm=150)
        assert zone.name == "Z2"
        assert zone.lower_bpm == 130
        assert zone.upper_bpm == 150


# ----------------------------------------------------------------------------
# Smoke test on a different athlete profile
# ----------------------------------------------------------------------------


def test_another_athlete_profile() -> None:
    """Smoke test: a different athlete still yields sensible zones."""
    # Lower HR_max, higher HR_rest (a less fit athlete).
    zones = calculate_heart_rate_zones(hr_max=180, hr_rest=70)
    assert zones.z1.lower_bpm == 70
    assert zones.z5.upper_bpm == 180
    # Walking ceiling should land between Z2 and Z3 territory.
    assert 130 < zones.walking_ceiling_bpm < 160

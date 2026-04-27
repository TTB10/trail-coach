"""Physiological zones based on heart rate (Karvonen method).

The Karvonen formula derives target heart rate zones from the runner's
heart rate reserve (HRR), defined as:

    HRR = HR_max - HR_rest

A target heart rate at intensity p (between 0 and 1) is then:

    HR_target = HR_rest + p * HRR

Zones used here follow the standard 5-zone endurance training model:

    Z1 (recovery)            <  60% HRR
    Z2 (endurance / aerobic) 60% to 75% HRR
    Z3 (tempo)               75% to 85% HRR
    Z4 (threshold)           85% to 95% HRR
    Z5 (VO2max)              >  95% HRR
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

# Zone thresholds, expressed as fractions of the heart rate reserve. ----------

Z1_UPPER: float = 0.60
Z2_LOWER: float = 0.60
Z2_UPPER: float = 0.75
Z3_LOWER: float = 0.75
Z3_UPPER: float = 0.85
Z4_LOWER: float = 0.85
Z4_UPPER: float = 0.95
Z5_LOWER: float = 0.95

#: Suggested heart rate ceiling when walking on steep climbs.
WALKING_CEILING_FRACTION: float = 0.70


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class HeartRateZone(BaseModel):
    """A single heart rate training zone, in beats per minute."""

    name: str = Field(..., description="Short label, e.g. 'Z2'.")
    lower_bpm: int = Field(..., ge=0, description="Lower bound, inclusive.")
    upper_bpm: int = Field(..., ge=0, description="Upper bound, inclusive.")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.upper_bpm < self.lower_bpm:
            raise ValueError(
                f"upper_bpm ({self.upper_bpm}) must be >= lower_bpm ({self.lower_bpm})"
            )
        return self


class HeartRateZones(BaseModel):
    """The five Karvonen zones plus the suggested walking ceiling."""

    z1: HeartRateZone
    z2: HeartRateZone
    z3: HeartRateZone
    z4: HeartRateZone
    z5: HeartRateZone
    walking_ceiling_bpm: int = Field(
        ..., ge=0, description="Recommended HR ceiling on steep walking sections."
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def calculate_heart_rate_zones(hr_max: int, hr_rest: int) -> HeartRateZones:
    """Compute the five Karvonen training zones for a runner.

    Args:
        hr_max: Maximum heart rate, in bpm. Typically 160-210.
        hr_rest: Resting heart rate, in bpm. Typically 40-80.

    Returns:
        Five contiguous heart rate zones (Z1-Z5) plus the suggested walking
        ceiling, all expressed in bpm.

    Raises:
        ValueError: If hr_max <= hr_rest, or if either value is non-positive.

    Examples:
        >>> zones = calculate_heart_rate_zones(hr_max=190, hr_rest=54)
        >>> zones.z2.lower_bpm
        135
        >>> zones.z2.upper_bpm
        156
    """
    if hr_max <= 0 or hr_rest <= 0:
        raise ValueError(f"Heart rates must be positive (hr_max={hr_max}, hr_rest={hr_rest}).")
    if hr_max <= hr_rest:
        raise ValueError(f"hr_max ({hr_max}) must be strictly greater than hr_rest ({hr_rest}).")

    reserve = hr_max - hr_rest

    def at(fraction: float) -> int:
        """Return the bpm corresponding to a given fraction of the reserve."""
        return int(hr_rest + fraction * reserve)

    return HeartRateZones(
        z1=HeartRateZone(name="Z1", lower_bpm=hr_rest, upper_bpm=at(Z1_UPPER)),
        z2=HeartRateZone(name="Z2", lower_bpm=at(Z2_LOWER), upper_bpm=at(Z2_UPPER)),
        z3=HeartRateZone(name="Z3", lower_bpm=at(Z3_LOWER), upper_bpm=at(Z3_UPPER)),
        z4=HeartRateZone(name="Z4", lower_bpm=at(Z4_LOWER), upper_bpm=at(Z4_UPPER)),
        z5=HeartRateZone(name="Z5", lower_bpm=at(Z5_LOWER), upper_bpm=hr_max),
        walking_ceiling_bpm=at(WALKING_CEILING_FRACTION),
    )

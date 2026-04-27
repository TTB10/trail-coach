"""Effort calculation for ultra-trail courses.

The "equivalent effort" score reduces a course (distance + elevation gain/loss)
to a single number representing how much it would cost on flat ground.

The rule of thumb used here is the standard Daniels-style heuristic:
100 meters of positive elevation gain ≈ 1 km of flat running.
A small bonus is added for steep descents, which also fatigue the legs.
"""

from __future__ import annotations

# Conversion factors ----------------------------------------------------------

#: 100 m of D+ is equivalent to 1 km of flat running.
METERS_OF_DPLUS_PER_KM_FLAT: float = 100.0

#: Below 10000 m of D-, the descent bonus scales linearly. Empirical.
DMINUS_REFERENCE_M: float = 10_000.0

#: At DMINUS_REFERENCE_M of D-, the effort is multiplied by 1.04 (a 4% bonus).
DMINUS_BONUS_AT_REFERENCE: float = 0.04


def calculate_corrected_effort(
    distance_km: float,
    elevation_gain_m: float,
    elevation_loss_m: float = 0.0,
) -> float:
    """Return the equivalent flat-running effort for a course.

    The score combines distance and elevation:
        effort = distance + D+ / 100
    A small bonus is then applied if the course also has significant D-:
        effort *= 1 + (D- / 10000) * 0.04

    Negative inputs are clamped to zero — they are physically meaningless.

    Args:
        distance_km: Total distance in kilometers. Must be >= 0.
        elevation_gain_m: Total positive elevation in meters. Must be >= 0.
        elevation_loss_m: Total negative elevation in meters (absolute value).
            Defaults to 0. Must be >= 0.

    Returns:
        The equivalent effort, in "flat kilometers".

    Examples:
        >>> calculate_corrected_effort(distance_km=10, elevation_gain_m=0)
        10.0
        >>> calculate_corrected_effort(distance_km=10, elevation_gain_m=500)
        15.0
        >>> # Negative values are clamped:
        >>> calculate_corrected_effort(distance_km=-5, elevation_gain_m=100)
        1.0
    """
    distance = max(float(distance_km), 0.0)
    dplus = max(float(elevation_gain_m), 0.0)
    dminus = max(float(elevation_loss_m), 0.0)

    effort = distance + (dplus / METERS_OF_DPLUS_PER_KM_FLAT)

    if dminus > 0:
        descent_factor = 1.0 + (dminus / DMINUS_REFERENCE_M) * DMINUS_BONUS_AT_REFERENCE
        effort *= descent_factor

    return effort

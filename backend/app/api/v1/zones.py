"""Heart rate zones endpoint — exposes the Karvonen calculator over HTTP."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.domain.physiology import HeartRateZones, calculate_heart_rate_zones

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("/heart-rate", response_model=HeartRateZones)
def get_heart_rate_zones(
    hr_max: int = Query(..., gt=0, le=250, description="Maximum heart rate, in bpm."),
    hr_rest: int = Query(..., gt=0, le=150, description="Resting heart rate, in bpm."),
) -> HeartRateZones:
    """Compute Karvonen heart rate training zones.

    Returns the five training zones (Z1-Z5) plus a recommended walking
    ceiling, based on the heart rate reserve method.
    """
    return calculate_heart_rate_zones(hr_max=hr_max, hr_rest=hr_rest)

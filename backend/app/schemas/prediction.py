"""HTTP schemas for the prediction endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.domain.predictor import ExperienceLevel, StrengthTrainingFrequency


class PredictionRequest(BaseModel):
    """Inputs needed to predict a race finish time.

    The frontend sends physical race parameters (distance, elevation) and
    runner state (CTL, experience). The endpoint translates them into the
    "equivalent effort" that the prediction model needs.
    """

    # Personal record (the reference performance)
    record_time_min: float = Field(..., gt=0, description="Personal record time in minutes.")
    record_distance_km: float = Field(..., gt=0, description="Distance of the record course in km.")
    record_elevation_gain_m: float = Field(
        ..., ge=0, description="Positive elevation gain of the record course in meters."
    )

    # Target race
    target_distance_km: float = Field(..., gt=0, description="Distance of the target race in km.")
    target_elevation_gain_m: float = Field(
        ..., ge=0, description="Positive elevation gain of the target race in meters."
    )

    # Athlete profile
    experience: ExperienceLevel = Field(
        default=ExperienceLevel.AMATEUR,
        description="Athlete's overall ultra-trail experience level.",
    )
    strength_training: StrengthTrainingFrequency = Field(
        default=StrengthTrainingFrequency.NEVER,
        description="Strength training frequency.",
    )

    # Physiological state
    current_ctl: float = Field(..., ge=0, description="Current chronic training load.")
    record_ctl: float = Field(..., ge=0, description="CTL at the time of the personal record.")
    projected_ctl: float = Field(..., ge=0, description="Expected CTL on race day (after taper).")
